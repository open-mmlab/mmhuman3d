import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from smplx.utils import find_joint_kin_chain

from mmhuman3d.core.conventions.keypoints_mapping import (
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.utils.geometry import weak_perspective_projection


class SMPLXHandMergeFunc():
    """This function use predictions from hand model to update the hand params
    (right_hand_pose, left_hand_pose, wrist_pose) in predictions from body
    model."""

    def __init__(self, body_model, convention='smplx'):
        self.body_model = body_model
        self.convention = convention
        self.left_hand_idxs = get_keypoint_idxs_by_part(
            'left_hand', self.convention)
        self.left_wrist_idx = get_keypoint_idx('left_wrist', self.convention)
        self.left_hand_idxs.append(self.left_wrist_idx)
        self.left_wrist_kin_chain = find_joint_kin_chain(
            self.left_wrist_idx, self.body_model.parents)

        self.right_hand_idxs = get_keypoint_idxs_by_part(
            'right_hand', self.convention)
        self.right_wrist_idx = get_keypoint_idx('right_wrist', self.convention)
        self.right_hand_idxs.append(self.right_wrist_idx)
        self.right_wrist_kin_chain = find_joint_kin_chain(
            self.right_wrist_idx, self.body_model.parents)

    def __call__(self, body_predictions, hand_predictions):
        """Function
        Args:
            body_predictions (dict): The prediction from body model.
            hand_predictions (dict): The prediction from hand model.
        Returns:
            dict: Merged prediction.
        """
        pred_param = body_predictions['pred_param']
        global_orient = pred_param['global_orient']
        body_pose = pred_param['body_pose']
        pred_cam = body_predictions['pred_cam']
        batch_size = pred_cam.shape[0]
        device = pred_cam.device
        hands_from_body_idxs = torch.arange(
            0, 2 * batch_size, dtype=torch.long, device=device)
        right_hand_from_body_idxs = hands_from_body_idxs[:batch_size]
        left_hand_from_body_idxs = hands_from_body_idxs[batch_size:]

        parent_rots = []
        right_wrist_parent_rot = find_joint_global_rotation(
            self.right_wrist_kin_chain[1:], global_orient, body_pose)

        left_wrist_parent_rot = find_joint_global_rotation(
            self.left_wrist_kin_chain[1:], global_orient, body_pose)
        left_to_right_wrist_parent_rot = flip_rotmat(left_wrist_parent_rot)

        parent_rots += [right_wrist_parent_rot, left_to_right_wrist_parent_rot]
        parent_rots = torch.cat(parent_rots, dim=0)

        wrist_pose_from_hand = hand_predictions['pred_param']['global_orient']
        # Undo the rotation of the parent joints to make the wrist rotation
        # relative again
        wrist_pose_from_hand = torch.matmul(
            parent_rots.reshape(-1, 3, 3).transpose(1, 2),
            wrist_pose_from_hand.reshape(-1, 3, 3))

        right_hand_wrist = wrist_pose_from_hand[right_hand_from_body_idxs]
        left_hand_wrist = flip_rotmat(
            wrist_pose_from_hand[left_hand_from_body_idxs])
        right_hand_pose = hand_predictions['pred_param']['right_hand_pose'][
            right_hand_from_body_idxs]
        left_hand_pose = flip_rotmat(
            hand_predictions['pred_param']['right_hand_pose']
            [left_hand_from_body_idxs])

        body_predictions['pred_param']['right_hand_pose'] = right_hand_pose
        body_predictions['pred_param']['left_hand_pose'] = left_hand_pose
        body_predictions['pred_param']['body_pose'][:, self.right_wrist_idx -
                                                    1] = right_hand_wrist
        body_predictions['pred_param']['body_pose'][:, self.left_wrist_idx -
                                                    1] = left_hand_wrist

        return body_predictions


class SMPLXFaceMergeFunc():
    """This function use predictions from face model to update the face params
    (jaw_pose, expression) in predictions from body model."""

    def __init__(self,
                 body_model,
                 convention='smplx',
                 num_expression_coeffs=10):
        self.body_model = body_model
        self.convention = convention
        self.num_expression_coeffs = num_expression_coeffs

    def __call__(self, body_predictions, face_predictions):
        """Function
        Args:
            body_predictions (dict): The prediction from body model.
            face_predictions (dict): The prediction from face model.
        Returns:
            dict: Merged prediction.
        """
        body_predictions['pred_param']['jaw_pose'] = face_predictions[
            'pred_param']['jaw_pose']
        body_predictions['pred_param']['expression'] = face_predictions[
            'pred_param']['expression'][:, :self.num_expression_coeffs]
        return body_predictions


def points_to_bbox(points, bbox_scale_factor: float = 1.0):
    """Get scaled bounding box from keypoints 2D."""
    min_coords, _ = torch.min(points, dim=1)
    xmin, ymin = min_coords[:, 0], min_coords[:, 1]
    max_coords, _ = torch.max(points, dim=1)
    xmax, ymax = max_coords[:, 0], max_coords[:, 1]

    center = torch.stack([xmax + xmin, ymax + ymin], dim=-1) * 0.5

    width = (xmax - xmin)
    height = (ymax - ymin)

    # Convert the bounding box to a square box
    size = torch.max(width, height) * bbox_scale_factor

    return center, size


def get_crop_info(points,
                  img_metas,
                  scale_factor: float = 1.0,
                  crop_size: int = 256):
    """Get the transformation of points on the cropped image to the points on
    the original image."""
    device = points.device
    dtype = points.dtype
    batch_size = points.shape[0]
    # Get the image to crop transformations and bounding box sizes
    crop_transforms = []
    img_bbox_sizes = []
    for img_meta in img_metas:
        crop_transforms.append(img_meta['crop_transform'])
        img_bbox_sizes.append(img_meta['scale'].max())

    img_bbox_sizes = torch.tensor(img_bbox_sizes, dtype=dtype, device=device)

    crop_transforms = torch.tensor(crop_transforms, dtype=dtype, device=device)

    crop_transforms = torch.cat([
        crop_transforms,
        torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).expand(
            [batch_size, 1, 3])
    ],
                                dim=1)

    inv_crop_transforms = torch.inverse(crop_transforms)

    # center on the cropped body image
    center_body_crop, bbox_size = points_to_bbox(
        points, bbox_scale_factor=scale_factor)

    orig_bbox_size = bbox_size / crop_size * img_bbox_sizes

    # Compute the center of the crop in the original image
    center = (
        torch.einsum('bij,bj->bi',
                     [inv_crop_transforms[:, :2, :2], center_body_crop]) +
        inv_crop_transforms[:, :2, 2])

    return {
        'center': center.reshape(-1, 2),
        'orig_bbox_size': orig_bbox_size,
        # 'bbox_size': bbox_size.reshape(-1),
        'inv_crop_transforms': inv_crop_transforms,
        # 'center_body_crop': 2 * center_body_crop / (crop_size-1) - 1,
    }


def concat_images(images: List[torch.Tensor]):
    """Concat images of different size."""
    sizes = [img.shape[1:] for img in images]
    H, W = [max(s) for s in zip(*sizes)]
    batch_size = len(images)
    batched_shape = (batch_size, images[0].shape[0], H, W)
    batched = torch.zeros(
        batched_shape, device=images[0].device, dtype=images[0].dtype)
    for ii, img in enumerate(images):
        shape = img.shape
        batched[ii, :shape[0], :shape[1], :shape[2]] = img
    return batched


def flip_rotmat(pose_rotmat):
    """Flip function.

    Flip rotmat.
    """
    rot_mats = pose_rotmat.reshape(-1, 9).clone()

    rot_mats[:, [1, 2, 3, 6]] *= -1
    return rot_mats.view_as(pose_rotmat)


def find_joint_global_rotation(kin_chain, root_pose, body_pose):
    """Computes the absolute rotation of a joint from the kinematic chain."""
    # Create a single vector with all the poses
    parents_pose = torch.cat([root_pose, body_pose], dim=1)[:, kin_chain]
    output_pose = parents_pose[:, 0]
    for idx in range(1, parents_pose.shape[1]):
        output_pose = torch.bmm(parents_pose[:, idx], output_pose)
    return output_pose


class CropSampler():
    """This function crops the HD images using bilinear interpolation."""

    def __init__(self, crop_size: int = 256) -> None:
        """Uses bilinear sampling to extract square crops.

        This module expects a high resolution image as input and a bounding
        box, described by its' center and size. It then proceeds to extract
        a sub-image using the provided information through bilinear
        interpolation.

        Parameters
        ----------
            crop_size: int
                The desired size for the crop.
        """
        super(CropSampler, self).__init__()

        self.crop_size = crop_size
        x = torch.arange(0, crop_size, dtype=torch.float32) / (crop_size - 1)
        grid_y, grid_x = torch.meshgrid(x, x)

        points = torch.stack([grid_y.flatten(), grid_x.flatten()], axis=1)

        self.grid = points.unsqueeze(dim=0)

    def _sample_padded(self, full_imgs, sampling_grid):
        """"""
        # Get the sub-images using bilinear interpolation
        return F.grid_sample(full_imgs, sampling_grid, align_corners=True)

    def __call__(self, full_imgs, center, bbox_size):
        """Crops the HD images using the provided bounding boxes.

        Parameters
        ----------
            full_imgs: ImageList
                An image list structure with the full resolution images
            center: torch.Tensor
                A Bx2 tensor that contains the coordinates of the center of
                the bounding box that will be cropped from the original
                image
            bbox_size: torch.Tensor
                A size B tensor that contains the size of the corp

        Returns
        -------
            cropped_images: torch.Tensoror
                The images cropped from the high resolution input
            sampling_grid: torch.Tensor
                The grid used to sample the crops
        """

        batch_size, _, H, W = full_imgs.shape
        self.grid = self.grid.to(device=full_imgs.device)
        transforms = torch.eye(
            3, dtype=full_imgs.dtype,
            device=full_imgs.device).reshape(1, 3,
                                             3).expand(batch_size, -1,
                                                       -1).contiguous()

        hd_to_crop = torch.eye(
            3, dtype=full_imgs.dtype,
            device=full_imgs.device).reshape(1, 3,
                                             3).expand(batch_size, -1,
                                                       -1).contiguous()

        # Create the transformation that maps crop pixels to image coordinates,
        # i.e. pixel (0, 0) from the crop_size x crop_size grid gets mapped to
        # the top left of the bounding box, pixel
        # (crop_size - 1, crop_size - 1) to the bottom right corner of the
        # bounding box
        transforms[:, 0, 0] = bbox_size  # / (self.crop_size - 1)
        transforms[:, 1, 1] = bbox_size  # / (self.crop_size - 1)
        transforms[:, 0, 2] = center[:, 0] - bbox_size * 0.5
        transforms[:, 1, 2] = center[:, 1] - bbox_size * 0.5

        hd_to_crop[:, 0, 0] = 2 * (self.crop_size - 1) / bbox_size
        hd_to_crop[:, 1, 1] = 2 * (self.crop_size - 1) / bbox_size
        hd_to_crop[:, 0,
                   2] = -(center[:, 0] - bbox_size * 0.5) * hd_to_crop[:, 0,
                                                                       0] - 1
        hd_to_crop[:, 1,
                   2] = -(center[:, 1] - bbox_size * 0.5) * hd_to_crop[:, 1,
                                                                       1] - 1

        size_bbox_sizer = torch.eye(
            3, dtype=full_imgs.dtype,
            device=full_imgs.device).reshape(1, 3,
                                             3).expand(batch_size, -1,
                                                       -1).contiguous()

        # Normalize the coordinates to [-1, 1] for the grid_sample function
        size_bbox_sizer[:, 0, 0] = 2.0 / (W - 1)
        size_bbox_sizer[:, 1, 1] = 2.0 / (H - 1)
        size_bbox_sizer[:, :2, 2] = -1

        #  full_transform = transforms
        full_transform = torch.bmm(size_bbox_sizer, transforms)

        batch_grid = self.grid.expand(batch_size, -1, -1)
        # Convert the grid to image coordinates using the transformations above
        sampling_grid = (
            torch.bmm(full_transform[:, :2, :2], batch_grid.transpose(1, 2)) +
            full_transform[:, :2, [2]]).transpose(1, 2)
        sampling_grid = sampling_grid.reshape(-1, self.crop_size,
                                              self.crop_size,
                                              2).transpose(1, 2)

        out_images = self._sample_padded(full_imgs, sampling_grid)

        return {
            'images': out_images,
            'sampling_grid': sampling_grid.reshape(batch_size, -1, 2),
            'transform': transforms,
            'hd_to_crop': hd_to_crop,
        }


class SMPLXHandCropFunc():
    """This function crop hand image from the original image.

    Use the output keypoints predicted by the body model to locate the hand
    position.
    """

    def __init__(self,
                 model_head,
                 body_model,
                 convention='smplx',
                 img_res=256,
                 scale_factor=2.0,
                 crop_size=224,
                 condition_hand_wrist_pose=True,
                 condition_hand_shape=False,
                 condition_hand_finger_pose=True):
        self.model_head = model_head
        self.body_model = body_model
        self.img_res = img_res
        self.convention = convention
        self.left_hand_idxs = get_keypoint_idxs_by_part(
            'left_hand', self.convention)
        left_wrist_idx = get_keypoint_idx('left_wrist', self.convention)
        self.left_hand_idxs.append(left_wrist_idx)
        self.left_wrist_kin_chain = find_joint_kin_chain(
            left_wrist_idx, self.body_model.parents)

        self.right_hand_idxs = get_keypoint_idxs_by_part(
            'right_hand', self.convention)
        right_wrist_idx = get_keypoint_idx('right_wrist', self.convention)
        self.right_hand_idxs.append(right_wrist_idx)
        self.right_wrist_kin_chain = find_joint_kin_chain(
            right_wrist_idx, self.body_model.parents)

        self.scale_factor = scale_factor
        self.hand_cropper = CropSampler(crop_size)

        self.condition_hand_wrist_pose = condition_hand_wrist_pose
        self.condition_hand_shape = condition_hand_shape
        self.condition_hand_finger_pose = condition_hand_finger_pose

    def build_hand_mean(self, global_orient, body_pose, betas, left_hand_pose,
                        raw_right_hand_pose, batch_size):
        """Builds the initial point for the iterative regressor of the hand."""
        hand_mean = []

        #  if self.condition_hand_on_body:
        # Convert the absolute pose to the latent representation
        if self.condition_hand_wrist_pose:
            # Compute the absolute pose of the right wrist
            right_wrist_pose_abs = find_joint_global_rotation(
                self.right_wrist_kin_chain, global_orient, body_pose)
            right_wrist_pose = right_wrist_pose_abs[:, :3, :2].contiguous(
            ).reshape(batch_size, -1)

            # Compute the absolute rotation for the left wrist
            left_wrist_pose_abs = find_joint_global_rotation(
                self.left_wrist_kin_chain, global_orient, body_pose)
            # Flip the left wrist to the right
            left_to_right_wrist_pose = flip_rotmat(left_wrist_pose_abs)

            # Convert to the latent representation
            left_to_right_wrist_pose = left_to_right_wrist_pose[:, :3, :
                                                                2].contiguous(
                                                                ).reshape(
                                                                    batch_size,
                                                                    -1)
        else:
            right_wrist_pose = self.model_head.get_mean(
                'global_orient', batch_size=batch_size)
            left_to_right_wrist_pose = self.model_head.get_mean(
                'global_orient', batch_size=batch_size)

        # Convert the pose of the left hand to the right hand and project
        # it to the encoder space
        left_to_right_hand_pose = flip_rotmat(
            left_hand_pose)[:, :, :3, :2].contiguous().reshape(batch_size, -1)
        right_hand_pose = raw_right_hand_pose.reshape(batch_size, -1)
        camera_mean = self.model_head.get_mean('camera', batch_size=batch_size)

        shape_condition = (
            betas if self.condition_hand_shape else self.model_head.get_mean(
                'shape', batch_size=batch_size))
        right_finger_pose_condition = (
            right_hand_pose if self.condition_hand_finger_pose else
            self.model_head.get_mean('right_hand_pose', batch_size=batch_size))
        right_hand_mean = torch.cat([
            right_wrist_pose, right_finger_pose_condition, shape_condition,
            camera_mean
        ],
                                    dim=1)

        left_finger_pose_condition = (
            left_to_right_hand_pose if self.condition_hand_finger_pose else
            self.model_head.get_mean('right_hand_pose', batch_size=batch_size))
        # Should be Bx31
        left_hand_mean = torch.cat([
            left_to_right_wrist_pose, left_finger_pose_condition,
            shape_condition, camera_mean
        ],
                                   dim=1)

        hand_mean += [right_hand_mean, left_hand_mean]
        hand_mean = torch.cat(hand_mean, dim=0)

        return hand_mean

    def __call__(self, body_predictions, img_metas):
        """Function
        Args:
            body_predictions (dict): The prediction from body model.
            img_metas (dict): Information of the input images.
        Returns:
            all_hand_imgs (torch.tensor): Cropped hand images.
            hand_mean (torch.tensor): Mean value of hand params.
            crop_info (dict): Hand crop transforms.
        """
        pred_param = body_predictions['pred_param']
        pred_cam = body_predictions['pred_cam']
        pred_raw = body_predictions['pred_raw']
        pred_output = self.body_model(**pred_param)

        pred_keypoints3d = pred_output['joints']
        pred_keypoints2d = weak_perspective_projection(
            pred_keypoints3d,
            scale=pred_cam[:, 0],
            translation=pred_cam[:, 1:3])
        # concat ori_img
        full_images = []
        for img_meta in img_metas:
            full_images.append(img_meta['ori_img'].to(device=pred_cam.device))
        full_imgs = concat_images(full_images)

        # left hand
        left_hand_joints = (pred_keypoints2d[:, self.left_hand_idxs] * 0.5 +
                            0.5) * (
                                self.img_res - 1)
        left_hand_points_to_crop = get_crop_info(left_hand_joints, img_metas,
                                                 self.scale_factor,
                                                 self.img_res)
        left_hand_center = left_hand_points_to_crop['center']
        left_hand_orig_bbox_size = left_hand_points_to_crop['orig_bbox_size']
        left_hand_inv_crop_transforms = left_hand_points_to_crop[
            'inv_crop_transforms']

        left_hand_cropper_out = self.hand_cropper(full_imgs, left_hand_center,
                                                  left_hand_orig_bbox_size)
        left_hand_crops = left_hand_cropper_out['images']
        # left_hand_points = left_hand_cropper_out['sampling_grid']
        left_hand_crop_transform = left_hand_cropper_out['transform']

        # right hand
        right_hand_joints = (pred_keypoints2d[:, self.right_hand_idxs] * 0.5 +
                             0.5) * (
                                 self.img_res - 1)
        right_hand_points_to_crop = get_crop_info(right_hand_joints, img_metas,
                                                  self.scale_factor,
                                                  self.img_res)
        right_hand_center = right_hand_points_to_crop['center']
        right_hand_orig_bbox_size = right_hand_points_to_crop['orig_bbox_size']
        # right_hand_inv_crop_transforms = right_hand_points_to_crop[
        #     'inv_crop_transforms']
        right_hand_cropper_out = self.hand_cropper(full_imgs,
                                                   right_hand_center,
                                                   right_hand_orig_bbox_size)
        right_hand_crops = right_hand_cropper_out['images']
        # right_hand_points = right_hand_cropper_out['sampling_grid']
        right_hand_crop_transform = right_hand_cropper_out['transform']

        # concat
        all_hand_imgs = []
        all_hand_imgs.append(right_hand_crops)
        all_hand_imgs.append(torch.flip(left_hand_crops, dims=(-1, )))

        # [right_hand , left hand]
        all_hand_imgs = torch.cat(all_hand_imgs, dim=0)
        hand_mean = self.build_hand_mean(
            pred_param['global_orient'],
            pred_param['body_pose'],
            pred_param['betas'],
            pred_param['left_hand_pose'],
            pred_raw['raw_right_hand_pose'],
            batch_size=full_imgs.shape[0])
        crop_info = dict(
            hand_inv_crop_transforms=left_hand_inv_crop_transforms,
            left_hand_crop_transform=left_hand_crop_transform,
            right_hand_crop_transform=right_hand_crop_transform)
        return all_hand_imgs, hand_mean, crop_info


class SMPLXFaceCropFunc():
    """This function crop face image from the original image.

    Use the output keypoints predicted by the facce model to locate the face
    position.
    """

    def __init__(self,
                 model_head,
                 body_model,
                 convention='smplx',
                 img_res=256,
                 scale_factor=2.0,
                 crop_size=256,
                 num_betas=10,
                 num_expression_coeffs=10,
                 condition_face_neck_pose=False,
                 condition_face_jaw_pose=True,
                 condition_face_shape=False,
                 condition_face_expression=True):
        self.model_head = model_head
        self.body_model = body_model
        self.img_res = img_res
        self.convention = convention
        self.num_betas = num_betas
        self.num_expression_coeffs = num_expression_coeffs

        self.face_idx = get_keypoint_idxs_by_part('head', self.convention)
        neck_idx = get_keypoint_idx('neck', self.convention)
        self.neck_kin_chain = find_joint_kin_chain(neck_idx,
                                                   self.body_model.parents)

        self.condition_face_neck_pose = condition_face_neck_pose
        self.condition_face_jaw_pose = condition_face_jaw_pose
        self.condition_face_shape = condition_face_shape
        self.condition_face_expression = condition_face_expression

        self.scale_factor = scale_factor
        self.face_cropper = CropSampler(crop_size)

    def build_face_mean(self, global_orient, body_pose, betas, raw_jaw_pose,
                        expression, batch_size):
        """Builds the initial point for the iterative regressor of the face."""
        face_mean = []
        # Compute the absolute pose of the right wrist
        neck_pose_abs = find_joint_global_rotation(self.neck_kin_chain,
                                                   global_orient, body_pose)
        # Convert the absolute neck pose to offsets
        neck_pose = neck_pose_abs[:, :3, :2].contiguous().reshape(
            batch_size, -1)

        camera_mean = self.model_head.get_mean('camera', batch_size=batch_size)

        neck_pose_condition = (
            neck_pose if self.condition_face_neck_pose else
            self.model_head.get_mean('global_orient', batch_size=batch_size))

        jaw_pose_condition = (
            raw_jaw_pose.reshape(batch_size, -1)
            if self.condition_face_jaw_pose else self.model_head.get_mean(
                'jaw_pose', batch_size=batch_size))
        face_num_betas = self.model_head.get_num_betas()
        shape_padding_size = face_num_betas - self.num_betas
        betas_condition = (
            F.pad(betas.reshape(batch_size, -1),
                  (0, shape_padding_size)) if self.condition_face_shape else
            self.model_head.get_mean('shape', batch_size=batch_size))

        face_num_expression_coeffs = self.model_head.get_num_expression_coeffs(
        )
        expr_padding_size = face_num_expression_coeffs \
            - self.num_expression_coeffs
        expression_condition = (
            F.pad(expression.reshape(batch_size, -1),
                  (0, expr_padding_size)) if self.condition_face_expression
            else self.model_head.get_mean('expression', batch_size=batch_size))

        # Should be Bx(Head pose params)
        face_mean.append(
            torch.cat([
                neck_pose_condition,
                jaw_pose_condition,
                betas_condition,
                expression_condition,
                camera_mean.reshape(batch_size, -1),
            ],
                      dim=1))

        face_mean = torch.cat(face_mean, dim=0)
        return face_mean

    def __call__(self, body_predictions, img_metas):
        """Function
        Args:
            body_predictions (dict): The prediction from body model.
            img_metas (dict): Information of the input images.
        Returns:
            all_face_imgs (torch.tensor): Cropped face images.
            face_mean (torch.tensor): Mean value of face params.
            crop_info (dict): Face crop transforms.
        """
        pred_param = body_predictions['pred_param']
        pred_cam = body_predictions['pred_cam']
        pred_raw = body_predictions['pred_raw']

        pred_output = self.body_model(**pred_param)

        pred_keypoints3d = pred_output['joints']
        pred_keypoints2d = weak_perspective_projection(
            pred_keypoints3d,
            scale=pred_cam[:, 0],
            translation=pred_cam[:, 1:3])
        # concat ori_img
        full_images = []
        for img_meta in img_metas:
            full_images.append(img_meta['ori_img'].to(device=pred_cam.device))
        full_imgs = concat_images(full_images)

        face_joints = (pred_keypoints2d[:, self.face_idx] * 0.5 + 0.5) * (
            self.img_res - 1)
        face_points_to_crop = get_crop_info(face_joints, img_metas,
                                            self.scale_factor, self.img_res)
        face_center = face_points_to_crop['center']
        face_orig_bbox_size = face_points_to_crop['orig_bbox_size']
        face_inv_crop_transforms = face_points_to_crop['inv_crop_transforms']

        face_cropper_out = self.face_cropper(full_imgs, face_center,
                                             face_orig_bbox_size)
        face_crops = face_cropper_out['images']
        # face_points = face_cropper_out['sampling_grid']
        face_crop_transform = face_cropper_out['transform']

        all_face_imgs = [face_crops]
        all_face_imgs = torch.cat(all_face_imgs, dim=0)

        face_mean = self.build_face_mean(
            pred_param['global_orient'],
            pred_param['body_pose'],
            pred_param['betas'],
            pred_raw['raw_jaw_pose'],
            pred_param['expression'],
            batch_size=full_imgs.shape[0])
        crop_info = dict(
            face_inv_crop_transforms=face_inv_crop_transforms,
            face_crop_transform=face_crop_transform)
        return all_face_imgs, face_mean, crop_info


def get_partial_smpl(partial_mesh_path: str = 'data/partial_mesh/'):
    """Get partial mesh of SMPL.

    Returns:
        part_vert_faces
    """
    part_vert_faces = {}

    for part in [
            'lhand', 'rhand', 'face', 'arm', 'forearm', 'larm', 'rarm',
            'lwrist', 'rwrist'
    ]:
        part_vid_fname = os.path.join(partial_mesh_path,
                                      f'smpl_{part}_vids.npz')
        if os.path.exists(part_vid_fname):
            part_vids = np.load(part_vid_fname)
            part_vert_faces[part] = {
                'vids': part_vids['vids'],
                'faces': part_vids['faces']
            }
        else:
            raise FileNotFoundError(f'{part_vid_fname} does not exist!')
    return part_vert_faces
