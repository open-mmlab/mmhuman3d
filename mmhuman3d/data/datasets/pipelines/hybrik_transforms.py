import math
import random

import cv2
import mmcv
import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import get_flip_pairs
from mmhuman3d.utils.demo_utils import box2cs, xyxy2xywh
from ..builder import PIPELINES
from .transforms import (
    _rotate_smpl_pose,
    affine_transform,
    get_affine_transform,
)


def get_bbox(bbox_xywh, w, h):
    """Obtain bbox in xyxy format given bbox in xywh format and applying
    clipping to ensure bbox is within image bounds.

    Args:
        xywh (list): bbox in format (x, y, w, h).
        w (int): image width
        h (int): image height

    Returns:
        xyxy (numpy.ndarray): Converted bboxes in format (xmin, ymin,
         xmax, ymax).
    """
    bbox_xywh = bbox_xywh.reshape(1, 4)
    xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(bbox_xywh), w, h)
    bbox = np.array([xmin, ymin, xmax, ymax])
    return bbox


def heatmap2coord(pred_jts,
                  pred_scores,
                  hm_shape,
                  bbox,
                  output_3d=False,
                  mean_bbox_scale=None):
    """Retrieve predicted keypoints and scores from heatmap."""
    hm_width, hm_height = hm_shape

    ndims = pred_jts.dim()
    assert ndims in [2, 3], 'Dimensions of input heatmap should be 2 or 3'
    if ndims == 2:
        pred_jts = pred_jts.unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0)

    coords = pred_jts.cpu().numpy()
    coords = coords.astype(float)
    pred_scores = pred_scores.cpu().numpy()
    pred_scores = pred_scores.astype(float)

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    preds = np.zeros_like(coords)
    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            preds[i, j, 0:2] = transform_preds(coords[i, j, 0:2], center,
                                               scale, [hm_width, hm_height])
            if output_3d:
                if mean_bbox_scale is not None:
                    zscale = scale[0] / mean_bbox_scale
                    preds[i, j, 2] = coords[i, j, 2] / zscale
                else:
                    preds[i, j, 2] = coords[i, j, 2]
    # maxvals = np.ones((*preds.shape[:2], 1), dtype=float)
    # score_mul = 1 if norm_name == 'sigmoid' else 5

    return preds, pred_scores


def transform_preds(coords, center, scale, output_size):
    """Transform heatmap coordinates to image coordinates."""
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(
        center, scale, 0, output_size, inv=1, pixel_std=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (x, y, w, h) to (xmin, ymin, xmax,
    ymax)

    Args:
        xywh (list, tuple or numpy.ndarray): bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns:
        xyxy (tuple or numpy.ndarray): Converted bboxes in format (xmin, ymin,
         xmax, ymax). Return numpy.ndarray if input is in the same format.
    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                'Bounding boxes must have 4 elements, given {}'.format(
                    len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return (xywh[0], xywh[1], xywh[0] + w, xywh[1] + h)
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(
                'Bounding boxes must have n * 4 elements, given {}'.format(
                    xywh.shape))
        xyxy = np.hstack(
            (xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.
            format(type(xywh)))


def bbox_clip_xyxy(xyxy, width, height):
    """Clip bounding box with format (xmin, ymin, xmax, ymax) to `(0, 0, width,
    height)`.

    Args:
    xyxy (list, tuple or numpy.ndarray): bbox in format (xmin, ymin,
     xmax, ymax). If numpy.ndarray is provided, we expect multiple bounding
     boxes with shape `(N, 4)`.
    width (int or float): Boundary width.
    height (int or float): Boundary height.

    Returns:
    xyxy (list, tuple or numpy.ndarray): clipped bbox in format (xmin, ymin,
     xmax, ymax) and input type
    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                'Bounding boxes must have 4 elements, given {}'.format(
                    len(xyxy)))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
        return (x1, y1, x2, y2)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                'Bounding boxes must have n * 4 elements, given {}'.format(
                    xyxy.shape))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[:, 0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[:, 1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[:, 2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[:, 3]))
        return np.hstack((x1, y1, x2, y2))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.
            format(type(xyxy)))


def cam2pixel(cam_coord, f, c):
    """Convert coordinates from camera to image frame given f and c
    Args:
        cam_coord (np.ndarray): Coordinates in camera frame
        f (list): focal length, fx, fy
        c (list): principal point offset, x0, y0

    Returns:
        img_coord (np.ndarray): Coordinates in image frame
    """

    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord


def get_intrinsic_matrix(f, c, inv=False):
    """Get intrisic matrix (or its inverse) given f and c.
    Args:
        f (list): focal length, fx, fy
        c (list): principal point offset, x0, y0
        inv (bool): Store True to get inverse. Default: False.

    Returns:
        intrinsic matrix (np.ndarray): 3x3 intrinsic matrix or its inverse
    """
    intrinsic_metrix = np.zeros((3, 3)).astype(np.float32)
    intrinsic_metrix[0, 0] = f[0]
    intrinsic_metrix[0, 2] = c[0]
    intrinsic_metrix[1, 1] = f[1]
    intrinsic_metrix[1, 2] = c[1]
    intrinsic_metrix[2, 2] = 1

    if inv:
        intrinsic_metrix = np.linalg.inv(intrinsic_metrix).astype(np.float32)
    return intrinsic_metrix


def aa_to_quat_numpy(axis_angle):
    """Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a np.ndarray of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as np.ndarray of shape (..., 4).
    """
    angles = np.linalg.norm(axis_angle, ord=2, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles])
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48)
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        axis=-1)
    return quaternions


def flip_thetas(thetas, theta_pairs):
    """Flip thetas.

    Args:
        thetas (np.ndarray): joints in shape (num_thetas, 3)
        theta_pairs (list): flip pairs for thetas

    Returns:
        thetas_flip (np.ndarray): flipped thetas with shape (num_thetas, 3)
    """
    thetas_flip = thetas.copy()
    # reflect horizontally
    thetas_flip[:, 1] = -1 * thetas_flip[:, 1]
    thetas_flip[:, 2] = -1 * thetas_flip[:, 2]
    # change left-right parts
    for pair in theta_pairs:
        thetas_flip[pair[0], :], thetas_flip[pair[1], :] = \
            thetas_flip[pair[1], :], thetas_flip[pair[0], :].copy()

    return thetas_flip


def flip_joints_3d(joints_3d, joints_3d_visible, width, flip_pairs):
    """Flip 3d joints.

    Args:
        joints_3d (np.ndarray): joints in shape (N, 3, 2)
        width (int): Image width
        joint_pairs (list): flip pairs for joints

    Returns:
        joints_3d_flipped (np.ndarray): flipped joints with shape (N, 3, 2)
        joints_3d_visible_flipped (np.ndarray): visibility of (N, 3, 2)
    """

    assert len(joints_3d) == len(joints_3d_visible)
    joints_3d[:, 0] = width - joints_3d[:, 0] - 1
    joints_3d_flipped = joints_3d.copy()
    joints_3d_visible_flipped = joints_3d_visible.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        joints_3d_flipped[left, :] = joints_3d[right, :]
        joints_3d_flipped[right, :] = joints_3d[left, :]

        joints_3d_visible_flipped[left, :] = joints_3d_visible[right, :]
        joints_3d_visible_flipped[right, :] = joints_3d_visible[left, :]

    joints_3d_flipped = joints_3d_flipped * joints_3d_visible_flipped

    return joints_3d_flipped, joints_3d_visible_flipped


def flip_xyz_joints_3d(joints_3d, flip_pairs):
    """Flip 3d xyz joints.

    Args:
        joints_3d (np.ndarray): Joints in shape (N, 3)
        joint_pairs (list): flip pairs for joints

    Returns:
        joints_3d_flipped (np.ndarray): flipped joints with shape (N, 3)
    """

    joints_3d[:, 0] = -1 * joints_3d[:, 0]
    joints_3d_flipped = joints_3d.copy()
    # change left-right parts
    for left, right in flip_pairs:
        joints_3d_flipped[left, :] = joints_3d[right, :]
        joints_3d_flipped[right, :] = joints_3d[left, :]

    return joints_3d_flipped


def flip_twist(twist_phi, twist_weight, twist_pairs):
    """Flip twist and weight.

    Args:
        twist_phi (np.ndarray): twist in shape (num_twist, 2)
        twist_weight (np.ndarray): weight in shape (num_twist, 2)
        twist_pairs (list): flip pairs for twist

    Returns:
        twist_flip (np.ndarray): flipped twist with shape (num_twist, 2)
        weight_flip (np.ndarray): flipped weights with shape (num_twist, 2)
    """
    # twist_flip = -1 * twist_phi.copy() # 23 x 2
    twist_flip = np.zeros_like(twist_phi)
    weight_flip = twist_weight.copy()

    twist_flip[:, 0] = twist_phi[:, 0].copy()  # cos
    twist_flip[:, 1] = -1 * twist_phi[:, 1].copy()  # sin
    for pair in twist_pairs:
        idx0 = pair[0] - 1
        idx1 = pair[1] - 1
        twist_flip[idx0, :], twist_flip[idx1, :] = \
            twist_flip[idx1, :], twist_flip[idx0, :].copy()

        weight_flip[idx0, :], weight_flip[idx1, :] = \
            weight_flip[idx1, :], weight_flip[idx0, :].copy()

    return twist_flip, weight_flip


def _center_scale_to_box(center, scale):
    """Flip twist and weight.

    Args:
        joints_3d (np.ndarray): Joints in shape (N, 3)
        joint_pairs (list): flip pairs for joints

    Returns:
        joints_3d_flipped (np.ndarray): flipped joints with shape (N, 3)
    """
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


@PIPELINES.register_module()
class RandomDPG(object):
    """Add dpg for data augmentation, including random crop and random sample
    Required keys: 'bbox', 'ann_info
    Modifies key: 'bbox', 'center', 'scale'
    Args:
        dpg_prob (float): Probability of dpg
    """

    def __init__(self, dpg_prob):
        self.dpg_prob = dpg_prob

    def __call__(self, results):
        if np.random.rand() > self.dpg_prob:
            return results

        bbox = results['bbox']
        imgwidth = results['ann_info']['width']
        imgheight = results['ann_info']['height']

        PatchScale = random.uniform(0, 1)
        width = bbox[2] - bbox[0]
        ht = bbox[3] - bbox[1]

        if PatchScale > 0.85:
            ratio = ht / width
            if (width < ht):
                patchWidth = PatchScale * width
                patchHt = patchWidth * ratio
            else:
                patchHt = PatchScale * ht
                patchWidth = patchHt / ratio

            xmin = bbox[0] + random.uniform(0, 1) * (width - patchWidth)
            ymin = bbox[1] + random.uniform(0, 1) * (ht - patchHt)
            xmax = xmin + patchWidth + 1
            ymax = ymin + patchHt + 1
        else:
            xmin = max(
                1,
                min(bbox[0] + np.random.normal(-0.0142, 0.1158) * width,
                    imgwidth - 3))
            ymin = max(
                1,
                min(bbox[1] + np.random.normal(0.0043, 0.068) * ht,
                    imgheight - 3))
            xmax = min(
                max(xmin + 2,
                    bbox[2] + np.random.normal(0.0154, 0.1337) * width),
                imgwidth - 3)
            ymax = min(
                max(ymin + 2,
                    bbox[3] + np.random.normal(-0.0013, 0.0711) * ht),
                imgheight - 3)
        bbox_xyxy = np.array([xmin, ymin, xmax, ymax])
        bbox_xywh = xyxy2xywh(bbox_xyxy)
        center, scale = box2cs(
            bbox_xywh, aspect_ratio=1.0, bbox_scale_factor=1.0)
        results['bbox'] = bbox_xyxy
        results['center'] = center
        results['scale'] = scale

        return results


@PIPELINES.register_module()
class HybrIKRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'keypoints3d', 'keypoints3d_vis', 'center',
    and 'ann_info', 'has_smpl'
    Additional keys required if has_smpl: 'keypoints3d17', 'keypoints3d17_vis',
    'keypoints3d_relative', 'keypoints3d17_relative', 'pose'

    Modifies key: 'img', 'keypoints3d', 'keypoints3d_vis', 'center', 'pose'
    Additional keys modified if has_smpl: 'keypoints3d17', 'keypoints3d17_vis',
    'keypoints3d_relative', 'keypoints3d17_relative', 'pose'

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        flip_pairs (list[int]): list of left-right keypoint pairs for flipping
    """

    def __init__(self, flip_prob=0.5, flip_pairs=None):
        assert 0 <= flip_prob <= 1
        self.flip_prob = flip_prob
        self.flip_pairs = flip_pairs

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        if np.random.rand() > self.flip_prob:
            results['is_flipped'] = np.array([0])
            return results

        results['is_flipped'] = np.array([1])

        # flip image
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imflip(results[key], direction='horizontal')

        width = results['img'][:, ::-1, :].shape[1]
        # flip bbox center
        center = results['center']
        center[0] = width - 1 - center[0]
        results['center'] = center

        keypoints3d = results['keypoints3d']
        keypoints3d_vis = results['keypoints3d_vis']

        keypoints3d, keypoints3d_vis = flip_joints_3d(keypoints3d,
                                                      keypoints3d_vis, width,
                                                      self.flip_pairs)

        if results['has_smpl']:
            pose = results['pose']
            smpl_flip_pairs = get_flip_pairs('smpl')
            pose = flip_thetas(pose, smpl_flip_pairs)

            keypoints3d17 = results['keypoints3d17']
            keypoints3d17_vis = results['keypoints3d17_vis']
            keypoints3d17_relative = results['keypoints3d17_relative']
            keypoints3d_relative = results['keypoints3d_relative']

            keypoints3d17, keypoints3d17_vis = flip_joints_3d(
                keypoints3d17, keypoints3d17_vis, width, self.flip_pairs)
            keypoints3d17_relative = flip_xyz_joints_3d(
                keypoints3d17_relative, self.flip_pairs)
            keypoints3d_relative = flip_xyz_joints_3d(keypoints3d_relative,
                                                      self.flip_pairs)
            twist_phi, twist_weight = results['target_twist'], results[
                'target_twist_weight']
            results['target_twist'], results[
                'target_twist_weight'] = flip_twist(twist_phi, twist_weight,
                                                    smpl_flip_pairs)

            results['keypoints3d17_relative'] = keypoints3d17_relative.astype(
                np.float32)
            results['keypoints3d_relative'] = keypoints3d_relative.astype(
                np.float32)
            results['keypoints3d17'] = keypoints3d17.astype(np.float32)
            results['keypoints3d17_vis'] = keypoints3d17_vis.astype(np.float32)
            results['pose'] = pose.astype(np.float32)

        results['keypoints3d'] = keypoints3d.astype(np.float32)
        results['keypoints3d_vis'] = keypoints3d_vis.astype(np.float32)
        return results


@PIPELINES.register_module()
class HybrIKAffine:
    """Affine transform the image to get input image. Affine transform the 2D
    keypoints, 3D kepoints and IUV image too.

    Required keys: 'img', 'keypoints3d', 'keypoints3d_vis', 'pose', 'ann_info',
    'scale', 'keypoints3d17', 'keypoints3d17_vis', 'rotation' and 'center'.
    Modifies key: 'img', 'keypoints3d','keypoints3d_vis', 'pose',
    'keypoints3d17', 'keypoints3d17_vis'
    """

    def __init__(self, img_res):
        self.image_size = np.array([img_res, img_res])

    def __call__(self, results):

        img = results['img']
        keypoints3d = results['keypoints3d']
        num_joints = len(keypoints3d)
        keypoints3d_vis = results['keypoints3d_vis']
        has_smpl = results['has_smpl']

        c = results['center']
        s = results['scale']
        r = results['rotation']
        trans = get_affine_transform(c, s, r, self.image_size, pixel_std=1)
        img = cv2.warpAffine(
            img,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(num_joints):
            if keypoints3d_vis[i, 0] > 0.0:
                keypoints3d[i, 0:2] = affine_transform(keypoints3d[i, 0:2],
                                                       trans)

        if has_smpl:

            keypoints3d17 = results['keypoints3d17']
            keypoints3d17_vis = results['keypoints3d17_vis']
            for i in range(17):
                if keypoints3d17_vis[i, 0] > 0.0:
                    keypoints3d17[i, 0:2] = affine_transform(
                        keypoints3d17[i, 0:2], trans)
            results['keypoints3d17'] = keypoints3d17
            results['keypoints3d17_vis'] = keypoints3d17_vis

            # to rotate poses
            pose = results['pose']
            pose = _rotate_smpl_pose(pose.reshape(-1), r)
            results['pose'] = pose.reshape(24, 3)

        results['img'] = img.astype(np.float32)
        results['keypoints3d_vis'] = keypoints3d_vis.astype(np.float32)
        results['keypoints3d'] = keypoints3d.astype(np.float32)

        return results


@PIPELINES.register_module()
class RandomOcclusion:
    """Add random occlusion.

    Add random occlusion based on occlusion probability.

    Args:
        occlusion_prob (float): probability of the image having
        occlusion. Default: 0.5
    """

    def __init__(self, occlusion_prob=0.5):
        self.occlusion_prob = occlusion_prob

    def __call__(self, results):

        if np.random.rand() > self.occlusion_prob:
            return results

        xmin, ymin, xmax, ymax = results['bbox']
        imgwidth = results['ann_info']['width']
        imgheight = results['ann_info']['height']
        img = results['img']

        area_min = 0.0
        area_max = 0.7
        synth_area = (random.random() *
                      (area_max - area_min) + area_min) * (xmax - xmin) * (
                          ymax - ymin)

        ratio_min = 0.3
        ratio_max = 1 / 0.3
        synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

        synth_h = math.sqrt(synth_area * synth_ratio)
        synth_w = math.sqrt(synth_area / synth_ratio)
        synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
        synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

        if synth_xmin >= 0 and synth_ymin >= 0 and \
            synth_xmin + synth_w < imgwidth and \
                synth_ymin + synth_h < imgheight:
            synth_xmin = int(synth_xmin)
            synth_ymin = int(synth_ymin)
            synth_w = int(synth_w)
            synth_h = int(synth_h)
            img[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin +
                synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255

        results['img'] = img

        return results


@PIPELINES.register_module()
class GenerateHybrIKTarget:
    """Generate the targets required for training.

    Required keys: 'keypoints3d', 'keypoints3d_vis', 'ann_info', 'depth_factor'
    Additional keys if has_smpl: 'keypoints3d17', 'keypoints3d17_vis',
    'keypoints3d_relative', 'keypoints3d17_relative' Add keys: 'target_uvd_29',
    'target_xyz_24', 'target_weight_24', 'target_weight_29', 'target_xyz_17',
    'target_weight_17', 'target_theta', 'target_beta', 'target_smpl_weight',
    'target_theta_weight', trans_inv', 'bbox'
    """

    def __init__(self, img_res, test_mode):
        self.test_mode = test_mode
        self.image_size = np.array([img_res, img_res])

    def _integral_uvd_target_generator(self,
                                       joints_3d,
                                       num_joints,
                                       patch_height,
                                       patch_width,
                                       depth_factor,
                                       test_mode=False):

        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / depth_factor

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_target_generator(self, joints_3d, num_joints, patch_height,
                                   patch_width, depth_factor):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]
        target_weight[:, 2] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / depth_factor

        target_weight[target[:, 0] > 0.5] = 0
        target_weight[target[:, 0] < -0.5] = 0
        target_weight[target[:, 1] > 0.5] = 0
        target_weight[target[:, 1] < -0.5] = 0
        target_weight[target[:, 2] > 0.5] = 0
        target_weight[target[:, 2] < -0.5] = 0

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_xyz_target_generator(self, joints_3d, joints_3d_vis,
                                       num_joints, depth_factor):
        target_weight = np.ones((num_joints, 3), dtype=np.float32)
        target_weight[:, 0] = joints_3d_vis[:, 0]
        target_weight[:, 1] = joints_3d_vis[:, 1]
        target_weight[:, 2] = joints_3d_vis[:, 2]

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0] / int(depth_factor)
        target[:, 1] = joints_3d[:, 1] / int(depth_factor)
        target[:, 2] = joints_3d[:, 2] / int(depth_factor)

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _integral_target_generator_coco(self, joints_3d, num_joints,
                                        patch_height, patch_width):
        target_weight = np.ones((num_joints, 2), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 2), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def __call__(self, results):

        has_smpl = results['has_smpl']
        inp_h, inp_w = self.image_size[0], self.image_size[1]

        keypoints3d = results['keypoints3d']
        num_joints = len(keypoints3d)
        keypoints3d_vis = results['keypoints3d_vis']
        depth_factor = results['depth_factor']

        c = results['center']
        s = results['scale']
        r = results['rotation']

        #  generate new keys
        trans_inv = get_affine_transform(
            c, s, r, self.image_size, inv=True, pixel_std=1).astype(np.float32)
        results['trans_inv'] = trans_inv.astype(np.float32)
        bbox = _center_scale_to_box(c, s)
        results['bbox'] = np.array(bbox, dtype=np.float32)

        if has_smpl:
            theta = results['pose']
            # aa to quat
            results['target_theta'] = aa_to_quat_numpy(theta).reshape(
                24 * 4).astype(np.float32)
            theta_24_weights = np.ones((24, 4))
            results['target_theta_weight'] = theta_24_weights.reshape(
                24 * 4).astype(np.float32)

            results['target_beta'] = results['beta'].astype(np.float32)
            results['target_smpl_weight'] = np.ones(1).astype(np.float32)

            keypoints3d17_vis = results['keypoints3d17_vis']
            keypoints3d17_relative = results['keypoints3d17_relative']
            joints24_relative_3d = results['keypoints3d_relative'][:24, :]

            gt_joints_29 = np.zeros((29, 3, 2), dtype=np.float32)
            gt_joints_29[:, :, 0] = keypoints3d.copy()
            gt_joints_29[:, :, 1] = keypoints3d_vis.copy()

            target_uvd_29, target_weight_29 = \
                self._integral_uvd_target_generator(
                    gt_joints_29, 29, inp_h, inp_w, depth_factor)
            target_xyz_17, target_weight_17 = \
                self._integral_xyz_target_generator(
                    keypoints3d17_relative, keypoints3d17_vis, 17,
                    depth_factor)
            target_xyz_24, target_weight_24 = \
                self._integral_xyz_target_generator(
                    joints24_relative_3d, keypoints3d_vis[:24, :], 24,
                    depth_factor)
            target_weight_29 *= keypoints3d_vis.reshape(-1)
            target_weight_24 *= keypoints3d_vis[:24, :].reshape(-1)
            target_weight_17 *= keypoints3d17_vis.reshape(-1)

            results['target_uvd_29'] = target_uvd_29.astype(np.float32)
            results['target_xyz_24'] = target_xyz_24.astype(np.float32)
            results['target_weight_29'] = target_weight_29.astype(np.float32)
            results['target_weight_24'] = target_weight_24.astype(np.float32)
            results['target_xyz_17'] = target_xyz_17.astype(np.float32)
            results['target_weight_17'] = target_weight_17.astype(np.float32)
        else:
            label_uvd_29 = np.zeros((29, 3))
            label_xyz_24 = np.zeros((24, 3))
            label_uvd_29_mask = np.zeros((29, 3))
            label_xyz_17 = np.zeros((17, 3))
            label_xyz_17_mask = np.zeros((17, 3))

            gt_joints = np.zeros((num_joints, 3, 2), dtype=np.float32)
            gt_joints[:, :, 0] = keypoints3d.copy()
            gt_joints[:, :, 1] = keypoints3d_vis.copy()
            mask_idx = [1, 2, 6, 9, 10, 11]

            if results['ann_info']['dataset_name'] == 'coco':
                target, target_weight = self._integral_target_generator_coco(
                    gt_joints, num_joints, inp_h, inp_w)

                label_jts_origin = target * target_weight
                label_jts_mask_origin = target_weight

                label_jts_origin = label_jts_origin.reshape(num_joints, 2)
                label_jts_mask_origin = label_jts_mask_origin.reshape(
                    num_joints, 2)
                label_jts_origin[mask_idx] = label_jts_origin[mask_idx] * 0
                label_jts_mask_origin[
                    mask_idx] = label_jts_origin[mask_idx] * 0
                label_uvd_29 = np.hstack([label_jts_origin, np.zeros([29, 1])])
                label_uvd_29_mask = np.hstack(
                    [label_jts_mask_origin,
                     np.zeros([29, 1])])

            elif results['ann_info']['dataset_name'] == 'mpi_inf_3dhp':
                if not self.test_mode:
                    target, target_weight = self._integral_target_generator(
                        gt_joints, num_joints, inp_h, inp_w, depth_factor)
                    target_weight *= keypoints3d_vis.reshape(-1)

                    label_jts_origin = target * target_weight
                    label_jts_mask_origin = target_weight

                    label_jts_origin = label_jts_origin.reshape(num_joints, 3)
                    label_jts_mask_origin = label_jts_mask_origin.reshape(
                        num_joints, 3)
                    label_jts_origin[mask_idx] = label_jts_origin[mask_idx] * 0
                    label_jts_mask_origin[
                        mask_idx] = label_jts_origin[mask_idx] * 0
                    label_uvd_29 = label_jts_origin
                    label_uvd_29_mask = label_jts_mask_origin

            label_uvd_29 = label_uvd_29.reshape(-1)
            label_xyz_24 = label_xyz_24.reshape(-1)
            label_uvd_24_mask = label_uvd_29_mask[:24, :].reshape(-1)
            label_uvd_29_mask = label_uvd_29_mask.reshape(-1)
            label_xyz_17 = label_xyz_17.reshape(-1)
            label_xyz_17_mask = label_xyz_17_mask.reshape(-1)

            results['target_uvd_29'] = label_uvd_29.astype(np.float32)
            results['target_xyz_24'] = label_xyz_24.astype(np.float32)
            results['target_weight_24'] = label_uvd_24_mask.astype(np.float32)
            results['target_weight_29'] = label_uvd_29_mask.astype(np.float32)
            results['target_xyz_17'] = label_xyz_17.astype(np.float32)
            results['target_weight_17'] = label_xyz_17_mask.astype(np.float32)
            results['target_theta'] = np.zeros(24 * 4).astype(np.float32)
            results['target_beta'] = np.zeros(10).astype(np.float32)
            results['target_smpl_weight'] = np.zeros(1).astype(np.float32)
            results['target_theta_weight'] = np.zeros(24 * 4).astype(
                np.float32)

        return results


@PIPELINES.register_module()
class NewKeypointsSelection:
    """Select keypoints.

    Modifies specified keys

    Args:
        map (dict): keypoints and index for selection
    """

    def __init__(self, maps):
        self.maps = maps

    def __call__(self, results):
        """Perform keypoints selection."""

        for map in self.maps:
            for keypoint in map['keypoints']:
                keypoints_index = map['keypoints_index']
                if keypoint in results:
                    results[keypoint] = results[keypoint][...,
                                                          keypoints_index, :]
        return results
