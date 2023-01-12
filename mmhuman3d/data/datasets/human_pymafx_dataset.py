# yapf: disable
import os
import os.path as osp
from abc import ABCMeta
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mmhuman3d.core.conventions.keypoints_mapping.mano import (
    MANO_RIGHT_REORDER_KEYPOINTS,
)
from mmhuman3d.utils.img_utils import (
    crop,
    flip_img,
    flip_kp,
    get_transform,
    transform,
)
from .base_dataset import BaseDataset
from .builder import DATASETS

# yapf: enable


@DATASETS.register_module()
class PyMAFXHumanImageDataset(BaseDataset, metaclass=ABCMeta):
    """Dataset for PyMAFX Inference."""

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 img_res: int,
                 hf_img_size,
                 image_folder,
                 frames,
                 bboxes=None,
                 joints2d=None,
                 scale_factor=1.0,
                 crop_size=224,
                 person_id_list=[],
                 wb_kps={},
                 test_mode: Optional[bool] = True):
        super().__init__(data_prefix, pipeline, test_mode)
        self.img_res = img_res
        self.hf_img_size = hf_img_size
        self.image_file_names = [
            osp.join(image_folder, x) for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = np.array(sorted(self.image_file_names))[frames]
        self.bboxes = bboxes
        self.joints2d = joints2d
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.frames = frames
        self.has_keypoints = True if joints2d is not None else False
        self.person_id_list = person_id_list
        self.norm_joints2d = np.zeros_like(self.joints2d)

        if self.has_keypoints:
            bboxes = []
            scales = []
            for j2d in self.joints2d:
                kp2d_valid = j2d[j2d[:, 2] > 0.]
                bbox_xyxy = [
                    min(kp2d_valid[:, 0]),
                    min(kp2d_valid[:, 1]),
                    max(kp2d_valid[:, 0]),
                    max(kp2d_valid[:, 1])
                ]
                center = [(bbox_xyxy[2] + bbox_xyxy[0]) / 2.,
                          (bbox_xyxy[3] + bbox_xyxy[1]) / 2.]
                scale = self.scale_factor * 1.2 * max(
                    bbox_xyxy[2] - bbox_xyxy[0],
                    bbox_xyxy[3] - bbox_xyxy[1]) / 200.

                res = [self.img_res, self.img_res]
                ul = np.array(transform([1, 1], center, scale, res,
                                        invert=1)) - 1
                # Bottom right point
                br = np.array(
                    transform(
                        [res[0] + 1, res[1] + 1], center, scale, res,
                        invert=1)) - 1

                center = [(ul[0] + br[0]) / 2., (ul[1] + br[1]) / 2.]
                width_height = [br[0] - ul[0], br[1] - ul[1]]

                bbox_xyxy = np.array(center + width_height)
                bboxes.append(bbox_xyxy)
                scales.append(scale)

            self.bboxes = np.stack(bboxes)
            self.scales = np.array(scales)

        joints2d_face = wb_kps['joints2d_face']
        joints2d_lhand = wb_kps['joints2d_lhand']
        joints2d_rhand = wb_kps['joints2d_rhand']

        joints_part = {
            'lhand': joints2d_lhand,
            'rhand': joints2d_rhand,
            'face': joints2d_face
        }

        self.bboxes_part = {}
        self.joints2d_part = {}

        for part, joints in joints_part.items():
            self.joints2d_part[part] = joints
            if len(self.joints2d_part[part]) == 0:
                exit()

    def __len__(self):
        return len(self.bboxes)

    def rgb_processing(self,
                       rgb_img,
                       center,
                       scale,
                       res=[224, 224],
                       rot=0.,
                       flip=0):
        """Process rgb image and do augmentation."""
        # crop
        crop_img_resized, crop_img, crop_shape = crop(
            rgb_img, center, scale, res, rot=rot)
        # flip the image
        if flip:
            crop_img_resized = flip_img(crop_img_resized)
            crop_img = flip_img(crop_img)
        # (224, 224, 3), float, [0, 1]
        crop_img_resized = crop_img_resized.astype('float32') / 255.0
        crop_img = crop_img.astype('float32') / 255.0
        return crop_img_resized, crop_img, crop_shape

    def j2d_processing(self,
                       kp,
                       t,
                       f,
                       is_smpl=False,
                       is_hand=False,
                       is_face=False,
                       is_feet=False):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        kp = kp.copy()
        nparts = kp.shape[0]
        for i in range(nparts):
            pt = kp[i, 0:2]
            new_pt = np.array([pt[0], pt[1], 1.]).T
            new_pt = np.dot(t, new_pt)
            kp[i, 0:2] = new_pt[:2]
        # convert to normalized coordinates
        kp[:, :-1] = 2. * kp[:, :-1] / self.img_res - 1.
        # flip the x coordinates
        if f:
            if is_hand:
                kp = flip_kp(kp, type='hand')
            elif is_face:
                kp = flip_kp(kp, type='face')
            elif is_feet:
                kp = flip_kp(kp, type='feet')
            else:
                kp = flip_kp(kp, is_smpl)
        kp = kp.astype('float32')
        return kp

    def load_annotations(self):
        """Load annotations from ``ann_file``"""
        pass

    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        img_orig = cv2.imread(
            self.image_file_names[idx])[:, :, ::-1].copy().astype(np.float32)
        orig_height, orig_width = img_orig.shape[:2]
        item = {}

        scale = self.scale_factor
        rot = 0.
        flip = 0

        j2d = self.joints2d[idx]

        kp2d_valid = j2d[j2d[:, 2] > 0.]
        bbox_xyxy = [
            min(kp2d_valid[:, 0]),
            min(kp2d_valid[:, 1]),
            max(kp2d_valid[:, 0]),
            max(kp2d_valid[:, 1])
        ]
        center = [(bbox_xyxy[2] + bbox_xyxy[0]) / 2.,
                  (bbox_xyxy[3] + bbox_xyxy[1]) / 2.]
        sc = 1.2 * max(bbox_xyxy[2] - bbox_xyxy[0],
                       bbox_xyxy[3] - bbox_xyxy[1]) / 200.

        img, _, _ = self.rgb_processing(img_orig, center, sc * scale,
                                        [self.img_res, self.img_res])

        # crop_img = np.transpose(img.astype('float32'), (1,2,0)) * 255.
        # cv2.imwrite('output/body_img.png', crop_img.astype(np.uint8))

        # Store image before normalization to use it in visualization
        item['img_body'] = self.pipeline({'img': img})['img'].float()
        item['orig_height'] = orig_height
        item['orig_width'] = orig_width
        item['person_id'] = self.person_id_list[idx]

        img_hr, img_crop, _ = self.rgb_processing(
            img_orig, center, sc * scale, [self.img_res * 8, self.img_res * 8])

        kps_transf = get_transform(
            center, sc * scale, [self.img_res, self.img_res], rot=rot)

        lhand_kp2d, rhand_kp2d, face_kp2d = self.joints2d_part['lhand'][
            idx], self.joints2d_part['rhand'][idx], self.joints2d_part['face'][
                idx]

        hand_kp2d = self.j2d_processing(
            np.concatenate([lhand_kp2d, rhand_kp2d]).copy(),
            kps_transf,
            flip,
            is_hand=True)
        face_kp2d = self.j2d_processing(
            face_kp2d.copy(), kps_transf, flip, is_face=True)

        n_hand_kp = len(MANO_RIGHT_REORDER_KEYPOINTS)
        part_kp2d_dict = {
            'lhand': hand_kp2d[:n_hand_kp],
            'rhand': hand_kp2d[n_hand_kp:],
            'face': face_kp2d
        }

        for part in ['lhand', 'rhand', 'face']:
            kp2d = part_kp2d_dict[part]
            # kp2d_valid = kp2d[kp2d[:, 2]>0.005]
            kp2d_valid = kp2d[kp2d[:, 2] > 0.]
            if len(kp2d_valid) > 0:
                bbox_xyxy = [
                    min(kp2d_valid[:, 0]),
                    min(kp2d_valid[:, 1]),
                    max(kp2d_valid[:, 0]),
                    max(kp2d_valid[:, 1])
                ]
                center_part = [(bbox_xyxy[2] + bbox_xyxy[0]) / 2.,
                               (bbox_xyxy[3] + bbox_xyxy[1]) / 2.]
                scale_part = 2. * max(bbox_xyxy[2] - bbox_xyxy[0],
                                      bbox_xyxy[3] - bbox_xyxy[1]) / 2

            # handle invalid part keypoints
            if len(kp2d_valid) < 1 or scale_part < 0.01:
                center_part = [0, 0]
                scale_part = 0.5
                kp2d[:, 2] = 0

            center_part = torch.tensor(center_part).float()

            theta_part = torch.zeros(1, 2, 3)
            theta_part[:, 0, 0] = scale_part
            theta_part[:, 1, 1] = scale_part
            theta_part[:, :, -1] = center_part
            crop_hf_img_size = torch.Size(
                [1, 3, self.hf_img_size, self.hf_img_size])
            grid = F.affine_grid(
                theta_part.detach(), crop_hf_img_size, align_corners=False)
            img_part = F.grid_sample(
                torch.from_numpy(img_crop.transpose(2, 0, 1)[None]),
                grid.cpu(),
                align_corners=False).squeeze(0)

            item[f'img_{part}'] = self.pipeline(
                {'img': img_part.numpy().transpose(1, 2, 0)})['img'].float()
            theta_i_inv = torch.zeros_like(theta_part)
            theta_i_inv[:, 0, 0] = 1. / theta_part[:, 0, 0]
            theta_i_inv[:, 1, 1] = 1. / theta_part[:, 1, 1]
            theta_i_inv[:, :,
                        -1] = -theta_part[:, :,
                                          -1] / theta_part[:, 0,
                                                           0].unsqueeze(-1)
            item[f'{part}_theta_inv'] = theta_i_inv[0]
            item['sample_idx'] = idx

        return item
