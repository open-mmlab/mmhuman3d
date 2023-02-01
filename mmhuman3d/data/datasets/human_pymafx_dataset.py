# yapf: disable
import os
import os.path as osp
from abc import ABCMeta
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mmhuman3d.core.conventions.keypoints_mapping.mano import (
    MANO_RIGHT_REORDER_KEYPOINTS,
)
from mmhuman3d.data.datasets.pipelines.pymafx_transforms import (
    crop,
    get_transform,
    transform,
)
from .base_dataset import BaseDataset
from .builder import DATASETS

# yapf: enable


@DATASETS.register_module()
class PyMAFXHumanImageDataset(BaseDataset, metaclass=ABCMeta):
    """Dataset for PyMAFX Inference.

    Args:

        data_prefix (str):
            Path to a directory where preprocessed datasets are held.
        pipeline (list[dict | callable]): A sequence of data transforms.
        img_res (int): The body image resolution.
        hf_img_size (int): The hand and head image resolution.
        image_folder (str): Path to images.
        frames (List[int]): The index of the frame.
        bboxes (List[np.ndarray], optional): Defaults to None.
        joints2d (List[np.ndarray], optional): The 2d joints. Defaults to None.
        scale_factor (float, optional): Defaults to 1.0.
        wb_kps (dict, optional):
            2d keypoints on the hands and face. Defaults to {}.
        test_mode (bool): Store True when building test dataset.
            Default to False.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 img_res: int,
                 hf_img_size: int,
                 image_folder: str,
                 frames: List[int],
                 bboxes: List[np.ndarray] = None,
                 joints2d: List[np.ndarray] = None,
                 scale_factor: float = 1.0,
                 wb_kps: dict = {},
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
        self.frames = frames
        self.has_keypoints = True if joints2d is not None else False
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

                bboxes.append(np.array(center + width_height))
                scales.append(scale)

            self.bboxes = np.stack(bboxes)
            self.scales = np.array(scales)

        joints2d_face = wb_kps['joints2d_face']
        joints2d_lhand = wb_kps['joints2d_lhand']
        joints2d_rhand = wb_kps['joints2d_rhand']

        self.joints2d_part = {
            'lhand': joints2d_lhand,
            'rhand': joints2d_rhand,
            'face': joints2d_face
        }

    def __len__(self):
        return len(self.bboxes)

    def process_rgb(self, rgb_img, center, scale, res=[224, 224]):
        """Process rgb image and do augmentation."""
        # crop
        crop_img_resized, crop_img = crop(rgb_img, center, scale, res)
        # (224, 224, 3), float, [0, 1]
        crop_img_resized = crop_img_resized.astype('float32') / 255.0
        crop_img = crop_img.astype('float32') / 255.0
        return crop_img_resized, crop_img

    def process_kps2d(self, kps2d, t):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        kps2d = kps2d.copy()
        n_kps2d = kps2d.shape[0]
        for i in range(n_kps2d):
            kp = kps2d[i, 0:2]
            new_kp = np.array([kp[0], kp[1], 1.]).T
            new_kp = np.dot(t, new_kp)
            kps2d[i, 0:2] = new_kp[:2]
        # convert to normalized coordinates
        kps2d[:, :-1] = 2. * kps2d[:, :-1] / self.img_res - 1.
        kps2d = kps2d.astype('float32')
        return kps2d

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

        img, _ = self.process_rgb(img_orig, center, sc * scale,
                                  [self.img_res, self.img_res])

        # Store image before normalization to use it in visualization
        item['img_body'] = self.pipeline({'img': img})['img'].float()
        item['orig_height'] = orig_height
        item['orig_width'] = orig_width

        img_hr, img_crop = self.process_rgb(
            img_orig, center, sc * scale, [self.img_res * 8, self.img_res * 8])

        kps_transf = get_transform(center, sc * scale,
                                   [self.img_res, self.img_res])

        lhand_kp2d, rhand_kp2d, face_kp2d = self.joints2d_part['lhand'][
            idx], self.joints2d_part['rhand'][idx], self.joints2d_part['face'][
                idx]

        hand_kp2d = self.process_kps2d(
            np.concatenate([lhand_kp2d, rhand_kp2d]).copy(), kps_transf)
        face_kp2d = self.process_kps2d(face_kp2d.copy(), kps_transf)

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
