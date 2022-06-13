import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, List, Optional, Union

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.core.evaluation import (
    keypoint_3d_auc,
    keypoint_3d_pck,
    keypoint_mpjpe,
    vertice_pve,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.human_data_cache import (
    HumanDataCacheReader,
    HumanDataCacheWriter,
)
from mmhuman3d.models.builder import build_body_model
from .human_image_dataset import HumanImageDataset
from .builder import DATASETS


@DATASETS.register_module()
class HumanImageSMPLXDataset(HumanImageDataset):
    def __init__(self, 
                 data_prefix: str, 
                 pipeline: list, 
                 dataset_name: str, 
                 body_model: Optional[Union[dict, None]] = None, 
                 ann_file: Optional[Union[str, None]] = None, 
                 convention: Optional[str] = 'human_data', 
                 cache_data_path: Optional[Union[str, None]] = None, 
                 test_mode: Optional[bool] = False,
                 num_betas: Optional[int] = 10,
                 num_expression: Optional[int] = 10):
        super().__init__(data_prefix, pipeline, dataset_name, body_model, ann_file, convention, cache_data_path, test_mode)
        self.num_betas = num_betas
        self.num_expression = num_expression

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        if self.cache_reader is not None:
            self.human_data = self.cache_reader.get_item(idx)
            idx = idx % self.cache_reader.slice_size
        info = {}
        info['img_prefix'] = None
        image_path = self.human_data['image_path'][idx]
        info['image_path'] = os.path.join(self.data_prefix, 'datasets', self.dataset_name, image_path)
        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = idx
        if 'bbox_xywh' in self.human_data:
            info['bbox_xywh'] = self.human_data['bbox_xywh'][idx]
            x, y, w, h, s = info['bbox_xywh']
            cx = x + w / 2
            cy = y + h / 2
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        else:
            info['bbox_xywh'] = np.zeros((5))
            info['center'] = np.zeros((2))
            info['scale'] = np.zeros((2))
        if 'keypoints2d' in self.human_data:
            info['keypoints2d'] = self.human_data['keypoints2d'][idx]
            info['has_keypoints2d'] = 1
        else:
            info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
            info['has_keypoints2d'] = 0
        if 'keypoints3d' in self.human_data:
            info['keypoints3d'] = self.human_data['keypoints3d'][idx]
            info['has_keypoints3d'] = 1
        else:
            info['keypoints3d'] = np.zeros((self.num_keypoints, 4))
            info['has_keypoints3d'] = 0
        
        if 'smplx' in self.human_data:
            smplx_dict = self.human_data['smplx']
            info['has_smplx'] = 1
        else:
            smplx_dict = {}
            info['has_smplx'] = 0
        if 'global_orient' in smplx_dict:
            info['smplx_global_orient'] = smplx_dict['global_orient'][idx]
            info['has_smplx_global_orient'] = 1
        else:
            info['smplx_global_orient'] = np.zeros((3),dtype=np.float32)
            info['has_smplx_global_orient'] = 0

        if 'body_pose' in smplx_dict:
            info['smplx_body_pose'] = smplx_dict['body_pose'][idx]
            info['has_smplx_body_pose'] = 1
        else:
            info['smplx_body_pose'] = np.zeros((21,3),dtype=np.float32)
            info['has_smplx_body_pose'] = 0

        if 'right_hand_pose' in smplx_dict:
            info['smplx_right_hand_pose'] = smplx_dict['right_hand_pose'][idx]
            info['has_smplx_right_hand_pose'] = 1
        else:
            info['smplx_right_hand_pose'] = np.zeros((15,3),dtype=np.float32)
            info['has_smplx_right_hand_pose'] = 0

        if 'left_hand_pose' in smplx_dict:
            info['smplx_left_hand_pose'] = smplx_dict['left_hand_pose'][idx]
            info['has_smplx_left_hand_pose'] = 1
        else:
            info['smplx_left_hand_pose'] = np.zeros((15,3),dtype=np.float32)
            info['has_smplx_left_hand_pose'] = 0

        if 'jaw_pose' in smplx_dict:
            info['smplx_jaw_pose'] = smplx_dict['jaw_pose'][idx]
            info['has_smplx_jaw_pose'] = 1
        else:
            info['smplx_jaw_pose'] = np.zeros((3),dtype=np.float32)
            info['has_smplx_jaw_pose'] = 0

        if 'betas' in smplx_dict:
            info['smplx_betas'] = smplx_dict['betas'][idx]
            info['has_smplx_betas'] = 1
        else:
            info['smplx_betas'] = np.zeros((self.num_betas),dtype=np.float32)
            info['has_smplx_betas'] = 0

        if 'expression' in smplx_dict:
            info['smplx_expression'] = smplx_dict['expression'][idx]
            info['has_smplx_expression'] = 1
        else:
            info['smplx_expression'] = np.zeros((self.num_expression),dtype=np.float32)
            info['has_smplx_expression'] = 0
        


        return info
