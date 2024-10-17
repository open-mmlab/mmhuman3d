import glob
import json
import os
import pdb
import random
import time
from multiprocessing import Pool
from typing import List

import cv2
import numpy as np
import pandas as pd
import smplx
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.core.conventions.segmentation.smpl import SMPL_SEGMENTATION_DICT
from mmhuman3d.data.data_structures.human_data import HumanData
# import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import transform_to_camera_frame
from mmhuman3d.core.conventions.cameras.convert_convention import convert_camera_matrix
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class ArcticConverter(BaseModeConverter):

    ACCEPTED_MODES = ['p1_test', 'p1_train', 
                      'p1_val', 'p2_test', 
                      'p2_train', 'p2_val']

    def __init__(self, modes: List = []) -> None:
        self.device = torch.device('cuda:0')
        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            bbox_source='keypoints2d_smplx',
            flat_hand_mean=True,
            cam_param_type='prespective',
            cam_param_source='original',
            smpl_source='original',
        )
        self.smplx_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 21, 3),
            'left_hand_pose': (-1, 15, 3),
            'right_hand_pose': (-1, 15, 3),
            'leye_pose': (-1, 3),
            'reye_pose': (-1, 3),
            'jaw_pose': (-1, 3),
            'expression': (-1, 10)
        }

        super(ArcticConverter, self).__init__(modes)
        
    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:
        
        
        # load split
        split_path = os.path.join(dataset_path, 'splits', mode + '.npy')
        split_info = np.load(split_path, allow_pickle=True).item()
        
        image_names = split_info['imgnames']
        data_dict = split_info['data_dict']
        
        # load meta
        meta_path = os.path.join(dataset_path, 'meta', 'misc.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        for image_name in image_names:
            
            image_path = image_name.replace('./arctic_data/', '')
            imgp = os.path.join(dataset_path, image_path)
        
            # load raw seqs
            sub_id = image_path.split('/')[1]
            seq_name = image_path.split('/')[2]
            cam_id = image_path.split('/')[3]
        
            raw_path = os.path.join(dataset_path, 'raw_seqs', sub_id)
            egocam_path = os.path.join(raw_path, f'{seq_name}.egocam.dist.npy')
            smplx_path = os.path.join(raw_path, f'{seq_name}.smplx.npy')
            
            # load egocam
            egocam_params = np.load(egocam_path, allow_pickle=True).item()
            smplx_params = np.load(smplx_path, allow_pickle=True).item()
            metainfo = metadata[sub_id]
            
            gender = metainfo[gender]
            
        
            pdb.set_trace()
        
        
        pass