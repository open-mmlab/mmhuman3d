import glob
import os
import pdb
import random
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class Hsc4dConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train']

    def __init__(self, modes=[], *args, **kwargs):

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.misc_config = dict(
            bbox_source='by_dataset',
            smpl_source='original',
            cam_param_type='prespective',
            kps3d_root_aligned=False,
        )
        INTRINSICS = [599.628, 599.466, 971.613, 540.258]
        DIST       = [0.003, -0.003, -0.001, 0.004, 0.0]
        LIDAR2CAM  = [[[-0.0355545576, -0.999323133, -0.0094419378, -0.00330376451], 
                    [0.00117895777, 0.00940596282, -0.999955068, -0.0498469479], 
                    [0.999367041, -0.0355640917, 0.00084373493, -0.0994979365], 
                    [0.0, 0.0, 0.0, 1.0]]]
        self.default_camera = {'fps':20, 'width': 1920, 'height':1080, 
                'intrinsics':INTRINSICS, 'lidar2cam':LIDAR2CAM, 'dist':DIST}
        self.smpl_shape = {
            'body_pose': (-1, 69),
            'betas': (-1, 10),
            'global_orient': (-1, 3),
            'transl': (-1, 3),}
        
        super(Hsc4dConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:
        print('Converting HSC4D dataset...')