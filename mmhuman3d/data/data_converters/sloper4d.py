import glob
import os
import pdb
import random

import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

smplx_shape = {
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

@DATA_CONVERTERS.register_module()
class Sloper4dConverter(BaseModeConverter):

    ACCEPTED_MODES = ['single', 'multiple']

    def __init__(self, modes=[], *args, **kwargs):
        super(Sloper4dConverter, self).__init__(modes, *args, **kwargs)

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.misc = dict(
            bbox_source='keypoints2d_smplx',
            smplx_source='smplifyx',
            cam_param_type='prespective',
            flat_hand_mean=True,
        )

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        pass