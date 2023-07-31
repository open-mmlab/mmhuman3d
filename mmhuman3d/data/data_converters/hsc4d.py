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

        self.smpl_shape = {
            'body_pose': (-1, 69),
            'betas': (-1, 10),
            'global_orient': (-1, 3),
            'transl': (-1, 3),}
        
        super(Hsc4dConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:
        print('Converting HSC4D dataset...')