import os
from typing import List

import time
import numpy as np
import pandas as pd
import json
import cv2
import glob
import random
from tqdm import tqdm
import torch
import smplx
import ast
import copy

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
# import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx, get_keypoint_idxs_by_part
from mmhuman3d.models.body_models.utils import transform_to_camera_frame, batch_transform_to_camera_frame
from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa


import pdb


@DATA_CONVERTERS.register_module()
class BehaveConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train', 'test']

    def __init__(self, modes: List = []) -> None:

        super(BehaveConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        pass
