import ast
import glob
import json
import os
import pdb
import random
import time
from typing import List

import cv2
import numpy as np
import pandas as pd
import smplx
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData
# import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import transform_to_camera_frame
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class BraceConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train']

    def __init__(self, modes: List = []) -> None:

        super(BraceConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask, keypoints3d, keypoints3d_mask, cam_param
                stored in HumanData() format
        """
        pass
