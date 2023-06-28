import ast
import glob
import json
import os
import pdb
import pickle
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
class MoyoConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train', 'val']

    def __init__(self, modes: List = []) -> None:

        super(MoyoConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:

        # parse all seqs
        pkl_ps = glob.glob(
            os.path.join(dataset_path, '*', 'mosh', '*', '*.pkl'))
        pkl_ps = sorted(pkl_ps)

        for pkl_p in pkl_ps:
            # load pickle
            with open(pkl_p, 'rb') as f:
                pickle_info = pickle.load(f, encoding='latin1')

            betas_400 = pickle_info['betas']  # (400,)
            fullpose_165 = pickle_info['fullpose']  # (frames, 165)
            trans = pickle_info['trans']  # (frames, 3)

            stagei_debug_details = pickle_info['stagei_debug_details']
            stageii_debug_details = pickle_info['stageii_debug_details']
            markers_latent = pickle_info['markers_latent']
            latent_labels = pickle_info['latent_labels']
            marker_meta = pickle_info['marker_meta']
            markers_latent_vids = pickle_info['markers_latent_vids']
            v_template_fname = pickle_info['v_template_fname']

            pdb.set_trace()

        pass
