import glob
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idxs_by_part


@DATA_CONVERTERS.register_module()
class SaaSManualConverter(BaseModeConverter):
    """SaaS Manual dataset (from yanjun)
    Args:
        modes (list): 'HuMMan_29', 'SAAS_29', 'SAAS_33', 'SAAS_53' for accepted modes
    """
    ACCEPTED_MODES = ['HuMMan_29', 'SAAS_29', 'SAAS_33', 'SAAS_53']

    def __init__(self, modes=[], *args, **kwargs):
        super(SaaSManualConverter, self).__init__(modes, *args, **kwargs)


    def convert_by_mode(self, dataset_path: str, out_path: str, mode: str) -> dict:
        pass