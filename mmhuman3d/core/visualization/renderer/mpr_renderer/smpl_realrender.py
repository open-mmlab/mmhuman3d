import minimal_pytorch_rasterizer as mpr
import numpy as np
import torch
import mmcv
import cv2
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils.demo_utils import conver_verts_to_cam_coord