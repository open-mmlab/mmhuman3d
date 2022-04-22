import warnings

import numpy as np
import scipy.signal as signal
import torch
from scipy.ndimage.filters import gaussian_filter1d

from ..builder import POST_PROCESSING


@POST_PROCESSING.register_module(name=['Gaus1dPostProcessing', 'gaus1d'])
class Gaus1dPostProcessing:
    """Applies median filter and then gaussian filter. code from:
    https://github.com/akanazawa/human_dynamics/blob/mas
    ter/src/util/smooth_bbox.py.

    Args:
        x (np.ndarray): input pose
        window_size (int, optional): for median filters (must be odd).
        sigma (float, optional): Sigma for gaussian smoothing.

    Returns:
        np.ndarray: Smoothed poses
    """

    def __init__(self, cfg, device=None):
        super(Gaus1dPostProcessing, self).__init__()

        self.window_size = cfg["window_size"]
        self.sigma = cfg["sigma"]

    def __call__(self, x=None):
        smooth_poses = gaussian_filter1d(x, self.sigma, axis=0)

        return smooth_poses
