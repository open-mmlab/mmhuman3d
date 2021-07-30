from typing import Optional, Tuple, Union

import numpy as np

from mmhuman3d.core.conventions.joints_mapping.kp_mapping import JOINTS_FACTORY
from mmhuman3d.core.conventions.joints_mapping.smplx import (
    SMPLX_LIMBS_INDEX,
    SMPLX_PALETTE,
)


def search_limbs(
    data_source: str,
    mask: Optional[Union[np.ndarray, tuple,
                         list]] = None) -> Tuple[dict, dict]:
    """Search the corresponding limbs following the basis smplx limbs.
            The mask could mask out the incorrect keypoints.

    Args:
        data_source (str): data source type.

        mask (Optional[Union[np.ndarray, tuple, list]], optional):
            refer to joints_mapping.kp_mapping. Defaults to None.

    Returns:
        Tuple[dict, dict]: (limbs_target, limbs_palette).
    """
    limbs_source = SMPLX_LIMBS_INDEX
    limbs_palette = SMPLX_PALETTE
    joints_source = JOINTS_FACTORY['smplx']
    joints_target = JOINTS_FACTORY[data_source]
    limbs_target = {}
    for k, part_limbs in limbs_source.items():
        limbs_target[k] = []
        for limb in part_limbs:
            flag = False
            if (joints_source[limb[0]]
                    in joints_target) and (joints_source[limb[1]]
                                           in joints_target):
                if mask is not None:
                    if mask[joints_target.index(joints_source[limb[0]])] \
                        != 0 and mask[joints_target.index(joints_source[
                            limb[1]])] != 0:
                        flag = True
                else:
                    flag = True
                if flag:
                    limbs_target.setdefault(k, []).append([
                        joints_target.index(joints_source[limb[0]]),
                        joints_target.index(joints_source[limb[1]])
                    ])
        if k in limbs_target:
            if k == 'body':
                np.random.seed(0)
                limbs_palette[k] = np.random.randint(
                    0, high=255, size=(len(limbs_target[k]), 3))
            else:
                limbs_palette[k] = np.array(limbs_palette[k])
    return limbs_target, limbs_palette
