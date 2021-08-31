import colorsys
from typing import Optional, Tuple, Union

import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY
from mmhuman3d.core.conventions.keypoints_mapping.smplx import (
    SMPLX_LIMBS_INDEX,
    SMPLX_PALETTE,
)


def search_limbs(
    data_source: str,
    mask: Optional[Union[np.ndarray, tuple,
                         list]] = None) -> Tuple[dict, dict]:
    """Search the corresponding limbs following the basis smplx limbs. The mask
    could mask out the incorrect keypoints.

    Args:
        data_source (str): data source type.

        mask (Optional[Union[np.ndarray, tuple, list]], optional):
            refer to keypoints_mapping. Defaults to None.

    Returns:
        Tuple[dict, dict]: (limbs_target, limbs_palette).
    """
    limbs_source = SMPLX_LIMBS_INDEX
    limbs_palette = SMPLX_PALETTE
    keypoints_source = KEYPOINTS_FACTORY['smplx']
    keypoints_target = KEYPOINTS_FACTORY[data_source]
    limbs_target = {}
    for k, part_limbs in limbs_source.items():
        limbs_target[k] = []
        for limb in part_limbs:
            flag = False
            if (keypoints_source[limb[0]]
                    in keypoints_target) and (keypoints_source[limb[1]]
                                              in keypoints_target):
                if mask is not None:
                    if mask[keypoints_target.index(keypoints_source[
                            limb[0]])] != 0 and mask[keypoints_target.index(
                                keypoints_source[limb[1]])] != 0:
                        flag = True
                else:
                    flag = True
                if flag:
                    limbs_target.setdefault(k, []).append([
                        keypoints_target.index(keypoints_source[limb[0]]),
                        keypoints_target.index(keypoints_source[limb[1]])
                    ])
        if k in limbs_target:
            if k == 'body':
                np.random.seed(0)
                limbs_palette[k] = np.random.randint(
                    0, high=255, size=(len(limbs_target[k]), 3))
            else:
                limbs_palette[k] = np.array(limbs_palette[k])
    return limbs_target, limbs_palette


def get_different_colors(number_of_colors, flag=0):
    nst0 = np.random.get_state()
    np.random.seed(flag)
    colors = []
    for i in np.arange(0., 360., 360. / number_of_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors_np = np.asarray(colors)
    colors_np = (255 * colors_np).astype(np.int32)
    # recover the random state
    np.random.set_state(nst0)
    return colors_np
