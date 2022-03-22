from typing import Optional, Tuple, Union

import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY
from mmhuman3d.core.conventions.keypoints_mapping.human_data import (
    HUMAN_DATA_LIMBS_INDEX,
    HUMAN_DATA_PALETTE,
)


def search_limbs(
        data_source: str,
        mask: Optional[Union[np.ndarray, tuple, list]] = None,
        keypoints_factory: dict = KEYPOINTS_FACTORY) -> Tuple[dict, dict]:
    """Search the corresponding limbs following the basis human_data limbs. The
    mask could mask out the incorrect keypoints.

    Args:
        data_source (str): data source type.
        mask (Optional[Union[np.ndarray, tuple, list]], optional):
            refer to keypoints_mapping. Defaults to None.
        keypoints_factory (dict, optional): Dict of all the conventions.
            Defaults to KEYPOINTS_FACTORY.
    Returns:
        Tuple[dict, dict]: (limbs_target, limbs_palette).
    """
    limbs_source = HUMAN_DATA_LIMBS_INDEX
    limbs_palette = HUMAN_DATA_PALETTE
    keypoints_source = keypoints_factory['human_data']
    keypoints_target = keypoints_factory[data_source]
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


def get_smpl_neighbor_triplets():
    return [
        [ 0,  1, 2 ],  # 0
        [ 1,  4, 0 ],  # 1
        [ 2,  0, 5 ],  # 2
        [ 3,  0, 6 ],  # 3
        [ 4,  7, 1 ],  # 4
        [ 5,  2, 8 ],  # 5
        [ 6,  3, 9 ],  # 6
        [ 7, 10, 4 ],  # 7
        [ 8,  5, 11],  # 8
        [ 9, 13, 14],  # 9
        [10,  7, 4 ],  # 10
        [11,  8, 5 ],  # 11
        [12,  9, 15],  # 12
        [13, 16, 9 ],  # 13
        [14,  9, 17],  # 14
        [15,  9, 12],  # 15
        [16, 18, 13],  # 16
        [17, 14, 19],  # 17
        [18, 20, 16],  # 18
        [19, 17, 21],  # 19
        [20, 22, 18],  # 20
        [21, 19, 23],  # 21
        [22, 20, 18],  # 22
        [23, 19, 21],  # 23
    ]