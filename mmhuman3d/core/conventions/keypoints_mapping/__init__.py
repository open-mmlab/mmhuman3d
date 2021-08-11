from typing import Tuple, Union

import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import (
    coco,
    h36m,
    mmpose,
    mpi_inf_3dhp,
    mpii,
    pw3d,
    smpl,
    smplx,
)

KEYPOINTS_FACTORY = {
    'coco': coco.COCO_KEYPOINTS,
    'smplx': smplx.SMPLX_KEYPOINTS,
    'smpl': smpl.SMPL_KEYPOINTS,
    'mmpose': mmpose.MMPOSE_KEYPOINTS,
    'mpi_inf_3dhp': mpi_inf_3dhp.MPI_INF_3DHP_KEYPOINTS,
    'mpi_inf_3dhp_test': mpi_inf_3dhp.MPI_INF_3DHP_TEST_KEYPOINTS,
    'h36m': h36m.H36M_KEYPOINTS,
    'pw3d': pw3d.PW3D_KEYPOINTS,
    'mpii': mpii.MPII_KEYPOINTS
}


def convert_kps(
    keypoints: Union[np.ndarray, torch.Tensor],
    src: str,
    dst: str,
    keypoints_factory: dict = KEYPOINTS_FACTORY,
) -> Tuple[np.ndarray, np.ndarray]:
    """[summary]

    Args:
        keypoints (np.ndarray): [input keypoints array, could be
                (f * n * J * 3/2) or (f * J * 3/2)]
        src (str): [source data type from keypoints_factory]
        dst (str): [destination data type from keypoints_factory]
        keypoints_factory (dict, optional): A class to store the attributes.
                Defaults to keypoints_factory.
    Returns:
        [Tuple(np.ndarray, np.ndarray)]: [out_keypoints, mask]
    """
    assert keypoints.ndim in [3, 4]
    if src == dst:
        return keypoints, np.ones((keypoints.shape[-2]))
    src_names = keypoints_factory[src.lower()]
    dst_names = keypoints_factory[dst.lower()]
    original_shape = keypoints.shape[:-2]
    keypoints = keypoints.reshape(-1, len(src_names), keypoints.shape[-1])
    out_keypoints = np.zeros(
        (keypoints.shape[0], len(dst_names), keypoints.shape[-1]))
    if isinstance(keypoints, np.ndarray):
        mask = np.zeros((len(dst_names)), dtype=np.uint8)
    elif isinstance(keypoints, torch.Tensor):
        mask = torch.zeros((len(dst_names)), dtype=torch.uint8)
    else:
        raise TypeError('keypoints should be torch.Tensor or np.ndarray')
    intersection = set(dst_names) & set(src_names)

    src_to_inter_idx = []
    inter_names = []
    for name in src_names:
        if name in intersection:
            index = src_names.index(name)
            if index not in src_to_inter_idx:
                src_to_inter_idx.append(index)
                inter_names.append(name)
    keypoints_inter = keypoints[:, src_to_inter_idx]
    dst_to_inter_index = []
    for name in inter_names:
        dst_to_inter_index.append(dst_names.index(name))

    out_keypoints[:, dst_to_inter_index] = keypoints_inter
    out_shape = original_shape + (len(dst_names), keypoints.shape[-1])
    out_keypoints = out_keypoints.reshape(out_shape)
    mask[dst_to_inter_index] = 1
    return out_keypoints, mask
