from typing import Tuple, Union

import numpy as np
import torch

from mmhuman3d.core.conventions.joints_mapping import (
    coco,
    h36m,
    mmpose,
    mpi_inf_3dhp,
    mpii,
    pw3d,
    smpl,
    smplx,
)

JOINTS_FACTORY = {
    'coco': coco.COCO_JOINTS,
    'smplx': smplx.SMPLX_JOINTS,
    'smpl': smpl.SMPL_JOINTS,
    'mmpose': mmpose.MMPOSE_JOINTS,
    'mpi_inf_3dhp': mpi_inf_3dhp.MPI_INF_3DHP_JOINTS,
    'mpi_inf_3dhp_test': mpi_inf_3dhp.MPI_INF_3DHP_TEST_JOINTS,
    'h36m': h36m.H36M_JOINTS,
    'pw3d': pw3d.PW3D_JOINTS,
    'mpii': mpii.MPII_JOINTS
}


def convert_kps(joints: Union[np.ndarray, torch.Tensor], src: str,
                dst: str) -> Tuple[np.ndarray, np.ndarray]:
    """[summary]

    Args:
        joints (np.ndarray): [input joints array, could be (f * n * J * 3/2)
                             or (f * J * 3/2)]
        src (str): [source data type from JOINTS_FACTORY]
        dst (str): [destination data type from JOINTS_FACTORY]
    Returns:
        [Tuple(np.ndarray, np.ndarray)]: [out_joints, mask ]
    """
    assert joints.ndim in [3, 4]
    if src == dst:
        return joints, np.ones((joints.shape[-2]))
    src_names = JOINTS_FACTORY[src.lower()]
    dst_names = JOINTS_FACTORY[dst.lower()]
    original_shape = joints.shape[:-2]
    joints = joints.reshape(-1, len(src_names), joints.shape[-1])
    out_joints = np.zeros((joints.shape[0], len(dst_names), joints.shape[-1]))
    if isinstance(joints, np.ndarray):
        mask = np.zeros((len(dst_names)), dtype=np.uint8)
    elif isinstance(joints, torch.Tensor):
        mask = torch.zeros((len(dst_names)), dtype=torch.uint8)
    else:
        raise TypeError('joints should be torch.Tensor or np.ndarray')
    intersection = set(dst_names) & set(src_names)

    src_to_inter_idx = []
    inter_names = []
    for name in src_names:
        if name in intersection:
            index = src_names.index(name)
            if index not in src_to_inter_idx:
                src_to_inter_idx.append(index)
                inter_names.append(name)
    joints_inter = joints[:, src_to_inter_idx]
    dst_to_inter_index = []
    for name in inter_names:
        dst_to_inter_index.append(dst_names.index(name))

    out_joints[:, dst_to_inter_index] = joints_inter
    out_shape = original_shape + (len(dst_names), joints.shape[-1])
    out_joints = out_joints.reshape(out_shape)
    mask[dst_to_inter_index] = 1
    return out_joints, mask
