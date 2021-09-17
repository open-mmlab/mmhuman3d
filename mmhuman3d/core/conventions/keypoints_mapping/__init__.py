from typing import Optional, Tuple, Union

import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import (
    agora,
    coco,
    coco_wholebody,
    h36m,
    lsp,
    mmpose,
    mpi_inf_3dhp,
    mpii,
    penn_action,
    pw3d,
    smpl,
    smplx,
)

KEYPOINTS_FACTORY = {
    'agora': agora.AGORA_KEYPOINTS,
    'coco': coco.COCO_KEYPOINTS,
    'coco_wholebody': coco_wholebody.COCO_WHOLEBODY_KEYPOINTS,
    'smplx': smplx.SMPLX_KEYPOINTS,
    'smpl': smpl.SMPL_KEYPOINTS,
    'mmpose': mmpose.MMPOSE_KEYPOINTS,
    'mpi_inf_3dhp': mpi_inf_3dhp.MPI_INF_3DHP_KEYPOINTS,
    'mpi_inf_3dhp_test': mpi_inf_3dhp.MPI_INF_3DHP_TEST_KEYPOINTS,
    'penn_action': penn_action.PENN_ACTION_KEYPOINTS,
    'h36m': h36m.H36M_KEYPOINTS,
    'pw3d': pw3d.PW3D_KEYPOINTS,
    'mpii': mpii.MPII_KEYPOINTS,
    'lsp': lsp.LSP_KEYPOINTS
}


def convert_kps(
    keypoints: Union[np.ndarray, torch.Tensor],
    src: str,
    dst: str,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    keypoints_factory: dict = KEYPOINTS_FACTORY,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Convert keypoints following the mapping correspondence between src and
    dst keypoints definition. Supported conventions by now: agora, coco, smplx,
    smpl, mmpose, mpi_inf_3dhp, mpi_inf_3dhp_test, h36m, pw3d, mpii, lsp.

    Args:
        keypoints (np.ndarray): input keypoints array, could be
            (f * n * J * 3/2) or (f * J * 3/2). You can set keypoints as
            np.zeros((1, J, 2)) if you only need mask.
        src (str): source data type from keypoints_factory.
        dst (str): destination data type from keypoints_factory.
        mask (Optional[Union[np.ndarray, torch.Tensor]], optional):
            The original mask to mark the existence of the keypoints.
            None represents all ones mask.
            Defaults to None.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.
    Returns:
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
            : tuple of (out_keypoints, mask). out_keypoints and mask will be of
            the same type.
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

    original_mask = mask
    if original_mask is not None:
        original_mask = original_mask.reshape(-1)
        assert original_mask.shape[0] == len(
            src_names), f'The length of mask should be {len(src_names)}'

    if isinstance(keypoints, np.ndarray):
        mask = np.zeros((len(dst_names)), dtype=np.uint8)
    elif isinstance(keypoints, torch.Tensor):
        mask = torch.zeros((len(dst_names)), dtype=torch.uint8)
    else:
        raise TypeError('keypoints should be torch.Tensor or np.ndarray')
    intersection = set(dst_names) & set(src_names)

    src_to_intersection_idx = []
    intersection_names = []
    for name in src_names:
        if name in intersection:
            index = src_names.index(name)
            if index not in src_to_intersection_idx:
                src_to_intersection_idx.append(index)
                intersection_names.append(name)
    keypoints_intersection = keypoints[:, src_to_intersection_idx]
    mask_intersection = original_mask[
        src_to_intersection_idx] if original_mask is not None else 1
    dst_to_intersection_index = []
    for name in intersection_names:
        dst_to_intersection_index.append(dst_names.index(name))

    out_keypoints[:, dst_to_intersection_index] = keypoints_intersection
    out_shape = original_shape + (len(dst_names), keypoints.shape[-1])
    out_keypoints = out_keypoints.reshape(out_shape)
    mask[dst_to_intersection_index] = mask_intersection
    return out_keypoints, mask
