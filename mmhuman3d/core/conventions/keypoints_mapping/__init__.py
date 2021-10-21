from typing import Optional, Tuple, Union

import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import (
    agora,
    coco,
    coco_wholebody,
    crowdpose,
    h36m,
    human_data_10,
    lsp,
    mmpose,
    mpi_inf_3dhp,
    mpii,
    openpose,
    penn_action,
    posetrack,
    pw3d,
    smpl,
    smplx,
)

KEYPOINTS_FACTORY = {
    'human_data_1.0': human_data_10.HUMAN_DATA_10,
    'agora': agora.AGORA_KEYPOINTS,
    'coco': coco.COCO_KEYPOINTS,
    'coco_wholebody': coco_wholebody.COCO_WHOLEBODY_KEYPOINTS,
    'crowdpose': crowdpose.CROWDPOSE_KEYPOINTS,
    'smplx': smplx.SMPLX_KEYPOINTS,
    'smpl': smpl.SMPL_KEYPOINTS,
    'mmpose': mmpose.MMPOSE_KEYPOINTS,
    'mpi_inf_3dhp': mpi_inf_3dhp.MPI_INF_3DHP_KEYPOINTS,
    'mpi_inf_3dhp_test': mpi_inf_3dhp.MPI_INF_3DHP_TEST_KEYPOINTS,
    'penn_action': penn_action.PENN_ACTION_KEYPOINTS,
    'h36m': h36m.H36M_KEYPOINTS,
    'pw3d': pw3d.PW3D_KEYPOINTS,
    'mpii': mpii.MPII_KEYPOINTS,
    'lsp': lsp.LSP_KEYPOINTS,
    'posetrack': posetrack.POSETRACK_KEYPOINTS,
    'openpose_25': openpose.OPENPOSE_25_KEYPOINTS,
    'openpose_135': openpose.OPENPOSE_135_KEYPOINTS
}

__KEYPOINTS_MAPPING_CACHE__ = {}


def convert_kps(
    keypoints: Union[np.ndarray, torch.Tensor],
    src: str,
    dst: str,
    compress: bool = False,
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
    assert keypoints.ndim in {3, 4}
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

    src_to_intersection_idx, \
        dst_to_intersection_index, \
        _ = get_mapping(src, dst, keypoints_factory)

    keypoints_intersection = keypoints[:, src_to_intersection_idx]
    mask_intersection = original_mask[
        src_to_intersection_idx] if original_mask is not None else 1
    out_keypoints[:, dst_to_intersection_index] = keypoints_intersection
    out_shape = original_shape + (len(dst_names), keypoints.shape[-1])
    out_keypoints = out_keypoints.reshape(out_shape)
    mask[dst_to_intersection_index] = mask_intersection

    if compress:
        sorted_dst_to_intersection_index = sorted(dst_to_intersection_index)
        out_keypoints = np.take(
            out_keypoints, sorted_dst_to_intersection_index, axis=1)
    return out_keypoints, mask


def compress_converted_kps(
    zero_pad_array: Union[np.ndarray, torch.Tensor],
    mask_array: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """Compress keypoints that are zero-padded after applying convert_kps.

    Args:
        keypoints (np.ndarray): input keypoints array, could be
            (f * n * J * 3/2) or (f * J * 3/2). You can set keypoints as
            np.zeros((1, J, 2)) if you only need mask.
        mask [Union[np.ndarray, torch.Tensor]]:
            The original mask to mark the existence of the keypoints.
    Returns:
        Union[np.ndarray, torch.Tensor]: out_keypoints
    """

    assert mask_array.shape[0] == zero_pad_array.shape[1]
    valid_mask_index = np.where(mask_array == 1)[0]
    compressed_array = np.take(zero_pad_array, valid_mask_index, axis=1)
    return compressed_array


def get_mapping(src: str,
                dst: str,
                keypoints_factory: dict = KEYPOINTS_FACTORY):
    """Get mapping list from src to dst.

    Args:
        src (str): source data type from keypoints_factory.
        dst (str): destination data type from keypoints_factory.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.

    Returns:
        list:
            [src_to_intersection_idx, dst_to_intersection_index,
             intersection_names]
    """
    if src in __KEYPOINTS_MAPPING_CACHE__ and \
            dst in __KEYPOINTS_MAPPING_CACHE__[src]:
        return __KEYPOINTS_MAPPING_CACHE__[src][dst]
    else:
        src_names = keypoints_factory[src.lower()]
        dst_names = keypoints_factory[dst.lower()]
        intersection = set(dst_names) & set(src_names)
        # for each name in intersection,
        # put its index in src_names into src_to_intersection_idx
        src_to_intersection_idx = []
        intersection_names = []
        for name in src_names:
            if name in intersection:
                index = src_names.index(name)
                if index not in src_to_intersection_idx:
                    src_to_intersection_idx.append(index)
                    intersection_names.append(name)
        # for each name in intersection,
        # put its index in dst_names into dst_to_intersection_index
        dst_to_intersection_index = []
        for name in intersection_names:
            dst_to_intersection_index.append(dst_names.index(name))
        mapping_list = [
            src_to_intersection_idx, dst_to_intersection_index,
            intersection_names
        ]
        if src not in __KEYPOINTS_MAPPING_CACHE__:
            __KEYPOINTS_MAPPING_CACHE__[src] = {}
        __KEYPOINTS_MAPPING_CACHE__[src][dst] = mapping_list
        return mapping_list
