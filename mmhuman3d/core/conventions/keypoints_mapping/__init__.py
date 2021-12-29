from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.utils import print_log

from mmhuman3d.core.conventions.keypoints_mapping import (
    agora,
    coco,
    coco_wholebody,
    crowdpose,
    gta,
    h36m,
    human_data,
    hybrik,
    instavariety,
    lsp,
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
    'human_data': human_data.HUMAN_DATA,
    'agora': agora.AGORA_KEYPOINTS,
    'coco': coco.COCO_KEYPOINTS,
    'coco_wholebody': coco_wholebody.COCO_WHOLEBODY_KEYPOINTS,
    'crowdpose': crowdpose.CROWDPOSE_KEYPOINTS,
    'smplx': smplx.SMPLX_KEYPOINTS,
    'smpl': smpl.SMPL_KEYPOINTS,
    'smpl_45': smpl.SMPL_45_KEYPOINTS,
    'smpl_54': smpl.SMPL_54_KEYPOINTS,
    'smpl_49': smpl.SMPL_49_KEYPOINTS,
    'smpl_24': smpl.SMPL_24_KEYPOINTS,
    'mpi_inf_3dhp': mpi_inf_3dhp.MPI_INF_3DHP_KEYPOINTS,
    'mpi_inf_3dhp_test': mpi_inf_3dhp.MPI_INF_3DHP_TEST_KEYPOINTS,
    'penn_action': penn_action.PENN_ACTION_KEYPOINTS,
    'h36m': h36m.H36M_KEYPOINTS,
    'h36m_mmpose': h36m.H36M_KEYPOINTS_MMPOSE,
    'pw3d': pw3d.PW3D_KEYPOINTS,
    'mpii': mpii.MPII_KEYPOINTS,
    'lsp': lsp.LSP_KEYPOINTS,
    'posetrack': posetrack.POSETRACK_KEYPOINTS,
    'instavariety': instavariety.INSTAVARIETY_KEYPOINTS,
    'openpose_25': openpose.OPENPOSE_25_KEYPOINTS,
    'openpose_135': openpose.OPENPOSE_135_KEYPOINTS,
    'hybrik_29': hybrik.HYBRIK_29_KEYPOINTS,
    'hybrik_hp3d': mpi_inf_3dhp.HYBRIK_MPI_INF_3DHP_KEYPOINTS,
    'gta': gta.GTA_KEYPOINTS
}

__KEYPOINTS_MAPPING_CACHE__ = defaultdict(dict)


def convert_kps(
    keypoints: Union[np.ndarray, torch.Tensor],
    src: str,
    dst: str,
    approximate: bool = False,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    keypoints_factory: dict = KEYPOINTS_FACTORY,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Convert keypoints following the mapping correspondence between src and
    dst keypoints definition. Supported conventions by now: agora, coco, smplx,
    smpl, mpi_inf_3dhp, mpi_inf_3dhp_test, h36m, h36m_mmpose, pw3d, mpii, lsp.
    Args:
        keypoints [Union[np.ndarray, torch.Tensor]]: input keypoints array,
            could be (f * n * J * 3/2) or (f * J * 3/2).
            You can set keypoints as np.zeros((1, J, 2))
            if you only need mask.
        src (str): source data type from keypoints_factory.
        dst (str): destination data type from keypoints_factory.
        approximate (bool): control whether approximate mapping is allowed.
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
    extra_dims = keypoints.shape[:-2]
    keypoints = keypoints.reshape(-1, len(src_names), keypoints.shape[-1])

    if isinstance(keypoints, np.ndarray):
        out_keypoints = np.zeros(
            (keypoints.shape[0], len(dst_names), keypoints.shape[-1]))
    else:
        out_keypoints = torch.zeros(
            (keypoints.shape[0], len(dst_names), keypoints.shape[-1]),
            device=keypoints.device,
            dtype=keypoints.dtype)

    original_mask = mask
    if original_mask is not None:
        original_mask = original_mask.reshape(-1)
        assert original_mask.shape[0] == len(
            src_names), f'The length of mask should be {len(src_names)}'

    if isinstance(keypoints, np.ndarray):
        mask = np.zeros((len(dst_names)), dtype=np.uint8)
    elif isinstance(keypoints, torch.Tensor):
        mask = torch.zeros((len(dst_names)),
                           dtype=torch.uint8,
                           device=keypoints.device)
    else:
        raise TypeError('keypoints should be torch.Tensor or np.ndarray')

    dst_idxs, src_idxs, _ = \
        get_mapping(src, dst, approximate, keypoints_factory)
    out_keypoints[:, dst_idxs] = keypoints[:, src_idxs]
    out_shape = extra_dims + (len(dst_names), keypoints.shape[-1])
    out_keypoints = out_keypoints.reshape(out_shape)

    mask[dst_idxs] = original_mask[src_idxs] \
        if original_mask is not None else 1.0

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
                approximate: bool = False,
                keypoints_factory: dict = KEYPOINTS_FACTORY):
    """Get mapping list from src to dst.

    Args:
        src (str): source data type from keypoints_factory.
        dst (str): destination data type from keypoints_factory.
        approximate (bool): control whether approximate mapping is allowed.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.

    Returns:
        list:
            [src_to_intersection_idx, dst_to_intersection_index,
             intersection_names]
    """
    if src in __KEYPOINTS_MAPPING_CACHE__ and \
        dst in __KEYPOINTS_MAPPING_CACHE__[src] and \
            __KEYPOINTS_MAPPING_CACHE__[src][dst][3] == approximate:
        return __KEYPOINTS_MAPPING_CACHE__[src][dst][:3]
    else:
        src_names = keypoints_factory[src.lower()]
        dst_names = keypoints_factory[dst.lower()]

        dst_idxs, src_idxs, intersection = [], [], []
        unmapped_names, approximate_names = [], []
        for dst_idx, dst_name in enumerate(dst_names):
            matched = False
            try:
                src_idx = src_names.index(dst_name)
            except ValueError:
                src_idx = -1
            if src_idx >= 0:
                matched = True
                dst_idxs.append(dst_idx)
                src_idxs.append(src_idx)
                intersection.append(dst_name)
            # approximate mapping
            if approximate and not matched:
                try:
                    part_list = human_data.APPROXIMATE_MAP[dst_name]
                except KeyError:
                    continue
                for approximate_name in part_list:
                    try:
                        src_idx = src_names.index(approximate_name)
                    except ValueError:
                        src_idx = -1
                    if src_idx >= 0:
                        dst_idxs.append(dst_idx)
                        src_idxs.append(src_idx)
                        intersection.append(dst_name)
                        unmapped_names.append(src_names[src_idx])
                        approximate_names.append(dst_name)
                        break

        if unmapped_names:
            warn_message = \
                f'Approximate mapping {unmapped_names}' +\
                f' to {approximate_names}'
            print_log(msg=warn_message)

        mapping_list = [dst_idxs, src_idxs, intersection, approximate]

        if src not in __KEYPOINTS_MAPPING_CACHE__:
            __KEYPOINTS_MAPPING_CACHE__[src] = {}
        __KEYPOINTS_MAPPING_CACHE__[src][dst] = mapping_list
        return mapping_list[:3]


def get_flip_pairs(convention: str = 'smplx',
                   keypoints_factory: dict = KEYPOINTS_FACTORY) -> List[int]:
    """Get indices of left, right keypoint pairs from specified convention.

    Args:
        convention (str): data type from keypoints_factory.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.
    Returns:
        List[int]: left, right keypoint indices
    """
    flip_pairs = []
    keypoints = keypoints_factory[convention]
    left_kps = [kp for kp in keypoints if 'left_' in kp]
    for left_kp in left_kps:
        right_kp = left_kp.replace('left_', 'right_')
        flip_pairs.append([keypoints.index(kp) for kp in [left_kp, right_kp]])
    return flip_pairs


def get_keypoint_idxs_by_part(
        part: str,
        convention: str = 'smplx',
        keypoints_factory: dict = KEYPOINTS_FACTORY) -> List[int]:
    """Get part keypoints indices from specified part and convention.

    Args:
        part (str): part to search from
        convention (str): data type from keypoints_factory.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.
    Returns:
        List[int]: part keypoint indices
    """
    humandata_parts = human_data.HUMAN_DATA_PARTS
    keypoints = keypoints_factory[convention]
    if part not in humandata_parts.keys():
        raise ValueError('part not in allowed parts')
    part_keypoints = list(set(humandata_parts[part]) & set(keypoints))
    part_keypoints_idx = [keypoints.index(kp) for kp in part_keypoints]
    return part_keypoints_idx


def get_keypoint_idx(name: str,
                     convention: str = 'smplx',
                     approximate: bool = False,
                     keypoints_factory: dict = KEYPOINTS_FACTORY) -> List[int]:
    """Get keypoint index from specified convention with keypoint name.

    Args:
        name (str): keypoint name
        convention (str): data type from keypoints_factory.
        approximate (bool): control whether approximate mapping is allowed.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.
    Returns:
        List[int]: keypoint index
    """
    keypoints = keypoints_factory[convention]
    try:
        idx = keypoints.index(name)
    except ValueError:
        idx = -1  # not matched
    if approximate and idx == -1:
        try:
            part_list = human_data.APPROXIMATE_MAP[name]
        except KeyError:
            return idx
        for approximate_name in part_list:
            try:
                idx = keypoints.index(approximate_name)
            except ValueError:
                idx = -1
            if idx >= 0:
                return idx
    return idx


def get_keypoint_num(convention: str = 'smplx',
                     keypoints_factory: dict = KEYPOINTS_FACTORY) -> List[int]:
    """Get number of keypoints of specified convention.

    Args:
        convention (str): data type from keypoints_factory.
        keypoints_factory (dict, optional): A class to store the attributes.
            Defaults to keypoints_factory.
    Returns:
        List[int]: part keypoint indices
    """
    keypoints = keypoints_factory[convention]
    return len(keypoints)
