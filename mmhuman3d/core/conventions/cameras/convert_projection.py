from typing import Tuple, Union

import numpy as np
import torch

from .convert_convention import convert_cameras


def convert_perspective_to_weakperspective(
        K: Union[torch.Tensor, np.ndarray],
        zmean: Union[torch.Tensor, np.ndarray, float, int],
        resolution: Union[int, Tuple[int, int], torch.Tensor,
                          np.ndarray] = None,
        in_ndc: bool = False,
        convention: str = 'opencv') -> Union[torch.Tensor, np.ndarray]:
    """Convert perspective to weakperspective intrinsic matrix.

    Args:
        K (Union[torch.Tensor, np.ndarray]): input intrinsic matrix, shape
            should be (batch, 4, 4) or (batch, 3, 3).
        zmean (Union[torch.Tensor, np.ndarray, int, float]): zmean for object.
            shape should be (batch, ) or singleton number.
        resolution (Union[int, Tuple[int, int], torch.Tensor, np.ndarray],
            optional): (height, width) of image. Defaults to None.
        in_ndc (bool, optional): whether defined in ndc. Defaults to False.
        convention (str, optional): camera convention. Defaults to 'opencv'.

    Returns:
        Union[torch.Tensor, np.ndarray]: output weakperspective pred_cam,
            shape is (batch, 4)
    """
    assert K is not None, 'K is required.'
    K, _, _ = convert_cameras(
        K=K,
        convention_src=convention,
        convention_dst='pytorch3d',
        is_perspective=True,
        in_ndc_src=in_ndc,
        in_ndc_dst=True,
        resolution_src=resolution)
    if isinstance(zmean, np.ndarray):
        zmean = torch.Tensor(zmean)
    elif isinstance(zmean, (float, int)):
        zmean = torch.Tensor([zmean])
    zmean = zmean.view(-1)
    num_frame = max(zmean.shape[0], K.shape[0])
    new_K = torch.eye(4, 4)[None].repeat(num_frame, 1, 1)
    fx = K[:, 0, 0]
    fy = K[:, 0, 0]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    new_K[:, 0, 0] = fx / zmean
    new_K[:, 1, 1] = fy / zmean
    new_K[:, 0, 3] = cx
    new_K[:, 1, 3] = cy
    return new_K


def convert_weakperspective_to_perspective(
        K: Union[torch.Tensor, np.ndarray],
        zmean: Union[torch.Tensor, np.ndarray, int, float],
        resolution: Union[int, Tuple[int, int], torch.Tensor,
                          np.ndarray] = None,
        in_ndc: bool = False,
        convention: str = 'opencv') -> Union[torch.Tensor, np.ndarray]:
    """Convert perspective to weakperspective intrinsic matrix.

    Args:
        K (Union[torch.Tensor, np.ndarray]): input intrinsic matrix, shape
            should be (batch, 4, 4) or (batch, 3, 3).
        zmean (Union[torch.Tensor, np.ndarray, int, float]): zmean for object.
            shape should be (batch, ) or singleton number.
        resolution (Union[int, Tuple[int, int], torch.Tensor, np.ndarray],
            optional): (height, width) of image. Defaults to None.
        in_ndc (bool, optional): whether defined in ndc. Defaults to False.
        convention (str, optional): camera convention. Defaults to 'opencv'.

    Returns:
        Union[torch.Tensor, np.ndarray]: output weakperspective pred_cam,
            shape is (batch, 4)
    """
    if K.ndim == 2:
        K = K[None]
    if isinstance(zmean, np.ndarray):
        zmean = torch.Tensor(zmean)
    elif isinstance(zmean, (float, int)):
        zmean = torch.Tensor([zmean])
    zmean = zmean.view(-1)
    _N = max(K.shape[0], zmean.shape[0])
    s1 = K[:, 0, 0]
    s2 = K[:, 1, 1]
    c1 = K[:, 0, 3]
    c2 = K[:, 1, 3]
    new_K = torch.zeros(_N, 4, 4)
    new_K[:, 0, 0] = zmean * s1
    new_K[:, 1, 1] = zmean * s2
    new_K[:, 0, 2] = c1
    new_K[:, 1, 2] = c2
    new_K[:, 2, 3] = 1
    new_K[:, 3, 2] = 1

    new_K, _, _ = convert_cameras(
        K=new_K,
        convention_src=convention,
        convention_dst='pytorch3d',
        is_perspective=True,
        in_ndc_src=in_ndc,
        in_ndc_dst=True,
        resolution_src=resolution)
    return new_K
