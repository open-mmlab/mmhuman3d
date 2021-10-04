import warnings
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch

from mmhuman3d.utils.transforms import ee_to_rotmat, rotmat_to_ee

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

CAMERA_CONVENTIONS = {
    'pytorch3d': {
        'axis': '-xyz',
        'left_mm_extrinsic': False,
        'world_to_view': False,
        'left_mm_intrinsic': True,
    },
    'pyrender': {
        'axis': 'xy-z',
        'left_mm_extrinsic': True,
        'world_to_view': False,
        'left_mm_intrinsic': True,
    },
    'opengl': {
        'axis': 'xy-z',
        'left_mm_extrinsic': True,
        'world_to_view': False,
        'left_mm_intrinsic': True,
    },
    'open3d': {
        'axis': 'x-yz',
        'left_mm_extrinsic': False,
        'world_to_view': False,
        'left_mm_intrinsic': False,
    },
    'opencv': {
        'axis': 'x-yz',
        'left_mm_extrinsic': True,
        'world_to_view': True,
        'left_mm_intrinsic': True,
    },
    'unity': {
        'axis': 'xyz',
        'left_mm_extrinsic': True,
        'world_to_view': False,
        'left_mm_intrinsic': True,
    },
    'blender': {
        'axis': 'xy-z',
        'left_mm_extrinsic': True,
        'world_to_view': False,
        'left_mm_intrinsic': True,
    },
    'maya': {
        'axis': 'xy-z',
        'left_mm_extrinsic': True,
        'world_to_view': False,
        'left_mm_intrinsic': True,
    }
}


def convert_cameras(
    K: Optional[Union[torch.Tensor, np.ndarray]] = None,
    R: Optional[Union[torch.Tensor, np.ndarray]] = None,
    T: Optional[Union[torch.Tensor, np.ndarray]] = None,
    projection: Literal['perspective', 'orthographics',
                        'weakperspective'] = 'perspective',
    convention_src: str = 'opencv',  # opencv definition
    convention_dst: str = 'pytorch3d',  # pytorch3d definition
    in_ndc_src: bool = True,
    in_ndc_dst: bool = True,
    image_size_src: Optional[Union[int, Tuple[int, int], torch.Tensor,
                                   np.ndarray]] = None,
    image_size_dst: Optional[Union[int, Tuple[int, int], torch.Tensor,
                                   np.ndarray]] = None,
    camera_conventions: dict = CAMERA_CONVENTIONS,
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray],
           Union[torch.Tensor, np.ndarray]]:
    """Convert the intrinsic matrix K and extrinsic matrix [R|T] from source
    convention to destination convention.

    Args:
        K (Union[torch.Tensor, np.ndarray]): Intrinsic matrix,
            shape should be (batch_size, 4, 4). Will be ignored if None.
        R (Optional[Union[torch.Tensor, np.ndarray]], optional):
            Extrinsic rotation matrix. Will be identity if None.
            Defaults to None.
        T (Optional[Union[torch.Tensor, np.ndarray]], optional):
            Extrinsic translation matrix. Will be zeros if None.
            Defaults to None.
        projection (Literal[, optional): Projection type of the camera.
            Defaults to 'perspective'.

        _____________________________________________________________________
        # Camera dependent args
        convention_src (str, optional): convention of source camera,

        convention_dst (str, optional): convention of destination camera,


        We define the convention of cameras by the order of right to, up to,
            and off screen.
            E.g., the first one is pyrender and its convention should be
            '+x+y+z'. '+' could be ignored.
            The second one is opencv and its convention should be '+x-y-z'.
            The third one is pytorch3d and its convention should be '-xyz'.
                    opengl(pyrender)     opencv            pytorch3d
                    y                   z                     y
                    |                  /                      |
                    |                 /                       |
                    |_______x        /________x     x________ |
                    /                |                        /
                   /                 |                       /
                z /                y |                    z /

        in_ndc_src (bool, optional): Whether is the source camera defined
            in ndc.
            Defaults to True.
        in_ndc_dst (bool, optional): Whether is the destination camera defined
            in ndc.
            Defaults to True.

        in camera_convention, we define these args as:
            1). `left_mm_ex` means extrinsic matrix `K` is left matrix
                multiplcation defined.
            2). `left_mm_in` means intrinsic matrix [`R`| `T`] is left
                matrix multiplcation defined.
            3) `world_to_view` means extrinsic matrix [`R`| `T`] is defined
                as world to view.

        image_size_src (Optional[Union[int, Tuple[int, int], torch.Tensor,
            np.ndarray]], optional):
            Source camera image size of (height, width).
            Required if defined in screen.
            Will be square if int.
            Shape should be (2,) if `array` or `tensor`.
            Defaults to None.
        image_size_dst (Optional[Union[int, Tuple[int, int], torch.Tensor,
            np.ndarray]], optional):
            Destination camera image size of (height, width).
            Required if defined in screen.
            Will be square if int.
            Shape should be (2,) if `array` or `tensor`.
            Defaults to None.
        camera_conventions: (dict, optional): `dict` containing
            pre-defined camera convention information.
            Defaults to CAMERA_CONVENTIONS.

    Raises:
        TypeError: K, R, T should all be `torch.Tensor` or `np.ndarray`.

    Returns:
        Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None],
            Union[torch.Tensor, None]]:
            Converted K, R, T matrix of `tensor`.
    """
    assert projection in [
        'perspective', 'weakperspective', 'orthographics', 'fovorthographics',
        'fovperspective'
    ]
    if projection in ['perspective', 'fovperspective']:
        projection = 'perspective'
    convention_dst = convention_dst.lower()
    convention_src = convention_src.lower()

    assert convention_dst in CAMERA_CONVENTIONS
    assert convention_src in CAMERA_CONVENTIONS

    # encode convention to sign and axis order
    def _enc_convention(convention):
        if convention in camera_conventions:
            convention = camera_conventions[convention]['axis']
        else:
            assert set(convention).issubset(
                {'x', 'y', 'z', '+',
                 '-'}), 'Wrong convention string, choose either in'
            f'set({camera_conventions.keys()}) or define by xyz.'
        sign = [1, 1, 1]
        convention = '_' + convention
        count = 0
        axis_order = ''
        for i in range(len(convention)):
            if convention[i] in 'xyz':
                axis_order += convention[i]
                if convention[i - 1] == '-':
                    sign[count] *= -1
                count += 1
        return sign, axis_order

    left_mm_ex_src = CAMERA_CONVENTIONS[convention_src].get(
        'left_mm_extrinsic', True)
    world_to_view_src = CAMERA_CONVENTIONS[convention_src].get(
        'world_to_view', False)
    left_mm_in_src = CAMERA_CONVENTIONS[convention_src].get(
        'left_mm_intrinsic', False)

    left_mm_ex_dst = CAMERA_CONVENTIONS[convention_dst].get(
        'left_mm_extrinsic', True)
    world_to_view_dst = CAMERA_CONVENTIONS[convention_dst].get(
        'world_to_view', False)
    left_mm_in_dst = CAMERA_CONVENTIONS[convention_dst].get(
        'left_mm_intrinsic', False)

    sign_src, axis_src = _enc_convention(convention_src)
    sign_dst, axis_dst = _enc_convention(convention_dst)
    sign = torch.Tensor(sign_dst) / torch.Tensor(sign_src)

    type_ = []
    for x in [K, R, T]:
        if x is not None:
            type_.append(type(x))
    if len(type_) > 0:
        if not all(x == type_[0] for x in type_):
            raise TypeError('Input type should be the same.')

    use_numpy = False
    if np.ndarray in type_:
        use_numpy = True
    # convert raw matrix to tensor
    if isinstance(K, np.ndarray):
        new_K = torch.Tensor(K)
    elif K is None:
        new_K = None
    elif isinstance(K, torch.Tensor):
        new_K = K.clone()
    else:
        raise TypeError(
            f'K should be `torch.Tensor` or `numpy.ndarray`, type(K): '
            f'{type(K)}')

    if isinstance(R, np.ndarray):
        new_R = torch.Tensor(R).view(-1, 3, 3)
    elif R is None:
        new_R = torch.eye(3, 3)[None]
    elif isinstance(R, torch.Tensor):
        new_R = R.view(-1, 3, 3).clone()
    else:
        raise TypeError(
            f'R should be `torch.Tensor` or `numpy.ndarray`, type(R): '
            f'{type(R)}')

    if isinstance(T, np.ndarray):
        new_T = torch.Tensor(T).view(-1, 3)
    elif T is None:
        new_T = torch.zeros(1, 3)
    elif isinstance(T, torch.Tensor):
        new_T = T.view(-1, 3).clone()
    else:
        raise TypeError(
            f'T should be `torch.Tensor` or `numpy.ndarray`, type(T): '
            f'{type(T)}')

    if axis_dst != axis_src:
        new_R = ee_to_rotmat(
            rotmat_to_ee(new_R, convention=axis_src), convention=axis_dst)

    # convert extrinsic to view_to_world
    if world_to_view_src is True:
        new_R, new_T = convert_world_view(new_R, new_T)

    # right mm to left mm
    if (not left_mm_ex_src) and left_mm_ex_dst:
        new_R *= sign
        new_R = new_R.permute(0, 2, 1)
    # left mm to right mm
    elif left_mm_ex_src and (not left_mm_ex_dst):
        new_R = new_R.permute(0, 2, 1)
        new_R *= sign
    # right_mm to right mm
    elif (not left_mm_ex_dst) and (not left_mm_ex_src):
        new_R *= sign
    # left mm to left mm
    elif left_mm_ex_src and left_mm_ex_dst:
        new_R *= sign.view(3, 1)
    new_T *= sign

    # convert extrinsic to as definition
    if world_to_view_dst is True:
        new_R, new_T = convert_world_view(new_R, new_T)

    # in ndc or in screen
    if in_ndc_dst is False and in_ndc_src is True:
        assert image_size_dst is not None, \
            'dst in screen, should specific image_size_dst.'

    if in_ndc_src is False and in_ndc_dst is True:
        assert image_size_src is not None, \
            'src in screen, should specific image_size_dst.'
    if image_size_dst is None:
        image_size_dst = 2.0
    if image_size_src is None:
        image_size_src = 2.0

    if new_K is not None:
        if left_mm_in_src is False and left_mm_in_dst is True:
            new_K = new_K.permute(0, 2, 1)
        if new_K.shape[-2:] == (3, 3):
            new_K = convert_K_3x3_to_4x4(new_K, projection)
        # src in ndc, dst in screen

        if in_ndc_src is True and (in_ndc_dst is False):
            convert_ndc_to_screen(
                K=new_K,
                projection=projection,
                sign=sign,
                image_size=image_size_dst)
        # src in screen, dst in ndc
        elif in_ndc_src is False and in_ndc_dst is True:
            convert_screen_to_ndc(
                K=new_K,
                projection=projection,
                sign=sign,
                image_size=image_size_src)
        # src in ndc, dst in ndc
        elif in_ndc_src is True and in_ndc_dst is True:
            if projection == 'perspective':
                new_K[:, 0, 2] *= sign[0]
                new_K[:, 1, 2] *= sign[1]
            else:
                new_K[:, 0, 3] *= sign[0]
                new_K[:, 1, 3] *= sign[1]
        # src in screen, dst in screen
        else:
            pass

        if left_mm_in_src is True and left_mm_in_dst is False:
            new_K = new_K.permute(0, 2, 1)
    if use_numpy:
        if isinstance(new_K, torch.Tensor):
            new_K = new_K.cpu().numpy()
        if isinstance(new_R, torch.Tensor):
            new_R = new_R.cpu().numpy()
        if isinstance(new_T, torch.Tensor):
            new_T = new_T.cpu().numpy()
    return new_K, new_R, new_T


def convert_K_3x3_to_4x4(K: Union[torch.Tensor, np.ndarray],
                         projection: Literal['perspective', 'weakperspective',
                                             'orthographics'] = 'perspective'):
    if not isinstance(K, (torch.Tensor, np.ndarray)):
        raise TypeError('K should be `torch.Tensor` or `numpy.ndarray`, '
                        f'type(K): {type(K)}.')
    if K.shape[-2:] == (4, 4):
        warnings.warn(
            f'shape of K already is {K.shape}, will pass converting.')
        return K
    use_numpy = False
    if K.ndim == 2:
        K = K[None].reshape(-1, 3, 3)
    elif K.ndim == 3:
        K = K.reshape(-1, 3, 3)
    else:
        raise ValueError(f'Wrong ndim of K: {K.ndim}')

    if isinstance(K, np.ndarray):
        use_numpy = True
    if projection == 'perspective':
        if use_numpy:
            K_ = np.zeros((K.shape[0], 4, 4))
        else:
            K_ = torch.zeros(K.shape[0], 4, 4)
        K_[:, :2, :3] = K[:, :2, :3]
        K_[:, 3, 2] = 1
        K_[:, 2, 3] = 1
    else:
        if use_numpy:
            K_ = np.eye(4, 4)[None].repeat(K.shape[0], 0)
        else:
            K_ = torch.eye(4, 4)[None].repeat(K.shape[0], 1, 1)
        K_[:, :2, :2] = K[:, :2, :2]
        K_[:, :2, 3:] = K[:, :2, 2:]
    return K_


def convert_K_4x4_to_3x3(K: Union[torch.Tensor, np.ndarray],
                         projection: Literal['perspective', 'weakperspective',
                                             'orthographics'] = 'perspective'):
    if not isinstance(K, (torch.Tensor, np.ndarray)):
        raise TypeError('K should be `torch.Tensor` or `numpy.ndarray`, '
                        f'type(K): {type(K)}.')
    if K.shape[-2:] == (3, 3):
        warnings.warn(
            f'shape of K already is {K.shape}, will pass converting.')
        return K
    use_numpy = True if isinstance(K, np.ndarray) else False
    if K.ndim == 2:
        K = K[None].reshape(-1, 4, 4)
    elif K.ndim == 3:
        K = K.reshape(-1, 4, 4)
    else:
        raise ValueError(f'Wrong ndim of K: {K.ndim}')

    if use_numpy:
        K_ = np.eye(3, 3)[None].repeat(K.shape[0], 0)
    else:
        K_ = torch.eye(3, 3)[None].repeat(K.shape[0], 1, 1)
    if projection == 'perspective':
        K_[:, :2, :3] = K[:, :2, :3]
    else:
        K_[:, :2, :2] = K[:, :2, :2]
        K_[:, :2, 2:3] = K[:, :2, 3:4]
    return K_


def convert_ndc_to_screen(K,
                          image_size,
                          sign: Optional[Iterable[int]] = None,
                          projection='perspective'):
    sign = [1, 1, 1] if sign is None else sign
    if not isinstance(K, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f'K should be `torch.Tensor` or `np.ndarray`, type(K): {type(K)}')
    if K.ndim == 2:
        K = K[None].reshape(-1, 4, 4)
    elif K.ndim == 3:
        K = K.reshape(-1, 4, 4)
    else:
        raise ValueError(f'Wrong ndim of K: {K.ndim}')

    if isinstance(image_size, (int, float)):
        w_dst = h_dst = image_size
    else:
        h_dst, w_dst = image_size
    K[:, 0, 0] *= w_dst / 2
    K[:, 1, 1] *= h_dst / 2
    K[:, 0, 0] /= w_dst / h_dst
    if projection == 'perspective':
        K[:, 0, 2] *= sign[0]
        K[:, 1, 2] *= sign[1]
        K[:, 0, 2] = (K[:, 0, 2] + 1) * (w_dst / 2)
        K[:, 1, 2] = (K[:, 1, 2] + 1) * (h_dst / 2)
    else:
        K[:, 0, 3] *= sign[0]
        K[:, 1, 3] *= sign[1]
        K[:, 0, 3] = (K[:, 0, 3] + 1) * (w_dst / 2)
        K[:, 1, 3] = (K[:, 1, 3] + 1) * (h_dst / 2)
    return K


def convert_screen_to_ndc(K,
                          image_size,
                          sign=[1, 1, 1],
                          projection='perspective'):
    if not isinstance(K, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f'K should be `torch.Tensor` or `np.ndarray`, type(K): {type(K)}')
    if K.ndim == 2:
        K = K[None].reshape(-1, 4, 4)
    elif K.ndim == 3:
        K = K.reshape(-1, 4, 4)
    else:
        raise ValueError(f'Wrong ndim of K: {K.ndim}')

    if isinstance(image_size, (int, float)):
        w_src = h_src = image_size
    else:
        h_src, w_src = image_size
    K[:, 0, 0] /= w_src / 2
    K[:, 1, 1] /= h_src / 2
    K[:, 0, 0] *= w_src / h_src
    if projection == 'perspective':
        K[:, 0, 2] = K[:, 0, 2] / (w_src / 2) - 1
        K[:, 1, 2] = K[:, 1, 2] / (h_src / 2) - 1
        K[:, 0, 2] *= sign[0]
        K[:, 1, 2] *= sign[1]
    else:
        K[:, 0, 3] = K[:, 0, 3] / (w_src / 2) - 1
        K[:, 1, 3] = K[:, 1, 3] / (h_src / 2) - 1
        K[:, 0, 3] *= sign[0]
        K[:, 1, 3] *= sign[1]
    return K


def convert_world_view(R, T):
    if not (type(R) is type(T)):
        raise TypeError(
            f'R: {type(R)}, T: {type(T)} should have the same type.')
    if isinstance(R, torch.Tensor):
        R = R.permute(0, 2, 1)
        T = -(R @ T.view(-1, 3, 1)).view(-1, 3)
    elif isinstance(R, np.ndarray):
        R = R.transpose(0, 2, 1)
        T = -(R @ T.reshape(-1, 3, 1)).reshape(-1, 3)
    else:
        raise TypeError(f'R: {type(R)}, T: {type(T)} should be torch.Tensor '
                        f'or numpy.ndarray.')
    return R, T
