import math
from typing import Iterable, List, Optional, Tuple, Union

import torch
from pytorch3d.renderer import cameras
from pytorch3d.transforms import Transform3d

from mmhuman3d.core.conventions.cameras import convert_cameras
from mmhuman3d.core.conventions.cameras.convert_convention import (
    convert_ndc_to_screen,
    convert_screen_to_ndc,
)
from mmhuman3d.utils.transforms import ee_to_rotmat
from .builder import CAMERAS


class NewAttributeCameras(cameras.CamerasBase):
    """Inherited from pytorch3d CamerasBase and provide some new functions."""

    def __init__(self, **kwargs) -> None:
        convention = kwargs.pop('convention', 'pytorch3d')
        self.convention = convention
        in_ndc = kwargs.pop('_in_ndc')
        is_perspective = kwargs.pop('_is_perspective')

        kwargs.update(_in_ndc=in_ndc)
        image_size = kwargs.get('image_size', kwargs.get('resolution', None))
        kwargs.update(image_size=image_size)
        if image_size is not None:
            if isinstance(image_size, (int, float)):
                image_size = (image_size, image_size)
            if isinstance(image_size, (tuple, list)):
                image_size = torch.Tensor(image_size)
            if isinstance(image_size, torch.Tensor):
                if image_size.numel() == 1:
                    image_size = image_size.repeat(2)
                image_size = image_size.view(-1, 2)
            kwargs.update(image_size=image_size)

        if kwargs.get('K') is None:
            focal_length = kwargs.get('focal_length', None)
            if focal_length is not None:
                if not isinstance(focal_length, Iterable):
                    focal_length = [focal_length, focal_length]
                if not torch.is_tensor(focal_length):
                    focal_length = torch.FloatTensor(focal_length).view(-1, 2)
                elif focal_length.numel() == 1:
                    focal_length = focal_length.repeat(2).view(-1, 2)
                kwargs.update(focal_length=focal_length)

            principal_point = kwargs.get('principal_point', None)
            if principal_point is not None:
                if isinstance(principal_point, (tuple, list)):
                    principal_point = torch.FloatTensor(principal_point)
                principal_point = principal_point.view(-1, 2)
                kwargs.update(principal_point=principal_point)
            K = self.get_default_projection_matrix(**kwargs)
            K, _, _ = convert_cameras(
                K=K,
                is_perspective=is_perspective,
                convention_src='pytorch3d',
                convention_dst=convention,
                in_ndc_src=in_ndc,
                in_ndc_dst=in_ndc,
                resolution_dst=image_size,
                resolution_src=image_size)
            kwargs.update(K=K)

        K, R, T = convert_cameras(
            K=kwargs.get('K'),
            R=kwargs.get('R', None),
            T=kwargs.get('T', None),
            convention_src=convention,
            convention_dst='pytorch3d',
            is_perspective=is_perspective,
            in_ndc_src=in_ndc,
            in_ndc_dst=in_ndc,
            resolution_src=image_size,
            resolution_dst=image_size)

        kwargs.update(K=K, R=R, T=T)
        super().__init__(**kwargs)

    def compute_depth_of_points(self, points: torch.Tensor) -> torch.Tensor:
        """Compute depth of points to the camera plane.

        Args:
            points ([torch.Tensor]): shape should be (batch_size, 3).

        Returns:
            torch.Tensor: shape will be (batch_size, 1)
        """
        world_to_view_transform = self.get_world_to_view_transform()
        world_to_view_points = world_to_view_transform.transform_points(
            points.to(self.device))
        return world_to_view_points[..., 2:3]

    def __repr__(self):
        main_str = super().__repr__()
        main_str = main_str.split(')')[0]
        main_str += f'N: {self.__len__()}, in_ndc: {self.in_ndc()}, '
        main_str += f'device: {self.device})'
        return main_str

    def get_image_size(self):
        """Returns the image size, if provided, expected in the form of
        (height, width) The image size is used for conversion of projected
        points to screen coordinates."""
        if hasattr(self, 'image_size'):
            return self.image_size
        else:
            return None

    def __getitem__(self, idx: Union[slice, int, torch.Tensor, List, Tuple]):
        if isinstance(idx, int):
            idx = [idx]
        return self.__class__(
            K=self.K[idx],
            R=self.R[idx],
            T=self.T[idx],
            image_size=self.get_image_size()[idx],
            _in_ndc=self.in_ndc(),
            _is_perspective=self.is_perspective(),
            convention=self.convention,
            device=self.device)

    def extend(self, N):
        return self.__class__(
            K=self.K.repeat(N, 1, 1),
            R=self.R.repeat(N, 1, 1),
            T=self.T.repeat(N, 1),
            image_size=self.get_image_size(),
            _in_ndc=self.in_ndc(),
            _is_perspective=self.is_perspective(),
            convention=self.convention,
            device=self.device)

    def extend_(self, N):
        self.K = self.K.repeat(N, 1, 1)
        self.R = self.R.repeat(N, 1, 1)
        self.T = self.T.repeat(N, 1)
        self._N = self._N * N

    @property
    def shape(self):
        return self.K.shape, self.R.shape, self.T.shape

    @classmethod
    def get_default_projection_matrix(cls, ):
        raise NotImplementedError()

    def to_screen(self, **kwargs):
        if self.in_ndc():
            if self.get_image_size() is None:
                self.image_size = kwargs.get('image_size')
            self.K = convert_ndc_to_screen(
                K=self.K,
                resolution=self.image_size,
                is_perspective=self.is_perspective())
            self._in_ndc = False
        else:
            print('Already in screen.')

    def to_ndc(self, **kwargs):
        if self.in_ndc():
            print('Already in ndc.')
        else:
            if self.get_image_size() is None:
                self.image_size = kwargs.get('image_size')
            self.K = convert_screen_to_ndc(
                K=self.K,
                resolution=self.image_size,
                is_perspective=self.is_perspective())


@CAMERAS.register_module(
    name=('WeakPerspectiveCameras', 'WeakPerspective', 'weakperspective'))
class WeakPerspectiveCameras(NewAttributeCameras):
    """Inherited from (pytorch3d cameras)[https://github.com/facebookresearch/
    pytorch3d/blob/main/pytorch3d/renderer/cameras.py] and mimiced the code
    style. And re-inmplemented functions: compute_projection_matrix,
    get_projection_transform, unproject_points, is_perspective, in_ndc for
    render.

    K modified from (VIBE)[https://github.com/mkocabas/VIBE/blob/master/
    lib/utils/renderer.py] and changed to opencv convention. This intrinsic
    matrix is orthographics indeed, but could serve as weakperspective for
    single smpl mesh.
    """
    is_perspective = False

    def __init__(
        self,
        scale_x: Union[torch.Tensor, float] = 1.0,
        scale_y: Union[torch.Tensor, float] = 1.0,
        transl_x: Union[torch.Tensor, float] = 0.0,
        transl_y: Union[torch.Tensor, float] = 0.0,
        znear: Union[torch.Tensor, float] = -1.0,
        aspect_ratio: Union[torch.Tensor, float] = 1.0,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        device: Union[torch.device, str] = 'cpu',
        convention: str = 'pytorch3d',
        **kwargs,
    ):
        """Initialize. If K is provided, don't need scale_x, scale_y, transl_x,
        transl_y, znear, aspect_ratio.

        Args:
            scale_x (Union[torch.Tensor, float], optional):
                Scale in x direction.
                Defaults to 1.0.
            scale_y (Union[torch.Tensor, float], optional):
                Scale in y direction.
                Defaults to 1.0.
            transl_x (Union[torch.Tensor, float], optional):
                Translation in x direction.
                Defaults to 0.0.
            transl_y (Union[torch.Tensor, float], optional):
                Translation in y direction.
                Defaults to 0.0.
            znear (Union[torch.Tensor, float], optional):
                near clipping plane of the view frustrum.
                Defaults to -1.0.
            aspect_ratio (Union[torch.Tensor, float], optional):
                aspect ratio of the image pixels. 1.0 indicates square pixels.
                Defaults to 1.0.
            K (Optional[torch.Tensor], optional): Intrinsic matrix of shape
                (N, 4, 4). If provided, don't need scale_x, scale_y, transl_x,
                transl_y, znear, aspect_ratio.
                Defaults to None.
            R (Optional[torch.Tensor], optional):
                Rotation matrix of shape (N, 3, 3).
                Defaults to None.
            T (Optional[torch.Tensor], optional):
                Translation matrix of shape (N, 3).
                Defaults to None.
            device (Union[torch.device, str], optional):
                torch device. Defaults to 'cpu'.
        """
        kwargs.update(
            _in_ndc=True,
            _is_perspective=False,
        )
        super().__init__(
            scale_x=scale_x,
            scale_y=scale_y,
            transl_x=transl_x,
            transl_y=transl_y,
            znear=znear,
            aspect_ratio=aspect_ratio,
            K=K,
            R=R,
            T=T,
            device=device,
            convention=convention,
            **kwargs)

    @staticmethod
    def convert_pred_cam_to_matrix(
            pred_cam: torch.Tensor,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute intrinsic camera matrix from pred camera parameter of smpl.

        Args:
            pred_cam (torch.Tensor): shape should be (N, 4).
            znear (Union[torch.Tensor, float], optional):
                near clipping plane of the view frustrum.
                Defaults to 0.0.
            aspect_ratio (Union[torch.Tensor, float], optional):
                aspect ratio of the image pixels. 1.0 indicates square pixels.
                Defaults to 1.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            opencv intrinsic matrix: (N, 4, 4)
            r > 1;
                K = [
                        [sx*r,   0,    0,   tx*sx*r],
                        [0,     sy,    0,   ty*sy],
                        [0,     0,     1,       0],
                        [0,     0,     0,       1],
                ]
            or r < 1:
                K = [
                        [sx,    0,     0,   tx*sx],
                        [0,   sy/r,    0,  ty*sy/r],
                        [0,     0,     1,      0],
                        [0,     0,     0,      1],
                ]
            rotation matrix: (N, 3, 3)
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ]
            translation matrix: (N, 3)
                [0, 0, -znear]
        """
        znear = kwargs.get('znear', -1.0)
        aspect_ratio = kwargs.get('aspect_ratio', 1.0)
        _N = pred_cam.shape[0]
        scale_x, scale_y, transl_x, transl_y = pred_cam[:, 0], pred_cam[:, 1],\
            pred_cam[:, 2], pred_cam[:, 3]
        K = torch.zeros((_N, 4, 4), dtype=torch.float32)
        if aspect_ratio >= 1.0:
            K[:, 0, 0] = scale_x * aspect_ratio
            K[:, 1, 1] = scale_y
            K[:, 0, 3] = transl_x * scale_x * aspect_ratio
            K[:, 1, 3] = transl_y * scale_y
        else:
            K[:, 0, 0] = scale_x
            K[:, 1, 1] = scale_y / aspect_ratio
            K[:, 0, 3] = transl_x * scale_x
            K[:, 1, 3] = transl_y * scale_y / aspect_ratio

        K[:, 3, 3] = 1
        K[:, 2, 2] = 1
        R = torch.eye(3, 3)[None].repeat(_N, 1, 1)
        T = torch.zeros(_N, 3)
        T[:, 2] = znear
        return K, R, T

    @staticmethod
    def convert_K_to_pred_cam(
        K: torch.Tensor,
        aspect_ratio: Union[torch.Tensor, float] = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute intrinsic camera matrix from pred camera parameter of smpl.

        Args:
            K (torch.Tensor):
                opencv orthographics intrinsic matrix: (N, 4, 4)
                K = [
                        [sx*r,   0,    0,   tx*sx*r],
                        [0,     sy,    0,   ty*sy],
                        [0,     0,     1,       0],
                        [0,     0,     0,       1],
                ]
            aspect_ratio (Union[torch.Tensor, float], optional):
                aspect ratio of the image pixels. 1.0 indicates square pixels.
                Defaults to 1.0.

        Returns:

            pred_cam (torch.Tensor): shape should be (N, 4).
        """
        _N = K.shape[0]
        s_x = K[:, 0, 0] / aspect_ratio
        s_y = K[:, 1, 1] / aspect_ratio
        t_x = K[:, 0, 3] / (aspect_ratio * s_x)
        t_y = K[:, 1, 3] / s_y
        pred_cam = torch.cat([s_x, s_y, t_x, t_y], -1).view(_N, 4)
        return pred_cam

    @classmethod
    def get_default_projection_matrix(cls, **args):
        pred_cam = args.get('pred_cam', None)
        scale_x = args.get('scale_x', 1.0)
        scale_y = args.get('scale_y', 1.0)
        transl_x = args.get('transl_x', 0.0)
        transl_y = args.get('transl_y', 0.0)
        aspect_ratio = args.get('aspect_ratio', 1.0)
        batch_size = args.get('batch_size', 1)
        device = args.get('device', 'cpu')

        if pred_cam is not None:
            K, _, _ = cls.convert_pred_cam_to_matrix(pred_cam, **args)
        else:
            K = torch.zeros((1, 4, 4), dtype=torch.float32)

            K[:, 0, 0] = scale_x * aspect_ratio
            K[:, 1, 1] = scale_y
            K[:, 3, 3] = 1
            K[:, 0, 3] = transl_x * scale_x * aspect_ratio
            K[:, 1, 3] = transl_y * scale_y
            K[:, 2, 2] = 1
            K = K.repeat(batch_size, 1, 1).to(device)
        return K

    def compute_projection_matrix(self, scale_x, scale_y, transl_x, transl_y,
                                  aspect_ratio) -> torch.Tensor:
        K = torch.zeros((self._N, 4, 4),
                        dtype=torch.float32,
                        device=self.device)

        K[:, 0, 0] = scale_x * aspect_ratio
        K[:, 1, 1] = scale_y
        K[:, 3, 3] = 1
        K[:, 0, 3] = transl_x * scale_x * aspect_ratio
        K[:, 1, 3] = transl_y * scale_y
        K[:, 2, 2] = 1
        return K

    def get_projection_transform(self, **kwargs) -> Transform3d:
        K = kwargs.get('K', self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = 'Expected K to have shape of (%r, 4, 4)'
                raise ValueError(msg % (self._N))
        else:
            K = self.compute_projection_matrix(
                kwargs.get('scale_x', self.scale_x),
                kwargs.get('scale_y', self.scale_y),
                kwargs.get('transl_x', self.trans_x),
                kwargs.get('transl_y', self.trans_y),
                kwargs.get('aspect_ratio', self.aspect_ratio))

        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device)
        return transform

    def unproject_points(self,
                         xy_depth: torch.Tensor,
                         world_coordinates: bool = True,
                         **kwargs) -> torch.Tensor:
        if world_coordinates:
            to_camera_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_camera_transform = self.get_projection_transform(**kwargs)

        unprojection_transform = to_camera_transform.inverse()
        return unprojection_transform.transform_points(xy_depth)

    def is_perspective(self):
        return False

    def in_ndc(self):
        return True

    def to_ndc(self, **kwargs):
        raise NotImplementedError()

    def to_screen(self, **kwargs):
        raise NotImplementedError()


@CAMERAS.register_module(
    name=('PerspectiveCameras', 'perspective', 'Perspective'))
class PerspectiveCameras(cameras.PerspectiveCameras, NewAttributeCameras):

    def __init__(
        self,
        focal_length=1.0,
        principal_point=((0.0, 0.0), ),
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        device: Union[torch.device, str] = 'cpu',
        in_ndc: bool = True,
        convention: str = 'pytorch3d',
        image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
        **kwargs,
    ) -> None:
        if image_size is not None:
            kwargs.update({'image_size': image_size})
        kwargs.update(
            _in_ndc=in_ndc,
            _is_perspective=True,
        )
        super(cameras.PerspectiveCameras, self).__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            convention=convention,
            **kwargs)
        if image_size is not None:
            if (self.image_size < 1).any():  # pyre-ignore
                raise ValueError('Image_size provided has invalid values')
        else:
            self.image_size = None

    def __getitem__(self, index: Union[slice, int, torch.Tensor, List, Tuple]):
        return super(cameras.PerspectiveCameras, self).__getitem__(index)

    @classmethod
    def get_default_projection_matrix(cls, **args) -> torch.Tensor:
        batch_size = args.get('batch_size', 1)
        device = args.get('device', 'cpu')
        focal_length = args.get('focal_length')
        principal_point = args.get('principal_point')
        return cameras._get_sfm_calibration_matrix(
            N=batch_size,
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            orthographic=False)


@CAMERAS.register_module(
    name=('FoVPerspectiveCameras', 'FoVPerspective', 'fovperspective'))
class FoVPerspectiveCameras(cameras.FoVPerspectiveCameras,
                            NewAttributeCameras):

    def __init__(
        self,
        znear=1.0,
        zfar=100.0,
        aspect_ratio=1.0,
        fov=60.0,
        degrees: bool = True,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        device: Union[torch.device, str] = 'cpu',
        convention: str = 'pytorch3d',
        **kwargs,
    ) -> None:
        kwargs.update(
            _in_ndc=True,
            _is_perspective=True,
        )
        super(cameras.FoVPerspectiveCameras, self).__init__(
            device=device,
            znear=znear,
            zfar=zfar,
            aspect_ratio=aspect_ratio,
            fov=fov,
            R=R,
            T=T,
            K=K,
            convention=convention,
            **kwargs,
        )
        self.degrees = degrees

    def __getitem__(self, index: Union[int, slice]):
        return super(cameras.FoVPerspectiveCameras, self).__getitem__(index)

    @classmethod
    def get_default_projection_matrix(cls, **args) -> torch.Tensor:
        znear = args.get('znear', 1.0)
        zfar = args.get('zfar', 100.0)
        aspect_ratio = args.get('aspect_ratio', 1.0)
        fov = args.get('fov', 60.0)
        degrees = args.get('degrees', True)
        batch_size = args.get('batch_size', 1)

        K = torch.zeros((1, 4, 4), dtype=torch.float32)
        if degrees:
            fov = (math.pi / 180) * fov

        if not torch.is_tensor(fov):
            fov = torch.tensor(fov)
        tanHalfFov = torch.tan((fov / 2))
        max_y = tanHalfFov * znear
        min_y = -max_y
        max_x = max_y * aspect_ratio
        min_x = -max_x

        z_sign = 1.0

        K[:, 0, 0] = 2.0 * znear / (max_x - min_x)
        K[:, 1, 1] = 2.0 * znear / (max_y - min_y)
        K[:, 0, 2] = (max_x + min_x) / (max_x - min_x)
        K[:, 1, 2] = (max_y + min_y) / (max_y - min_y)
        K[:, 3, 2] = z_sign

        K[:, 2, 2] = z_sign * zfar / (zfar - znear)
        K[:, 2, 3] = -(zfar * znear) / (zfar - znear)
        K = K.repeat(batch_size, 1, 1)
        return K

    def to_ndc(self, **kwargs):
        raise NotImplementedError()

    def to_screen(self, **kwargs):
        raise NotImplementedError()


@CAMERAS.register_module(
    name=('OrthographicCameras', 'Orthographic', 'orthographic'))
class OrthographicCameras(cameras.OrthographicCameras, NewAttributeCameras):

    def __init__(
        self,
        focal_length=1.0,
        principal_point=((0.0, 0.0), ),
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        device: Union[torch.Tensor, str] = 'cpu',
        in_ndc: bool = True,
        image_size: Optional[torch.Tensor] = None,
        convention: str = 'pytorch3d',
        **kwargs,
    ) -> None:
        if image_size is not None:
            kwargs.update({'image_size': image_size})
        kwargs.update(
            _is_perspective=False,
            _in_ndc=in_ndc,
        )
        super(cameras.OrthographicCameras, self).__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            convention=convention,
            **kwargs)
        if image_size is not None:
            if (self.image_size < 1).any():  # pyre-ignore
                raise ValueError('Image_size provided has invalid values')
        else:
            self.image_size = None

    def __getitem__(self, index: Union[int, slice]):
        return super(cameras.OrthographicCameras, self).__getitem__(index)

    @classmethod
    def get_default_projection_matrix(cls, **args) -> torch.Tensor:
        batch_size = args.get('batch_size', 1)
        device = args.get('device', 'cpu')
        focal_length = args.get('focal_length')
        principal_point = args.get('principal_point')

        return cameras._get_sfm_calibration_matrix(
            N=batch_size,
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            orthographic=True)


@CAMERAS.register_module(
    name=('FoVOrthographicCameras', 'FoVOrthographic', 'fovorthographic'))
class FoVOrthographicCameras(cameras.FoVOrthographicCameras,
                             NewAttributeCameras):

    def __init__(
            self,
            znear=1.0,
            zfar=100.0,
            max_y=1.0,
            min_y=-1.0,
            max_x=1.0,
            min_x=-1.0,
            scale_xyz=((1.0, 1.0, 1.0), ),  # (1, 3)
            R: Optional[torch.Tensor] = None,
            T: Optional[torch.Tensor] = None,
            K: Optional[torch.Tensor] = None,
            device: Union[torch.device, str] = 'cpu',
            convention: str = 'pytorch3d',
            **kwargs):
        kwargs.update(_is_perspective=True, _in_ndc=True)
        super(cameras.FoVOrthographicCameras, self).__init__(
            device=device,
            znear=znear,
            zfar=zfar,
            max_y=max_y,
            min_y=min_y,
            max_x=max_x,
            min_x=min_x,
            scale_xyz=scale_xyz,
            R=R,
            T=T,
            K=K,
            convention=convention,
            **kwargs)

    def __getitem__(self, index: Union[int, slice]):
        return super(cameras.FoVOrthographicCameras, self).__getitem__(index)

    @classmethod
    def get_default_projection_matrix(cls, **args) -> torch.Tensor:
        """
            scale_x = 2 / (max_x - min_x)
            scale_y = 2 / (max_y - min_y)
            scale_z = 2 / (far-near)
            mid_x = (max_x + min_x) / (max_x - min_x)
            mix_y = (max_y + min_y) / (max_y - min_y)
            mid_z = (far + near) / (far - near)

            K = [
                    [scale_x,        0,         0,  -mid_x],
                    [0,        scale_y,         0,  -mix_y],
                    [0,              0,  -scale_z,  -mid_z],
                    [0,              0,         0,       1],
            ]
        """
        znear = args.get('znear', 1.0)
        zfar = args.get('zfar', 100.0)
        max_y = args.get('max_y', 1.0)
        min_y = args.get('min_y', -1.0)
        max_x = args.get('max_x', 1.0)
        min_x = args.get('min_x', -1.0)
        scale_xyz = args.get(
            'scale_xyz',
            ((1.0, 1.0, 1.0), ),
        )
        batch_size = args.get('batch_size', 1)

        K = torch.zeros((1, 4, 4), dtype=torch.float32)
        ones = torch.ones((1), dtype=torch.float32)
        z_sign = +1.0

        if not isinstance(scale_xyz, torch.Tensor):
            scale_xyz = torch.Tensor(scale_xyz)
        K[:, 0, 0] = (2.0 / (max_x - min_x)) * scale_xyz[:, 0]
        K[:, 1, 1] = (2.0 / (max_y - min_y)) * scale_xyz[:, 1]
        K[:, 0, 3] = -(max_x + min_x) / (max_x - min_x)
        K[:, 1, 3] = -(max_y + min_y) / (max_y - min_y)
        K[:, 3, 3] = ones

        # NOTE: This maps the z coordinate to the range [0, 1] and replaces the
        # the OpenGL z normalization to [-1, 1]
        K[:, 2, 2] = z_sign * (1.0 / (zfar - znear)) * scale_xyz[:, 2]
        K[:, 2, 3] = -znear / (zfar - znear)
        K = K.repeat(batch_size, 1, 1)
        return K

    def to_ndc(self, **kwargs):
        raise NotImplementedError()

    def to_screen(self, **kwargs):
        raise NotImplementedError()


def compute_orbit_cameras(
    elev: float = 0,
    azim: float = 0,
    dist: float = 2.7,
    at: Union[torch.Tensor, List, Tuple] = (0, 0, 0),
    batch_size: int = 1,
    orbit_speed: Union[float, Tuple[float, float]] = 0,
    dist_speed: Optional[float] = 0,
    convention: str = 'pytorch3d',
):
    """Generate a sequence of moving cameras following an orbit.

    Args:
        elev (float, optional):  This is the angle between the
            vector from the object to the camera, and the horizontal
            plane y = 0 (xz-plane).
             Defaults to 0.
        azim (float, optional): angle in degrees or radians. The vector
            from the object to the camera is projected onto a horizontal
            plane y = 0. azim is the angle between the projected vector and a
            reference vector at (0, 0, 1) on the reference plane (the
            horizontal plane).
            Defaults to 0.
        dist (float, optional): distance of the camera from the object.
            Defaults to 2.7.
        at (Union[torch.Tensor, List, Tuple], optional):
            the position of the object(s) in world coordinates.
            Defaults to (0, 0, 0).
        batch_size (int, optional): batch size. Defaults to 1.
        orbit_speed (Union[float, Tuple[float, float]], optional):
            degree speed of camera moving along the orbit.
            Could be one or two number. One number for only elev speed,
            two number for both.
            Defaults to 0.
        dist_speed (Optional[float], optional):
            speed of camera moving along the center line.
            Defaults to 0.
        convention (str, optional): Camera convention. Defaults to 'pytorch3d'.

    Returns:
        Union[torch.Tensor, torch.Tensor, torch.Tensor]: computed K, R, T.
    """
    if not isinstance(orbit_speed, Iterable):
        orbit_speed = (orbit_speed, 0.0)
    if not isinstance(at, torch.Tensor):
        at = torch.Tensor(at)
    at = at.view(1, 3)
    if batch_size > 1 and orbit_speed[0] != 0:
        azim = torch.linspace(azim, azim + batch_size * orbit_speed[0],
                              batch_size)
    if batch_size > 1 and orbit_speed[1] != 0:
        elev = torch.linspace(elev, elev + batch_size * orbit_speed[1],
                              batch_size)
    if batch_size > 1 and dist_speed != 0:
        dist = torch.linspace(dist, dist + batch_size * dist_speed, batch_size)

    if convention == 'opencv':
        rotation_compensate = ee_to_rotmat(
            torch.Tensor([math.pi, 0, 0]).view(1, 3))
        at = rotation_compensate.permute(0, 2, 1) @ at.view(-1, 3, 1)
        at = at.view(1, 3)
    R, T = cameras.look_at_view_transform(
        dist=dist, elev=elev, azim=azim, at=at)

    K = FoVPerspectiveCameras.get_default_projection_matrix(
        batch_size=batch_size)
    if convention == 'opencv':
        rotation_compensate = ee_to_rotmat(
            torch.Tensor([math.pi, 0, 0]).view(1, 3))
        R = rotation_compensate.permute(0, 2, 1) @ R
    return K, R, T
