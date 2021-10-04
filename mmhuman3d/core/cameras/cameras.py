from typing import Iterable, List, Optional, Tuple, Union

import torch
from pytorch3d.renderer import cameras
from pytorch3d.transforms import Transform3d

from mmhuman3d.core.conventions.cameras import (
    CAMERA_CONVENTIONS,
    convert_world_view,
)


class WeakPerspectiveCamerasVibe(cameras.CamerasBase):
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
        )

    @staticmethod
    def compute_matrix_from_pred_cam(
        pred_cam: torch.Tensor,
        znear: Union[torch.Tensor, float] = -1.0,
        aspect_ratio: Union[torch.Tensor, float] = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computer intrinsic camera matrix from pred camera parameter of smpl.

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
                K = [
                        [sx*r,   0,    0,   tx*sx*r],
                        [0,     sy,    0,   ty*sy],
                        [0,     0,     1,       0],
                        [0,     0,     0,       1],
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
        _N = pred_cam.shape[0]
        scale_x, scale_y, transl_x, transl_y = pred_cam[:, 0], pred_cam[:, 1],\
            pred_cam[:, 2], pred_cam[:, 3]
        K = torch.zeros((_N, 4, 4), dtype=torch.float32)

        K[:, 0, 0] = scale_x * aspect_ratio
        K[:, 1, 1] = scale_y
        K[:, 3, 3] = 1
        K[:, 0, 3] = transl_x * scale_x * aspect_ratio
        K[:, 1, 3] = transl_y * scale_y
        K[:, 2, 2] = 1
        R = torch.eye(3, 3)[None].repeat(_N, 1, 1)
        T = torch.zeros(_N, 3)
        T[:, 2] = znear
        return K, R, T

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

    def get_camera_center_plane(**kwargs):  # TODO
        pass


def orbit_camera_extrinsic(
    elev: float = 0,
    azim: float = 0,
    dist: float = 2.7,
    at: Union[torch.Tensor, List, Tuple] = (0, 0, 0),
    batch: int = 1,
    orbit_speed: Union[float, Tuple[float, float]] = 0,
    convention: str = 'pytorch3d',
):
    if convention == 'opencv':
        azim += 180

    if not isinstance(orbit_speed, Iterable):
        orbit_speed = (orbit_speed, 0.0)
    if not isinstance(at, torch.Tensor):
        at = torch.Tensor(at)
    at = at.view(1, 3)
    if batch > 1 and orbit_speed[0] != 0:
        azim = torch.linspace(azim, azim + batch * orbit_speed[0], batch)
    if batch > 1 and orbit_speed[1] != 0:
        elev = torch.linspace(elev, elev + batch * orbit_speed[1], batch)
    R, T = cameras.look_at_view_transform(
        dist=dist, elev=elev, azim=azim, at=at)
    if CAMERA_CONVENTIONS[convention]['world_to_view']:
        R, T = convert_world_view(R, T)
    return R, T
