import torch
from pytorch3d.common.types import Device
from pytorch3d.renderer import cameras
from pytorch3d.transforms import Transform3d


class WeakPerspectiveCameras(cameras.CamerasBase):
    """Inherited from https://github.com/facebookresearch/
    pytorch3d/blob/main/pytorch3d/renderer/cameras.py and mimiced the code
    style.

    And reinplemented functions: compute_projection_matrix,
    get_projection_transform, unproject_points, is_perspective, in_ndc.
    """

    def __init__(
        self,
        scale_x,
        scale_y,
        trans_x,
        trans_y,
        trans_z,
        device: Device = 'cpu',
    ):
        super().__init__(
            device=device,
            scale_x=scale_x,
            scale_y=scale_y,
            trans_x=trans_x,
            trans_y=trans_y,
        )
        K = torch.zeros((self._N, 4, 4),
                        dtype=torch.float32,
                        device=self.device)
        ones = torch.ones(self._N, dtype=torch.float32, device=self.device)

        K[:, 0, 0] = scale_x
        K[:, 1, 1] = scale_y
        K[:, 3, 3] = ones

        K[:, 2, 2] = -1
        self.K = K.to(device)
        self.R = torch.eye(3, 3)[None].repeat(self._N, 1, 1).to(device)
        self.T = torch.zeros(self._N, 3).to(device)
        self.T[:, 0] = trans_x
        self.T[:, 1] = trans_y
        self.T[:, 2] = trans_z

    def compute_projection_matrix(self, scale_x, scale_y, trans_x,
                                  trans_y) -> torch.Tensor:
        K = torch.zeros((self._N, 4, 4),
                        dtype=torch.float32,
                        device=self.device)
        ones = torch.ones(self._N, dtype=torch.float32, device=self.device)

        K[:, 0, 0] = scale_x
        K[:, 1, 1] = scale_y
        K[:, 3, 3] = ones

        K[:, 2, 2] = -1
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
                kwargs.get('trans_x', self.trans_x),
                kwargs.get('trans_y', self.trans_y))

        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device)
        return transform

    def unproject_points(self,
                         xy_depth: torch.Tensor,
                         world_coordinates: bool = True,
                         scaled_depth_input: bool = False,
                         **kwargs) -> torch.Tensor:
        if world_coordinates:
            to_ndc_transform = self.get_full_projection_transform(
                **kwargs.copy())
        else:
            to_ndc_transform = self.get_projection_transform(**kwargs.copy())

        if scaled_depth_input:
            # the input depth is already scaled
            xy_sdepth = xy_depth
        else:
            K = self.get_projection_transform(**kwargs).get_matrix()
            unsqueeze_shape = [1] * K.dim()
            unsqueeze_shape[0] = K.shape[0]
            mid_z = K[:, 3, 2].reshape(unsqueeze_shape)
            scale_z = K[:, 2, 2].reshape(unsqueeze_shape)
            scaled_depth = scale_z * xy_depth[..., 2:3] + mid_z
            # cat xy and scaled depth
            xy_sdepth = torch.cat((xy_depth[..., :2], scaled_depth), dim=-1)
        # finally invert the transform
        unprojection_transform = to_ndc_transform.inverse()
        return unprojection_transform.transform_points(xy_sdepth)

    def is_perspective(self):
        return False

    def in_ndc(self):
        return True
