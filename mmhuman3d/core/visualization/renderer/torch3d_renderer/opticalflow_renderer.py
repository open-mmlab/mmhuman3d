from typing import Optional, Union, Iterable, Literal
from mmhuman3d.core.visualization.renderer.torch3d_renderer.base_renderer import MeshBaseRenderer

import torch
import torch.nn as nn

from pytorch3d.structures import Meshes

from .builder import RENDERER


@RENDERER.register_module(name=['opticalflow', 'optical_flow', 'OpticalFlow'])
class OpticalFlowRenderer(MeshBaseRenderer):

    def __init__(
        self,
        resolution: Iterable[int] = None,
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        out_img_format: str = '%06d.png',
        projection: Literal['weakperspective', 'fovperspective',
                            'orthographics', 'perspective',
                            'fovorthographics'] = 'weakperspective',
        in_ndc: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            resolution=resolution,
            device=device,
            output_path=output_path,
            obj_path=None,
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            **kwargs)

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if self.rasterizer.cameras is not None:
            self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        self.shader = self.shader.to(device)
        return self

    def forward(
        self,
        meshes_source: Optional[Meshes] = None,
        meshes_target: Optional[Meshes] = None,
        cameras=None,
        cameras_source=None,
        cameras_target=None,
        **kwargs,
    ) -> Union[torch.Tensor, None]:
        """[summary]

        Args:
            meshes_source (Optional[Meshes], optional):
                [description]. Defaults to None.
            meshes_target (Optional[Meshes], optional):
                [description]. Defaults to None.
            cameras ([type], optional): [description].
                Defaults to None.
            cameras_source ([type], optional):
                [description]. Defaults to None.
            cameras_target ([type], optional):
                [description]. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            Union[torch.Tensor, None]: [description]
        """
        assert len(meshes_source) == len(meshes_target)

        if cameras_source is None:
            cameras_source = cameras
        if cameras_target is None:
            cameras_target = cameras

        fragments_source = self.rasterizer(meshes_source,
                                           **{'cameras': cameras_source})
        fragments_target = self.rasterizer(meshes_target,
                                           **{'cameras': cameras_target})

        if cameras_source is None or cameras_target is None:
            msg = 'Cameras must be specified either at initialization \
                or in the forward pass of OpticalFLowShader'

            raise ValueError(msg)

        verts_source_ndc = cameras_source.transform_points_ndc(
            meshes_source.verts_padded())
        verts_target_ndc = cameras_target.transform_points_ndc(
            meshes_target.verts_padded())

        verts_scene_flow = verts_target_ndc - verts_source_ndc
        pixel_scene_flow = self.shader(
            fragments=fragments_target,
            meshes=meshes_target,
            verts_scene_flow=verts_scene_flow,
            **kwargs)

        mask_target = (fragments_target.pix_to_face >= 0).long()

        visible_face_idx_source = fragments_source.pix_to_face.unique()[1:]
        visible_face_idx_target = fragments_target.pix_to_face.unique()[1:]
        face_visibility_packed_source = torch.zeros(
            meshes_source.faces_packed().shape[0]).long().to(
                meshes_source.device)
        face_visibility_packed_source[visible_face_idx_source] = 1

        face_visibility_packed_target = torch.zeros(
            meshes_target.faces_packed().shape[0]).long().to(
                meshes_target.device)
        face_visibility_packed_target[visible_face_idx_target] = 1

        face_visibility_packed = face_visibility_packed_source * \
            face_visibility_packed_target
        shape = fragments_target.pix_to_face.shape
        visiblity_mask = face_visibility_packed[
            fragments_target.pix_to_face.view(-1)].view(shape) * mask_target

        pixel_scene_flow = torch.cat([pixel_scene_flow, visiblity_mask], -1)
        return pixel_scene_flow
