import math
import os.path as osp
from typing import Iterable, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from mmcv.visualization import flow2rgb
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from tqdm import trange

from .base_renderer import MeshBaseRenderer
from .builder import RENDERER


@RENDERER.register_module(name=['opticalflow', 'optical_flow', 'OpticalFlow'])
class OpticalFlowRenderer(MeshBaseRenderer):

    def __init__(self,
                 resolution: Tuple[int, int] = [1024, 1024],
                 device: Union[torch.device, str] = 'cpu',
                 obj_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 return_type: Optional[List] = None,
                 out_img_format: str = '%06d.png',
                 projection: Literal['weakperspective', 'fovperspective',
                                     'orthographics', 'perspective',
                                     'fovorthographics'] = 'weakperspective',
                 in_ndc: bool = True,
                 **kwargs) -> None:
        """OpticalFlowRenderer for neural rendering, visualization and warping.

        Args:
            resolution (Iterable[int]):
                (width, height) of the rendered images resolution.
            device (Union[torch.device, str], optional):
                You can pass a str or torch.device for cpu or gpu render.
                Defaults to 'cpu'.
            output_path (Optional[str], optional):
                Output path of the video or images to be saved.
                Defaults to None.
            return_tensor (bool, optional):
                Boolean of whether return the rendered tensor.
                Defaults to False.
            out_img_format (str, optional): The image format string for
                saving the images.
                Defaults to '%06d.png'.
            projection (str, optional):
                Projection type of camera.
                Defaults to 'weakperspetive'.
            in_ndc (bool, optional): Whether defined in NDC.
                Defaults to True.
        Returns:
            None
        """
        super().__init__(
            resolution=resolution,
            device=device,
            obj_path=obj_path,
            output_path=output_path,
            return_type=return_type,
            out_img_format=out_img_format,
            projection=projection,
            in_ndc=in_ndc,
            **kwargs)

    def forward(
        self,
        meshes: Optional[Meshes] = None,
        cameras=None,
        indexes: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, None]:
        """Render Meshes.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Defaults to None.
            K (Optional[torch.Tensor], optional): Camera intrinsic matrixs.
                Defaults to None.
            R (Optional[torch.Tensor], optional): Camera rotation matrixs.
                Defaults to None.
            T (Optional[torch.Tensor], optional): Camera tranlastion matrixs.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for the
                images.
                Defaults to None.
        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        meshes = meshes.to(self.device)

        renderer = self.init_renderer(cameras, self.lights)

        rendered_images = renderer(meshes)
        rgbs = rendered_images.clone()[..., :3]
        rgbs = rgbs / rgbs.max()

        if self.output_path is not None:
            optical_flow = rgbs[..., :3].detach().cpu().numpy()
            for idx in range(len(optical_flow)):
                flow_image = (flow2rgb(optical_flow[idx, ..., :2]) *
                              255).astype(np.uint8)
                # flow_image = (optical_flow[idx] * 255).astype(np.uint8)
                real_idx = indexes[idx]
                folder = self.temp_path if self.temp_path is not None else\
                    self.output_path
                cv2.imwrite(
                    osp.join(folder, self.out_img_format % real_idx),
                    flow_image)
        if self.return_tensor:
            return rendered_images
        else:
            return None

    def forward_by_batch(self, meshes, K, R, T, batch_size):
        meshes = meshes.to(self.device)
        num_frames = len(meshes)
        verts = meshes.verts_padded()
        cameras = self.init_cameras(K=K, R=R, T=T)
        num_verts = verts.shape[1]
        verts_motion = cameras[1:].transform_points_ndc(
            verts[1:]) - cameras[:-1].transform_points_ndc(verts[:-1])
        verts_motion = torch.cat(
            [verts_motion,
             torch.zeros(1, num_verts, 3).to(self.device)], 0)
        min_, max_ = verts_motion.min(), verts_motion.max()

        verts_motion_norm = (verts_motion - min_) / (max_ - min_)

        textures = TexturesVertex(verts_features=verts_motion_norm)
        meshes.textures = textures
        results = []
        for i in trange(math.ceil(num_frames // batch_size)):
            indexes = list(
                range(i * batch_size, min((i + 1) * batch_size, num_frames)))
            rendered_images_batch = self.forward(
                meshes[indexes], cameras=cameras[indexes], indexes=indexes)
            if self.return_tensor:
                results.append(rendered_images_batch)
        self.export()

        if self.return_tensor:
            results = torch.cat(results)
            mask = results[..., 3:] > 0
            results = results * (max_ - min_) + min_
            results = results[..., :2] * mask
            return results
