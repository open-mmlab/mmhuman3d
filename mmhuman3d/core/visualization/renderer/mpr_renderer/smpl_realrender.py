# These codes brought from `https://github.com/rmbashirov/rgbd-kinect-pose/`

import cv2
import numpy as np
import torch

from mmhuman3d.core.visualization.renderer.mpr_renderer.camera import Pinhole2D
from mmhuman3d.core.visualization.renderer.mpr_renderer.rasterizer import \
    estimate_normals  # noqa: E501
from mmhuman3d.core.visualization.renderer.mpr_renderer.utils import \
    vis_normals  # noqa: E501


class VisualizerMeshSMPL:

    def __init__(self,
                 device=None,
                 body_models=None,
                 focal_length=5000.,
                 camera_center=[112., 112.],
                 resolution=None,
                 scale=None):
        self.body_models = body_models
        self.pinhole2d = Pinhole2D(
            fx=focal_length,
            fy=focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            w=resolution[1],
            h=resolution[0])
        self.device = torch.device(device)
        self.faces = self.body_models.faces_tensor.to(
            dtype=torch.int32, device=self.device)

    def __call__(self, vertices, bg=None, **kwargs):
        assert vertices.device == self.faces.device
        vertices = vertices.clone()
        coords, normals = estimate_normals(
            vertices=vertices, faces=self.faces, pinhole=self.pinhole2d)
        vis = vis_normals(coords, normals)
        if bg is not None:
            mask = coords[:, :, [2]] <= 0
            vis = (
                vis[:, :, None] +
                torch.tensor(bg).to(mask.device) * mask).cpu().numpy().astype(
                    np.uint8)
        else:
            # convert gray to 3 channel img
            vis = vis.detach().cpu().numpy()
            vis = cv2.merge((vis, vis, vis))
        # z_buffer = project_mesh(
        #     vertices=vertices,
        #     faces=self.faces,
        #     vertice_values=vertices[:, [2]],
        #     pinhole=self.pinhole2d)
        # z_buffer = z_buffer[:, :, 0].cpu().numpy()
        # vis = colormap_z(z_buffer, percentile=1)
        # vis = (vis * 255).round().clip(0,255).astype(np.uint8)[..., :3]
        return vis
