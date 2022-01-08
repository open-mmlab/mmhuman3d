import math
import os
from typing import Optional, Union

import torch
from pytorch3d.structures.meshes import Meshes
from tqdm import trange

from mmhuman3d.core.cameras.builder import build_cameras
from mmhuman3d.core.cameras.cameras import NewAttributeCameras
from mmhuman3d.core.visualization.renderer import MeshBaseRenderer
from .builder import build_renderer

osj = os.path.join


def render(output_path: Optional[str] = None,
           device: Union[str, torch.device, None] = None,
           meshes: Meshes = None,
           cameras: Optional[NewAttributeCameras] = None,
           camera_config: Optional[dict] = None,
           renderer: Optional[MeshBaseRenderer] = None,
           render_config: Optional[dict] = None,
           batch_size: int = 5,
           no_grad: bool = False):
    if renderer is None:
        renderer = build_renderer(render_config)
    if output_path is not None:
        renderer.set_output_path(output_path)
    if device is not None:
        renderer.device = device
    num_frames = len(meshes)
    if cameras is None:
        cameras = build_cameras(camera_config)

    tensors = []
    for i in trange(math.ceil(num_frames // batch_size)):
        indexes = list(
            range(i * batch_size, min((i + 1) * batch_size, len(meshes))))
        if no_grad:
            with torch.no_grad():
                images_batch = renderer(
                    meshes=meshes[indexes],
                    cameras=cameras[indexes],
                    indexes=indexes)
        else:
            images_batch = renderer(
                meshes=meshes[indexes],
                cameras=cameras[indexes],
                indexes=indexes)
        tensors.append(images_batch['tensor'])
    renderer.export()
    tensors = torch.cat(tensors)
    renderer.export()
    return tensors
