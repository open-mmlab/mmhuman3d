import math
import os
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.lighting import (
    AmbientLights,
    DirectionalLights,
    PointLights,
)
from pytorch3d.structures.meshes import Meshes
from tqdm import trange

from mmhuman3d.core.cameras import MMCamerasBase
from mmhuman3d.core.cameras.builder import build_cameras
from mmhuman3d.core.visualization.renderer import build_lights

osj = os.path.join


def render(output_path: Optional[str] = None,
           device: Union[str, torch.device, None] = None,
           meshes: Meshes = None,
           cameras: Union[MMCamerasBase, dict, None] = None,
           lights: Union[AmbientLights, DirectionalLights, PointLights, dict,
                         None] = None,
           renderer: Union[nn.Module, dict, None] = None,
           batch_size: int = 5,
           return_tensor: bool = False,
           resolution: Union[Iterable[int], int] = None,
           no_grad: bool = False):

    if device is not None:
        renderer = renderer.to(device)

    if isinstance(cameras, dict):
        cameras = build_cameras(cameras)
    elif isinstance(cameras, MMCamerasBase):
        cameras = cameras
    elif isinstance(cameras,
                    CamerasBase) and not isinstance(cameras, MMCamerasBase):
        cameras = build_cameras(
            dict(
                type=cameras.__class__.__name__,
                K=cameras.K,
                R=cameras.R,
                T=cameras.T,
                in_ndc=cameras.in_ndc(),
                resolution=resolution))
    cameras = cameras.to(device)

    if isinstance(lights, dict):
        lights = build_lights(lights)
    elif cameras is None:
        lights = AmbientLights()
    elif isinstance(cameras, Union[AmbientLights, DirectionalLights,
                                   PointLights]):
        lights = lights
    else:
        raise ValueError('Wrong light type.')
    lights = lights.to(device)

    if output_path is not None:
        renderer._set_output_path(output_path)

    num_frames = len(meshes)
    if len(cameras) == 1:
        cameras = cameras.extend(num_frames)
    batch_size = min(batch_size, num_frames)
    tensors = []
    for i in trange(math.ceil(num_frames // batch_size)):
        indexes = list(
            range(i * batch_size, min((i + 1) * batch_size, len(meshes))))
        if no_grad:
            with torch.no_grad():
                images_batch = renderer(
                    meshes=meshes[indexes],
                    cameras=cameras[indexes],
                    lights=lights,
                    indexes=indexes)
        else:
            images_batch = renderer(
                meshes=meshes[indexes],
                cameras=cameras[indexes],
                lights=lights,
                indexes=indexes)
        if return_tensor:
            tensors.append(images_batch)

    renderer.export()

    if return_tensor:
        tensors = torch.cat(tensors)
        return tensors
