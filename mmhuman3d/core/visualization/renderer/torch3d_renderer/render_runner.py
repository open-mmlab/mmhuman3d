import math
import os
from typing import Optional, Union

import torch
import torch.nn as nn
from pytorch3d.renderer.lighting import (
    AmbientLights,
    DirectionalLights,
    PointLights,
)
from pytorch3d.structures.meshes import Meshes
from tqdm import trange

from mmhuman3d.core.cameras.cameras import MMCamerasBase

osj = os.path.join


def render(output_path: Optional[str] = None,
           device: Union[str, torch.device, None] = None,
           meshes: Meshes = None,
           cameras: Optional[MMCamerasBase] = None,
           lights: Union[AmbientLights, DirectionalLights, PointLights] = None,
           renderer: Optional[nn.Module] = None,
           batch_size: int = 5,
           return_tensor=False,
           no_grad: bool = False):

    renderer = renderer.to(device)

    cameras = cameras.to(device)
    lights = AmbientLights() if lights is None else lights
    lights = lights.to(device)
    if output_path is not None:
        renderer._set_output_path(output_path)
    if device is not None:
        renderer.device = device

    num_frames = len(meshes)
    if len(cameras) == 1:
        cameras = cameras.extend(num_frames)

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
