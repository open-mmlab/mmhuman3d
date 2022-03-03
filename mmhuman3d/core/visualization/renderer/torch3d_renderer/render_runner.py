import math
import os
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
from pytorch3d.renderer import MeshRenderer, SoftSilhouetteShader
from pytorch3d.renderer.cameras import CamerasBase
from tqdm import trange

from mmhuman3d.core.cameras import MMCamerasBase
from mmhuman3d.core.cameras.builder import build_cameras
from .base_renderer import BaseRenderer
from .builder import build_lights, build_renderer
from .lights import AmbientLights, DirectionalLights, PointLights

osj = os.path.join


def render(renderer: Union[nn.Module, dict],
           output_path: Optional[str] = None,
           resolution: Union[Iterable[int], int] = None,
           device: Union[str, torch.device] = 'cpu',
           cameras: Union[MMCamerasBase, CamerasBase, dict, None] = None,
           lights: Union[AmbientLights, DirectionalLights, PointLights, dict,
                         None] = None,
           batch_size: int = 5,
           return_tensor: bool = False,
           no_grad: bool = False,
           **forward_params):

    if isinstance(renderer, dict):
        renderer = build_renderer(renderer)
    elif isinstance(renderer, MeshRenderer):
        if isinstance(renderer.shader, SoftSilhouetteShader):
            renderer = build_renderer(
                dict(
                    type='silhouette',
                    resolution=resolution,
                    shader=renderer.shader,
                    rasterizer=renderer.rasterizer))
        else:
            renderer = build_renderer(
                dict(
                    type='mesh',
                    resolution=resolution,
                    shader=renderer.shader,
                    rasterizer=renderer.rasterizer))
    elif isinstance(renderer, BaseRenderer):
        renderer = renderer
    else:
        raise TypeError('Wrong input renderer type.')

    renderer = renderer.to(device)
    if output_path is not None:
        renderer._set_output_path(output_path)

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
    else:
        raise TypeError('Wrong input cameras type.')

    if isinstance(lights, dict):
        lights = build_lights(lights)
    elif isinstance(lights, (AmbientLights, DirectionalLights, PointLights)):
        lights = lights
    elif lights is None:
        lights = AmbientLights()
    else:
        raise ValueError('Wrong light type.')

    meshes = renderer._prepare_meshes(device=device, **forward_params)

    num_frames = len(meshes)

    if len(cameras) == 1:
        cameras = cameras.extend(num_frames)
    if len(lights) == 1:
        lights = lights.extend(num_frames)

    forward_params.update(lights=lights, cameras=cameras, meshes=meshes)

    batch_size = min(batch_size, num_frames)
    tensors = []
    for i in trange(math.ceil(num_frames // batch_size)):
        indexes = list(
            range(i * batch_size, min((i + 1) * batch_size, len(meshes))))
        foward_params_batch = {}

        for k in forward_params:
            if hasattr(forward_params[k], '__getitem__'):
                foward_params_batch[k] = forward_params[k][indexes].to(device)

        if no_grad:
            with torch.no_grad():
                images_batch = renderer(indexes=indexes, **foward_params_batch)

        else:
            images_batch = renderer(indexes=indexes, **foward_params_batch)
        if return_tensor:
            tensors.append(images_batch)

    renderer.export()

    if return_tensor:
        tensors = torch.cat(tensors)
        return tensors
