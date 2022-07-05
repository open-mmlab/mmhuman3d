import math
import os
from typing import Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import MeshRenderer, SoftSilhouetteShader
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes
from tqdm import trange

from mmhuman3d.core.cameras import MMCamerasBase
from mmhuman3d.core.cameras.builder import build_cameras
from .base_renderer import BaseRenderer
from .builder import build_renderer
from .lights import AmbientLights, MMLights, build_lights

osj = os.path.join


def render(renderer: Union[nn.Module, dict],
           meshes: Union[Meshes, None] = None,
           output_path: Optional[str] = None,
           resolution: Union[Iterable[int], int] = None,
           device: Union[str, torch.device] = 'cpu',
           cameras: Union[MMCamerasBase, CamerasBase, dict, None] = None,
           lights: Union[MMLights, dict, None] = None,
           batch_size: int = 5,
           return_tensor: bool = False,
           no_grad: bool = False,
           verbose: bool = True,
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
    num_frames = len(meshes)
    if isinstance(lights, dict):
        lights = build_lights(lights)
    elif isinstance(lights, MMLights):
        lights = lights
    elif lights is None:
        lights = AmbientLights(device=device).extend(num_frames)
    else:
        raise ValueError('Wrong light type.')

    if len(cameras) == 1:
        cameras = cameras.extend(num_frames)
    if len(lights) == 1:
        lights = lights.extend(num_frames)

    forward_params.update(lights=lights, cameras=cameras, meshes=meshes)

    batch_size = min(batch_size, num_frames)
    tensors = []
    for k in forward_params:
        if isinstance(forward_params[k], np.ndarray):
            forward_params.update(
                {k: torch.tensor(forward_params[k]).to(device)})
    if verbose:
        iter_func = trange
    else:
        iter_func = range
    for i in iter_func(math.ceil(num_frames // batch_size)):
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
