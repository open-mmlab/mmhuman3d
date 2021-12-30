import math
import os
from pathlib import Path
from typing import Iterable, Union

import mmcv
import torch
from pytorch3d.structures.meshes import Meshes
from tqdm import trange

import mmhuman3d
from mmhuman3d.core.cameras import compute_orbit_cameras
from mmhuman3d.core.conventions.cameras import convert_cameras
from .builder import build_renderer

osj = os.path.join


def render(
    output_path: str,
    device: Union[str, torch.device] = 'cpu',
    meshes: Meshes = None,
    render_choice: str = 'base',
    batch_size: int = 5,
    K: torch.Tensor = None,
    R: torch.Tensor = None,
    T: torch.Tensor = None,
    projection: str = 'perspective',
    orbit_speed: Union[Iterable[float], float] = 0.0,
    dist: float = 2.7,
    dist_speed=1.0,
    in_ndc: bool = True,
    resolution=[1024, 1024],
    convention: str = 'pytorch3d',
    no_grad: bool = False,
    return_tensor: bool = True,
):
    RENDER_CONFIGS = mmcv.Config.fromfile(
        os.path.join(
            Path(mmhuman3d.__file__).parents[1],
            'configs/render/smpl.py'))['RENDER_CONFIGS'][render_choice]
    renderer = build_renderer(
        dict(
            type=RENDER_CONFIGS['renderer_type'],
            device=device,
            resolution=resolution,
            projection=projection,
            output_path=output_path,
            return_type=['tensor'] if return_tensor else None,
            in_ndc=in_ndc,
            **RENDER_CONFIGS))

    num_frames = len(meshes)
    if K is None or R is None or T is None:
        K, R, T = compute_orbit_cameras(
            orbit_speed=orbit_speed,
            dist=dist,
            batch_size=num_frames,
            dist_speed=dist_speed)

    if projection in ['perspective', 'fovperspective']:
        is_perspective = True
    else:
        is_perspective = False

    K, R, T = convert_cameras(
        resolution_dst=resolution,
        resolution_src=resolution,
        in_ndc_dst=in_ndc,
        in_ndc_src=in_ndc,
        K=K,
        R=R,
        T=T,
        is_perspective=is_perspective,
        convention_src=convention,
        convention_dst='pytorch3d')

    tensors = []
    for i in trange(math.ceil(num_frames // batch_size)):
        indexes = list(
            range(i * batch_size, min((i + 1) * batch_size, len(meshes))))
        if no_grad:
            with torch.no_grad():
                images_batch = renderer(
                    meshes=meshes[indexes],
                    K=K[indexes],
                    R=R[indexes],
                    T=T[indexes],
                    indexes=indexes)
        else:
            images_batch = renderer(
                meshes=meshes[indexes],
                K=K[indexes],
                R=R[indexes],
                T=T[indexes],
                indexes=indexes)
        tensors.append(images_batch['tensor'])

    tensors = torch.cat(tensors)
    renderer.export()
    return tensors
