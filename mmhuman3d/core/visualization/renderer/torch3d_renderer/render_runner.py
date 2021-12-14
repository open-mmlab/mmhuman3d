import os
from pathlib import Path
from typing import Iterable, Union

import mmcv
import torch
from pytorch3d.structures.meshes import Meshes

import mmhuman3d
from mmhuman3d.core.cameras import compute_orbit_cameras
from mmhuman3d.core.conventions.cameras import convert_cameras
from .builder import build_renderer


def render(
    output_path: str,
    device: Union[str, torch.device] = 'cpu',
    meshes: Meshes = None,
    render_choice: str = 'mesh',
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
):
    RENDER_CONFIGS = mmcv.Config.fromfile(
        os.path.join(
            Path(mmhuman3d.__file__).parents[1],
            'configs/render/smpl.py'))['RENDER_CONFIGS']
    renderer = build_renderer(
        dict(
            type=render_choice,
            device=device,
            resolution=resolution,
            projection=projection,
            output_path=output_path,
            return_tensor=True,
            in_ndc=in_ndc,
            **RENDER_CONFIGS[render_choice]))
    num_frames = len(meshes)
    if K is None or R is None or T is None:
        K, R, T = compute_orbit_cameras(
            orbit_speed=orbit_speed,
            dist=dist,
            batch_size=num_frames,
            dist_speed=dist_speed)
    # batch_size = 5
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
    print(K.shape, R.shape, T.shape)
    # for i in trange(math.ceil(num_frames // batch_size)):
    #     indexs = list(
    #         range(i * batch_size, min((i + 1) * batch_size, num_frames)))
    from pytorch3d.io import load_objs_as_meshes
    meshes_texture = load_objs_as_meshes(
        ['/home/SENSETIME/wangwenjia/programs/data/cow_mesh/cow.obj'])
    base_renderer = build_renderer(
        dict(
            type='base',
            device=device,
            resolution=resolution,
            projection=projection,
            output_path=output_path,
            in_ndc=in_ndc,
            return_tensor=True,
            **RENDER_CONFIGS['hq']))

    image_texture = base_renderer(
        meshes_texture.extend(2), K=K, R=R, T=T, indexs=[0, 1])

    # base_renderer.forward(meshes, K=K, R=R, T=T, indexs=[1, 2])
    flow = renderer.forward_by_batch(meshes, K=K, R=R, T=T, batch_size=2)
    from IPython import embed
    from torch.nn.functional import grid_sample
    import cv2

    new_h = torch.linspace(-1, 1,
                           resolution[0]).view(-1, 1).repeat(1, resolution[1])
    new_w = torch.linspace(-1, 1, resolution[1]).repeat(resolution[0], 1)
    base_grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim=2)
    base_grid = base_grid.unsqueeze(0).to(device)

    image = image_texture[:1, ..., :3].permute(0, 3, 1, 2)
    grid_flow = base_grid + flow

    out = grid_sample(image, grid=grid_flow[:1], mode='bilinear')
    out_image = out[0].permute(1, 2, 0).cpu().numpy() * 255
    cv2.imwirte('cow_grid2.png', out_image)
    embed()
