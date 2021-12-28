import torch
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.utils import ico_sphere

from mmhuman3d.core.visualization import render_runner


def test_render_runner():
    device = torch.device('cuda:0')
    meshes = ico_sphere(3, device)
    meshes.textures = TexturesVertex(
        verts_features=torch.zeros_like(meshes.verts_padded()).to(device))
    for render_choice in [
            'hq', 'lq', 'mq', 'pointcloud', 'normal', 'depth', 'silhouette'
    ]:
        render_runner.render(
            meshes=meshes.extend(2),
            render_choice=render_choice,
            orbit_speed=1.0,
            dist_speed=0.0,
            device=device,
            batch_size=2,
            output_path=f'/tmp/{render_choice}.mp4')
