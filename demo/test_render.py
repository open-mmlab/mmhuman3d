import cv2
import mmcv
import numpy as np
import torch
# from pyrender import Material
from pytorch3d.renderer import FoVOrthographicCameras, FoVPerspectiveCameras
# from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes

# from mmhuman3d.core.cameras import MMCamerasBase
from mmhuman3d.core.cameras.builder import build_cameras
from mmhuman3d.core.conventions.cameras.convert_convention import \
    convert_camera_matrix  # prevent yapf isort conflict
# from mmhuman3d.core.visualization import render_runner
from mmhuman3d.core.visualization.renderer import build_renderer
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.builder import build_body_model
from mmhuman3d.utils.demo_utils import conver_verts_to_cam_coord

result_path = 'data/demo_result/inference_result.npz'


def realtime_visualization(
        verts,
        faces,
        textures,
        renderer,
        pred_cam,
        # lights,
        perps=True):

    device = verts.device
    mesh = Meshes(verts, faces, textures)

    if pred_cam is not None:
        if perps:
            R, T, fov = pred_cam
            pred_cam = FoVPerspectiveCameras(R=R, T=T, fov=fov, device=device)
        else:
            R, T, xyz_ranges = pred_cam
            pred_cam = FoVOrthographicCameras(
                R=R, T=T, **xyz_ranges, device=device)

    image = renderer(meshes=mesh, cameras=pred_cam)
    return image


if __name__ == '__main__':
    device = 'cuda'
    data = HumanData.fromfile(result_path)
    render_cfg = mmcv.Config.fromfile(
        'configs/render/smpl.py')['RENDER_CONFIGS']
    renderer = build_renderer(render_cfg['lq']).to('cuda')
    # light = build_lights(render_cfg['lq']['lights'])
    pred_cam = data['pred_cams']
    bboxes_xy = data['bboxes_xyxy']
    image = mmcv.imread('data/demo_result/' + data['image_path'][0])
    render_resolution = [image.shape[0], image.shape[1]]
    verts, K = conver_verts_to_cam_coord(
        data['verts'], pred_cam, bboxes_xy, focal_length=5000.)
    verts = torch.tensor(verts[0], dtype=torch.float32)[None].to(device)
    # K = np.array([[5000,0,112],
    #               [0,5000,112],
    #               [0, 0, 1]])
    K, R, T = convert_camera_matrix(
        convention_dst='pytorch3d',
        K=K,
        R=None,
        T=None,
        is_perspective=True,
        convention_src='opencv',
        resolution_src=render_resolution,
        in_ndc_src=False,
        in_ndc_dst=False)
    cameras = build_cameras(
        dict(
            type='perspective',
            in_ndc=False,
            device='cuda',
            K=K,
            R=R,
            T=T,
            resolution=render_resolution)).to(device)
    # renderer.lights.location = \
    # cameras.get_camera_center()-cameras.get_camera_plane_normals()
    body_model = build_body_model(
        dict(type='SMPL', model_path='data/body_models/smpl'))
    faces = torch.tensor(body_model.faces.astype(np.float32))[None].to(device)
    textures = TexturesVertex(
        verts_features=torch.ones(verts.shape)).to(device)
    mesh = Meshes(verts, faces, textures)

    images = renderer(meshes=mesh, cameras=cameras)
    rgba = renderer.tensor2rgba(images).cpu().numpy()
    rgbs, valid_masks = rgba[..., :3] * 255, rgba[..., 3:]
    output_images = (rgbs * valid_masks + (1 - valid_masks) * image).astype(
        np.uint8)
    cv2.imwrite('test.png', output_images[0])

    # realtime_visualization(verts, faces, textures, renderer, pred_cam)

#     base_point_light = {
#     'type': 'point',
#     'ambient_color': [[0.56, 0.56, 0.56]],
#     # 'ambient_color': [[1, 1, 1]],
#     # 'diffuse_color': [[0.3, 0.3, 0.3]],
#     # 'specular_color': [[0.5, 0.5, 0.5]],
#     'location': [[0., 0., 0.]],
# }