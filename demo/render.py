from mmhuman3d.models.builder import build_body_model
from mmhuman3d.core.visualization.renderer import build_renderer
from mmhuman3d.core.visualization.renderer import build_renderer
from mmhuman3d.core.conventions.cameras.convert_convention import \
convert_camera_matrix  # prevent yapf isort conflict
from mmhuman3d.utils.demo_utils import conver_verts_to_cam_coord, get_default_hmr_intrinsic, smooth_process
from mmhuman3d.core.cameras.builder import build_cameras
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes
from mmhuman3d.data.data_structures.human_data import HumanData
import mmcv
import cv2
import os
import torch
import numpy as np
result_path = 'data/demo_result/inference_result.npz'
root_path = 'data/demo_result'

data = HumanData.fromfile(result_path)
vertices = data['verts']
img_paths = data['image_path']
bbox_xyxy = data['bboxes_xyxy']
pred_cams = data['pred_cams']

device = 'cuda'
resolution = list(cv2.imread(os.path.join(root_path,'images/000000.png')).shape[:2])

# prepare mesh
body_model = build_body_model(
        dict(
            type='SMPL',
            gender='neutral',
            num_betas=10,
            model_path='data/body_models/smpl'))

faces = torch.tensor(body_model.faces.astype(np.int32))[None]

# build renderer

render_cfg = mmcv.Config.fromfile(
'configs/render/smpl.py')['RENDER_CONFIGS']['lq']
render_cfg['device'] = 'cuda'
render_cfg['resolution'] = resolution

renderer = build_renderer(render_cfg)

# build camera
K = get_default_hmr_intrinsic(num_frame=1, focal_length=5000.)[0]
K, R, T = convert_camera_matrix(
    convention_dst='pytorch3d',
    K=K,
    R=None,
    T=None,
    is_perspective=True,
    convention_src='opencv',
    resolution_src=resolution,
    in_ndc_src=False,
    in_ndc_dst=False)
# # camera
cameras = build_cameras(
    dict(
        type='perspective',
        in_ndc=False,
        device=device,
        K=K,
        R=R,
        T=T,
        resolution=resolution)).to(device)
textures = TexturesVertex(verts_features=torch.ones([1,6890,3])).to(device)
import time
T1 = time.time()
for i in range(35):
    print(f'visualize {i}th frame...')
    origin_img = cv2.imread(os.path.join(root_path,img_paths[i]))
    vert, _ = conver_verts_to_cam_coord(
        vertices[i], pred_cams[i], bbox_xyxy[i], focal_length=5000.) 
    # generate mesh
    # if i==0:
    vert = torch.tensor(vert.squeeze(), dtype=torch.float32)[None]
    mesh = Meshes(vert, faces, textures)
    rendered_images = renderer(meshes=mesh, cameras=cameras)
    # rgba = renderer.tensor2rgba(rendered_images).cpu().numpy().squeeze()

    # rgbs, valid_masks = rgba[..., :3] * 255, rgba[..., 3:]
    # img = (rgbs * valid_masks + (1 - valid_masks) * origin_img).astype(np.uint8)
    # cv2.imshow('vis', img)
    # cv2.waitKey(1)

print(time.time()-T1)
cv2.destroyAllWindows()
















