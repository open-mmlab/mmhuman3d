import numpy as np
import torch

from mmhuman3d.apis import inference_model, init_model
from mmhuman3d.utils.demo_utils import conver_verts_to_cam_coord


def test_inference_model():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    mesh_model = init_model(
        'configs/spin/resnet50_spin_pw3d.py', None, device=device_name)

    if torch.cuda.is_available():
        frames_iter = np.ones([224, 224, 3])
        person_results = [{'track_id': 0, 'bbox': [0, 0, 224, 224, 1]}]
        mesh_results = inference_model(mesh_model, frames_iter, person_results)
        pred_cams = mesh_results[0]['camera'][None]
        verts = mesh_results[0]['vertices'][None]
        bboxes_xy = mesh_results[0]['bbox'][None]
        _, _ = conver_verts_to_cam_coord(verts, pred_cams, bboxes_xy)
