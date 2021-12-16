import mmcv
import numpy as np
import torch

from mmhuman3d.apis import (
    feature_extract,
    inference_image_based_model,
    inference_video_based_model,
    init_model,
)
from mmhuman3d.utils.demo_utils import (
    conver_verts_to_cam_coord,
    extract_feature_sequence,
)


def test_inference_image_based_model():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    mesh_model, _ = init_model(
        'configs/spin/resnet50_spin_pw3d.py', None, device=device_name)

    if torch.cuda.is_available():
        frames_iter = np.ones([224, 224, 3])
        person_results = [{'track_id': 0, 'bbox': [0, 0, 224, 224, 1]}]
        mesh_results = inference_image_based_model(mesh_model, frames_iter,
                                                   person_results)
        pred_cams = mesh_results[0]['camera'][None]
        verts = mesh_results[0]['vertices'][None]
        bboxes_xy = mesh_results[0]['bbox'][None]
        _, _ = conver_verts_to_cam_coord(verts, pred_cams, bboxes_xy)


def test_inference_video_based_model():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'

    config = mmcv.Config.fromfile('configs/vibe/resnet50_vibe_pw3d.py')
    config.extractor.checkpoint = None
    mesh_model, extractor = init_model(config, None, device=device_name)
    if torch.cuda.is_available():
        frames_iter = np.ones([224, 224, 3])
        person_results = [{'track_id': 0, 'bbox': [0, 0, 224, 224, 1]}]
        person_results = feature_extract(extractor, frames_iter,
                                         person_results)
        person_results_list = [person_results]
        feature_results_seq = extract_feature_sequence(
            person_results_list, frame_idx=0, causal=True, seq_len=16, step=1)
        inference_video_based_model(
            mesh_model,
            extracted_results=feature_results_seq,
            with_track_id=True)
        inference_video_based_model(
            mesh_model,
            extracted_results=feature_results_seq,
            with_track_id=False)
