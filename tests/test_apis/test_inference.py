import mmcv
import numpy as np
import pytest
import torch

from mmhuman3d.apis import (
    feature_extract,
    inference_image_based_model,
    inference_video_based_model,
    init_model,
)
from mmhuman3d.utils.demo_utils import (
    conver_verts_to_cam_coord,
    convert_crop_cam_to_orig_img,
    extract_feature_sequence,
    get_speed_up_interval,
    prepare_frames,
    process_mmdet_results,
    process_mmtracking_results,
    smooth_process,
    speed_up_process,
)


def test_inference_image_based_model():
    device_name = 'cpu'
    config = mmcv.Config.fromfile('configs/spin/resnet50_spin_pw3d.py')
    config.model.backbone.norm_cfg = dict(type='BN', requires_grad=True)
    mesh_model, _ = init_model(config, None, device=device_name)

    frames_iter = np.ones([224, 224, 3])
    person_results = [{'track_id': 0, 'bbox': [0, 0, 224, 224, 1]}]
    mesh_results = inference_image_based_model(mesh_model, frames_iter,
                                               person_results)
    pred_cams = mesh_results[0]['camera'][None]
    verts = mesh_results[0]['vertices'][None]
    bboxes_xy = mesh_results[0]['bbox'][None]
    smooth_process(verts.repeat(20, 0))
    speed_up_process(torch.ones(100, 24, 3, 3))
    get_speed_up_interval('deciwatch')
    _, _ = conver_verts_to_cam_coord(verts, pred_cams, bboxes_xy)


def test_inference_video_based_model():
    device_name = 'cpu'
    config = mmcv.Config.fromfile('configs/vibe/resnet50_vibe_pw3d.py')
    config.extractor.checkpoint = None
    config.extractor.backbone.norm_cfg = dict(type='BN', requires_grad=True)
    mesh_model, extractor = init_model(config, None, device=device_name)

    frames_iter = np.ones([224, 224, 3])
    person_results = [{'track_id': 0, 'bbox': [0, 0, 224, 224, 1]}]
    person_results = feature_extract(extractor, frames_iter, person_results)
    person_results_list = [person_results]
    feature_results_seq = extract_feature_sequence(
        person_results_list, frame_idx=0, causal=True, seq_len=16, step=1)

    inference_video_based_model(
        mesh_model, extracted_results=feature_results_seq, with_track_id=True)

    inference_video_based_model(
        mesh_model,
        extracted_results=feature_results_seq,
        with_track_id=True,
        causal=False)

    inference_video_based_model(
        mesh_model, extracted_results=feature_results_seq, with_track_id=False)


def test_process_mmdet_results():
    det_results = [[np.array([0, 0, 100, 100, 0.99])]]
    det_mask_results = None

    _ = process_mmdet_results(
        mmdet_results=(det_results, det_mask_results), cat_id=1, bbox_thr=0.9)

    _ = process_mmdet_results(
        mmdet_results=det_results, cat_id=1, bbox_thr=0.9)


def test_convert_crop_cam_to_orig_img():
    pred_cam = np.ones([10, 10, 10, 3])
    bbox = np.ones([10, 10, 5])
    img_width, img_height = 224, 224
    convert_crop_cam_to_orig_img(
        pred_cam, bbox, img_width, img_height, bbox_format='xyxy')
    convert_crop_cam_to_orig_img(
        pred_cam, bbox, img_width, img_height, bbox_format='xywh')
    convert_crop_cam_to_orig_img(
        pred_cam, bbox, img_width, img_height, bbox_format='cs')
    with pytest.raises(ValueError):
        convert_crop_cam_to_orig_img(
            pred_cam, bbox, img_width, img_height, bbox_format='xy')


def test_prepare_frames():
    image_path = 'demo/resources/S1_Directions_1.54138969_000001.jpg'
    prepare_frames(image_path)

    video_path = 'demo/resources/single_person_demo.mp4'
    prepare_frames(video_path)

    image_folder = 'demo/resources/'
    prepare_frames(image_folder)


def test_process_mmtracking_results():
    track_bboxes = {
        'track_bboxes': [[np.array([1, 0, 0, 100, 100, 0.99])]],
    }
    process_mmtracking_results(track_bboxes, max_track_id=0, bbox_thr=0.9)

    person_results = {
        'track_results': [[np.array([1, 0, 0, 100, 100, 0.99])]],
    }
    process_mmtracking_results(person_results, max_track_id=0, bbox_thr=0.9)
