from scipy.spatial.transform import Rotation as R

import os
import os.path as osp
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
import torch
import warnings

from mmhuman3d.apis import (
    feature_extract,
    inference_video_based_model,
    init_model,
)
from mmhuman3d.core.visualization import visualize_smpl_vibe
from mmhuman3d.utils import array_to_images
from mmhuman3d.utils.demo_utils import (
    extract_feature_sequence,
    prepare_frames,
    process_mmdet_results,
    process_mmtracking_results,
    inference_post_processing,
)

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    from mmtrack.apis import inference_mot
    from mmtrack.apis import init_model as init_tracking_model
    has_mmtrack = True
except (ImportError, ModuleNotFoundError):
    has_mmtrack = False


def single_person_with_mmdet(args, frames_iter):
    """Estimate smpl parameters from single-person
        images with mmdetection
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames

    """
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    mesh_model, extractor = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())

    # Used to save the img index
    img_index = []
    person_results_list = []
    pred_cams, verts, bboxes_xyxy, poses, betas = [], [], [], [], []

    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, frame)

        # keep the person class bounding boxes.
        person_results = \
            process_mmdet_results(
                mmdet_results,
                args.det_cat_id)

        # extract features from the input video or image sequences
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator' \
                and extractor is not None:
            person_results = feature_extract(
                extractor, frame, person_results, args.bbox_thr, format='xyxy')

        # drop the frame with no detected results
        if person_results == []:
            continue

        # vis bboxes
        if args.draw_bbox:
            bboxes = [res['bbox'] for res in person_results]
            bboxes = np.vstack(bboxes)
            mmcv.imshow_bboxes(
                frame, bboxes, top_k=-1, thickness=2, show=False)

        person_results_list.append(person_results)
        img_index.append(i)

    for i, person_results in enumerate(
            mmcv.track_iter_progress(person_results_list)):
        feature_results_seq = extract_feature_sequence(
            person_results_list, frame_idx=i, causal=True, seq_len=16, step=1)

        mesh_results = inference_video_based_model(
            mesh_model,
            extracted_results=feature_results_seq,
            with_track_id=False)
        # only single person
        det_result = person_results[0]
        pred_cams.append(mesh_results[0]['camera'])
        verts.append(mesh_results[0]['vertices'])
        bboxes_xyxy.append(det_result['bbox'])
        poses.append(mesh_results[0]['smpl_pose'])
        betas.append(mesh_results[0]['smpl_beta'])

    pred_cams = np.array(pred_cams)
    verts = np.array(verts)
    bboxes_xyxy = np.array(bboxes_xyxy)
    poses = np.array(poses)
    betas = np.array(betas)

    del mesh_model
    del extractor
    del person_det_model
    torch.cuda.empty_cache()

    # smooth
    if args.post_processing is not None:
        if args.post_processing == "deciwatch":
            warnings.warn(
                'deciwatch is not a good choice for video based estimation, '
                'please try gaus1d, oneeuro, and savgol post processing methods.'
            )
        poses = inference_post_processing(
            poses,
            args.post_processing,
            cfg=args.mesh_reg_config,
            device=args.device.lower())

    if args.show_path is not None:
        frames_folder = osp.join(Path(args.show_path).parent, 'images')
        os.makedirs(frames_folder, exist_ok=True)
        array_to_images(
            np.array(frames_iter)[img_index], output_folder=frames_folder)

        body_model_config = dict(model_path=args.body_model_dir, type='smpl')
        visualize_smpl_vibe(
            poses=R.from_matrix(poses.reshape(-1, 3, 3)).as_rotvec().reshape(
                -1, 24 * 3),
            betas=betas,
            pred_cam=pred_cams,
            bbox=bboxes_xyxy,
            output_path=args.show_path,
            render_choice=args.render_choice,
            resolution=frames_iter[0].shape[:2],
            origin_frames=frames_folder,
            body_model_config=body_model_config,
            overwrite=True,
            palette=args.palette,
            read_frames_batch=True)


def multi_person_with_mmtracking(args, frames_iter):
    """Estimate smpl parameters from multi-person
        images with mmtracking
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames
    """
    tracking_model = init_tracking_model(
        args.tracking_config, None, device=args.device.lower())

    mesh_model, extractor = \
        init_model(args.mesh_reg_config, args.mesh_reg_checkpoint,
                   device=args.device.lower())

    max_track_id = 0
    max_instance = 0
    person_results_list = []
    frame_num = 0
    img_index = []

    # First stage: person tracking
    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        mmtracking_results = inference_mot(tracking_model, frame, frame_id=i)

        # keep the person class bounding boxes.
        person_results, max_track_id, instance_num = \
            process_mmtracking_results(mmtracking_results, max_track_id)

        # extract features from the input video or image sequences
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator' \
                and extractor is not None:
            person_results = feature_extract(
                extractor, frame, person_results, args.bbox_thr, format='xyxy')

        # drop the frame with no detected results
        if person_results == []:
            continue

        # update max_instance
        if instance_num > max_instance:
            max_instance = instance_num

        # vis bboxes
        if args.draw_bbox:
            bboxes = [res['bbox'] for res in person_results]
            bboxes = np.vstack(bboxes)
            mmcv.imshow_bboxes(
                frame, bboxes, top_k=-1, thickness=2, show=False)

        person_results_list.append(person_results)
        img_index.append(i)
        frame_num += 1

    verts = np.zeros([frame_num, max_track_id + 1, 6890, 3])
    pred_cams = np.zeros([frame_num, max_track_id + 1, 3])
    bboxes_xyxy = np.zeros([frame_num, max_track_id + 1, 5])
    poses = np.zeros([frame_num, max_track_id + 1, 24, 3, 3])
    betas = np.zeros([frame_num, max_track_id + 1, 10])
    track_ids_lists = []
    # Second stage: estimate smpl parameters
    for i, person_results in enumerate(
            mmcv.track_iter_progress(person_results_list)):
        feature_results_seq = extract_feature_sequence(
            person_results_list, frame_idx=i, causal=True, seq_len=16, step=1)

        mesh_results = inference_video_based_model(
            mesh_model,
            extracted_results=feature_results_seq,
            with_track_id=True)

        track_ids = []
        for idx, mesh_result in enumerate(mesh_results):
            det_result = person_results[idx]
            instance_id = det_result['track_id']
            bboxes_xyxy[i, instance_id] = det_result['bbox']
            pred_cams[i, instance_id] = mesh_result['camera']
            verts[i, instance_id] = mesh_result['vertices']
            poses[i, instance_id] = mesh_result['smpl_pose']
            betas[i, instance_id] = mesh_result['smpl_beta']
            track_ids.append(instance_id)

        track_ids_lists.append(track_ids)

    del mesh_model
    del extractor
    del tracking_model
    torch.cuda.empty_cache()

    # smooth
    if args.post_processing is not None:
        if args.post_processing == "deciwatch":
            warnings.warn(
                'deciwatch is not a good choice for video based estimation, '
                'please try gaus1d, oneeuro, and savgol post processing methods.'
            )
        poses = inference_post_processing(
            poses,
            args.post_processing,
            cfg=args.mesh_reg_config,
            device=args.device.lower())

    # To compress vertices array
    compressed_verts = np.zeros([frame_num, max_instance, 6890, 3])
    compressed_cams = np.zeros([frame_num, max_instance, 3])
    compressed_bboxs = np.zeros([frame_num, max_instance, 5])
    compressed_poses = np.zeros([frame_num, max_instance, 24, 3, 3])
    compressed_betas = np.zeros([frame_num, max_instance, 10])
    for i, track_ids_list in enumerate(track_ids_lists):
        instance_num = len(track_ids_list)
        compressed_verts[i, :instance_num] = verts[i, track_ids_list]
        compressed_cams[i, :instance_num] = pred_cams[i, track_ids_list]
        compressed_bboxs[i, :instance_num] = bboxes_xyxy[i, track_ids_list]
        compressed_poses[i, :instance_num] = poses[i, track_ids_list]
        compressed_betas[i, :instance_num] = betas[i, track_ids_list]
    assert len(img_index) > 0

    if args.show_path is not None:
        frames_folder = osp.join(Path(args.show_path).parent, 'images')
        os.makedirs(frames_folder, exist_ok=True)
        array_to_images(
            np.array(frames_iter)[img_index], output_folder=frames_folder)
        body_model_config = dict(model_path=args.body_model_dir, type='smpl')
        visualize_smpl_vibe(
            poses=R.from_matrix(compressed_poses.reshape(
                -1, 3, 3)).as_rotvec().reshape(-1, max_instance, 24 * 3),
            betas=compressed_betas,
            pred_cam=compressed_cams,
            bbox=compressed_bboxs,
            output_path=args.show_path,
            render_choice=args.render_choice,
            resolution=frames_iter[0].shape[:2],
            origin_frames=frames_folder,
            body_model_config=body_model_config,
            overwrite=True,
            palette=args.palette,
            read_frames_batch=True)


def main(args):

    # prepare input
    frames_iter = prepare_frames(args.input_path)

    if args.single_person_demo:
        # verts, pred_cams, bboxes, img_index = \
        single_person_with_mmdet(args, frames_iter)
    elif args.multi_person_demo:
        multi_person_with_mmtracking(args, frames_iter)
    else:
        raise ValueError(
            'Only supports single_person_demo or multi_person_demo')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        'mesh_reg_config',
        type=str,
        default=None,
        help='Config file for mesh regression')
    parser.add_argument(
        'mesh_reg_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for mesh regression')
    parser.add_argument(
        '--single_person_demo',
        action='store_true',
        help='Single person demo with MMDetection')
    parser.add_argument('--det_config', help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--multi_person_demo',
        action='store_true',
        help='Multi person demo with MMTracking')
    parser.add_argument('--tracking_config', help='Config file for tracking')
    parser.add_argument(
        '--body_model_dir',
        type=str,
        default='data/body_models/',
        help='Body models file path')
    parser.add_argument(
        '--input_path', type=str, default=None, help='Input path')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='directory to save output result file')
    parser.add_argument(
        '--show_path',
        type=str,
        default=None,
        help='directory to save rendered images or video')
    parser.add_argument(
        '--render_choice',
        type=str,
        default='hq',
        help='Render choice parameters')
    parser.add_argument(
        '--palette', type=str, default='segmentation', help='Color theme')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.99,
        help='Bounding box score threshold')
    parser.add_argument(
        '--draw_bbox',
        action='store_true',
        help='Draw a bbox for each detected instance')
    parser.add_argument(
        '--post_processing',
        type=str,
        default=None,
        help='Post processing the data through the specified type.')
    parser.add_argument(
        '--focal_length', type=float, default=5000., help='Focal lenght')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()

    if args.single_person_demo:
        assert has_mmdet, 'Please install mmdet to run the demo.'
        assert args.det_config is not None
        assert args.det_checkpoint is not None

    if args.multi_person_demo:
        assert has_mmtrack, 'Please install mmtrack to run the demo.'
        assert args.tracking_config is not None

    main(args)
