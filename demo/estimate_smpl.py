import os
import os.path as osp
import shutil
import warnings
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
import torch

from mmhuman3d.apis import (
    feature_extract,
    inference_image_based_model,
    inference_video_based_model,
    init_model,
)
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_hmr
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.utils.demo_utils import (
    extract_feature_sequence,
    get_speed_up_interval,
    prepare_frames,
    process_mmdet_results,
    process_mmtracking_results,
    smooth_process,
    speed_up_interpolate,
    speed_up_process,
)
from mmhuman3d.utils.ffmpeg_utils import array_to_images
from mmhuman3d.utils.transforms import rotmat_to_aa

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


def get_tracking_result(args, frames_iter, mesh_model, extractor):
    tracking_model = init_tracking_model(
        args.tracking_config, None, device=args.device.lower())

    max_track_id = 0
    max_instance = 0
    result_list = []
    frame_id_list = []

    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        mmtracking_results = inference_mot(tracking_model, frame, frame_id=i)

        # keep the person class bounding boxes.
        result, max_track_id, instance_num = \
            process_mmtracking_results(
                mmtracking_results,
                max_track_id=max_track_id,
                bbox_thr=args.bbox_thr)

        # extract features from the input video or image sequences
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator' \
                and extractor is not None:
            result = feature_extract(
                extractor, frame, result, args.bbox_thr, format='xyxy')

        # drop the frame with no detected results
        if result == []:
            continue

        # update max_instance
        if instance_num > max_instance:
            max_instance = instance_num

        # vis bboxes
        if args.draw_bbox:
            bboxes = [res['bbox'] for res in result]
            bboxes = np.vstack(bboxes)
            mmcv.imshow_bboxes(
                frame, bboxes, top_k=-1, thickness=2, show=False)

        result_list.append(result)
        frame_id_list.append(i)

    return max_track_id, max_instance, frame_id_list, result_list


def get_detection_result(args, frames_iter, mesh_model, extractor):
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    frame_id_list = []
    result_list = []
    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        mmdet_results = inference_detector(person_det_model, frame)
        # keep the person class bounding boxes.
        results = process_mmdet_results(
            mmdet_results, cat_id=args.det_cat_id, bbox_thr=args.bbox_thr)

        # extract features from the input video or image sequences
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator' \
                and extractor is not None:
            results = feature_extract(
                extractor, frame, results, args.bbox_thr, format='xyxy')
        # drop the frame with no detected results
        if results == []:
            continue
        # vis bboxes
        if args.draw_bbox:
            bboxes = [res['bbox'] for res in results]
            bboxes = np.vstack(bboxes)
            mmcv.imshow_bboxes(
                frame, bboxes, top_k=-1, thickness=2, show=False)
        frame_id_list.append(i)
        result_list.append(results)

    return frame_id_list, result_list


def single_person_with_mmdet(args, frames_iter):
    """Estimate smpl parameters from single-person
        images with mmdetection
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames

    """

    mesh_model, extractor = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())

    pred_cams, verts, smpl_poses, smpl_betas, bboxes_xyxy = \
        [], [], [], [], []

    frame_id_list, result_list = \
        get_detection_result(args, frames_iter, mesh_model, extractor)

    frame_num = len(frame_id_list)
    # speed up
    if args.speed_up_type:
        speed_up_interval = get_speed_up_interval(args.speed_up_type)
        speed_up_frames = (frame_num -
                           1) // speed_up_interval * speed_up_interval

    for i, result in enumerate(mmcv.track_iter_progress(result_list)):
        frame_id = frame_id_list[i]
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator':
            if args.speed_up_type:
                warnings.warn(
                    'Video based models do not support speed up. '
                    'By default we will inference with original speed.',
                    UserWarning)
            feature_results_seq = extract_feature_sequence(
                result_list, frame_idx=i, causal=True, seq_len=16, step=1)
            mesh_results = inference_video_based_model(
                mesh_model,
                extracted_results=feature_results_seq,
                with_track_id=False)
        elif mesh_model.cfg.model.type == 'ImageBodyModelEstimator':
            if args.speed_up_type and i % speed_up_interval != 0\
                 and i <= speed_up_frames:
                mesh_results = [{
                    'bbox': np.zeros((5)),
                    'camera': np.zeros((3)),
                    'smpl_pose': np.zeros((24, 3, 3)),
                    'smpl_beta': np.zeros((10)),
                    'vertices': np.zeros((6890, 3)),
                    'keypoints_3d': np.zeros((17, 3)),
                }]
            else:
                mesh_results = inference_image_based_model(
                    mesh_model,
                    frames_iter[frame_id],
                    result,
                    bbox_thr=args.bbox_thr,
                    format='xyxy')
        else:
            raise Exception(
                f'{mesh_model.cfg.model.type} is not supported yet')

        smpl_betas.append(mesh_results[0]['smpl_beta'])
        smpl_pose = mesh_results[0]['smpl_pose']
        smpl_poses.append(smpl_pose)
        pred_cams.append(mesh_results[0]['camera'])
        verts.append(mesh_results[0]['vertices'])
        bboxes_xyxy.append(mesh_results[0]['bbox'])

    smpl_poses = np.array(smpl_poses)
    smpl_betas = np.array(smpl_betas)
    pred_cams = np.array(pred_cams)
    verts = np.array(verts)
    bboxes_xyxy = np.array(bboxes_xyxy)

    # release GPU memory
    del mesh_model
    del extractor
    torch.cuda.empty_cache()

    # speed up
    if args.speed_up_type:
        smpl_poses = speed_up_process(
            torch.tensor(smpl_poses).to(args.device.lower()),
            args.speed_up_type)

        selected_frames = np.arange(0, len(frames_iter), speed_up_interval)
        smpl_poses, smpl_betas, pred_cams, bboxes_xyxy = speed_up_interpolate(
            selected_frames, speed_up_frames, smpl_poses, smpl_betas,
            pred_cams, bboxes_xyxy)

    # smooth
    if args.smooth_type is not None:
        smpl_poses = smooth_process(
            smpl_poses.reshape(frame_num, 24, 9),
            smooth_type=args.smooth_type).reshape(frame_num, 24, 3, 3)
        verts = smooth_process(verts, smooth_type=args.smooth_type)
        pred_cams = smooth_process(
            pred_cams[:, np.newaxis],
            smooth_type=args.smooth_type).reshape(frame_num, 3)

    if smpl_poses.shape[1:] == (24, 3, 3):
        smpl_poses = rotmat_to_aa(smpl_poses)
    elif smpl_poses.shape[1:] == (24, 3):
        smpl_poses = smpl_pose
    else:
        raise Exception(f'Wrong shape of `smpl_pose`: {smpl_pose.shape}')

    if args.output is not None:
        body_pose_, global_orient_, smpl_betas_, verts_, pred_cams_, \
            bboxes_xyxy_, image_path_, person_id_, frame_id_ = \
            [], [], [], [], [], [], [], [], []
        human_data = HumanData()
        frames_folder = osp.join(args.output, 'images')
        os.makedirs(frames_folder, exist_ok=True)
        array_to_images(
            np.array(frames_iter)[frame_id_list], output_folder=frames_folder)

        for i, img_i in enumerate(sorted(os.listdir(frames_folder))):
            body_pose_.append(smpl_poses[i][1:])
            global_orient_.append(smpl_poses[i][:1])
            smpl_betas_.append(smpl_betas[i])
            verts_.append(verts[i])
            pred_cams_.append(pred_cams[i])
            bboxes_xyxy_.append(bboxes_xyxy[i])
            image_path_.append(os.path.join('images', img_i))
            person_id_.append(0)
            frame_id_.append(frame_id_list[i])

        smpl = {}
        smpl['body_pose'] = np.array(body_pose_).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(global_orient_).reshape((-1, 3))
        smpl['betas'] = np.array(smpl_betas_).reshape((-1, 10))
        human_data['smpl'] = smpl
        human_data['verts'] = verts_
        human_data['pred_cams'] = pred_cams_
        human_data['bboxes_xyxy'] = bboxes_xyxy_
        human_data['image_path'] = image_path_
        human_data['person_id'] = person_id_
        human_data['frame_id'] = frame_id_
        human_data.dump(osp.join(args.output, 'inference_result.npz'))

    if args.show_path is not None:
        if args.output is not None:
            frames_folder = os.path.join(args.output, 'images')
        else:
            frames_folder = osp.join(Path(args.show_path).parent, 'images')
            os.makedirs(frames_folder, exist_ok=True)
            array_to_images(
                np.array(frames_iter)[frame_id_list],
                output_folder=frames_folder)

        body_model_config = dict(model_path=args.body_model_dir, type='smpl')
        visualize_smpl_hmr(
            poses=smpl_poses.reshape(-1, 24 * 3),
            betas=smpl_betas,
            cam_transl=pred_cams,
            bbox=bboxes_xyxy,
            output_path=args.show_path,
            render_choice=args.render_choice,
            resolution=frames_iter[0].shape[:2],
            origin_frames=frames_folder,
            body_model_config=body_model_config,
            overwrite=True,
            palette=args.palette,
            read_frames_batch=True)
        if args.output is None:
            shutil.rmtree(frames_folder)


def multi_person_with_mmtracking(args, frames_iter):
    """Estimate smpl parameters from multi-person
        images with mmtracking
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames
    """
    mesh_model, extractor = \
        init_model(args.mesh_reg_config, args.mesh_reg_checkpoint,
                   device=args.device.lower())

    max_track_id, max_instance, frame_id_list, result_list = \
        get_tracking_result(args, frames_iter, mesh_model, extractor)

    frame_num = len(frame_id_list)
    verts = np.zeros([frame_num, max_track_id + 1, 6890, 3])
    pred_cams = np.zeros([frame_num, max_track_id + 1, 3])
    bboxes_xyxy = np.zeros([frame_num, max_track_id + 1, 5])
    smpl_poses = np.zeros([frame_num, max_track_id + 1, 24, 3, 3])
    smpl_betas = np.zeros([frame_num, max_track_id + 1, 10])

    # speed up
    if args.speed_up_type:
        speed_up_interval = get_speed_up_interval(args.speed_up_type)
        speed_up_frames = (frame_num -
                           1) // speed_up_interval * speed_up_interval

    track_ids_lists = []
    for i, result in enumerate(mmcv.track_iter_progress(result_list)):
        frame_id = frame_id_list[i]
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator':
            if args.speed_up_type:
                warnings.warn(
                    'Video based models do not support speed up. '
                    'By default we will inference with original speed.',
                    UserWarning)
            feature_results_seq = extract_feature_sequence(
                result_list, frame_idx=i, causal=True, seq_len=16, step=1)

            mesh_results = inference_video_based_model(
                mesh_model,
                extracted_results=feature_results_seq,
                with_track_id=True)
        elif mesh_model.cfg.model.type == 'ImageBodyModelEstimator':
            if args.speed_up_type and i % speed_up_interval != 0\
                 and i <= speed_up_frames:
                mesh_results = []
                for idx in range(len(result)):
                    mesh_result = result[idx].copy()
                    mesh_result['bbox'] = np.zeros((5))
                    mesh_result['camera'] = np.zeros((3))
                    mesh_result['smpl_pose'] = np.zeros((24, 3, 3))
                    mesh_result['smpl_beta'] = np.zeros((10))
                    mesh_result['vertices'] = np.zeros((6890, 3))
                    mesh_result['keypoints_3d'] = np.zeros((17, 3))
                    mesh_results.append(mesh_result)

            else:
                mesh_results = inference_image_based_model(
                    mesh_model,
                    frames_iter[frame_id],
                    result,
                    bbox_thr=args.bbox_thr,
                    format='xyxy')
        else:
            raise Exception(
                f'{mesh_model.cfg.model.type} is not supported yet')

        track_ids = []
        for mesh_result in mesh_results:
            instance_id = mesh_result['track_id']
            bboxes_xyxy[i, instance_id] = mesh_result['bbox']
            pred_cams[i, instance_id] = mesh_result['camera']
            verts[i, instance_id] = mesh_result['vertices']
            smpl_betas[i, instance_id] = mesh_result['smpl_beta']
            smpl_poses[i, instance_id] = mesh_result['smpl_pose']
            track_ids.append(instance_id)
        track_ids_lists.append(track_ids)

    # release GPU memory
    del mesh_model
    del extractor
    torch.cuda.empty_cache()

    # speed up
    if args.speed_up_type:
        smpl_poses = speed_up_process(
            torch.tensor(smpl_poses).to(args.device.lower()),
            args.speed_up_type)

        selected_frames = np.arange(0, len(frames_iter), speed_up_interval)
        smpl_poses, smpl_betas, pred_cams, bboxes_xyxy = speed_up_interpolate(
            selected_frames, speed_up_frames, smpl_poses, smpl_betas,
            pred_cams, bboxes_xyxy)

    # smooth
    if args.smooth_type is not None:
        smpl_poses = smooth_process(
            smpl_poses.reshape(frame_num, -1, 24, 9),
            smooth_type=args.smooth_type).reshape(frame_num, -1, 24, 3, 3)
        verts = smooth_process(verts, smooth_type=args.smooth_type)
        pred_cams = smooth_process(
            pred_cams[:, np.newaxis],
            smooth_type=args.smooth_type).reshape(frame_num, -1, 3)

    if smpl_poses.shape[2:] == (24, 3, 3):
        smpl_poses = rotmat_to_aa(smpl_poses)
    elif smpl_poses.shape[2:] == (24, 3):
        smpl_poses = smpl_poses
    else:
        raise Exception(f'Wrong shape of `smpl_pose`: {smpl_poses.shape}')

    if args.output is not None:
        body_pose_, global_orient_, smpl_betas_, verts_, pred_cams_, \
            bboxes_xyxy_, image_path_, frame_id_, person_id_ = \
            [], [], [], [], [], [], [], [], []
        human_data = HumanData()
        frames_folder = osp.join(args.output, 'images')
        os.makedirs(frames_folder, exist_ok=True)
        array_to_images(
            np.array(frames_iter)[frame_id_list], output_folder=frames_folder)

        for i, img_i in enumerate(sorted(os.listdir(frames_folder))):
            for person_i in track_ids_lists[i]:
                body_pose_.append(smpl_poses[i][person_i][1:])
                global_orient_.append(smpl_poses[i][person_i][:1])
                smpl_betas_.append(smpl_betas[i][person_i])
                verts_.append(verts[i][person_i])
                pred_cams_.append(pred_cams[i][person_i])
                bboxes_xyxy_.append(bboxes_xyxy[i][person_i])
                image_path_.append(os.path.join('images', img_i))
                person_id_.append(person_i)
                frame_id_.append(frame_id_list[i])

        smpl = {}
        smpl['body_pose'] = np.array(body_pose_).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(global_orient_).reshape((-1, 3))
        smpl['betas'] = np.array(smpl_betas_).reshape((-1, 10))
        human_data['smpl'] = smpl
        human_data['verts'] = verts_
        human_data['pred_cams'] = pred_cams_
        human_data['bboxes_xyxy'] = bboxes_xyxy_
        human_data['image_path'] = image_path_
        human_data['person_id'] = person_id_
        human_data['frame_id'] = frame_id_
        human_data.dump(osp.join(args.output, 'inference_result.npz'))

    # To compress vertices array
    compressed_verts = np.zeros([frame_num, max_instance, 6890, 3])
    compressed_cams = np.zeros([frame_num, max_instance, 3])
    compressed_bboxs = np.zeros([frame_num, max_instance, 5])
    compressed_poses = np.zeros([frame_num, max_instance, 24, 3])
    compressed_betas = np.zeros([frame_num, max_instance, 10])

    for i, track_ids_list in enumerate(track_ids_lists):
        instance_num = len(track_ids_list)
        compressed_verts[i, :instance_num] = verts[i, track_ids_list]
        compressed_cams[i, :instance_num] = pred_cams[i, track_ids_list]
        compressed_bboxs[i, :instance_num] = bboxes_xyxy[i, track_ids_list]
        compressed_poses[i, :instance_num] = smpl_poses[i, track_ids_list]
        compressed_betas[i, :instance_num] = smpl_betas[i, track_ids_list]

    assert len(frame_id_list) > 0

    if args.show_path is not None:
        if args.output is not None:
            frames_folder = os.path.join(args.output, 'images')
        else:
            frames_folder = osp.join(Path(args.show_path).parent, 'images')
            os.makedirs(frames_folder, exist_ok=True)
            array_to_images(
                np.array(frames_iter)[frame_id_list],
                output_folder=frames_folder)
        body_model_config = dict(model_path=args.body_model_dir, type='smpl')
        visualize_smpl_hmr(
            poses=compressed_poses.reshape(-1, max_instance, 24 * 3),
            betas=compressed_betas,
            cam_transl=compressed_cams,
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
        '--smooth_type',
        type=str,
        default=None,
        help='Smooth the data through the specified type.'
        'Select in [oneeuro,gaus1d,savgol].')
    parser.add_argument(
        '--speed_up_type',
        type=str,
        default=None,
        help='Speed up data processing through the specified type.'
        'Select in [deciwatch].')
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
