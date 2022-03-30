import os
import os.path as osp
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
import torch

from mmhuman3d.apis import inference_image_based_model, init_model
from mmhuman3d.core.visualization import visualize_smpl_hmr
from mmhuman3d.utils import array_to_images
from mmhuman3d.utils.demo_utils import (
    prepare_frames,
    process_mmdet_results,
    process_mmtracking_results,
    smooth_process,
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

    mesh_model, _ = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())

    # Used to save the img index
    img_index = []
    pred_cams, verts, bboxes_xyxy = [], [], []
    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, frame)

        # keep the person class bounding boxes.
        person_results = \
            process_mmdet_results(
                mmdet_results,
                args.det_cat_id)

        # test a single image, with a list of bboxes.
        mesh_results = inference_image_based_model(
            mesh_model,
            frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy')

        # drop the frame with no detected results
        if mesh_results == []:
            continue
        # vis bboxes
        if args.draw_bbox:
            bboxes = person_results[0]['bbox'][None]
            mmcv.imshow_bboxes(
                frame, bboxes, top_k=-1, thickness=2, show=False)
        img_index.append(i)

        pred_cams.append(mesh_results[0]['camera'])
        verts.append(mesh_results[0]['vertices'])
        bboxes_xyxy.append(mesh_results[0]['bbox'])

    pred_cams = np.array(pred_cams)
    verts = np.array(verts)
    bboxes_xyxy = np.array(bboxes_xyxy)

    del mesh_model
    del person_det_model
    torch.cuda.empty_cache()

    # smooth
    if args.smooth_type is not None:
        verts = smooth_process(verts, smooth_type=args.smooth_type)

    if args.show_path is not None:
        frames_folder = osp.join(Path(args.show_path).parent, 'images')
        os.makedirs(frames_folder, exist_ok=True)
        array_to_images(
            np.array(frames_iter)[img_index], output_folder=frames_folder)
        body_model_config = dict(model_path=args.body_model_dir, type='smpl')
        # Visualization
        visualize_smpl_hmr(
            cam_transl=pred_cams,
            bbox=bboxes_xyxy,
            verts=verts,
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

    mesh_model, _ = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())

    max_track_id = 0
    max_instance = 0
    mesh_results_list = []
    frame_num = 0
    img_index = []

    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):

        mmtracking_results = inference_mot(tracking_model, frame, frame_id=i)

        # keep the person class bounding boxes.
        person_results, max_track_id, instance_num = \
            process_mmtracking_results(mmtracking_results, max_track_id)

        # test a single image, with a list of bboxes.
        mesh_results = inference_image_based_model(
            mesh_model,
            frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy')

        # drop the frame with no detected results
        if mesh_results == []:
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

        mesh_results_list.append(mesh_results)
        img_index.append(i)
        frame_num += 1

    verts = np.zeros([frame_num, max_track_id + 1, 6890, 3])
    pred_cams = np.zeros([frame_num, max_track_id + 1, 3])
    bboxes_xyxy = np.zeros([frame_num, max_track_id + 1, 5])

    track_ids_lists = []
    for i, mesh_results in enumerate(
            mmcv.track_iter_progress(mesh_results_list)):
        track_ids = []
        for mesh_result in mesh_results:
            instance_id = mesh_result['track_id']
            bboxes_xyxy[i, instance_id] = mesh_result['bbox']
            pred_cams[i, instance_id] = mesh_result['camera']
            verts[i, instance_id] = mesh_result['vertices']
            track_ids.append(instance_id)

        track_ids_lists.append(track_ids)

    del mesh_model
    del tracking_model
    torch.cuda.empty_cache()

    # smooth
    if args.smooth_type is not None:
        verts = smooth_process(verts, smooth_type=args.smooth_type)

    # To compress vertices array
    compressed_verts = np.zeros([frame_num, max_instance, 6890, 3])
    compressed_cams = np.zeros([frame_num, max_instance, 3])
    compressed_bboxs = np.zeros([frame_num, max_instance, 5])
    for i, track_ids_list in enumerate(
            mmcv.track_iter_progress(track_ids_lists)):
        instance_num = len(track_ids_list)
        compressed_verts[i, :instance_num] = verts[i, track_ids_list]
        compressed_cams[i, :instance_num] = pred_cams[i, track_ids_list]
        compressed_bboxs[i, :instance_num] = bboxes_xyxy[i, track_ids_list]

    if args.show_path is not None:
        frames_folder = osp.join(Path(args.show_path).parent, 'images')
        os.makedirs(frames_folder, exist_ok=True)
        array_to_images(
            np.array(frames_iter)[img_index], output_folder=frames_folder)

        body_model_config = dict(model_path=args.body_model_dir, type='smpl')

        # Visualization
        visualize_smpl_hmr(
            cam_transl=compressed_cams,
            bbox=compressed_bboxs,
            verts=compressed_verts,
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
        help='Smooth the data through the specified type.\
         Select in [oneeuro,gaus1d,savgol].')
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
