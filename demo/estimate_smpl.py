from argparse import ArgumentParser

import mmcv
import numpy as np

from mmhuman3d.apis import inference_model, init_model
from mmhuman3d.core.visualization.visualize_smpl import render_smpl
from mmhuman3d.utils.demo_utils import (
    conver_verts_to_cam_coord,
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

    Returns:
        (np.ndarray): estimated vertices.
        (np.ndarray): intrinsic parameters.
        (list): index of the prepared frames.
    """
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    pose_model = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())

    # Used to save the img index
    img_index = []
    pred_cams, verts, bboxes_xy = [], [], []
    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, frame)

        # keep the person class bounding boxes.
        person_det_results = \
            process_mmdet_results(
                mmdet_results,
                args.det_cat_id)

        # test a single image, with a list of bboxes.
        mesh_results = inference_model(
            pose_model,
            frame,
            person_det_results,
            bbox_thr=args.bbox_thr,
            format='xyxy')

        # drop the frame with no detected results
        if mesh_results == []:
            continue

        img_index.append(i)

        pred_cams.append(mesh_results[0]['camera'])
        verts.append(mesh_results[0]['vertices'])
        bboxes_xy.append(mesh_results[0]['bbox'])

    pred_cams = np.array(pred_cams)
    verts = np.array(verts)
    bboxes_xy = np.array(bboxes_xy)

    # convert vertices from world to camera
    verts, K0 = conver_verts_to_cam_coord(
        verts, pred_cams, bboxes_xy, focal_length=args.focal_length)

    # smooth
    if args.smooth_type is not None:
        verts = smooth_process(verts, smooth_type=args.smooth_type)

    return verts, K0, img_index


def multi_person_with_mmtracking(args, frames_iter):
    """Estimate smpl parameters from multi-person
        images with mmtracking
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames
    Returns:
        (np.ndarray): estimated vertices.
        (np.ndarray): intrinsic parameters.
        (list): index of the prepared frames.
    """
    tracking_model = init_tracking_model(
        args.tracking_config, None, device=args.device.lower())

    mesh_model = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())

    # The total number of people detected in a video or image sequence
    det_instances = 0
    # Maximum number of people appearing in the same frame
    max_instances = 0
    # Used to save inference results
    mesh_results_list = []
    # Used to save the value of the total number of frames
    frame_num = 0
    # Used to save the img index
    img_index = []
    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):

        mmtracking_results = inference_mot(tracking_model, frame, frame_id=i)

        # keep the person class bounding boxes.
        person_results = process_mmtracking_results(mmtracking_results)

        # test a single image, with a list of bboxes.
        mesh_results = inference_model(
            mesh_model,
            frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy')

        # drop the frame with no detected results
        if mesh_results == []:
            continue

        for mesh_result in mesh_results:
            track_id = mesh_result['track_id']
            if track_id > det_instances:
                det_instances = track_id
        mesh_results_list.append(mesh_results)
        img_index.append(i)
        frame_num += 1

    verts = np.zeros([frame_num, det_instances + 1, 6890, 3])
    pred_cams = np.zeros([frame_num, det_instances + 1, 3])
    bboxes_xy = np.zeros([frame_num, det_instances + 1, 5])

    track_ids_lists = []
    for i, mesh_results in enumerate(
            mmcv.track_iter_progress(mesh_results_list)):
        track_ids = []
        for mesh_result in mesh_results:
            instance_id = mesh_result['track_id']
            pred_cams[i, instance_id] = mesh_result['camera']
            verts[i, instance_id] = mesh_result['vertices']
            bboxes_xy[i, instance_id] = mesh_result['bbox']
            track_ids.append(instance_id)
        # max instances
        max_instances = max(max_instances, len(track_ids))
        track_ids_lists.append(track_ids)

    verts, K0 = conver_verts_to_cam_coord(
        verts, pred_cams, bboxes_xy, focal_length=args.focal_length)

    # smooth
    if args.smooth_type is not None:
        verts = smooth_process(verts, smooth_type=args.smooth_type)

    # To compress vertices array
    V = np.zeros([frame_num, max_instances, 6890, 3])
    for i, track_ids_list in enumerate(track_ids_lists):
        instance_num = len(track_ids_list)
        V[i, :instance_num] = verts[i, track_ids_list]

    return V, K0, img_index


def main(args):

    # prepare input
    frames_iter = prepare_frames(args.image_path, args.video_path)

    if args.single_person_demo:
        verts, K0, img_index = single_person_with_mmdet(args, frames_iter)
    elif args.multi_person_demo:
        verts, K0, img_index = multi_person_with_mmtracking(args, frames_iter)
    else:
        raise ValueError(
            'Only supports single_person_demo or multi_person_demo')

    # Visualization
    render_smpl(
        verts=verts,
        model_path=args.body_model_dir,
        model_type='smpl',
        img_format=None,
        K=K0,
        output_path=args.output_path,
        render_choice=args.render_choice,
        resolution=frames_iter[0].shape[:2],
        image_array=np.array(frames_iter)[img_index],
        overwrite=True,
        in_ndc=False,
        convention='opencv',
        projection='perspective',
        no_grad=True,
        return_tensor=False,
        palette=args.palette)


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
        '--video_path', type=str, default=None, help='Video path')
    parser.add_argument(
        '--image_path', type=str, default=None, help='Image path')
    parser.add_argument(
        '--output_path', type=str, default=None, help='Output path')
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
