from argparse import ArgumentParser

import mmcv
import numpy as np

from mmhuman3d.apis import (
    feature_extract,
    inference_video_based_model,
    init_model,
)
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization import visualize_smpl_vibe
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.utils.demo_utils import (
    extract_feature_sequence,
    prepare_frames,
    process_mmdet_results,
    process_mmtracking_results,
    smooth_process,
    xyxy2xywh,
)
from mmhuman3d.utils.path_utils import prepare_output_path
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

    mesh_model, extrator = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())

    # Used to save the img index
    img_index = []
    person_results_list = []
    pred_cams, verts, poses, betas, kp3d, bboxes_xyxy = \
        [], [], [], [], [], []

    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, frame)

        # keep the person class bounding boxes.
        person_det_results = \
            process_mmdet_results(
                mmdet_results,
                args.det_cat_id)

        # extract features from the input video or image sequences
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator' \
                and extrator is not None:
            person_results = feature_extract(
                extrator,
                frame,
                person_det_results,
                args.bbox_thr,
                format='xyxy')

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
        smpl_pose = rotmat_to_aa(mesh_results[0]['smpl_pose'])
        poses.append(smpl_pose)
        betas.append(mesh_results[0]['smpl_beta'])
        kp3d.append(mesh_results[0]['keypoints_3d'])
        bboxes_xyxy.append(det_result['bbox'])

    pred_cams = np.array(pred_cams)
    verts = np.array(verts)
    poses = np.array(poses)
    betas = np.array(betas)
    kp3d = np.array(kp3d)
    bboxes_xyxy = np.array(bboxes_xyxy)

    if args.output is not None:
        prepare_output_path(
            output_path=args.output,
            path_type='file',
            allowed_suffix=['.npz'],
            overwrite=True)

        smpl = {}
        smpl['body_pose'] = poses[..., 1:, :]
        smpl['global_orient'] = poses[..., 0, :]
        smpl['betas'] = betas

        bboxes_xywh = xyxy2xywh(bboxes_xyxy)
        conf = np.ones(kp3d.shape[:-1])[..., None]
        kp3d = np.concatenate([kp3d, conf], axis=-1)
        keypoints3d_, keypoints3d_mask = \
            convert_kps(kp3d, 'h36m', 'human_data')

        human_data = HumanData()
        human_data['bbox_xywh'] = bboxes_xywh
        human_data['keypoints3d'] = keypoints3d_
        human_data['keypoints3d_mask'] = keypoints3d_mask
        human_data['smpl'] = smpl
        human_data.compress_keypoints_by_mask()

        human_data.dump(args.output)

    # smooth
    if args.smooth_type is not None:
        verts = smooth_process(verts, smooth_type=args.smooth_type)

    if args.show_path is not None:
        prepare_output_path(
            output_path=args.show_path,
            path_type='file',
            allowed_suffix=['.mp4'],
            overwrite=True)

        visualize_smpl_vibe(
            verts=verts,
            pred_cam=pred_cams,
            bbox=bboxes_xyxy,
            model_path=args.body_model_dir,
            model_type='smpl',
            output_path=args.show_path,
            render_choice=args.render_choice,
            resolution=frames_iter[0].shape[:2],
            image_array=np.array(frames_iter)[img_index],
            overwrite=True,
            palette=args.palette)


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

    mesh_model, extrator = \
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
                and extrator is not None:
            person_results = feature_extract(
                extrator, frame, person_results, args.bbox_thr, format='xyxy')

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
            labels = [
                'track_id_' + str(res['track_id']) for res in person_results
            ]
            labels = np.array(labels)
            mmcv.imshow_det_bboxes(frame, bboxes, labels, show=False)

        person_results_list.append(person_results)
        img_index.append(i)
        frame_num += 1

    poses = np.zeros([frame_num, max_track_id + 1, 24, 3])
    betas = np.zeros([frame_num, max_track_id + 1, 10])
    kp3d = np.zeros([frame_num, max_track_id + 1, 17, 3])
    verts = np.zeros([frame_num, max_track_id + 1, 6890, 3])
    pred_cams = np.zeros([frame_num, max_track_id + 1, 3])
    bboxes_xyxy = np.zeros([frame_num, max_track_id + 1, 5])
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
            smpl_pose = rotmat_to_aa(mesh_result['smpl_pose'])
            poses[i, instance_id] = smpl_pose
            betas[i, instance_id] = mesh_result['smpl_beta']
            kp3d[i, instance_id] = mesh_result['keypoints_3d']
            track_ids.append(instance_id)

        track_ids_lists.append(track_ids)

    if args.output is not None:
        prepare_output_path(
            output_path=args.output,
            path_type='file',
            allowed_suffix=['.npz'],
            overwrite=True)
        smpl = {}
        smpl['body_pose'] = poses[..., 1:, :]
        smpl['global_orient'] = poses[..., 0, :]
        smpl['betas'] = betas

        bboxes_xywh = xyxy2xywh(bboxes_xyxy)
        conf = np.ones(kp3d.shape[:-1])[..., None]
        kp3d = np.concatenate([kp3d, conf], axis=-1)
        keypoints3d_, keypoints3d_mask = \
            convert_kps(kp3d, 'h36m', 'human_data')

        human_data = HumanData()
        human_data['bbox_xywh'] = bboxes_xywh
        human_data['keypoints3d'] = keypoints3d_
        human_data['keypoints3d_mask'] = keypoints3d_mask
        human_data['smpl'] = smpl
        human_data.compress_keypoints_by_mask()

        human_data.dump(args.output)

    # smooth
    if args.smooth_type is not None:
        verts = smooth_process(verts, smooth_type=args.smooth_type)

    # To compress vertices array
    V = np.zeros([frame_num, max_instance, 6890, 3])
    C = np.zeros([frame_num, max_instance, 3])
    B = np.zeros([frame_num, max_instance, 5])
    for i, track_ids_list in enumerate(track_ids_lists):
        instance_num = len(track_ids_list)
        V[i, :instance_num] = verts[i, track_ids_list]
        C[i, :instance_num] = pred_cams[i, track_ids_list]
        B[i, :instance_num] = bboxes_xyxy[i, track_ids_list]
    assert len(img_index) > 0

    if args.show_path is not None:
        prepare_output_path(
            output_path=args.show_path,
            path_type='file',
            allowed_suffix=['.mp4'],
            overwrite=True)

        visualize_smpl_vibe(
            verts=verts,
            pred_cam=pred_cams,
            bbox=bboxes_xyxy,
            model_path=args.body_model_dir,
            model_type='smpl',
            output_path=args.show_path,
            render_choice=args.render_choice,
            resolution=frames_iter[0].shape[:2],
            image_array=np.array(frames_iter)[img_index],
            overwrite=True,
            palette=args.palette)


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
