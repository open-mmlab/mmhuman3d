# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
import smplx
import torch
from mmpose.apis import (
    extract_pose_sequence,
    get_track_id,
    inference_pose_lifter_model,
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
)
from mmpose.datasets.dataset_info import DatasetInfo

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.filter import Gaus1dFilter, OneEuroFilter, SGFilter
from mmhuman3d.core.parametric_model import build_registrant
from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d
from mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.utils.misc import torch_to_numpy
from mmhuman3d.utils.path_utils import check_input_path, prepare_output_path

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def smooth_keypoints3d(kp3d, smooth_type='savgol'):
    """Smooth the detected 3D keypoints with the specified smoothing type.

    Args:
        kp3d (np.ndarray): shape should be (f * J * 3)
        smooth_type (str, optional): smooth type.
            choose in ['oneeuro', 'gaus1d', 'savgol'].
            Defaults to 'savgol'.
    Raises:
        ValueError: check the input smoothing type.

    Returns:
        np.ndarray: smoothed 3D keypoints (f * J * 3).
    """
    kp3d = kp3d.copy()
    if smooth_type == 'oneeuro':
        smooth_kp = OneEuroFilter()
    elif smooth_type == 'gaus1d':
        smooth_kp = Gaus1dFilter()
    elif smooth_type == 'savgol':
        smooth_kp = SGFilter()
    else:
        raise ValueError('Please input valid smooth type.')

    kp3d_smooth = smooth_kp(kp3d)
    return kp3d_smooth


def get_keypoints(args):
    """To detect human keypoints from the image or the video.

    Args:
        args (argparse.Namespace):  object of argparse.Namespace.

    Raises:
        ValueError: check the input path.

    Returns:
        np.ndarray: The bbox: (left, top, width, height, [score])
        np.ndarray: The detected 2D keypoints (f * J * 2).
        np.ndarray: The detected 3D keypoints (f * J * 3).
    """

    if (args.image_path is not None) and (args.video_path is not None):
        warnings.warn('Redundant input, will ignore video')
    # prepare input
    if args.image_path is not None:
        file_list = []
        if Path(args.image_path).is_file():
            file_list = [args.image_path]
        elif Path(args.image_path).is_dir():
            file_list = [
                os.path.join(args.image_path, fn)
                for fn in os.listdir(args.image_path)
                if fn.lower().endswith(('.png', '.jpg'))
            ]
        else:
            raise ValueError('Image path should be an image or image folder.'
                             f' Got invalid image path: {args.image_path}')
        file_list.sort()
        img_list = [mmcv.imread(img_path) for img_path in file_list]
        assert len(img_list), f'Failed to load image from {args.image_path}'
    elif args.video_path is not None:
        check_input_path(
            input_path=args.video_path,
            path_type='file',
            allowed_suffix=['.mp4'])
        video = mmcv.VideoReader(args.video_path)
        assert video.opened, f'Failed to load video file {args.video_path}'
    else:
        raise ValueError('No image path or video path provided.')
    # First stage: 2D pose detection
    print('Stage 1: 2D pose detection.')
    # Build the person detector model from a config file and a checkpoint file.
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # Build the 2d pose model from a config file and a checkpoint file.
    pose_det_model = init_pose_model(
        args.pose_detector_config,
        args.pose_detector_checkpoint,
        device=args.device.lower())

    assert pose_det_model.cfg.model.type == 'TopDown', 'Only "TopDown"' \
        ' model is supported for the 1st stage (2D pose detection)'

    pose_det_dataset = pose_det_model.cfg.data['test']['type']
    pose_det_dataset_info = pose_det_model.cfg.data['test'].get(
        'dataset_info', None)

    if pose_det_dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config. Check'
            ' https://github.com/open-mmlab/mmpose/pull/663'
            ' for details.', DeprecationWarning)
    else:
        pose_det_dataset_info = DatasetInfo(pose_det_dataset_info)

    pose_det_results_list = []
    next_id = 0
    pose_det_results = []
    frames_iter = img_list if args.image_path is not None else video
    for frame in frames_iter:
        pose_det_results_last = pose_det_results

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, frame)

        # keep the person class bounding boxes.
        person_det_results = process_mmdet_results(mmdet_results,
                                                   args.det_cat_id)

        # make person results for single image
        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_model,
            frame,
            person_det_results,
            bbox_thr=args.bbox_thr,
            format='xywh',
            dataset=pose_det_dataset,
            dataset_info=pose_det_dataset_info,
            return_heatmap=False,
            outputs=None)

        # get track id for each person instance
        pose_det_results, next_id = get_track_id(
            pose_det_results,
            pose_det_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr,
            use_one_euro=args.euro)

        pose_det_results_list.append(copy.deepcopy(pose_det_results))

    # Second stage: Pose lifting
    print('Stage 2: 2D-to-3D pose lifting.')

    # Build the pose lifting model from a config file and a checkpoint file.
    pose_lift_model = init_pose_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert pose_lift_model.cfg.model.type == 'PoseLifter', \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'

    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']
    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get(
        'dataset_info', None)

    if pose_lift_dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config. Check '
            'https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)

    # convert keypoint definition
    for pose_det_results in pose_det_results_list:
        for res in pose_det_results:
            if args.keypoint2d_type != args.keypoint3d_type:
                keypoints = res['keypoints'][None]
                res['keypoints'] = convert_kps(
                    keypoints, args.keypoint2d_type,
                    args.keypoint3d_type)[0].squeeze()
    bbox_xywh = []
    kp2d = []
    kp3d = []
    for i, pose_det_results in enumerate(
            mmcv.track_iter_progress(pose_det_results_list)):
        if args.input_type == 'video':
            # load temporal padding config from model.data_cfg
            if hasattr(pose_lift_model.cfg, 'test_data_cfg'):
                data_cfg = pose_lift_model.cfg.test_data_cfg
            else:
                data_cfg = pose_lift_model.cfg.data_cfg

            # the loaded pose lifter model was trained from video
            pose_results_2d = extract_pose_sequence(
                pose_det_results_list,
                frame_idx=i,
                causal=data_cfg.causal,
                seq_len=data_cfg.seq_len,
                step=data_cfg.seq_frame_interval)
        elif args.input_type == 'image':
            pose_results_2d = [pose_det_results]

        # 2D-to-3D pose lifting
        pose_lift_results = inference_pose_lifter_model(
            pose_lift_model,
            pose_results_2d=pose_results_2d,
            dataset=pose_lift_dataset,
            dataset_info=pose_lift_dataset_info,
            with_track_id=True,
            image_size=frame.shape[:2],
            norm_pose_2d=args.norm_pose_2d)

        # Pose processing
        # pose_lift_results: [person_index,{track_id,kp2d,kp3d,title,bbox}]
        for idx, res in enumerate(pose_lift_results):
            keypoints_3d = res['keypoints_3d'][..., :3]
            det_res = pose_det_results[idx]
            keypoints_2d = det_res['keypoints']
            bbox = det_res['bbox']
        bbox_xywh.append(bbox)
        kp2d.append(keypoints_2d)
        kp3d.append(keypoints_3d)

    bbox_xywh = np.array(bbox_xywh)
    kp2d = np.array(kp2d)
    kp3d = np.array(kp3d)

    # smooth
    if args.smooth_type is not None:
        print('smooth 3D keypoints ...')
        kp3d = smooth_keypoints3d(kp3d, smooth_type=args.smooth_type)

    # visualize kp2d
    if args.show_kp2d_path is not None:
        # check show path
        prepare_output_path(
            output_path=args.show_kp2d_path,
            path_type='auto',
            allowed_suffix=['.mp4', ''],
            overwrite=True)
        # visualize 2D keypoints
        print('visualize 2D keypoints ...')
        origin_frames = Path(file_list[0]).parent \
            if args.image_path else args.video_path
        visualize_kp2d(
            kp2d,
            output_path=args.show_kp2d_path,
            data_source=args.keypoint2d_type,
            origin_frames=origin_frames,
            overwrite=True)

    # visualize kp3d
    if args.show_kp3d_path is not None:
        # check show path
        prepare_output_path(
            output_path=args.show_kp3d_path,
            path_type='auto',
            allowed_suffix=['.mp4', ''],
            overwrite=True)
        # visualize 3D keypoints
        print('visualize 3D keypoints ...')
        visualize_kp3d(
            kp3d,
            output_path=args.show_kp3d_path,
            data_source=args.keypoint3d_type,
            value_range=None)

    # save keypoints
    if args.save_kp_path is not None:
        # check save path
        prepare_output_path(
            output_path=args.save_kp_path,
            path_type='file',
            allowed_suffix=['.npz', '.npy', '.pickle'],
            overwrite=True)
        print('save keypoints ...')
        np.savez(args.save_kp_path, kp3d=kp3d, kp2d=kp2d)

    # # HumanData
    # # convert kp2d
    # keypoints2d,keypoints2d_mask=convert_kps(kp2d,'h36m','smplx')
    # # convert kp3d
    # keypoints3d,keypoints3d_mask=convert_kps(kp3d,'h36m','smplx')
    # data_dict = {
    #     'bbox_xywh': bbox_xywh,
    #     'keypoints2d':keypoints2d,
    #     'keypoints2d_mask':keypoints2d_mask,
    #     'keypoints3d':keypoints3d,
    #     'keypoints3d_mask':keypoints3d_mask
    #     }

    # human_data = HumanData().new(source_dict=data_dict)

    return bbox_xywh, kp2d, kp3d


def smplify_proc(args, keypoints2d=None, keypoints3d=None):
    """To regress SMPL/SMPL-X parameter from the input 2D or 3D keypoint.

    Args:
        args (argparse.Namespace):  object of argparse.Namespace.
        kp2d (np.ndarray, optional):
            shape should be (f * J * 2). Defaults to None.
        kp3d (np.ndarray, optional):
            shape should be (f * J * 3). Defaults to None.

    Returns:
        None.

    Raises:
        ValueError: check the value of keypoints2d and keypoints3d.
    """
    cfg = mmcv.Config.fromfile(args.config)
    assert cfg.body_model_type in ['smpl', 'smplx']
    assert cfg.smplify_method in ['smplify', 'smplifyx']

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    if keypoints2d is not None:
        keypoints_src = keypoints2d.copy()
        assert keypoints2d.shape[-1] == 2
    elif keypoints3d is not None:
        keypoints_src = keypoints3d.copy()
        assert keypoints3d.shape[-1] == 3
    else:
        raise ValueError('Only support keypoints2d and keypoints3d')
    # convert keypoints
    keypoints, mask = convert_kps(
        keypoints_src, src=args.keypoint3d_type, dst=cfg.body_model_type)
    keypoints_conf = mask

    batch_size = args.batch_size if args.batch_size else keypoints.shape[0]

    keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
    keypoints_conf = torch.tensor(
        keypoints_conf, dtype=torch.float32, device=device)

    human_data = dict(keypoints3d=keypoints, keypoints3d_conf=keypoints_conf)

    # create body model
    cfg_body_model = dict(
        model_path=args.body_model_dir,
        model_type=cfg.body_model_type,
        gender=args.gender,
        num_betas=args.num_betas,
        batch_size=batch_size,
    )
    if cfg.body_model_type == 'smplx':
        cfg_body_model.update(
            dict(
                use_face_contour=True,  # 127 -> 144
                use_pca=False,  # current vis do not supports use_pca
            ))
    body_model = smplx.create(**cfg_body_model)

    cfg.update(
        dict(
            type=cfg.smplify_method,
            body_model=body_model,
            use_one_betas_per_video=args.use_one_betas_per_video,
            num_epochs=args.num_epochs))

    # run SMPLify(X)
    smplify = build_registrant(dict(cfg))
    print('run SMPLify...')
    smplify_output = smplify(**human_data)

    # TODO: read keypoints3d directly from smplify_output
    if cfg.body_model_type == 'smpl':
        body_model_output = body_model(
            global_orient=smplify_output['global_orient'],
            transl=smplify_output['transl'],
            body_pose=smplify_output['body_pose'],
            betas=smplify_output['betas'])
    else:
        body_model_output = body_model(
            global_orient=smplify_output['global_orient'],
            transl=smplify_output['transl'],
            body_pose=smplify_output['body_pose'],
            betas=smplify_output['betas'],
            left_hand_pose=smplify_output['left_hand_pose'],
            right_hand_pose=smplify_output['right_hand_pose'],
            expression=smplify_output['expression'],
            jaw_pose=smplify_output['jaw_pose'],
            leye_pose=smplify_output['leye_pose'],
            reye_pose=smplify_output['reye_pose'])

    body_model_keypoints3d = torch_to_numpy(body_model_output.joints)

    results = {k: torch_to_numpy(v) for k, v in smplify_output.items()}
    results.update({'keypoints3d': body_model_keypoints3d})

    # save results
    if args.save_mesh_result is not None:
        print(f'writing results to {args.save_mesh_result}')
        mmcv.dump(results, args.save_mesh_result)

    # save rendered results
    if args.show_mesh_path is not None:
        # check show path
        prepare_output_path(
            output_path=args.show_mesh_path,
            allowed_suffix=['.mp4', ''],
            path_type='auto',
            overwrite=True)
        # TODO: use results directly after !42 is merged
        poses = {k: v.detach().cpu() for k, v in smplify_output.items()}
        # visualize mesh
        print('visualize mesh...')
        visualize_smpl_pose(
            poses=poses,
            body_model_dir=args.body_model_dir,
            output_path=args.show_mesh_path,
            model_type=cfg.body_model_type,
            overwrite=True)


def main(args):
    # TODO: support 2D keypoints
    _, _, kp3d = get_keypoints(args)
    # smplify
    smplify_proc(args, keypoints3d=kp3d)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose_detector_config',
        type=str,
        default=None,
        help='Config file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_detector_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_lifter_config',
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        'pose_lifter_checkpoint',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--video_path', type=str, default=None, help='Video path')
    parser.add_argument(
        '--image_path', type=str, default=None, help='Image path')
    parser.add_argument(
        '--input_type',
        type=str,
        default='video',
        help='The input type of the pose lifter model.\
        Select in [video,image]')
    parser.add_argument(
        '--norm_pose_2d',
        action='store_true',
        help='Scale the bbox (along with the 2D pose) to the average bbox '
        'scale of the dataset, and move the bbox (along with the 2D pose) to '
        'the average bbox center of the dataset. This is useful when bbox '
        'is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--det_cat_id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.9, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--save_kp_path',
        type=str,
        default=None,
        help='Save 3D keypoints path')
    parser.add_argument(
        '--show_kp3d_path',
        type=str,
        default=None,
        help='Directory to save 3D keypoints video')
    parser.add_argument(
        '--show_kp2d_path',
        type=str,
        default=None,
        help='Directory to save 2D keypoints video')
    parser.add_argument(
        '--keypoint2d_type',
        type=str,
        default='coco',
        help='The source type of 2D keypoints')
    parser.add_argument(
        '--keypoint3d_type',
        type=str,
        default='h36m',
        help='The source type of 3D keypoints')
    # smooth config
    parser.add_argument(
        '--smooth_type',
        type=str,
        default=None,
        help='The type of smoothing 3D keypoints.\
         Select in [oneeuro,gaus1d,savgol].')
    # smplify configs
    parser.add_argument('--config', help='Config file for the SMPLify stage')
    parser.add_argument('--body_model_dir', help='Body models file path')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_betas', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument(
        '--use_one_betas_per_video',
        action='store_true',
        help='use one betas to keep shape consistent through a video')
    parser.add_argument(
        '--gender',
        choices=['neutral', 'male', 'female'],
        default='neutral',
        help='gender of SMPL model')
    parser.add_argument(
        '--save_mesh_result',
        type=str,
        default=None,
        help='output result file')
    parser.add_argument(
        '--show_mesh_path',
        type=str,
        default=None,
        help='directory to save rendered images or video')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')

    args = parser.parse_args()
    assert has_mmdet, 'Please install mmdet to run the demo.'
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    main(args)
