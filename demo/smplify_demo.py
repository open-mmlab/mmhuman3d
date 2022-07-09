# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from argparse import ArgumentParser

import mmcv
import numpy as np
import torch
from mmpose.apis import (
    get_track_id,
    inference_top_down_pose_model,
    init_pose_model,
    process_mmdet_results,
)
from mmpose.datasets.dataset_info import DatasetInfo

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization import visualize_kp2d, visualize_smpl_pose
from mmhuman3d.models.registrants.builder import build_registrant
from mmhuman3d.utils.demo_utils import prepare_frames
from mmhuman3d.utils.path_utils import prepare_output_path

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def covert_keypoint_definition(keypoints, pose_det_dataset, pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition.

    Args:
        keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.
    """
    if pose_det_dataset == 'TopDownH36MDataset' and \
            pose_lift_dataset == 'Body3DH36MDataset':
        return keypoints
    elif pose_det_dataset == 'TopDownCocoDataset' and \
            pose_lift_dataset == 'Body3DH36MDataset':
        keypoints_new = np.zeros((17, keypoints.shape[1]))
        # pelvis is in the middle of l_hip and r_hip
        keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
        # head is in the middle of l_eye and r_eye
        keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
        # spine is in the middle of thorax and pelvis
        keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
        # rearrange other keypoints
        keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        return keypoints_new
    else:
        raise NotImplementedError


def estimate_keypoints(args, frames_iter):
    """To detect human keypoints from the image or the video.

    Args:
        args (argparse.Namespace):  object of argparse.Namespace.
        frames_iter (list[np.ndarray,]): prepared frames.
    Raises:
        ValueError: check the input path.

    Returns:
        np.ndarray: The bbox: (left, top, width, height, [score])
        np.ndarray: The detected 2D keypoints (f * J * 2).
        np.ndarray: The detected 3D keypoints (f * J * 3).
    """

    # 2D pose detection
    # Build the person detector model from a config file and a checkpoint file.
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # Build the 2d pose model from a config file and a checkpoint file.
    pose_det_model = init_pose_model(
        args.pose_detector_config,
        args.pose_detector_checkpoint,
        device=args.device.lower())

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
            format='xyxy',
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

    bbox_xyxy = []
    keypoints2d = []
    # mask = []
    # kp3d = []
    for i, pose_det_results in enumerate(
            mmcv.track_iter_progress(pose_det_results_list)):
        # Pose processing
        # pose_lift_results: [person_index,{track_id,kp2d,kp3d,title,bbox}]
        for res in pose_det_results:
            # keypoints_3d = res['keypoints_3d'][..., :3]
            kp2d = res['keypoints']
            bbox = res['bbox']

        bbox_xyxy.append(bbox)
        keypoints2d.append(kp2d)

    bbox_xyxy = np.array(bbox_xyxy)
    keypoints2d = np.array(keypoints2d)

    # mask = np.array(mask)
    # smooth
    # if args.smooth_type is not None:
    #     keypoints2d = smooth_process(
    #          keypoints2d,
    #          smooth_type=args.smooth_type)
    # visualize kp2d
    if args.show_kp2d_path is not None:
        prepare_output_path(
            output_path=args.show_kp2d_path,
            path_type='auto',
            allowed_suffix=['.mp4', ''],
            overwrite=True)
        visualize_kp2d(
            keypoints2d,
            output_path=args.show_kp2d_path,
            data_source=args.keypoint_dst,
            img_format=None,
            image_array=np.array(frames_iter),
            overwrite=True)

    # save keypoints
    if args.save_kp_path is not None:
        prepare_output_path(
            output_path=args.save_kp_path,
            path_type='file',
            allowed_suffix=['.npz', '.npy', '.pickle'],
            overwrite=True)
        print('save keypoints ...')
        np.savez(args.save_kp_path, keypoints2d=keypoints2d)

    return keypoints2d


def smplify_proc(args, frames_iter, keypoints2d):
    """To regress SMPL/SMPL-X parameter from the input 2D or 3D keypoint.

    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (list[np.ndarray,]): prepared frames.
        keypoints2d (np.ndarray, optional):
            shape should be (f * J * 2). Defaults to None.
        keypoints2d_conf(np.ndarray, optional):
            confidence of the keypoints2d, shape should be (f * J * 1).

    Returns:
        None.

    Raises:
        ValueError: check the value of keypoints2d and keypoints3d.
    """
    smplify_config = mmcv.Config.fromfile(args.config)
    assert smplify_config.body_model.type.lower() in ['smpl', 'smplx']
    assert smplify_config.type.lower() in ['smplify', 'smplifyx']
    device = torch.device(args.device)

    if not (args.keypoint_dst == args.keypoint_src):
        keypoints2d, keypoints2d_conf = convert_kps(
            keypoints2d,
            src=args.keypoint_dst,
            dst=args.keypoint_src,
            approximate=True)
        if keypoints2d.shape[-1] == 3:
            keypoints2d_conf = keypoints2d[..., -1] * keypoints2d_conf.astype(
                keypoints2d.dtype)
            keypoints2d = keypoints2d[..., :-1]

    if isinstance(keypoints2d, np.ndarray):
        keypoints2d = torch.tensor(
            keypoints2d, dtype=torch.float32, device=device)
    if isinstance(keypoints2d_conf, np.ndarray):
        keypoints2d_conf = torch.tensor(
            keypoints2d_conf, dtype=torch.float32, device=device)

    # build SMPLify
    body_model_config = dict(
        type=smplify_config.body_model.type,
        gender=args.gender,
        num_betas=args.num_betas,
        keypoint_src=args.keypoint_src,  # J_regressor type
        keypoint_dst=args.keypoint_src,  #
        model_path=args.body_model_dir,
        joints_regressor=args.J_regressor,
        extra_joints_regressor=args.J_regressor_extra)
    if smplify_config.body_model.type.lower() == 'smplx':
        body_model_config.update(
            dict(
                use_face_contour=True,  # 127 -> 144
                use_pca=False,  # current vis do not supports use_pca
            ))
    smplify_config.update(
        dict(
            body_model=body_model_config,
            use_one_betas_per_video=args.use_one_betas_per_video,
            num_epochs=args.num_epochs,
            verbose=True,
            img_res=1000))
    smplify = build_registrant(dict(smplify_config))
    print('run SMPLify...')
    smplify_output = smplify(
        keypoints2d=keypoints2d,
        keypoints2d_conf=keypoints2d_conf,
        return_joints=True,
        return_verts=True)
    # get smpl parameters directly from smplify output
    poses = {k: v.detach().cpu() for k, v in smplify_output.items()}
    poses = {
        'global_orient': smplify_output['global_orient'].detach().cpu(),
        'transl': smplify_output['transl'].detach().cpu(),
        'body_pose': smplify_output['body_pose'].detach().cpu(),
        'betas': smplify_output['betas'].detach().cpu()
    }
    joints = smplify_output['joints'].detach().cpu()
    from mmhuman3d.utils.geometry import perspective_projection
    bs = joints.shape[0]
    projected_joints_xyd = perspective_projection(
        joints,
        torch.eye(3).expand((bs, 3, 3)).to(joints.device),
        torch.zeros((bs, 3)).to(joints.device), 5000.0,
        torch.Tensor([1000 / 2, 1000 / 2]).to(joints.device))
    visualize_kp2d(
        projected_joints_xyd.detach().cpu().numpy(),
        output_path='data/smplify_result/smplify_kp2d.mp4',
        data_source='smpl_54',
        img_format=None,
        # mask=mask,
        image_array=np.array(frames_iter),
        overwrite=True)
    if args.show_mesh_path is not None:
        # check show path
        prepare_output_path(
            output_path=args.show_mesh_path,
            allowed_suffix=['.mp4', ''],
            path_type='auto',
            overwrite=True)

        # visualize mesh
        print('visualize mesh...')
        body_model_config = dict(model_path=args.body_model_dir, type='smpl')
        visualize_smpl_pose(
            poses=poses,
            body_model_config=body_model_config,
            transl=smplify_output['transl'].detach().cpu(),
            output_path='data/smplify_result/h36m_mesh_.mp4',
            convention='opencv',
            overwrite=True)


def main(args):
    # prepare input
    frames_iter = prepare_frames(args.input_path)
    keypoints2d = estimate_keypoints(args, frames_iter)
    # data = np.load('data/smplify_result/coco_wholebody.npz')
    # keypoints2d = data['keypoints2d'][:30, ...]
    # smplify
    smplify_proc(args, frames_iter, keypoints2d=keypoints2d)


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
        '--J_regressor',
        type=str,
        default=None,
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--J_regressor_extra',
        type=str,
        default=None,
        # default='data/body_models/J_regressor_extra.npy',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--keypoint_src',
        type=str,
        default='h36m',
        help='The keypoint convention used by SMPlify, '
        'which is related to J_regressor')
    parser.add_argument(
        '--keypoint_dst',
        type=str,
        default='coco_wholebody',
        help='The keypoint convention detected by MMPose')
    parser.add_argument(
        '--keypoint_approximate',
        type=str,
        default=False,
        help='whether to use approximate matching in '
        'convention conversion for keypoints.')
    parser.add_argument(
        '--input_path', type=str, default=None, help='Video path')
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
        default=0.9,
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
        '--show_kp2d_path',
        type=str,
        default=None,
        help='Directory to save 2D keypoints video')
    # parser.add_argument(
    #     '--keypoint2d_type',
    #     type=str,
    #     default='coco',
    #     help='The source type of 2D keypoints')
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
    parser.add_argument('--num_epochs', type=int, default=20)
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
