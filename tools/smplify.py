import argparse
import os
import time

import mmcv
import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.evaluation import keypoint_mpjpe
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.registrants.builder import build_registrant


def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d smplify tool')
    parser.add_argument(
        '--input',
        help=('input file path.'
              'Input shape should be [N, J, D] or [N, M, J, D],'
              ' where N is the sequence length, M is the number of persons,'
              ' J is the number of joints and D is the dimension.'))
    parser.add_argument(
        '--input_type',
        choices=['keypoints2d', 'keypoints3d'],
        default='keypoints3d',
        help='input type')
    parser.add_argument(
        '--J_regressor',
        type=str,
        default=None,
        help='the path of the J_regressor')
    parser.add_argument(
        '--keypoint_type',
        default='human_data',
        help='the source type of input keypoints')
    parser.add_argument('--config', help='smplify config file path')
    parser.add_argument('--body_model_dir', help='body models file path')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_betas', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument(
        '--use_one_betas_per_video',
        action='store_true',
        help='use one betas to keep shape consistent through a video')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for smplify')
    parser.add_argument(
        '--gender',
        choices=['neutral', 'male', 'female'],
        default='neutral',
        help='gender of SMPL model')
    parser.add_argument('--output', help='output result file')
    parser.add_argument(
        '--show_path', help='directory to save rendered images or video')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to overwrite if there is already a result file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    smplify_config = mmcv.Config.fromfile(args.config)
    assert smplify_config.body_model.type.lower() in ['smpl', 'smplx']
    assert smplify_config.type.lower() in ['smplify', 'smplifyx']

    # set cudnn_benchmark
    if smplify_config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    human_data = HumanData.fromfile(args.input)
    keypoints_src = human_data[args.input_type]
    keypoints_src_mask = human_data[args.input_type + '_mask']
    if args.input_type == 'keypoints2d':
        assert keypoints_src.shape[-1] in {2, 3}
    elif args.input_type == 'keypoints3d':
        assert keypoints_src.shape[-1] in {3, 4}
        keypoints_src = keypoints_src[..., :3]
    else:
        raise KeyError('Only support keypoints2d and keypoints3d')

    keypoints, mask = convert_kps(
        keypoints_src,
        mask=keypoints_src_mask,
        src=args.keypoint_type,
        dst=smplify_config.body_model['keypoint_dst'])
    keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)

    batch_size = args.batch_size if args.batch_size else keypoints.shape[0]

    keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
    keypoints_conf = torch.tensor(
        keypoints_conf, dtype=torch.float32, device=device)

    if args.input_type == 'keypoints3d':
        human_data = dict(
            keypoints3d=keypoints, keypoints3d_conf=keypoints_conf)
    elif args.input_type == 'keypoints2d':
        human_data = dict(
            keypoints2d=keypoints, keypoints2d_conf=keypoints_conf)
    else:
        raise TypeError(f'Unsupported input type: {args.input_type}')

    # create body model
    body_model_config = dict(
        type=smplify_config.body_model.type.lower(),
        gender=args.gender,
        num_betas=args.num_betas,
        model_path=args.body_model_dir,
        batch_size=batch_size,
    )

    if args.J_regressor is not None:
        body_model_config.update(dict(joints_regressor=args.J_regressor))

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
            num_epochs=args.num_epochs))

    smplify = build_registrant(dict(smplify_config))

    # run SMPLify(X)
    t0 = time.time()
    smplify_output = smplify(**human_data, return_joints=True)
    t1 = time.time()
    print(f'Time:  {t1 - t0:.2f} s')

    # test MPJPE
    pred = smplify_output['joints'].cpu().numpy()
    gt = keypoints.cpu().numpy()
    mask = mask.reshape(1, -1).repeat(gt.shape[0], axis=0).astype(bool)
    mpjpe = keypoint_mpjpe(pred=pred, gt=gt, mask=mask)
    print(f'SMPLify MPJPE: {mpjpe * 1000:.2f} mm')

    # get smpl parameters directly from smplify output
    poses = {k: v.detach().cpu() for k, v in smplify_output.items()}
    smplify_results = HumanData(dict(smpl=poses))

    if args.output is not None:
        print(f'Dump results to {args.output}')
        smplify_results.dump(args.output, overwrite=args.overwrite)

    if args.show_path is not None:
        # visualize smpl pose
        body_model_dir = os.path.dirname(args.body_model_dir.rstrip('/'))
        body_model_config.update(
            model_path=body_model_dir,
            model_type=smplify_config.body_model.type.lower())
        visualize_smpl_pose(
            poses=poses,
            body_model_config=body_model_config,
            output_path=args.show_path,
            orbit_speed=1,
            overwrite=True)


if __name__ == '__main__':
    main()
