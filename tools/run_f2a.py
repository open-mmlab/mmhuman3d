import argparse
import pickle

import mmcv
import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization import visualize_smpl_pose
from mmhuman3d.models.builder import build_registrant

from mmhuman3d.core.visualization.renderer import render_runner
import os

osj = os.path.join


def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d smplify tool')
    parser.add_argument(
        '--keypoint',
        default=None,
        help=('input file path.'
              'Input shape should be [N, J, D] or [N, M, J, D],'
              ' where N is the sequence length, M is the number of persons,'
              ' J is the number of joints and D is the dimension.'))
    parser.add_argument(
        '--keypoint_type',
        choices=['keypoints2d', 'keypoints3d'],
        default='keypoints3d',
        help='input type')
    parser.add_argument(
        '--keypoint_src',
        default='coco_wholebody',
        help='the source type of input keypoints')
    parser.add_argument('--config', help='smplify config file path')
    parser.add_argument('--model_path', help='body models file path')
    parser.add_argument('--uv_param_path', type=int, default=None)
    parser.add_argument('--num_betas', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=20)
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
    parser.add_argument('--output_path', help='output result file')
    parser.add_argument('--exp_dir', help='tmp dir for writing some results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    flow2avatar_config = mmcv.Config.fromfile(args.config)
    assert flow2avatar_config.body_model.type.lower() in ['smpld']

    # set cudnn_benchmark
    if flow2avatar_config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    if args.keypoint is not None:
        with open(args.keypoint, 'rb') as f:
            keypoints_src = pickle.load(f, encoding='latin1')
            if args.input_type == 'keypoints2d':
                assert keypoints_src.shape[-1] == 2
            elif args.input_type == 'keypoints3d':
                assert keypoints_src.shape[-1] == 3
            else:
                raise KeyError('Only support keypoints2d and keypoints3d')

        keypoints, mask = convert_kps(
            keypoints_src,
            src=args.keypoint_type,
            dst=flow2avatar_config.body_model['keypoint_dst'])
        keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)

        keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
        keypoints_conf = torch.tensor(
            keypoints_conf, dtype=torch.float32, device=device)

        if args.keypoint_type == 'keypoints3d':
            human_data = dict(
                keypoints3d=keypoints, keypoints3d_conf=keypoints_conf)

    flow2avatar = build_registrant(dict(flow2avatar_config))

    body_model_config = dict(
        type=smplify_config.body_model.type,
        gender=args.gender,
        num_betas=args.num_betas,
        model_path=args.body_model_dir,
        batch_size=batch_size,
    )
    

    # run SMPLify(X)
    flow2avatar_output = flow2avatar(**human_data)

    avatar = flow2avatar_output.pop('meshes')
    # get smpl parameters directly from smplify output
    with open(osj(output_path, 'smpld.pkl'), 'wb') as f:
        pickle.dump(flow2avatar_output, f)

    if args.visualize:
        render_runner.render(
            renderer=flow2avatar.renderer_rgb,
            device=device,
            meshes=avatar,
            cameras=cameras,
            no_grad=True,
            return_tensor=False,
            output_path=osj(output_path, 'demo.mp4'),
        )


if __name__ == '__main__':
    main()
