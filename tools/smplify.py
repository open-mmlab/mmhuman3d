import argparse
import pickle

import mmcv
import smplx
import torch

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.parametric_model import build_registrant
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.utils.misc import torch_to_numpy


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
        '--keypoint_type',
        choices=['mmpose', 'coco'],
        default='mmpose',
        help='the source type of input keypoints')
    parser.add_argument('--config', help='smplify config file path')
    parser.add_argument('--body_model_dir', help='body models file path')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_betas', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=2)
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
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    assert cfg.body_model_type in ['smpl', 'smplx']
    assert cfg.smplify_method in ['smplify', 'smplifyx']
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # TODO: support keypoints2d
    assert args.input_type == 'keypoints3d', 'Only support keypoints3d!'

    with open(args.input, 'rb') as f:
        keypoints_src = pickle.load(f, encoding='latin1')
        if args.input_type == 'keypoints2d':
            assert keypoints_src.shape[-1] == 2
        elif args.input_type == 'keypoints3d':
            assert keypoints_src.shape[-1] == 3
        else:
            raise KeyError('Only support keypoints2d and keypoints3d')

    keypoints, mask = convert_kps(
        keypoints_src, src=args.keypoint_type, dst=cfg.body_model_type)
    keypoints_conf = mask

    batch_size = args.batch_size if args.batch_size else keypoints.shape[0]

    keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
    keypoints_conf = torch.tensor(
        keypoints_conf, dtype=torch.float32, device=device)

    if args.input_type == 'keypoints3d':
        human_data = dict(
            keypoints3d=keypoints, keypoints3d_conf=keypoints_conf)

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
    if args.output:
        print(f'writing results to {args.output}')
        mmcv.dump(results, args.output)

    # save rendered results
    if args.show_path:
        # TODO: use results directly after !42 is merged
        poses = {k: v.detach().cpu() for k, v in smplify_output.items()}
        visualize_smpl_pose(
            poses=poses,
            body_model_dir=args.body_model_dir,
            output_path=args.show_path,
            model_type=cfg.body_model_type,
            overwrite=True)


if __name__ == '__main__':
    main()
