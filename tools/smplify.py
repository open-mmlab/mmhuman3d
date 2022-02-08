import argparse
import os
import time

import mmcv
import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.data.data_structures import HumanData
from mmhuman3d.models.builder import build_registrant


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
    parser.add_argument('--output', help='output result file')
    parser.add_argument(
        '--show_path', help='directory to save rendered images or video')
    # parser.add_argument('--tmpdir', help='tmp dir for writing some results')
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

    # keypoints_src = pickle.load(f, encoding='latin1')
    # data = np.load(args.input, allow_pickle=True)
    data = HumanData.fromfile(args.input)
    keypoints_src = data[args.input_type]
    keypoints_src_mask = data[args.input_type + '_mask']
    if args.input_type == 'keypoints2d':
        assert keypoints_src.shape[-1] in {2, 3}
    elif args.input_type == 'keypoints3d':
        assert keypoints_src.shape[-1] in {3, 4}
        keypoints_src = keypoints_src[..., :3]
    else:
        raise KeyError('Only support keypoints2d and keypoints3d')
    # with open(args.input, 'rb') as f:
    #     keypoints_src = pickle.load(f, encoding='latin1')
    #     if args.input_type == 'keypoints2d':
    #         assert keypoints_src.shape[-1] == 2
    #     elif args.input_type == 'keypoints3d':
    #         assert keypoints_src.shape[-1] == 3
    #     else:
    #         raise KeyError('Only support keypoints2d and keypoints3d')

    keypoints, mask = convert_kps(
        keypoints_src,
        mask=keypoints_src_mask,
        src=args.keypoint_type,
        dst=smplify_config.body_model['keypoint_dst'])
    keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)

    batch_size = args.batch_size if args.batch_size else keypoints.shape[0]

    print('keypoints.shape', keypoints.shape)
    keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
    keypoints_conf = torch.tensor(
        keypoints_conf, dtype=torch.float32, device=device)

    # TODO: support keypoints2d
    if args.input_type == 'keypoints3d':
        human_data = dict(
            keypoints3d=keypoints, keypoints3d_conf=keypoints_conf)

    # create body model
    body_model_config = dict(
        type=smplify_config.body_model.type,
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
    smplify_output = smplify(**human_data)
    t1 = time.time()
    print(f'{t1 - t0} s')

    # get smpl parameters directly from smplify output
    poses = {k: v.detach().cpu() for k, v in smplify_output.items()}
    print(poses.keys())
    smplify_results = HumanData(dict(smpl=poses))

    if args.output is not None:
        print(f'Dump results to {args.output}')
        smplify_results.dump(args.output, overwrite=args.overwrite)
        # np.savez_compressed(args.output, **poses)

    if args.show_path is not None:
        # visualize mesh
        body_model_dir = os.path.dirname(args.body_model_dir.rstrip('/'))
        body_model_config = dict(model_path=body_model_dir, model_type='smpl')
        print(body_model_config)
        visualize_smpl_pose(
            poses=poses,
            # model_path=args.body_model_dir,
            body_model_config=body_model_config,
            output_path=args.show_path,
            model_type=smplify_config.body_model.type.lower(),
            orbit_speed=1,
            overwrite=True)


if __name__ == '__main__':
    main()
