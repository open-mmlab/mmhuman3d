import argparse
import pickle

import mmcv
from mmhuman3d.core.cameras.builder import build_cameras
from mmhuman3d.utils.mesh_utils import save_meshes_as_objs
import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.visualization import visualize_smpl_pose
from mmhuman3d.models.builder import build_registrant

from mmhuman3d.core.visualization.renderer import render_runner
import os
import glob

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
        '--keypoint_src',
        default='coco_wholebody',
        help='the source type of input keypoints')
    parser.add_argument('--config', default='f2a_config.py')
    parser.add_argument('--camera_path', help='smplify config file path')
    parser.add_argument('--image_folder', help='smplify config file path')
    parser.add_argument(
        '--model_path',
        default='/mnt/lustre/share/sugar/SMPLmodels/',
        help='smplify config file path')
    parser.add_argument(
        '--uv_param_path',
        default='/mnt/lustre/share/sugar/smpl_uv.pkl',
        help='smplify config file path')

    parser.add_argument('--num_betas', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument(
        '--use_one_betas_per_video',
        default=True,
        type=bool,
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
    parser.add_argument('--exp_dir', help='tmp dir for writing some results')
    parser.add_argument(
        '--verbose',
        action='store_true',
    )
    parser.add_argument(
        '--visualize',
        type=bool,
        default=True,
    )
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

    body_model_type = flow2avatar_config.body_model.type
    body_model_config = dict(
        type=body_model_type,
        gender=args.gender,
        num_betas=args.num_betas,
        model_path=osj(args.model_path, 'smpl'),
        uv_param_path=args.uv_param_path,
    )

    flow2avatar_config.renderer_uv.update(uv_param_path=args.uv_param_path)
    flow2avatar_config.update(
        device=device,
        verbose=args.verbose,
        experiment_dir=args.exp_dir,
        body_model=body_model_config,
        use_one_betas_per_video=args.use_one_betas_per_video)

    if args.image_folder is not None:
        image_paths = glob.glob(osj(args.image_folder, '*.png'))
    else:
        image_paths = None

    cameras = None
    if args.camera_path is not None:
        with open(args.camera_path, 'rb') as f:
            cameras_config = pickle.load(f)
        cameras = build_cameras(cameras_config).to(device)

    data = dict(
        image_paths=image_paths,
        cameras=cameras,
        return_texture=True,
        return_mesh=True)

    d = np.load('/mnt/lustre/wangwenjia/mesh/m2m_smpl.npz')
    init_global_orient = torch.Tensor(d['global_orient'])[0:1].to(device)
    init_transl = torch.Tensor(d['transl'])[0:1].to(device)
    init_body_pose = torch.Tensor(d['body_pose'])[0:1].to(device)
    init_betas = torch.Tensor(d['betas'])[0:1].to(device)
    data.update(
        init_global_orient=init_global_orient,
        init_transl=init_transl,
        init_body_pose=init_body_pose,
        init_betas=init_betas)

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
            src=args.keypoint_src,
            dst=flow2avatar_config.body_model['keypoint_dst'])
        keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)

        keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
        keypoints_conf = torch.tensor(
            keypoints_conf, dtype=torch.float32, device=device)

        if args.keypoint_type == 'keypoints3d':
            data.update(
                dict(keypoints3d=keypoints, keypoints3d_conf=keypoints_conf))

    flow2avatar = build_registrant(dict(flow2avatar_config))

    flow2avatar_output = flow2avatar(**data)

    avatar = flow2avatar_output.pop('meshes')

    pose = torch.zeros(1, 72).to(device)
    pose = flow2avatar.body_model.tensor2dict(pose)

    flow2avatar_output = flow2avatar.body_model(
        displacement=flow2avatar_output['displacement'],
        texture_image=flow2avatar_output['texture_image'],
        return_mesh=True,
        return_texture=True,
        **pose)

    T_avatar = flow2avatar_output['meshes']
    save_meshes_as_objs(T_avatar[0], [osj(args.exp_dir, 'T_pose.obj')])
    # get smpl parameters directly from smplify output
    with open(osj(args.exp_dir, 'smpld.pkl'), 'wb') as f:
        pickle.dump(flow2avatar_output, f)

    # if args.visualize:
    #     render_runner.render(
    #         renderer=flow2avatar.renderer_rgb,
    #         device=device,
    #         meshes=avatar,
    #         cameras=cameras,
    #         no_grad=True,
    #         return_tensor=False,
    #         output_path=osj(args.exp_dir, 'demo.mp4'),
    #     )


if __name__ == '__main__':
    main()
