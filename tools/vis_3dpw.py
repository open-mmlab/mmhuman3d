import argparse
import os
import os.path as osp

import mmcv
from mmhuman3d.models.builder import build_body_model
from numpy import int8
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
print('importint')
from mmhuman3d.core.visualization import visualize_smpl_hmr
import cv2
from mmhuman3d.apis import multi_gpu_test, single_gpu_test
from mmhuman3d.data.datasets import build_dataloader, build_dataset
from mmhuman3d.models import build_architecture
import numpy as np
from mmhuman3d.utils.transforms import rotmat_to_ee
from torchvision.utils import save_image
def denormalize_images(images):
    images = images * np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = images + np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    return images

def bgr2rgb(rgbs):
    if isinstance(rgbs, torch.Tensor):
        bgrs = torch.cat(
            [rgbs[..., 2, None], rgbs[..., 1, None], rgbs[..., 0, None]],
            -1)
    elif isinstance(rgbs, np.ndarray):
        bgrs = np.concatenate(
            [rgbs[..., 2, None], rgbs[..., 1, None], rgbs[..., 0, None]],
            -1)
    return bgrs
def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default='pa-mpjpe',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "pa-mpjpe" for H36M')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--result_dir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        default={},
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
        
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    result = mmcv.load(args.result_dir)
    result = result[0]
    # poses = rotmat_to_ee(torch.tensor(result['poses'])).reshape(-1,72)# .cuda()
    poses = torch.tensor(result['poses'])
    betas = torch.tensor(result['betas'])# .cuda()
    camera = torch.tensor(result['betas']).numpy()
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1000,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    body_model_render1=dict(
            type='smpl',
            
            model_path='/mnt/lustre/wangyanjun/data/body_models',
            )
    body_model = build_body_model(cfg.model.body_model_test)# .numpy()
    batch_idx = 0
    # save_dir = '/mnt/lustre/wangyanjun/pare_log/3dpw/batch%d'
    for batch in data_loader:
        
        idx = batch['sample_idx'].squeeze(-1)
        bs = idx.shape[0]
        batch_pose = poses[idx]
        batch_betas = betas[idx]
        batch_camera = camera[idx]
        batch_img = (batch['img']).numpy()
        batch_img = denormalize_images(batch_img)
        batch_img = (batch_img.transpose(0,2,3,1)*255).astype(np.int8)
        batch_img = np.concatenate([batch_img,batch_img],axis=2)
        # batch_img = (batch['img'].permute(0,2,3,1)*255).detach().cpu().numpy().astype(np.int8)
        bboxes_xyxy = torch.tensor([[0,0,224,224]]).repeat(bs,1).numpy()
        out = body_model(
            betas=batch_betas,
            body_pose=batch_pose[:,1:],
            global_orient=batch_pose[:,:1],
            pose2rot=False,
        )
        save_dir = '/mnt/lustre/wangyanjun/pare_log/3dpw/batch%d'%batch_idx
        batch_img = bgr2rgb(batch_img)
   
        visualize_smpl_hmr(
            verts=out['vertices'],
            cam_transl=batch_camera,
            bbox=bboxes_xyxy,
            body_model_config=body_model_render1,
            output_path=save_dir,
            overwrite = True,
            # image_array=batch_img,
            render_choice='lq',
            resolution=[224,448],
            return_tensor=True,
        )
        batch_idx += 1
    
        im_save = (render_res.detach().cpu().numpy()*255).astype(int8)
        cv2.imwrite('m.png',im_save[0,:,:,:3])
        # print(render_res.shape)
    
    # # build the model and load checkpoint
    # model = build_architecture(cfg.model)
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    # if not distributed:
    #     if args.device == 'cpu':
    #         model = model.cpu()
    #     else:
    #         model = MMDataParallel(model, device_ids=[0])
    #     outputs = single_gpu_test(model, data_loader)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)

    # rank, _ = get_dist_info()
    # eval_cfg = cfg.get('evaluation', args.eval_options)
    # eval_cfg.update(dict(metric=args.metrics))
    # if rank == 0:
    #     mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    #     results = dataset.evaluate(outputs, args.work_dir, **eval_cfg)
    #     for k, v in results.items():
    #         print(f'\n{k} : {v:.2f}')

    # if args.out and rank == 0:
    #     print(f'\nwriting results to {args.out}')
    #     mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
