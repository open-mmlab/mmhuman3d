# yapf: disable

import argparse
import copy
import os
import os.path as osp
import pickle as pkl

import joblib
import mmcv
import numpy as np
import torch
from openpifpaf import decoder as ppdecoder
from openpifpaf import network as ppnetwork
from openpifpaf.predictor import Predictor
from openpifpaf.stream import Stream
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.data.datasets import build_dataset
from mmhuman3d.models.architectures.builder import build_architecture
from mmhuman3d.utils.ffmpeg_utils import video_to_images
from mmhuman3d.utils.geometry import convert_to_full_img_cam
from mmhuman3d.utils.transforms import rotmat_to_aa

# yapf: enable


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    """Convert predicted camera from cropped image coordinates to original
    image coordinates.

    Args:
        cam (ndarray, shape=(3,)):
            weak perspective camera in cropped img coordinates.
        bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h).
        img_width (int): original image width.
        img_height (int): original image height.
    """

    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:, 0] * (1. / (img_width / h))
    sy = cam[:, 0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:, 1]
    ty = ((cy - hh) / hh / sy) + cam[:, 2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def prepare_data_with_pifpaf_detection(args):
    if args.image_folder is None:
        video_file = args.vid_file
        if not os.path.isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')

        output_path = os.path.join(
            args.output_folder,
            os.path.basename(video_file).replace('.mp4', ''))
        image_folder = osp.join(output_path, 'images')
        os.makedirs(image_folder, exist_ok=True)
        video_to_images(video_file, image_folder)
        num_frames = len(os.listdir(image_folder))
    else:
        image_folder = args.image_folder
        num_frames = len(os.listdir(image_folder))
        output_path = os.path.join(args.output_folder,
                                   osp.split(image_folder)[-1])

    os.makedirs(output_path, exist_ok=True)
    # pifpaf person detection
    pp_det_file_path = os.path.join(output_path, 'pp_det_results.pkl')
    pp_args = copy.deepcopy(args)
    pp_args.force_complete_pose = True
    ppdecoder.configure(pp_args)
    ppnetwork.Factory.configure(pp_args)
    ppnetwork.Factory.checkpoint = pp_args.detector_checkpoint
    Predictor.configure(pp_args)
    Stream.configure(pp_args)

    Predictor.batch_size = pp_args.detector_batch_size
    if pp_args.detector_batch_size > 1:
        Predictor.long_edge = 1000
    Predictor.loader_workers = 1
    predictor = Predictor()
    if args.vid_file is not None:
        capture = Stream(args.vid_file, preprocess=predictor.preprocess)
        capture = predictor.dataset(capture)
    elif args.image_folder is not None:
        image_file_names = sorted([
            osp.join(image_folder, x) for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])
        capture = predictor.images(image_file_names)

    tracking_results = {}
    print('Running openpifpaf for person detection...')
    for preds, _, meta in tqdm(
            capture, total=num_frames // args.detector_batch_size):
        for pid, ann in enumerate(preds):
            if ann.score > args.detection_threshold:
                frame_i = meta['frame_i'] - 1 if 'frame_i' in meta else meta[
                    'dataset_index']
                file_name = meta[
                    'file_name'] if 'file_name' in meta else image_folder
                person_id = file_name.split('/')[-1].split(
                    '.')[0] + '_f' + str(frame_i) + '_p' + str(pid)
                det_wb_kps = ann.data
                det_face_kps = det_wb_kps[23:91]
                tracking_results[person_id] = {
                    'frames': [frame_i],
                    'joints2d': [det_wb_kps[:17]],
                    'joints2d_lhand': [det_wb_kps[91:112]],
                    'joints2d_rhand': [det_wb_kps[112:133]],
                    'joints2d_face':
                    [np.concatenate([det_face_kps[17:], det_face_kps[:17]])],
                }
    pkl.dump(tracking_results, open(pp_det_file_path, 'wb'))
    return tracking_results, image_folder, output_path


def main(args):
    device = torch.device(args.device)
    args.device = device
    args.pin_memory = True if torch.cuda.is_available() else False
    pymaf_config = dict(mmcv.Config.fromfile(args.mesh_reg_config))
    # prepare input
    tracking_results, image_folder, output_path = \
        prepare_data_with_pifpaf_detection(args)

    # ========= Define model ========= #
    model = build_architecture(pymaf_config['model'])
    model = model.to(device)

    # ========= Load pretrained weights ========= #
    if args.mesh_reg_checkpoint is not None:
        print(
            f'Loading pretrained weights from \"{args.mesh_reg_checkpoint}\"')
        checkpoint = torch.load(args.mesh_reg_checkpoint)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f'loaded checkpoint: {args.mesh_reg_checkpoint}')

    model.eval()

    # ========= Run pred on each person ========= #
    print('Running reconstruction on each tracklet...')
    pred_results = {}
    bboxes = joints2d = []
    frames = []
    if args.tracking_method == 'pose':
        wb_kps = {
            'joints2d_lhand': [],
            'joints2d_rhand': [],
            'joints2d_face': [],
        }
    person_id_list = list(tracking_results.keys())
    for person_id in person_id_list:
        if args.tracking_method == 'pose':
            joints2d.extend(tracking_results[person_id]['joints2d'])
            wb_kps['joints2d_lhand'].extend(
                tracking_results[person_id]['joints2d_lhand'])
            wb_kps['joints2d_rhand'].extend(
                tracking_results[person_id]['joints2d_rhand'])
            wb_kps['joints2d_face'].extend(
                tracking_results[person_id]['joints2d_face'])

        frames.extend(tracking_results[person_id]['frames'])
    pymaf_config['data']['test']['image_folder'] = image_folder
    pymaf_config['data']['test']['frames'] = frames
    pymaf_config['data']['test']['bboxes'] = bboxes
    pymaf_config['data']['test']['joints2d'] = joints2d
    pymaf_config['data']['test']['person_id_list'] = person_id_list
    pymaf_config['data']['test']['wb_kps'] = wb_kps
    test_dataset = build_dataset(pymaf_config['data']['test'],
                                 dict(test_mode=True))
    bboxes = test_dataset.bboxes
    scales = test_dataset.scales
    frames = test_dataset.frames

    dataloader = DataLoader(
        test_dataset, batch_size=args.model_batch_size, num_workers=0)

    with torch.no_grad():
        pred_cam, pred_verts, pred_smplx_verts, pred_pose = [], [], [], []
        pred_betas, pred_joints3d = [], []
        orig_height, orig_width = [], []
        person_ids = []
        smplx_params = []

        for batch in tqdm(dataloader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            person_ids.extend(batch['person_id'])
            orig_height.append(batch['orig_height'])
            orig_width.append(batch['orig_width'])
            preds_dict = model.forward_test(batch)

            output = preds_dict['mesh_out'][-1]

            pred_cam.append(output['theta'][:, :3])
            pred_verts.append(output['verts'])
            pred_smplx_verts.append(output['smplx_verts'])
            pred_pose.append(output['theta'][:, 13:85])
            pred_betas.append(output['theta'][:, 3:13])
            pred_joints3d.append(output['kp_3d'])

            smplx_params.append({
                'shape': output['pred_shape'],
                'body_pose': output['rotmat'],
                'left_hand_pose': output['pred_lhand_rotmat'],
                'right_hand_pose': output['pred_rhand_rotmat'],
                'jaw_pose': output['pred_face_rotmat'][:, 0:1],
                'leye_pose': output['pred_face_rotmat'][:, 1:2],
                'reye_pose': output['pred_face_rotmat'][:, 2:3],
                'expression': output['pred_exp'],
            })

        pred_cam = torch.cat(pred_cam, dim=0)
        pred_verts = torch.cat(pred_verts, dim=0)
        pred_smplx_verts = torch.cat(pred_smplx_verts, dim=0)
        pred_pose = torch.cat(pred_pose, dim=0)
        pred_betas = torch.cat(pred_betas, dim=0)
        pred_joints3d = torch.cat(pred_joints3d, dim=0)

        orig_height = torch.cat(orig_height, dim=0)
        orig_width = torch.cat(orig_width, dim=0)

        del batch

    # ========= Save results to a pickle file ========= #
    pred_cam = pred_cam.cpu().numpy()
    pred_verts = pred_verts.cpu().numpy()
    pred_smplx_verts = pred_smplx_verts.cpu().numpy()
    pred_pose = pred_pose.cpu().numpy()
    pred_betas = pred_betas.cpu().numpy()
    pred_joints3d = pred_joints3d.cpu().numpy()
    orig_height = orig_height.cpu().numpy()
    orig_width = orig_width.cpu().numpy()

    orig_cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=bboxes,
        img_width=orig_width,
        img_height=orig_height)

    camera_translation = convert_to_full_img_cam(
        pare_cam=pred_cam,
        bbox_height=scales * 200.,
        bbox_center=bboxes[:, :2],
        img_w=orig_width,
        img_h=orig_height,
        focal_length=5000.,
    )

    pred_results = {
        'pred_cam': pred_cam,
        'orig_cam': orig_cam,
        'orig_cam_t': camera_translation,
        'verts': pred_verts,
        'smplx_verts': pred_smplx_verts,
        'pose': pred_pose,
        'betas': pred_betas,
        'joints3d': pred_joints3d,
        'joints2d': joints2d,
        'bboxes': bboxes,
        'frame_ids': frames,
        'person_ids': person_ids,
        'smplx_params': smplx_params,
    }

    del model
    fullpose = []
    for smplx_params in pred_results['smplx_params']:
        global_orient = rotmat_to_aa(
            smplx_params['body_pose'].cpu().numpy()[:, :1])
        body_pose = rotmat_to_aa(smplx_params['body_pose'].cpu().numpy()[:,
                                                                         1:22])
        jaw_pose = rotmat_to_aa(smplx_params['jaw_pose'].cpu().numpy())
        leye_pose = rotmat_to_aa(smplx_params['leye_pose'].cpu().numpy())
        reye_pose = rotmat_to_aa(smplx_params['reye_pose'].cpu().numpy())
        left_hand_pose = rotmat_to_aa(
            smplx_params['left_hand_pose'].cpu().numpy())
        right_hand_pose = rotmat_to_aa(
            smplx_params['right_hand_pose'].cpu().numpy())
        fullpose.append(
            np.concatenate((
                global_orient,
                body_pose,
                jaw_pose,
                leye_pose,
                reye_pose,
                left_hand_pose,
                right_hand_pose,
            ),
                           axis=1))
    fullpose = np.concatenate(fullpose, axis=0)
    # create body model
    body_model_config = dict(
        type='smplx',
        num_betas=10,
        use_face_contour=True,
        use_pca=False,
        flat_hand_mean=True,
        model_path='data/body_models/',
        keypoint_src='smplx',
        keypoint_dst='smplx',
    )
    visualize_smpl_pose(
        poses=fullpose.reshape(-1, 165),
        body_model_config=body_model_config,
        output_path=os.path.join('smplx.mp4'),
        resolution=(1024, 1024),
        overwrite=True)
    joblib.dump(pred_results, os.path.join(output_path, 'output.pkl'))

    print('================= END =================')


def init_openpifpaf(parser):
    ppnetwork.Factory.cli(parser)
    ppdecoder.cli(parser)
    Predictor.cli(parser)
    Stream.cli(parser)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
    init_openpifpaf(parser)

    parser.add_argument(
        '--mesh_reg_config',
        type=str,
        default='configs/pymafx/pymafx.py',
        help='Config file for mesh regression')
    parser.add_argument(
        '--mesh_reg_checkpoint',
        type=str,
        default='data/pretrained_models/PyMAF-X_model_checkpoint.pt',
        help='Checkpoint file for mesh regression')
    parser.add_argument(
        '--tracking_method',
        type=str,
        default='pose',
        help='tracking method to calculate the tracklet of a subject')
    parser.add_argument(
        '--detector_checkpoint',
        type=str,
        default='shufflenetv2k30-wholebody',
        help='detector checkpoint for openpifpaf')
    parser.add_argument(
        '--detector_batch_size',
        type=int,
        default=1,
        help='batch size of person detection')
    parser.add_argument(
        '--detection_threshold',
        type=float,
        default=0.3,
        help='pifpaf detection score threshold.')
    parser.add_argument(
        '--vid_file', type=str, default=None, help='input video path')
    parser.add_argument(
        '--image_folder', type=str, default=None, help='input image folder')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='output',
        help='output folder to write results')

    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    parser.add_argument(
        '--model_batch_size',
        type=int,
        default=8,
        help='batch size for SMPL prediction')
    args = parser.parse_args()
    main(args)
