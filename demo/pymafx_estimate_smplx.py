# yapf: disable

import argparse
import copy
import os
import os.path as osp
import pickle as pkl

import cv2
import mmcv
import numpy as np
import torch
from openpifpaf import decoder as ppdecoder
from openpifpaf import network as ppnetwork
from openpifpaf.predictor import Predictor
from openpifpaf.stream import Stream
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmhuman3d.apis import init_model
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_vibe
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.datasets import build_dataset
from mmhuman3d.utils.ffmpeg_utils import video_to_images
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
    max_instance = 0
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

    Predictor.batch_size = 1
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
    for preds, _, meta in tqdm(capture, total=num_frames):
        num_person = 0
        for pid, ann in enumerate(preds):
            if ann.score > args.detection_threshold:
                num_person += 1
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
        if num_person > max_instance:
            max_instance = num_person
    pkl.dump(tracking_results, open(pp_det_file_path, 'wb'))
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
    return bboxes, joints2d, frames, wb_kps, person_id_list,\
        image_folder, output_path, max_instance


def main(args):
    # Define model
    mesh_model, _ = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())

    device = torch.device(args.device)
    args.device = device
    args.pin_memory = True if torch.cuda.is_available() else False
    pymaf_config = dict(mmcv.Config.fromfile(args.mesh_reg_config))
    # Prepare input
    bboxes, joints2d, frames, wb_kps, person_id_list, image_folder, \
        output_path, max_instance = prepare_data_with_pifpaf_detection(args)

    pymaf_config['data']['test']['image_folder'] = image_folder
    pymaf_config['data']['test']['frames'] = frames
    pymaf_config['data']['test']['bboxes'] = bboxes
    pymaf_config['data']['test']['joints2d'] = joints2d
    pymaf_config['data']['test']['person_id_list'] = person_id_list
    pymaf_config['data']['test']['wb_kps'] = wb_kps
    test_dataset = build_dataset(pymaf_config['data']['test'],
                                 dict(test_mode=True))
    bboxes = test_dataset.bboxes
    frame_ids = test_dataset.frames
    dataloader = DataLoader(
        test_dataset, batch_size=args.model_batch_size, num_workers=0)

    # Run pred on each person
    with torch.no_grad():
        pred_cam, orig_height, orig_width, person_ids,\
            smplx_params = [], [], [], [], []

        for batch in tqdm(dataloader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            person_ids.extend(batch['person_id'])
            orig_height.append(batch['orig_height'])
            orig_width.append(batch['orig_width'])

            preds_dict = mesh_model.forward_test(batch)
            output = preds_dict['mesh_out'][-1]

            pred_cam.append(output['theta'][:, :3])
            smplx_params.append({
                'betas': output['pred_shape'],
                'body_pose': output['rotmat'],
                'left_hand_pose': output['pred_lhand_rotmat'],
                'right_hand_pose': output['pred_rhand_rotmat'],
                'jaw_pose': output['pred_face_rotmat'][:, 0:1],
                'leye_pose': output['pred_face_rotmat'][:, 1:2],
                'reye_pose': output['pred_face_rotmat'][:, 2:3],
                'expression': output['pred_exp'],
            })

        pred_cam = torch.cat(pred_cam, dim=0)
        orig_height = torch.cat(orig_height, dim=0)
        orig_width = torch.cat(orig_width, dim=0)
        del batch

    pred_cam = pred_cam.cpu().numpy()
    orig_height = orig_height.cpu().numpy()
    orig_width = orig_width.cpu().numpy()

    orig_cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=bboxes,
        img_width=orig_width,
        img_height=orig_height)

    del mesh_model
    fullpose = []
    betas = []
    for data in smplx_params:
        smplx_results = dict(
            global_orient=[],
            body_pose=[],
            leye_pose=[],
            reye_pose=[],
            jaw_pose=[],
            left_hand_pose=[],
            right_hand_pose=[],
            betas=[],
            expression=[])
        for key in smplx_results:
            if key == 'global_orient':
                smplx_results[key].append(
                    rotmat_to_aa(data['body_pose'].cpu().numpy()[:, :1]))
            elif key == 'body_pose':
                smplx_results[key].append(
                    rotmat_to_aa(data['body_pose'].cpu().numpy()[:, 1:22]))
            elif key == 'betas':
                smplx_results[key].append(data['betas'].cpu().numpy())
            elif key == 'expression':
                smplx_results[key].append(data['expression'].cpu().numpy())
            else:
                smplx_results[key].append(
                    rotmat_to_aa(data[key].cpu().numpy()))
        for key in smplx_results:
            smplx_results[key] = np.array(smplx_results[key][0])

        fullpose.append(
            np.concatenate((
                smplx_results['global_orient'],
                smplx_results['body_pose'],
                smplx_results['jaw_pose'],
                smplx_results['leye_pose'],
                smplx_results['reye_pose'],
                smplx_results['left_hand_pose'],
                smplx_results['right_hand_pose'],
            ),
                           axis=1))
        betas.append(smplx_results['betas'])
    fullpose = np.concatenate(fullpose, axis=0)
    betas = np.concatenate(betas, axis=0)
    if output_path is not None:
        human_data = HumanData()
        smplx = {}
        smplx['fullpose'] = fullpose
        smplx['betas'] = betas
        human_data['smplx'] = smplx
        human_data.dump(osp.join(output_path, 'inference_result.npz'))
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
    # To compress vertices array
    frame_num = len(os.listdir(image_folder))
    compressed_cams = np.zeros([frame_num, max_instance, 4])
    compressed_fullpose = np.zeros([frame_num, max_instance, 55, 3])
    compressed_betas = np.zeros([frame_num, max_instance, 10])
    for idx, frame_id in enumerate(frame_ids):
        if idx == 0:
            saved_frame_id = frame_id
            n_person = frame_ids.count(frame_id)
            compressed_fullpose[frame_id, :n_person] = fullpose[idx:idx +
                                                                n_person]
            compressed_betas[frame_id, :n_person] = betas[idx:idx + n_person]
            compressed_cams[frame_id, :n_person] = orig_cam[idx:idx + n_person]
        else:
            if saved_frame_id != frame_id:
                saved_frame_id = frame_id
                n_person = frame_ids.count(frame_id)
                compressed_fullpose[frame_id, :n_person] = fullpose[idx:idx +
                                                                    n_person]
                compressed_betas[frame_id, :n_person] = betas[idx:idx +
                                                              n_person]
                compressed_cams[frame_id, :n_person] = orig_cam[idx:idx +
                                                                n_person]
    if args.visualization and args.image_folder is None:
        image_file_names = sorted([
            osp.join(image_folder, x) for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])
        image_array = []
        for path in image_file_names:
            img = cv2.imread(path)
            image_array.append(img)
        visualize_smpl_vibe(
            poses=compressed_fullpose.reshape(-1, max_instance, 165),
            betas=compressed_betas,
            orig_cam=compressed_cams,
            output_path=os.path.join(output_path, 'smplx.mp4'),
            image_array=np.array(image_array),
            body_model_config=body_model_config,
            resolution=(orig_height[0], orig_width[0]),
            overwrite=True,
        )


def init_openpifpaf(parser):
    ppnetwork.Factory.cli(parser)
    ppdecoder.cli(parser)
    Predictor.cli(parser)
    Stream.cli(parser)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    init_openpifpaf(parser)

    parser.add_argument(
        '--mesh_reg_config',
        type=str,
        default='configs/pymafx/pymafx.py',
        help='Config file for mesh regression')
    parser.add_argument(
        '--mesh_reg_checkpoint',
        type=str,
        default='data/pretrained_models/PyMAF-X_model_checkpoint.pth',
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
        '--detection_threshold',
        type=float,
        default=0.4,
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
        '--visualization', action='store_true', help='SMPLX Visualization')

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
