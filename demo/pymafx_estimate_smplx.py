# yapf: disable

import argparse
import copy
import os
import os.path as osp
import shutil
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mmhuman3d.apis import init_model
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_vibe
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.datasets import build_dataset
from mmhuman3d.utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    prepare_frames,
    process_mmdet_results,
)
from mmhuman3d.utils.ffmpeg_utils import video_to_images
from mmhuman3d.utils.transforms import rotmat_to_aa

try:
    from openpifpaf import decoder as ppdecoder
    from openpifpaf import network as ppnetwork
    from openpifpaf.predictor import Predictor
    from openpifpaf.stream import Stream
    has_openpifpaf = True
except (ImportError, ModuleNotFoundError):
    has_openpifpaf = False

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

try:
    from mmpose.apis import (
        get_track_id,
        inference_top_down_pose_model,
        init_pose_model,
    )
    has_mmpose = True
except (ImportError, ModuleNotFoundError):
    has_mmpose = False

# yapf: enable


def process_tracking_results(tracking_results_all_frames):
    """Process mmtracking results."""
    tracklet = []
    final_results = []
    max_instance = 0

    for frame_id, tracking_results in enumerate(tracking_results_all_frames):
        num_person = len(tracking_results)
        if num_person > max_instance:
            max_instance = num_person
        for result in tracking_results:
            tracklet.append(frame_id)
            final_results.append([result])

    return tracklet, final_results, max_instance


def prepare_data_with_mmpose_detection(args, frames_iter):
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []
    all_results = []
    for frame_id, img in tqdm(
            enumerate(mmcv.track_iter_progress(frames_iter))):
        pose_results_last = pose_results
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results)

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.mmpose_bbox_thr,
            format='xyxy',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_results, next_id = get_track_id(pose_results, pose_results_last,
                                             next_id)
        all_results.append(pose_results.copy())
    joints2d = []
    person_id_list = []
    wb_kps = {
        'joints2d_lhand': [],
        'joints2d_rhand': [],
        'joints2d_face': [],
    }
    frames_idx, final_results, max_instance = process_tracking_results(
        all_results)
    for results in final_results:
        joints2d.append(results[0]['keypoints'])
        person_id_list.append(results[0]['track_id'])
        wb_kps['joints2d_lhand'].append(results[0]['keypoints'][91:112])
        wb_kps['joints2d_rhand'].append(results[0]['keypoints'][112:133])
        wb_kps['joints2d_face'].append(results[0]['keypoints'][23:91])
    if Path(args.input_path).is_file():
        image_folder = osp.join(args.output_path, 'images')
        os.makedirs(image_folder, exist_ok=True)
        video_to_images(args.input_path, image_folder)
    elif Path(args.input_path).is_dir():
        image_folder = args.input_path
    return joints2d, frames_idx, wb_kps, image_folder, max_instance


def prepare_data_with_pifpaf_detection(args, frames_iter):
    max_instance = 0
    num_frames = len(frames_iter)

    # pifpaf person detection
    pp_args = copy.deepcopy(args)
    pp_args.force_complete_pose = True
    ppdecoder.configure(pp_args)
    ppnetwork.Factory.configure(pp_args)
    ppnetwork.Factory.checkpoint = pp_args.openpifpaf_checkpoint
    Predictor.configure(pp_args)
    Stream.configure(pp_args)

    Predictor.batch_size = 1
    Predictor.loader_workers = 1
    predictor = Predictor()
    if Path(args.input_path).is_file():
        image_folder = osp.join(args.output_path, 'images')
        os.makedirs(image_folder, exist_ok=True)
        video_to_images(args.input_path, image_folder)
        capture = Stream(args.input_path, preprocess=predictor.preprocess)
        capture = predictor.dataset(capture)
    elif Path(args.input_path).is_dir():
        image_folder = args.input_path
        image_file_names = sorted([
            osp.join(args.input_path, x) for x in os.listdir(args.input_path)
            if x.endswith('.png') or x.endswith('.jpg')
        ])
        capture = predictor.images(image_file_names)

    tracking_results = {}
    for preds, _, meta in tqdm(capture, total=num_frames):
        num_person = 0
        for pid, ann in enumerate(preds):
            if ann.score > args.openpifpaf_threshold:
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
    joints2d = []
    frames = []
    wb_kps = {
        'joints2d_lhand': [],
        'joints2d_rhand': [],
        'joints2d_face': [],
    }
    person_id_list = list(tracking_results.keys())
    for person_id in person_id_list:
        joints2d.extend(tracking_results[person_id]['joints2d'])
        wb_kps['joints2d_lhand'].extend(
            tracking_results[person_id]['joints2d_lhand'])
        wb_kps['joints2d_rhand'].extend(
            tracking_results[person_id]['joints2d_rhand'])
        wb_kps['joints2d_face'].extend(
            tracking_results[person_id]['joints2d_face'])

        frames.extend(tracking_results[person_id]['frames'])
    return joints2d, frames, wb_kps, image_folder, max_instance


def main(args):
    # Define model
    pymafx_config = mmcv.Config.fromfile(args.mesh_reg_config)
    pymafx_config.model['device'] = args.device
    mesh_model, _ = init_model(
        pymafx_config, args.mesh_reg_checkpoint, device=args.device.lower())
    frames_iter = prepare_frames(args.input_path)
    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device(args.device)

    if args.use_openpifpaf:
        args.device = device
        args.pin_memory = True if torch.cuda.is_available() else False
        # Prepare input
        joints2d, frames, wb_kps, image_folder, max_instance = \
            prepare_data_with_pifpaf_detection(args, frames_iter)
    else:
        joints2d, frames, wb_kps, image_folder, max_instance = \
            prepare_data_with_mmpose_detection(args, frames_iter)

    pymafx_config['data']['test']['image_folder'] = image_folder
    pymafx_config['data']['test']['frames'] = frames
    pymafx_config['data']['test']['joints2d'] = joints2d
    pymafx_config['data']['test']['wb_kps'] = wb_kps
    test_dataset = build_dataset(pymafx_config['data']['test'],
                                 dict(test_mode=True))
    bboxes_cs = test_dataset.bboxes
    frame_ids = test_dataset.frames
    dataloader = DataLoader(
        test_dataset, batch_size=args.model_batch_size, num_workers=0)

    # Run pred on each person
    with torch.no_grad():
        pred_cam, orig_height, orig_width, smplx_params = [], [], [], []

        for batch in tqdm(dataloader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
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
        bbox=bboxes_cs,
        img_width=orig_width,
        img_height=orig_height,
        bbox_format='cs')

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
    human_data = HumanData()
    smplx = {}
    smplx['fullpose'] = fullpose
    smplx['betas'] = betas
    human_data['smplx'] = smplx
    human_data.dump(osp.join(args.output_path, 'inference_result.npz'))
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
    image_file_names = sorted([
        osp.join(image_folder, x) for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])
    frame_num = len(image_file_names)
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
    if args.visualization:
        image_array = []
        for path in image_file_names:
            img = cv2.imread(path)
            image_array.append(img)
        if Path(args.input_path).is_dir():
            for i, image in enumerate(image_array):
                image_path = osp.join(
                    args.output_path,
                    image_file_names[i].split('/')[-1].split('.')[0])
                visualize_smpl_vibe(
                    poses=compressed_fullpose.reshape(-1, max_instance,
                                                      165)[i:i + 1],
                    betas=compressed_betas[i:i + 1],
                    orig_cam=compressed_cams[i:i + 1],
                    output_path=image_path,
                    image_array=image,
                    body_model_config=body_model_config,
                    resolution=(image.shape[0], image.shape[1]),
                    overwrite=True,
                )
        elif Path(args.input_path).is_file():
            visualize_smpl_vibe(
                poses=compressed_fullpose.reshape(-1, max_instance, 165),
                betas=compressed_betas,
                orig_cam=compressed_cams,
                output_path=osp.join(args.output_path, 'smplx.mp4'),
                image_array=np.array(image_array),
                body_model_config=body_model_config,
                resolution=(orig_height[0], orig_width[0]),
                overwrite=True,
            )
            shutil.rmtree(image_folder)


def init_openpifpaf(parser):
    ppnetwork.Factory.cli(parser)
    ppdecoder.cli(parser)
    Predictor.cli(parser)
    Stream.cli(parser)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
    # openpifpaf
    parser.add_argument(
        '--openpifpaf_checkpoint',
        type=str,
        default='shufflenetv2k30-wholebody',
        help='detector checkpoint for openpifpaf')
    parser.add_argument(
        '--openpifpaf_threshold',
        type=float,
        default=0.35,
        help='pifpaf detection score threshold.')
    parser.add_argument(
        '--use_openpifpaf', action='store_true', help='pifpaf detection')
    # mmpose
    parser.add_argument(
        '--mmpose_bbox_thr',
        type=float,
        default=0.8,
        help='Bounding box score threshold')
    parser.add_argument(
        '--det_config',
        default='demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
        help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint',
        default='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
        'faster_rcnn_r50_fpn_1x_coco/'
        'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        help='Checkpoint file for detection')
    parser.add_argument(
        '--pose_config',
        default='demo/mmpose_cfg/'
        'hrnet_w48_coco_wholebody_384x288_dark_plus.py',
        help='Config file for pose')
    parser.add_argument(
        '--pose_checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/'
        'hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',
        help='Checkpoint file for pose')

    parser.add_argument(
        '--input_path', type=str, default=None, help='input folder')
    parser.add_argument(
        '--output_path',
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
    if args.use_openpifpaf:
        init_openpifpaf(parser)
        args = parser.parse_args()
        assert has_openpifpaf, 'Please install openpifpaf to run the demo.'
        assert args.det_config is not None
        assert args.det_checkpoint is not None
    else:
        assert has_mmdet, 'Please install mmdet to run the demo.'
        assert has_mmpose, 'Please install mmpose to run the demo.'
        assert args.det_config is not None
        assert args.pose_config is not None
    main(args)
