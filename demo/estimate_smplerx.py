import os
import os.path as osp
import shutil
from argparse import ArgumentParser

import mmcv
from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict

import numpy as np
import torch
from collections import OrderedDict

from mmhuman3d.apis import (
    feature_extract,
    inference_image_based_model,
    init_model,
)
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_hmr
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.utils.demo_utils import (
    prepare_frames,
    process_mmdet_results,
)
from mmhuman3d.utils.ffmpeg_utils import array_to_images

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False



def get_detection_result(args, frames_iter):
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    frame_id_list = []
    result_list = []
    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        mmdet_results = inference_detector(person_det_model, frame)
        # keep the person class bounding boxes.
        results = process_mmdet_results(
            mmdet_results, cat_id=args.det_cat_id, bbox_thr=args.bbox_thr)
        # drop the frame with no detected results
        if results == []:
            continue
        # vis bboxes
        if args.draw_bbox:
            bboxes = [res['bbox'] for res in results]
            bboxes = np.vstack(bboxes)
            mmcv.imshow_bboxes(
                frame, bboxes, top_k=-1, thickness=2, show=False)
        frame_id_list.append(i)
        result_list.append(results)

    return frame_id_list, result_list


def single_person_with_mmdet(args, frames_iter):
    """Estimate smplx parameters with mmdetection
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames

    """
    smplx_results = dict(
        global_orient=[],
        body_pose=[],
        betas=[],
        left_hand_pose=[],
        right_hand_pose=[],
        jaw_pose=[],
        expression=[])
    pred_cams, bboxes_xyxy = [], []

    frame_id_list, result_list = get_detection_result(args, frames_iter)

    frame_num = len(frame_id_list)
    mesh_model, extractor = init_model(
        args.mesh_reg_config,
        device=args.device.lower())

    ckpt = torch.load(args.mesh_reg_checkpoint)
    new_state_dict = OrderedDict()
    for k, v in ckpt['network'].items():
        k = k.replace('module.', '').replace('backbone', 'encoder').replace('body_rotation_net', 'body_regressor').replace(
            'hand_rotation_net', 'hand_regressor')
        new_state_dict[k] = v
    mesh_model.load_state_dict(new_state_dict, strict=False)

    for i, result in enumerate(mmcv.track_iter_progress(result_list)):
        frame_id = frame_id_list[i]
        mesh_results = inference_image_based_model(
            mesh_model,
            frames_iter[frame_id],
            result,
            bbox_thr=args.bbox_thr,
            format='xyxy')
        smplx_results['global_orient'].append(mesh_results[0]['smplx_root_pose'].reshape(-1,3).cpu().numpy())
        smplx_results['body_pose'].append(mesh_results[0]['smplx_body_pose'].reshape(-1,3).cpu().numpy())
        smplx_results['betas'].append(mesh_results[0]['smplx_shape'].reshape(-1,10).cpu().numpy())
        smplx_results['left_hand_pose'].append(mesh_results[0]['smplx_lhand_pose'].reshape(-1,3).cpu().numpy())
        smplx_results['right_hand_pose'].append(mesh_results[0]['smplx_rhand_pose'].reshape(-1,3).cpu().numpy())
        smplx_results['jaw_pose'].append(mesh_results[0]['smplx_jaw_pose'].reshape(-1,3).cpu().numpy())
        smplx_results['expression'].append(mesh_results[0]['smplx_expr'].reshape(-1,10).cpu().numpy())

        pred_cams.append(mesh_results[0]['cam_trans'].reshape(-1,3).cpu().numpy())
        bboxes_xyxy.append(mesh_results[0]['bbox'])

    for key in smplx_results:
        smplx_results[key] = np.array(smplx_results[key])
    pred_cams = np.array(pred_cams)
    bboxes_xyxy = np.array(bboxes_xyxy)
    # release GPU memory
    del mesh_model
    torch.cuda.empty_cache()

    fullpose = np.concatenate(
        (
            smplx_results['global_orient'],
            smplx_results['body_pose'],
            smplx_results['jaw_pose'],
            # Use zero for leye_pose and reye_pose
            np.zeros((frame_num, 2, 3), dtype=smplx_results['jaw_pose'].dtype),
            smplx_results['left_hand_pose'],
            smplx_results['right_hand_pose'],
        ),
        axis=1)

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
        human_data = HumanData()
        smplx = {}
        smplx['fullpose'] = fullpose
        smplx['betas'] = smplx_results['betas']
        human_data['smplx'] = smplx
        human_data['pred_cams'] = pred_cams
        human_data.dump(osp.join(args.output, 'inference_result.npz'))

    if args.show_path is not None:
        frames_folder = osp.join(args.show_path, 'images')
        os.makedirs(frames_folder, exist_ok=True)
        array_to_images(
            np.array(frames_iter)[frame_id_list], output_folder=frames_folder)
        # create body model
        body_model_config = dict(
            type='smplx',
            num_betas=10,
            use_face_contour=True,
            use_pca=False,
            flat_hand_mean=True,
            model_path=args.body_model_dir,
            keypoint_src='smplx',
            keypoint_dst='smplx',
        )
        visualize_smpl_hmr(
            poses=fullpose.reshape(-1, 1, 165),
            betas=smplx_results['betas'],
            cam_transl=pred_cams,
            bbox=bboxes_xyxy,
            output_path=os.path.join(args.show_path, 'smplx.mp4'),
            render_choice=args.render_choice,
            resolution=frames_iter[0].shape[:2],
            origin_frames=frames_folder,
            body_model_config=body_model_config,
            overwrite=True)
        shutil.rmtree(frames_folder)

def main(args):

    # prepare input
    frames_iter = prepare_frames(args.input_path)

    single_person_with_mmdet(args, frames_iter)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--mesh_reg_config',
        type=str,
        default='configs/expose/expose.py',
        help='Config file for mesh regression')
    parser.add_argument(
        '--mesh_reg_checkpoint',
        type=str,
        default='data/pretrained_models/expose-d9d5dbf7_20220708.pth',
        help='Checkpoint file for mesh regression')
    parser.add_argument(
        '--single_person_demo',
        action='store_true',
        help='Single person demo with MMDetection')
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
        '--det_cat_id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')

    parser.add_argument(
        '--body_model_dir',
        type=str,
        default='data/body_models/',
        help='Body models file path')
    parser.add_argument(
        '--input_path',
        type=str,
        default='demo/resources/single_person_demo.mp4',
        help='Input path')
    parser.add_argument(
        '--output',
        type=str,
        default='demo_result',
        help='directory to save output result file')
    parser.add_argument(
        '--show_path',
        type=str,
        default='demo_result',
        help='directory to save rendered images or video')
    parser.add_argument(
        '--render_choice',
        type=str,
        default='hq',
        help='Render choice parameters')
    parser.add_argument(
        '--bbox_thr',
        type=float,
        default=0.99,
        help='Bounding box score threshold')
    parser.add_argument(
        '--draw_bbox',
        action='store_true',
        help='Draw a bbox for each detected instance')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()

    if args.single_person_demo:
        assert has_mmdet, 'Please install mmdet to run the demo.'
        assert args.det_config is not None
        assert args.det_checkpoint is not None
    main(args)
