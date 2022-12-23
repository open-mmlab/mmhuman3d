import os
import os.path as osp
import shutil
from argparse import ArgumentParser

import mmcv
import numpy as np
import torch

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
    smooth_process,
)
from mmhuman3d.utils.ffmpeg_utils import array_to_images
from mmhuman3d.utils.transforms import rotmat_to_aa

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def get_detection_result(args, frames_iter, mesh_model, extractor):
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    frame_id_list = []
    result_list = []
    for i, frame in enumerate(mmcv.track_iter_progress(frames_iter)):
        mmdet_results = inference_detector(person_det_model, frame)
        # keep the person class bounding boxes.
        results = process_mmdet_results(
            mmdet_results, cat_id=args.det_cat_id, bbox_thr=args.bbox_thr)

        # extract features from the input video or image sequences
        if mesh_model.cfg.model.type == 'VideoBodyModelEstimator' \
                and extractor is not None:
            results = feature_extract(
                extractor, frame, results, args.bbox_thr, format='xyxy')
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
    """Estimate smplx parameters from single-person
        images with mmdetection
    Args:
        args (object):  object of argparse.Namespace.
        frames_iter (np.ndarray,): prepared frames

    """
    mesh_model, extractor = init_model(
        args.mesh_reg_config,
        args.mesh_reg_checkpoint,
        device=args.device.lower())
    smplx_results = dict(
        global_orient=[],
        body_pose=[],
        betas=[],
        left_hand_pose=[],
        right_hand_pose=[],
        jaw_pose=[],
        expression=[])
    pred_cams, bboxes_xyxy = [], []

    frame_id_list, result_list = get_detection_result(args, frames_iter,
                                                      mesh_model, extractor)

    frame_num = len(frame_id_list)

    for i, result in enumerate(mmcv.track_iter_progress(result_list)):
        frame_id = frame_id_list[i]
        if mesh_model.cfg.model.type == 'SMPLXImageBodyModelEstimator':
            mesh_results = inference_image_based_model(
                mesh_model,
                frames_iter[frame_id],
                result,
                bbox_thr=args.bbox_thr,
                format='xyxy')
        else:
            raise Exception(
                f'{mesh_model.cfg.model.type} is not supported yet')
        for key in smplx_results:
            smplx_results[key].append(
                mesh_results[0]['param'][key].cpu().numpy())
        pred_cams.append(mesh_results[0]['camera'])
        bboxes_xyxy.append(mesh_results[0]['bbox'])

    for key in smplx_results:
        smplx_results[key] = np.array(smplx_results[key])
    pred_cams = np.array(pred_cams)
    bboxes_xyxy = np.array(bboxes_xyxy)
    # release GPU memory
    del mesh_model
    del extractor
    torch.cuda.empty_cache()

    # smooth
    if args.smooth_type is not None:
        for key in smplx_results:
            if key not in ['betas', 'expression']:
                dim = smplx_results[key].shape[1]
                smplx_results[key] = smooth_process(
                    smplx_results[key].reshape(frame_num, -1, dim, 9),
                    smooth_type=args.smooth_type).reshape(
                        frame_num, dim, 3, 3)
        pred_cams = smooth_process(
            pred_cams[:, np.newaxis],
            smooth_type=args.smooth_type).reshape(frame_num, 3)

    if smplx_results['body_pose'].shape[1:] == (21, 3, 3):
        for key in smplx_results:
            if key not in ['betas', 'expression']:
                smplx_results[key] = rotmat_to_aa(smplx_results[key])
    else:
        raise Exception('Wrong shape of `smpl_pose`')
    fullpose = np.concatenate((
        smplx_results['global_orient'].reshape(frame_num, 1, 3),
        smplx_results['body_pose'].reshape(frame_num, 21, 3),
        smplx_results['jaw_pose'].reshape(frame_num, 1, 3),
        np.zeros((frame_num, 2, 3), dtype=smplx_results['jaw_pose'].dtype),
        smplx_results['left_hand_pose'].reshape(frame_num, 15, 3),
        smplx_results['right_hand_pose'].reshape(frame_num, 15, 3),
    ),
                              axis=1)

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
        human_data = HumanData()
        smplx = {}
        smplx['fullpose'] = fullpose
        smplx['betas'] = np.array(smplx_results['betas']).reshape((-1, 10))
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

    if args.single_person_demo:
        single_person_with_mmdet(args, frames_iter)
    else:
        raise ValueError('Only supports single_person_demo')


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
        '--palette', type=str, default='segmentation', help='Color theme')
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
        '--smooth_type',
        type=str,
        default=None,
        help='Smooth the data through the specified type.'
        'Select in [oneeuro,gaus1d,savgol].')
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
