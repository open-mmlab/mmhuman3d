import argparse

import mmcv
import torch
from demo.pymafx_estimate_smplx import prepare_data_with_pifpaf_detection
from openpifpaf import decoder as ppdecoder
from openpifpaf import network as ppnetwork
from openpifpaf.predictor import Predictor
from openpifpaf.stream import Stream
from torch.utils.data import DataLoader

from mmhuman3d.apis import init_model
from mmhuman3d.data.datasets import build_dataset
from mmhuman3d.utils.keypoint_utils import process_kps2d


def _init_openpifpaf(parser):
    ppnetwork.Factory.cli(parser)
    ppdecoder.cli(parser)
    Predictor.cli(parser)
    Stream.cli(parser)


def _setup_parser():
    parser = argparse.ArgumentParser()
    _init_openpifpaf(parser)

    parser.add_argument(
        '--mesh_reg_config',
        type=str,
        default='configs/pymafx/pymafx.py',
        help='Config file for mesh regression')
    parser.add_argument(
        '--mesh_reg_checkpoint',
        type=str,
        default='https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/'
        'mmhuman3d/models/pymaf_x/PyMAF-X_model_checkpoint.pth',
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
        default=0.35,
        help='pifpaf detection score threshold.')
    parser.add_argument(
        '--vid_file', type=str, default=None, help='input video path')
    parser.add_argument('--image_folder', type=str, default='demo/resources')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='output',
        help='output folder to write results')
    parser.add_argument(
        '--device', default='cpu', help='device used for testing')
    parser.add_argument('tests/test_pymafx_pipeline.py', default=True)
    args = parser.parse_args()
    return args


def test_process_kps2d():
    pred_hand_proj = torch.ones((1, 390, 2))
    kps2d = process_kps2d(pred_hand_proj, torch.ones((1, 2, 3)))
    assert kps2d.shape == (1, 390, 2)


def test_pymafx_inference():
    args = _setup_parser()
    config = mmcv.Config.fromfile(args.mesh_reg_config)
    config.model['device'] = args.device
    for bhf_mode, global_mode in [('full_body', True), ('body_hand', False)]:
        config['model']['head']['bhf_mode'] = bhf_mode
        config['model']['regressor']['bhf_mode'] = bhf_mode
        config['model']['bhf_mode'] = bhf_mode
        config['model']['backbone']['global_mode'] = global_mode
        mesh_model, _ = init_model(
            config, args.mesh_reg_checkpoint, device=args.device)
        args.device = torch.device(args.device)
        args.pin_memory = False
        # Prepare input
        joints2d, frames, wb_kps, person_id_list, image_folder, \
            _, _ = prepare_data_with_pifpaf_detection(args)
        assert len(joints2d) == 1 and joints2d[0].shape == (17, 3)
        assert wb_kps['joints2d_lhand'][0].shape == (21, 3)
        assert wb_kps['joints2d_rhand'][0].shape == (21, 3)
        assert wb_kps['joints2d_face'][0].shape == (68, 3)

        config['data']['test']['image_folder'] = image_folder
        config['data']['test']['frames'] = frames
        config['data']['test']['joints2d'] = joints2d
        config['data']['test']['person_id_list'] = person_id_list
        config['data']['test']['wb_kps'] = wb_kps
        test_dataset = build_dataset(config['data']['test'],
                                     dict(test_mode=True))

        dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)

        # Run pred on each person
        with torch.no_grad():
            orig_height, orig_width, person_ids = [], [], []

            for batch in dataloader:
                batch = {
                    k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                person_ids.extend(batch['person_id'])
                orig_height.append(batch['orig_height'])
                orig_width.append(batch['orig_width'])

                preds_dict = mesh_model.forward_test(batch)
                output = preds_dict['mesh_out'][-1]
                assert output['pred_shape'] is not None
                assert output['rotmat'] is not None
                assert output['pred_lhand_rotmat'] is not None
                assert output['pred_rhand_rotmat'] is not None
                assert output['pred_face_rotmat'][:, 0:1] is not None
                assert output['pred_face_rotmat'][:, 1:2] is not None
                assert output['pred_face_rotmat'][:, 2:3] is not None
                assert output['pred_exp'] is not None
