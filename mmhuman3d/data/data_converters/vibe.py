import os
from collections import OrderedDict
from typing import List

import mmcv
import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.datasets.pipelines import Compose
from mmhuman3d.models import build_architecture
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class VibeConverter(BaseModeConverter):
    """VIBE datasets `VIBE: Video Inference for Human Body Pose and Shape
    Estimation' CVPR`2020 More details can be found in the `paper.

    <https://arxiv.org/pdf/1912.05656.pdf>`__.

    Args:
        modes (list):  'mpi_inf_3dhp' and/or 'pw3d' for accepted modes
    """

    ACCEPTED_MODES = ['mpi_inf_3dhp', 'pw3d']

    def __init__(self, modes: List = [], pretrained_ckpt: dict = None) -> None:
        super(VibeConverter, self).__init__(modes)
        self.mapping_dict = {
            'mpi_inf_3dhp': 'mpi_inf_3dhp_train.npz',
            'pw3d': '3dpw_test.npz'
        }
        self.img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        self.transforms = Compose([
            dict(type='LoadImageFromFile'),
            dict(type='MeshAffine', img_res=224),
            dict(type='Normalize', **self.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
        ])
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = 128
        self.pretrained_ckpt = pretrained_ckpt

    @staticmethod
    def init_model(model_config, ckpt_path, device):
        config = mmcv.Config.fromfile(model_config)
        # modify config to support single gpu/cpu
        config.model.backbone.norm_cfg = {'type': 'BN', 'requires_grad': True}
        model = build_architecture(config.model)

        head_keys = [
            'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias',
            'decpose.weight', 'decpose.bias', 'decshape.weight',
            'decshape.bias', 'deccam.weight', 'deccam.bias'
        ]

        def key_transformation(old_key):
            if 'extractor' in old_key:
                old_key = old_key.replace('extractor', 'backbone')
            if old_key in head_keys:
                old_key = 'head.' + old_key
            return old_key

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path)['state_dict']
            new_state_dict = OrderedDict()

            for key, value in state_dict.items():
                new_key = key_transformation(key)
                new_state_dict[new_key] = value

            model.load_state_dict(new_state_dict, strict=False)
        return model.to(device)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where spin preprocessed
            npz files are stored
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask,stored in HumanData() format. keypoints3d,
                keypoints3d_mask, smpl are added if available.

        """
        # use HumanData to store all data
        human_data = HumanData()

        image_path_, keypoints2d_, bbox_xywh_ = [], [], []

        seq_file = self.mapping_dict[mode]
        seq_path = os.path.join(dataset_path, seq_file)
        data = np.load(seq_path)

        image_path_ = data['imgname']
        num_data = len(image_path_)

        if mode == 'mpi_inf_3dhp':
            image_path_ = np.array([
                s.replace('imageFrames/', '').replace('frame_', '')
                for s in data['imgname']
            ])

        # center scale to bbox
        w = h = data['scale'] * 200
        x = data['center'][:, 0] - w / 2
        y = data['center'][:, 1] - h / 2

        bbox_xywh_ = np.column_stack((x, y, w, h))

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])

        if 'openpose' in data:
            keypoints2d_ = np.hstack([data['openpose'], data['part']])
            keypoints2d_ = np.array(keypoints2d_).reshape((-1, 49, 3))
            keypoints2d_, keypoints2d_mask = convert_kps(
                keypoints2d_, 'smpl_49', 'human_data')
            human_data['keypoints2d_mask'] = keypoints2d_mask
            human_data['keypoints2d'] = keypoints2d_

        if 'S' in data:
            keypoints3d_ = data['S']
            keypoints3d_ = np.array(keypoints3d_).reshape((-1, 24, 4))
            keypoints3d_, keypoints3d_mask = convert_kps(
                keypoints3d_, 'smpl_24', 'human_data')
            human_data['keypoints3d_mask'] = keypoints3d_mask
            human_data['keypoints3d'] = keypoints3d_

        if 'pose' in data:
            smpl = {}
            smpl['body_pose'] = np.array(data['pose'][:, 3:]).reshape(
                (-1, 23, 3))
            smpl['global_orient'] = np.array(data['pose'][:, :3]).reshape(
                (-1, 3))
            smpl['betas'] = np.array(data['shape']).reshape((-1, 10))
            human_data['smpl'] = smpl
            human_data['has_smpl'] = np.ones(num_data)

        if 'gender' in data:
            meta = {}
            meta['gender'] = data['gender']
            human_data['meta'] = meta

        # get features
        model_config = 'configs/hmr/resnet50_hmr_pw3d.py'
        model = self.init_model(model_config, self.pretrained_ckpt,
                                self.device)
        model.eval()
        img_root_path = os.path.join(dataset_path, '..', mode)
        video = []
        for center, scale, img_name in tqdm(
                zip(data['center'], data['scale'], image_path_),
                total=num_data):
            results = {}
            results['center'] = center
            results['scale'] = np.array([scale, scale]) * 200
            results['rotation'] = 0.
            results['img_prefix'] = img_root_path
            results['image_path'] = img_name
            transformed_results = self.transforms(results)
            video.append(transformed_results['img'].unsqueeze(0))

        video = torch.cat(video)
        features = []
        # split video into batches of frames
        img_batch_list = torch.split(video, self.batch_size)
        with torch.no_grad():
            for img_batch in img_batch_list:
                pred = model.backbone(img_batch.to(self.device))
                pred = pred[0].mean(dim=-1).mean(dim=-1)
                features.append(pred.cpu())
                del pred, img_batch
        features = torch.cat(features, dim=0).numpy()

        # store
        human_data['image_path'] = image_path_.tolist()
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['config'] = mode
        human_data['features'] = features
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        seq_file = seq_file.replace('3dpw', 'pw3d')
        out_file = os.path.join(out_path, f'vibe_{seq_file}')
        human_data.dump(out_file)
