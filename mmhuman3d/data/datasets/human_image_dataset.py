import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict

import numpy as np
import torch

from mmhuman3d.core.evaluation.mpjpe import keypoint_mpjpe
from mmhuman3d.models.builder import build_body_model
from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class HumanImageDataset(BaseDataset, metaclass=ABCMeta):

    def __init__(self,
                 data_prefix,
                 pipeline,
                 dataset_name,
                 body_model=None,
                 ann_file=None,
                 test_mode=False):
        super(HumanImageDataset,
              self).__init__(data_prefix, pipeline, ann_file, test_mode,
                             dataset_name)
        if body_model is not None:
            self.body_model = build_body_model(body_model)
        else:
            self.body_model = None

    def get_annotation_file(self):
        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        self.ann_file = os.path.join(ann_prefix, self.ann_file)

    def load_annotations(self):

        self.get_annotation_file()
        data = np.load(self.ann_file, allow_pickle=True)

        self.image_path = data['image_path']

        num_data = len(self.image_path)

        try:
            self.bbox_xywh = data['bbox_xywh']
        except KeyError:
            self.bbox_xywh = np.zeros((num_data, 4))

        try:
            self.config = data['config']
        except KeyError:
            self.config = None

        try:
            self.keypoints2d = data['keypoints2d']
        except KeyError:
            self.keypoints2d = np.zeros((num_data, 144, 3))

        try:
            self.keypoints3d = data['keypoints3d']
        except KeyError:
            self.keypoints3d = np.zeros((num_data, 144, 4))

        try:
            self.smpl = data['smpl'].item()
            if 'transl' not in self.smpl:
                self.smpl['transl'] = np.zeros((num_data, 3))
            if 'has_smpl' not in data.keys():
                self.has_smpl = np.ones((num_data)).astype(np.float32)
            else:
                self.has_smpl = data['has_smpl'].astype(np.float32)
        except KeyError:
            self.smpl = {
                'body_pose': np.zeros((num_data, 23, 3)),
                'global_orient': np.zeros((num_data, 3)),
                'betas': np.zeros((num_data, 10)),
                'transl': np.zeros((num_data, 3))
            }
            self.has_smpl = np.zeros((num_data)).astype(np.float32)

        try:
            self.smplx = data['smplx'].item()
            self.has_smplx = np.ones((num_data))
        except KeyError:
            self.smplx = {
                'body_pose': np.zeros((num_data, 21, 3)),
                'global_orient': np.zeros((num_data, 3)),
                'betas': np.zeros((num_data, 10)),
                'transl': np.zeros((num_data, 3)),
                'left_hand_pose': np.zeros((num_data, 15, 3)),
                'right_hand_pose': np.zeros((num_data, 15, 3)),
                'expression': np.zeros((num_data, 10)),
                'leye_pose': np.zeros((num_data, 3)),
                'reye_pose': np.zeros((num_data, 3)),
                'jaw_pose': np.zeros((num_data, 3))
            }
            self.has_smplx = np.zeros((num_data))

        try:
            self.meta = data['meta'].item()
            self.gender = self.meta['gender']
        except KeyError:
            self.meta = None
            self.gender = None

        try:
            self.mask = data['mask']
        except KeyError:
            self.mask = np.zeros((144))

        try:
            self.keypoints2d_mask = data['keypoints2d_mask']
        except KeyError:
            self.keypoints2d_mask = np.zeros((144))

        try:
            self.keypoints3d_mask = data['keypoints3d_mask']
        except KeyError:
            self.keypoints3d_mask = np.zeros((144))

        data_infos = []
        for idx in range(num_data):
            info = {}

            info['dataset_name'] = self.dataset_name
            info['sample_idx'] = idx

            info['img_prefix'] = None
            info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                              self.dataset_name,
                                              self.image_path[idx])
            info['bbox_xywh'] = self.bbox_xywh[idx]
            x, y, w, h = self.bbox_xywh[idx]
            info['center'] = np.array([x + w / 2, y + h / 2])
            info['scale'] = np.array([w, h])
            info['keypoints2d'] = self.keypoints2d[idx][:, :2]
            info['keypoints3d'] = self.keypoints3d[idx][:, :3]
            for k, v in self.smpl.items():
                info['smpl_' + k] = v[idx]
            info['has_smpl'] = int(self.has_smpl[idx])

            for k, v in self.smplx.items():
                info['smplx_' + k] = v[idx]
            info['has_smplx'] = self.has_smplx[idx]

            info['mask'] = self.mask
            info['keypoints2d_mask'] = self.keypoints2d[idx][:, 2]
            info['keypoints3d_mask'] = self.keypoints3d[idx][:, 3]

            data_infos.append(info)

        return data_infos

    def evaluate(self, outputs, res_folder, metric='joint_error', logger=None):
        """Evaluate 3D keypoint results."""
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['joint_error']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        kpts = []
        for out in outputs:
            for (keypoints, image_path) in zip(out['keypoints_3d'],
                                               out['image_path']):
                kpts.append({
                    'keypoints': keypoints.tolist(),
                    'image': image_path,
                })

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file)
        name_value = OrderedDict(info_str)
        return name_value

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self, res_file):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.data_infos)

        pred_keypoints3d = [pred['keypoints'] for pred in preds]
        pred_keypoints3d = np.array(pred_keypoints3d)

        if self.dataset_name == 'pw3d':
            betas = []
            body_pose = []
            global_orient = []
            gender = []
            for idx, item in enumerate(self.data_infos):
                betas.append(item['smpl_betas'])
                body_pose.append(item['smpl_body_pose'])
                global_orient.append(item['smpl_global_orient'])
                if self.gender[idx] == 'm':
                    gender.append(0)
                else:
                    gender.append(1)
            betas = torch.FloatTensor(betas)
            body_pose = torch.FloatTensor(body_pose).view(-1, 69)
            global_orient = torch.FloatTensor(global_orient)
            gender = torch.Tensor(gender)
            gt_output = self.body_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                gender=gender)
            gt_keypoints3d = gt_output['joints'].numpy()
            gt_keypoints3d_mask = np.ones((len(preds), 24))
        elif self.dataset_name == 'h36m':
            gt_keypoints3d = [item['keypoints3d'] for item in self.data_infos]
            gt_keypoints3d_mask = \
                [item['keypoints3d_mask'] for item in self.data_infos]
            gt_keypoints3d = np.array(gt_keypoints3d)
            gt_keypoints3d_mask = np.array(gt_keypoints3d_mask)
        else:
            raise NotImplementedError()

        # we only evaluate on 14 lsp joints
        joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
        pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]
        pred_pelvis = (pred_keypoints3d[:, 2] + pred_keypoints3d[:, 3]) / 2
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]

        gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
        gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2
        gt_keypoints3d = gt_keypoints3d - gt_pelvis[:, None, :]
        gt_keypoints3d_mask = gt_keypoints3d_mask[:, joint_mapper] > 0

        mpjpe = keypoint_mpjpe(pred_keypoints3d, gt_keypoints3d,
                               gt_keypoints3d_mask)
        mpjpe_pa = keypoint_mpjpe(
            pred_keypoints3d,
            gt_keypoints3d,
            gt_keypoints3d_mask,
            alignment='procrustes')

        info_str = []
        info_str.append(('MPJPE', mpjpe * 1000))
        info_str.append(('MPJPE-PA', mpjpe_pa * 1000))
        return info_str
