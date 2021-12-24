import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict
from typing import Any, Optional, Union

import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.core.evaluation.mpjpe import keypoint_mpjpe
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.builder import build_body_model
from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class HumanImageDataset(BaseDataset, metaclass=ABCMeta):
    """Human Image Dataset.

    Args:
        data_prefix (str): the prefix of data path.
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmhuman3d.datasets.pipelines`.
        dataset_name (str | None): the name of dataset. It is used to
            identify the type of evaluation metric. Default: None.
        body_model (dict | None, optional): the config for body model,
            which will be used to generate meshes and keypoints.
            Default: None.
        ann_file (str | None, optional): the annotation file. When ann_file
            is str, the subclass is expected to read from the ann_file.
            When ann_file is None, the subclass is expected to read
            according to data_prefix.
        convention (str, optional): keypoints convention. Keypoints will be
            converted from "human_data" to the given one.
            Default: "human_data"
        test_mode (bool, optional): in train mode or test mode.
            Default: False.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: str,
                 body_model: Optional[Union[dict, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 convention: Optional[str] = 'human_data',
                 test_mode: Optional[bool] = False):
        self.convention = convention
        self.num_keypoints = get_keypoint_num(convention)
        super(HumanImageDataset,
              self).__init__(data_prefix, pipeline, ann_file, test_mode,
                             dataset_name)
        if body_model is not None:
            self.body_model = build_body_model(body_model)
        else:
            self.body_model = None

    def get_annotation_file(self):
        """Get path of the annotation file."""
        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        self.ann_file = os.path.join(ann_prefix, self.ann_file)

    def load_annotations(self):
        """Load annotation from the annotation file.

        Here we simply use :obj:`HumanData` to parse the annotation.
        """
        self.get_annotation_file()
        # change keypoint from 'human_data' to the given convention
        self.human_data = HumanData.fromfile(self.ann_file)
        if self.human_data.check_keypoints_compressed():
            self.human_data.decompress_keypoints()
        if 'keypoints3d' in self.human_data:
            keypoints3d = self.human_data['keypoints3d']
            assert 'keypoints3d_mask' in self.human_data
            keypoints3d_mask = self.human_data['keypoints3d_mask']
            keypoints3d, keypoints3d_mask = \
                convert_kps(
                    keypoints3d,
                    src='human_data',
                    dst=self.convention,
                    mask=keypoints3d_mask)
            self.human_data.__setitem__('keypoints3d', keypoints3d)
            self.human_data.__setitem__('keypoints3d_mask', keypoints3d_mask)
        if 'keypoints2d' in self.human_data:
            keypoints2d = self.human_data['keypoints2d']
            assert 'keypoints2d_mask' in self.human_data
            keypoints2d_mask = self.human_data['keypoints2d_mask']
            keypoints2d, keypoints2d_mask = \
                convert_kps(
                    keypoints2d,
                    src='human_data',
                    dst=self.convention,
                    mask=keypoints2d_mask)
            self.human_data.__setitem__('keypoints2d', keypoints2d)
            self.human_data.__setitem__('keypoints2d_mask', keypoints2d_mask)
        self.num_data = self.human_data.temporal_len

    def prepare_raw_data(self, idx: int):
        """Get item from self.human_data."""
        info = {}
        info['img_prefix'] = None
        image_path = self.human_data['image_path'][idx]
        info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                          self.dataset_name, image_path)
        info['dataset_name'] = self.dataset_name
        info['sample_idx'] = idx
        if 'bbox_xywh' in self.human_data:
            info['bbox_xywh'] = self.human_data['bbox_xywh'][idx]
            x, y, w, h, s = info['bbox_xywh']
            cx = x + w / 2
            cy = y + h / 2
            w = h = max(w, h)
            info['center'] = np.array([cx, cy])
            info['scale'] = np.array([w, h])
        else:
            info['bbox_xywh'] = np.zeros((5))
            info['center'] = np.zeros((2))
            info['scale'] = np.zeros((2))

        # in later modules, we will check validity of each keypoint by
        # its confidence. Therefore, we do not need the mask of keypoints.

        if 'keypoints2d' in self.human_data:
            info['keypoints2d'] = self.human_data['keypoints2d'][idx]
        else:
            info['keypoints2d'] = np.zeros((self.num_keypoints, 3))
        if 'keypoints3d' in self.human_data:
            info['keypoints3d'] = self.human_data['keypoints3d'][idx]
        else:
            info['keypoints3d'] = np.zeros((self.num_keypoints, 4))

        if 'smpl' in self.human_data:
            smpl_dict = self.human_data['smpl']
        else:
            smpl_dict = {}

        if 'smpl' in self.human_data:
            if 'has_smpl' in self.human_data:
                info['has_smpl'] = int(self.human_data['has_smpl'][idx])
            else:
                info['has_smpl'] = 1
        else:
            info['has_smpl'] = 0
        if 'body_pose' in smpl_dict:
            info['smpl_body_pose'] = smpl_dict['body_pose'][idx]
        else:
            info['smpl_body_pose'] = np.zeros((23, 3))

        if 'global_orient' in smpl_dict:
            info['smpl_global_orient'] = smpl_dict['global_orient'][idx]
        else:
            info['smpl_global_orient'] = np.zeros((3))

        if 'betas' in smpl_dict:
            info['smpl_betas'] = smpl_dict['betas'][idx]
        else:
            info['smpl_betas'] = np.zeros((10))

        if 'transl' in smpl_dict:
            info['smpl_transl'] = smpl_dict['transl'][idx]
        else:
            info['smpl_transl'] = np.zeros((3))

        return info

    def prepare_data(self, idx: int):
        """Generate and transform data."""
        info = self.prepare_raw_data(idx)
        return self.pipeline(info)

    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[str] = 'joint_error'):
        """Evaluate 3D keypoint results.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            metric (str): the type of metric. Default: 'joint_error'

        Returns:
            dict:
                A dict of all evaluation results.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['joint_error']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        # for keeping correctness during multi-gpu test, we sort all results
        kpts_dict = {}
        for out in outputs:
            for (keypoints, idx) in zip(out['keypoints_3d'], out['image_idx']):
                kpts_dict[int(idx)] = keypoints.tolist()
        kpts = []
        for i in range(self.num_data):
            kpts.append(kpts_dict[i])
        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file)
        name_value = OrderedDict(info_str)
        return name_value

    @staticmethod
    def _write_keypoint_results(keypoints: Any, res_file: str):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self, res_file: str):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """

        with open(res_file, 'r') as fin:
            pred_keypoints3d = json.load(fin)
        assert len(pred_keypoints3d) == self.num_data

        pred_keypoints3d = np.array(pred_keypoints3d)
        if self.dataset_name == 'pw3d':
            betas = []
            body_pose = []
            global_orient = []
            gender = []
            smpl_dict = self.human_data['smpl']
            for idx in range(self.num_data):
                betas.append(smpl_dict['betas'][idx])
                body_pose.append(smpl_dict['body_pose'][idx])
                global_orient.append(smpl_dict['global_orient'][idx])
                if self.human_data['meta']['gender'][idx] == 'm':
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
            gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
            gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 24))
        elif self.dataset_name == 'h36m':
            gt_keypoints3d = self.human_data['keypoints3d'][:, :, :3]
            gt_keypoints3d_mask = np.ones((len(pred_keypoints3d), 17))
        else:
            raise NotImplementedError()
        # SMPL_49 only!
        if gt_keypoints3d.shape[1] == 49:
            assert pred_keypoints3d.shape[1] == 49

            gt_keypoints3d = gt_keypoints3d[:, 25:, :]
            pred_keypoints3d = pred_keypoints3d[:, 25:, :]

            joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
            gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
            pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

            # we only evaluate on 14 lsp joints
            pred_pelvis = (pred_keypoints3d[:, 2] + pred_keypoints3d[:, 3]) / 2
            gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

        # H36M for testing!
        elif gt_keypoints3d.shape[1] == 17:
            assert pred_keypoints3d.shape[1] == 17

            H36M_TO_J17 = [
                6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9
            ]
            H36M_TO_J14 = H36M_TO_J17[:14]
            joint_mapper = H36M_TO_J14

            pred_pelvis = pred_keypoints3d[:, 0]
            gt_pelvis = gt_keypoints3d[:, 0]

            gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
            pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

        # keypoint 24
        elif gt_keypoints3d.shape[1] == 24:
            assert pred_keypoints3d.shape[1] == 24

            joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
            gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
            pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]

            # we only evaluate on 14 lsp joints
            pred_pelvis = (pred_keypoints3d[:, 2] + pred_keypoints3d[:, 3]) / 2
            gt_pelvis = (gt_keypoints3d[:, 2] + gt_keypoints3d[:, 3]) / 2

        else:
            pass
        pred_keypoints3d = pred_keypoints3d - pred_pelvis[:, None, :]
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
