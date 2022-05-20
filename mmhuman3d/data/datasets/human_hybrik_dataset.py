import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict
from typing import List, Optional, Union

import mmcv
import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import get_mapping
from mmhuman3d.core.evaluation import (
    keypoint_3d_auc,
    keypoint_3d_pck,
    keypoint_mpjpe,
    vertice_pve,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.utils.demo_utils import box2cs, xyxy2xywh
from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class HybrIKHumanImageDataset(BaseDataset, metaclass=ABCMeta):
    """Dataset for HybrIK training. The dataset loads raw features and apply
    specified transforms to return a dict containing the image tensors and
    other information.

    Args:

        data_prefix (str): Path to a directory where preprocessed datasets are
         held.
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_name (str): accepted names include 'h36m', 'pw3d',
         'mpi_inf_3dhp', 'coco'
        ann_file (str): Name of annotation file.
        test_mode (bool): Store True when building test dataset.
         Default: False.
    """
    # metric
    ALLOWED_METRICS = {
        'mpjpe', 'pa-mpjpe', 'pve', '3dpck', 'pa-3dpck', '3dauc', 'pa-3dauc'
    }

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: str,
                 body_model: Optional[Union[dict, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 test_mode: Optional[bool] = False):
        if dataset_name is not None:
            self.dataset_name = dataset_name
        self.test_mode = test_mode
        super(HybrIKHumanImageDataset, self).__init__(data_prefix, pipeline,
                                                      ann_file, test_mode)
        if body_model is not None:
            self.body_model = build_body_model(body_model)
        else:
            self.body_model = None

    def get_annotation_file(self):
        """Obtain annotation file path from data prefix."""
        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        self.ann_file = os.path.join(ann_prefix, self.ann_file)

    @staticmethod
    def get_3d_keypoints_vis(keypoints):
        """Get 3d keypoints and visibility mask
        Args:
            keypoints (np.ndarray): 2d (NxKx3) or 3d (NxKx4) keypoints with
             visibility. N refers to number of datapoints, K refers to number
             of keypoints.

        Returns:
            joint_img (np.ndarray): (NxKx3) 3d keypoints
            joint_vis (np.ndarray): (NxKx3) visibility mask for keypoints
        """
        keypoints, keypoints_vis = keypoints[:, :, :-1], keypoints[:, :, -1]
        num_datapoints, num_keypoints, dim = keypoints.shape
        joint_img = np.zeros((num_datapoints, num_keypoints, 3),
                             dtype=np.float32)
        joint_vis = np.zeros((num_datapoints, num_keypoints, 3),
                             dtype=np.float32)
        joint_img[:, :, :dim] = keypoints
        joint_vis[:, :, :dim] = np.tile(
            np.expand_dims(keypoints_vis, axis=2), (1, dim))
        return joint_img, joint_vis

    def load_annotations(self):
        """Load annotations."""
        self.get_annotation_file()
        data = HumanData()
        data.load(self.ann_file)

        self.image_path = data['image_path']
        self.num_data = len(self.image_path)

        self.bbox_xyxy = data['bbox_xywh']
        self.width = data['image_width']
        self.height = data['image_height']
        self.depth_factor = data['depth_factor']

        try:
            self.keypoints3d, self.keypoints3d_vis = self.get_3d_keypoints_vis(
                data['keypoints2d'])
        except KeyError:
            self.keypoints3d, self.keypoints3d_vis = self.get_3d_keypoints_vis(
                data['keypoints3d'])

        try:
            self.smpl = data['smpl']
            if 'has_smpl' not in data.keys():
                self.has_smpl = np.ones((self.num_data)).astype(np.float32)
            else:
                self.has_smpl = data['has_smpl'].astype(np.float32)
            self.thetas = self.smpl['thetas'].astype(np.float32)
            self.betas = self.smpl['betas'].astype(np.float32)

            self.keypoints3d_relative, _ = self.get_3d_keypoints_vis(
                data['keypoints3d_relative'])
            self.keypoints3d17, self.keypoints3d17_vis = \
                self.get_3d_keypoints_vis(data['keypoints3d17'])
            self.keypoints3d17_relative, _ = self.get_3d_keypoints_vis(
                data['keypoints3d17_relative'])

            if self.test_mode:
                self.keypoints3d_cam, _ = self.get_3d_keypoints_vis(
                    data['keypoints3d_cam'])
        except KeyError:
            self.has_smpl = np.zeros((self.num_data)).astype(np.float32)
            if self.test_mode:
                self.keypoints3d, self.keypoints3d_vis = \
                    self.get_3d_keypoints_vis(data['keypoints3d'])
                self.keypoints3d_cam, _ = self.get_3d_keypoints_vis(
                    data['keypoints3d_cam'])

        try:
            self.intrinsic = data['cam_param']['intrinsic']
        except KeyError:
            self.intrinsic = np.zeros((self.num_data, 3, 3))

        try:
            self.target_twist = data['phi']
            # self.target_twist_weight = np.ones_like((self.target_twist))
            self.target_twist_weight = data['phi_weight']
        except KeyError:
            self.target_twist = np.zeros((self.num_data, 23, 2))
            self.target_twist_weight = np.zeros_like((self.target_twist))

        try:
            self.root_cam = data['root_cam']
        except KeyError:
            self.root_cam = np.zeros((self.num_data, 3))

        self.data_infos = []

        for idx in range(self.num_data):
            info = {}
            info['ann_info'] = {}
            info['img_prefix'] = None
            info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                              self.dataset_name,
                                              self.image_path[idx])
            bbox_xyxy = self.bbox_xyxy[idx]
            info['bbox'] = bbox_xyxy[:4]
            bbox_xywh = xyxy2xywh(bbox_xyxy)
            center, scale = box2cs(
                bbox_xywh, aspect_ratio=1.0, bbox_scale_factor=1.25)

            info['center'] = center
            info['scale'] = scale
            info['rotation'] = 0
            info['ann_info']['dataset_name'] = self.dataset_name
            info['ann_info']['height'] = self.height[idx]
            info['ann_info']['width'] = self.width[idx]
            info['depth_factor'] = float(self.depth_factor[idx])
            info['has_smpl'] = int(self.has_smpl[idx])
            info['joint_root'] = self.root_cam[idx].astype(np.float32)
            info['intrinsic_param'] = self.intrinsic[idx].astype(np.float32)
            info['target_twist'] = self.target_twist[idx].astype(
                np.float32)  # twist_phi
            info['target_twist_weight'] = self.target_twist_weight[idx].astype(
                np.float32)
            info['keypoints3d'] = self.keypoints3d[idx]
            info['keypoints3d_vis'] = self.keypoints3d_vis[idx]

            if info['has_smpl']:
                info['pose'] = self.thetas[idx]
                info['beta'] = self.betas[idx].astype(np.float32)
                info['keypoints3d_relative'] = self.keypoints3d_relative[idx]
                info['keypoints3d17'] = self.keypoints3d17[idx]
                info['keypoints3d17_vis'] = self.keypoints3d17_vis[idx]
                info['keypoints3d17_relative'] = self.keypoints3d17_relative[
                    idx]

                if self.test_mode:
                    info['joint_relative_17'] = self.keypoints3d17_relative[
                        idx].astype(np.float32)

            else:
                if self.test_mode:
                    info['joint_relative_17'] = self.keypoints3d_cam[
                        idx].astype(np.float32)

            self.data_infos.append(info)

    def evaluate(self,
                 outputs: list,
                 res_folder: str,
                 metric: Optional[Union[str, List[str]]] = 'pa-mpjpe',
                 **kwargs: dict):
        """Evaluate 3D keypoint results.

        Args:
            outputs (list): results from model inference.
            res_folder (str): path to store results.
            metric (Optional[Union[str, List(str)]]):
                the type of metric. Default: 'pa-mpjpe'
            kwargs (dict): other arguments.
        Returns:
            dict:
                A dict of all evaluation results.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in self.ALLOWED_METRICS:
                raise ValueError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        res_dict = {}
        for out in outputs:
            target_id = out['image_idx']
            batch_size = len(out['xyz_17'])
            for i in range(batch_size):
                res_dict[int(target_id[i])] = dict(
                    keypoints=out['xyz_17'][i],
                    poses=out['smpl_pose'][i],
                    betas=out['smpl_beta'][i],
                )

        keypoints, poses, betas = [], [], []
        for i in range(self.num_data):
            keypoints.append(res_dict[i]['keypoints'])
            poses.append(res_dict[i]['poses'])
            betas.append(res_dict[i]['betas'])

        res = dict(keypoints=keypoints, poses=poses, betas=betas)
        mmcv.dump(res, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(res)
            elif _metric == 'pa-mpjpe':
                _nv_tuples = self._report_mpjpe(res, metric='pa-mpjpe')
            elif _metric == '3dpck':
                _nv_tuples = self._report_3d_pck(res)
            elif _metric == 'pa-3dpck':
                _nv_tuples = self._report_3d_pck(res, metric='pa-3dpck')
            elif _metric == '3dauc':
                _nv_tuples = self._report_3d_auc(res)
            elif _metric == 'pa-3dauc':
                _nv_tuples = self._report_3d_auc(res, metric='pa-3dauc')
            elif _metric == 'pve':
                _nv_tuples = self._report_pve(res)
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        name_value = OrderedDict(name_value_tuples)
        return name_value

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""
        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _parse_result(self, res, mode='keypoint'):
        """Parse results."""
        gts = self.data_infos
        if mode == 'vertice':
            pred_pose = torch.FloatTensor(res['poses'])
            pred_beta = torch.FloatTensor(res['betas'])
            pred_output = self.body_model(
                betas=pred_beta,
                body_pose=pred_pose[:, 1:],
                global_orient=pred_pose[:, 0].unsqueeze(1),
                pose2rot=False)
            pred_vertices = pred_output['vertices'].detach().cpu().numpy()

            gt_pose = torch.FloatTensor([gt['pose']
                                         for gt in gts]).view(-1, 72)
            gt_beta = torch.FloatTensor([gt['beta'] for gt in gts])
            gt_output = self.body_model(
                betas=gt_beta,
                body_pose=gt_pose[:, 3:],
                global_orient=gt_pose[:, :3])
            gt_vertices = gt_output['vertices'].detach().cpu().numpy()
            gt_mask = np.ones(gt_vertices.shape[:-1])
            assert len(pred_vertices) == self.num_data

            return pred_vertices * 1000., gt_vertices * 1000., gt_mask
        elif mode == 'keypoint':
            pred_keypoints3d = res['keypoints']
            assert len(pred_keypoints3d) == self.num_data
            # (B, 17, 3)
            pred_keypoints3d = np.array(pred_keypoints3d)
            factor, root_idx_17 = 1, 0

            if self.dataset_name == 'mpi_inf_3dhp':
                _, hp3d_idxs, _ = get_mapping('human_data',
                                              'mpi_inf_3dhp_test')
                gt_keypoints3d = np.array(
                    [gt['joint_relative_17'][hp3d_idxs] for gt in gts])
                joint_mapper = [
                    14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4
                ]
                gt_keypoints3d_mask = np.ones(
                    (len(gt_keypoints3d), len(joint_mapper)))
            else:
                _, h36m_idxs, _ = get_mapping('human_data', 'h36m')
                gt_keypoints3d = np.array(
                    [gt['joint_relative_17'][h36m_idxs] for gt in gts])
                joint_mapper = [
                    6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10
                ]
                gt_keypoints3d_mask = np.ones(
                    (len(gt_keypoints3d), len(joint_mapper)))
                if self.dataset_name == 'pw3d':
                    factor = 1000

            assert len(pred_keypoints3d) == self.num_data

            pred_keypoints3d = pred_keypoints3d * (2000 / factor)
            if self.dataset_name == 'mpi_inf_3dhp':
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
            # root joint alignment
            pred_keypoints3d = (
                pred_keypoints3d -
                pred_keypoints3d[:, None, root_idx_17]) * factor
            gt_keypoints3d = (gt_keypoints3d -
                              gt_keypoints3d[:, None, root_idx_17]) * factor

            if self.dataset_name == 'pw3d' or self.dataset_name == 'h36m':
                # select eval 14 joints
                pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]
                gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]

            gt_keypoints3d_mask = gt_keypoints3d_mask > 0

            return pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask

        else:
            raise NotImplementedError()

    def _report_mpjpe(self, res_file, metric='mpjpe'):
        """Cauculate mean per joint position error (MPJPE) or its variants PA-
        MPJPE.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (PA-MPJPE)
        """
        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file, mode='keypoint')

        err_name = metric.upper()
        if metric == 'mpjpe':
            alignment = 'none'
        elif metric == 'pa-mpjpe':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_mpjpe(pred_keypoints3d, gt_keypoints3d,
                               gt_keypoints3d_mask, alignment)
        info_str = [(err_name, error)]

        return info_str

    def _report_3d_pck(self, res_file, metric='3dpck'):
        """Cauculate Percentage of Correct Keypoints (3DPCK) w. or w/o
        Procrustes alignment.
        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            metric (str): Specify mpjpe variants. Supported options are:
                - ``'3dpck'``: Standard 3DPCK.
                - ``'pa-3dpck'``:
                    3DPCK after aligning prediction to groundtruth
                    via a rigid transformation (scale, rotation and
                    translation).
        """

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file, mode='keypoint')

        err_name = metric.upper()
        if metric == '3dpck':
            alignment = 'none'
        elif metric == 'pa-3dpck':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_pck(pred_keypoints3d, gt_keypoints3d,
                                gt_keypoints3d_mask, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_3d_auc(self, res_file, metric='3dauc'):
        """Cauculate the Area Under the Curve (AUC) computed for a range of
        3DPCK thresholds.
        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DMpiInf3dhpDataset.evaluate' for details.
            metric (str): Specify mpjpe variants. Supported options are:
                - ``'3dauc'``: Standard 3DAUC.
                - ``'pa-3dauc'``: 3DAUC after aligning prediction to
                    groundtruth via a rigid transformation (scale, rotation and
                    translation).
        """

        pred_keypoints3d, gt_keypoints3d, gt_keypoints3d_mask = \
            self._parse_result(res_file, mode='keypoint')

        err_name = metric.upper()
        if metric == '3dauc':
            alignment = 'none'
        elif metric == 'pa-3dauc':
            alignment = 'procrustes'
        else:
            raise ValueError(f'Invalid metric: {metric}')

        error = keypoint_3d_auc(pred_keypoints3d, gt_keypoints3d,
                                gt_keypoints3d_mask, alignment)
        name_value_tuples = [(err_name, error)]

        return name_value_tuples

    def _report_pve(self, res_file):
        """Cauculate per vertex error."""
        pred_verts, gt_verts, _ = \
            self._parse_result(res_file, mode='vertice')
        error = vertice_pve(pred_verts, gt_verts)
        return [('PVE', error)]
