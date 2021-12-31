import json
import os
import os.path
from abc import ABCMeta
from collections import OrderedDict

import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import get_mapping
from mmhuman3d.core.evaluation.mpjpe import keypoint_mpjpe
from mmhuman3d.data.data_structures.human_data import HumanData
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

    def __init__(self,
                 data_prefix,
                 pipeline,
                 dataset_name,
                 ann_file,
                 test_mode=False):
        if dataset_name is not None:
            self.dataset_name = dataset_name
        self.test_mode = test_mode
        super(HybrIKHumanImageDataset, self).__init__(data_prefix, pipeline,
                                                      ann_file, test_mode)

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

    def evaluate(self, outputs, res_folder, metric='joint_error', logger=None):
        """Evaluate 3D keypoint results."""
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['joint_error']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        kpts_dict = {}
        for out in outputs:
            for (keypoints, idx) in zip(out['xyz_17'], out['image_idx']):
                kpts_dict[int(idx)] = keypoints.tolist()
        kpts = []
        for i in range(self.num_data):
            kpts.append(kpts_dict[i])
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
            pred_keypoints3d = json.load(fin)
        assert len(pred_keypoints3d) == len(self.data_infos)

        pred_keypoints3d = np.array(pred_keypoints3d)
        factor, root_idx_17 = 1, 0
        gts = self.data_infos
        if self.dataset_name == 'mpi_inf_3dhp':
            _, hp3d_idxs, _ = get_mapping('human_data', 'mpi_inf_3dhp_test')
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
            joint_mapper = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
            gt_keypoints3d_mask = np.ones(
                (len(gt_keypoints3d), len(joint_mapper)))
            if self.dataset_name == 'pw3d':
                factor = 1000

        print('Evaluation start...')
        assert len(gts) == len(pred_keypoints3d)

        pred_keypoints3d = pred_keypoints3d * (2000 / factor)
        if self.dataset_name == 'mpi_inf_3dhp':
            gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]
        # root joint alignment
        pred_keypoints3d = pred_keypoints3d - pred_keypoints3d[:, None,
                                                               root_idx_17]
        gt_keypoints3d = gt_keypoints3d - gt_keypoints3d[:, None, root_idx_17]

        if self.dataset_name == 'pw3d' or self.dataset_name == 'h36m':
            # select eval 14 joints
            pred_keypoints3d = pred_keypoints3d[:, joint_mapper, :]
            gt_keypoints3d = gt_keypoints3d[:, joint_mapper, :]

        gt_keypoints3d_mask = gt_keypoints3d_mask > 0

        mpjpe = keypoint_mpjpe(pred_keypoints3d, gt_keypoints3d,
                               gt_keypoints3d_mask)
        mpjpe_pa = keypoint_mpjpe(
            pred_keypoints3d,
            gt_keypoints3d,
            gt_keypoints3d_mask,
            alignment='procrustes')

        info_str = []
        info_str.append(('MPJPE', mpjpe * factor))
        info_str.append(('MPJPE-PA', mpjpe_pa * factor))
        return info_str
