import os
import os.path
from abc import ABCMeta, abstractmethod

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


class BaseHumanImageDataset(BaseDataset, metaclass=ABCMeta):

    @abstractmethod
    def get_annotation_file(self):
        pass

    def load_annotations(self):

        self.get_annotation_file()
        data = np.load(self.ann_file, allow_pickle=True)

        self.image_path = data['image_path']
        self.bbox_xywh = data['bbox_xywh']
        num_data = len(self.image_path)

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
            self.has_smpl = np.ones((num_data))
        except KeyError:
            self.smpl = {
                'body_pose': np.zeros((num_data, 23, 3)),
                'global_orient': np.zeros((num_data, 3)),
                'betas': np.zeros((num_data, 10)),
                'transl': np.zeros((num_data, 3))
            }
            self.has_smpl = np.zeros((num_data))

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
            self.gender = data['gender']
        except KeyError:
            self.meta = None
            self.gender = None

        try:
            self.mask = data['mask']
        except KeyError:
            self.mask = np.zeros((num_data, 144, 1))

        data_infos = []

        for idx in range(num_data):
            info = {}
            info['img_prefix'] = None
            info['image_path'] = os.path.join(self.data_prefix, 'datasets',
                                              self.dataset_name,
                                              self.image_path[idx])
            info['bbox_xywh'] = self.bbox_xywh[idx]
            info['keypoints2d'] = self.keypoints2d[idx]
            info['keypoints3d'] = self.keypoints3d[idx]
            for k, v in self.smpl.items():
                info['smpl_' + k] = v[idx]
            info['has_smpl'] = self.has_smpl[idx]

            for k, v in self.smplx.items():
                info['smplx_' + k] = v[idx]
            info['has_smplx'] = self.has_smplx[idx]

            info['mask'] = self.mask[idx]

            data_infos.append(info)

        return data_infos


@DATASETS.register_module()
class PW3D(BaseHumanImageDataset):

    dataset_name = '3dpw'

    def get_annotation_file(self):
        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        if self.ann_file is not None:
            self.ann_file = os.path.join(ann_prefix, self.ann_file)
        elif not self.test_mode:
            self.ann_file = os.path.join(ann_prefix, '3dpw_train.npz')
        else:
            self.ann_file = os.path.join(ann_prefix, '3dpw_test.npz')
