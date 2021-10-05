import os
from abc import ABCMeta

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class MeshDataset(BaseDataset, metaclass=ABCMeta):

    def __init__(self,
                 data_prefix,
                 pipeline,
                 dataset_name,
                 ann_file=None,
                 test_mode=False):
        if dataset_name is not None:
            self.dataset_name = dataset_name
        super(MeshDataset, self).__init__(data_prefix, pipeline, ann_file,
                                          test_mode)

    def get_annotation_file(self):
        ann_prefix = os.path.join(self.data_prefix, 'preprocessed_datasets')
        self.ann_file = os.path.join(ann_prefix, self.ann_file)

    def load_annotations(self):

        self.get_annotation_file()
        data = np.load(self.ann_file, allow_pickle=True)

        self.smpl = data['smpl'].item()
        num_data = self.smpl['global_orient'].shape[0]
        if 'transl' not in self.smpl:
            self.smpl['transl'] = np.zeros((num_data, 3))
        self.has_smpl = np.ones((num_data))

        data_infos = []

        for idx in range(num_data):
            info = {}
            for k, v in self.smpl.items():
                info['smpl_' + k] = v[idx]

            data_infos.append(info)

        return data_infos
