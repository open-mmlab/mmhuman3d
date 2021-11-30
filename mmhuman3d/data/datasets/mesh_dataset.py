import os
from abc import ABCMeta
from typing import Optional, Union

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class MeshDataset(BaseDataset, metaclass=ABCMeta):
    """Mesh Dataset. This dataset only contains smpl data.

    Args:
        data_prefix (str): the prefix of data path.
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmhuman3d.datasets.pipelines`.
        dataset_name (str | None): the name of dataset. It is used to
            identify the type of evaluation metric. Default: None.
        ann_file (str | None, optional): the annotation file. When ann_file
            is str, the subclass is expected to read from the ann_file. When
            ann_file is None, the subclass is expected to read according
            to data_prefix.
        test_mode (bool, optional): in train mode or test mode. Default: False.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: str,
                 ann_file: Optional[Union[str, None]] = None,
                 test_mode: Optional[bool] = False):
        self.dataset_name = dataset_name
        super(MeshDataset, self).__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            ann_file=ann_file,
            test_mode=test_mode)

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
        self.num_data = len(data_infos)
        self.data_infos = data_infos
