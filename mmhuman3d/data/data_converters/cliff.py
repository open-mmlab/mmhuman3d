import os
from typing import List

import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.multi_human_data import MultiHumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class CliffConverter(BaseModeConverter):
    """CLIFF datasets converter `Carrying Location Information in Full Frames
    into Human Pose and Shape Estimation' More details can be found in the
    `paper.

    <https://arxiv.org/pdf/2208.00571.pdf>`__.
    Args:
        modes (list):  'coco', 'mpii'
        for accepted modes
    """

    ACCEPTED_MODES = ['coco', 'mpii']

    def __init__(self, modes: List = []) -> None:
        super(CliffConverter, self).__init__(modes)

        # def __init__(self) -> None:
        self.mapping_dict = {
            'coco': 'coco2014part_cliffGT.npz',
            'mpii': 'mpii_cliffGT.npz',
        }

    def convert_by_mode(self,
                        dataset_path: str,
                        out_path: str,
                        mode: str,
                        enable_multi_human_data: bool = False) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where spin preprocessed
            npz files are stored
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes
            enable_multi_human_data (bool):
                Whether to generate a multi-human data. If set to True,
                stored in MultiHumanData() format.
                Default: False, stored in HumanData() format.

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask,stored in HumanData() format. keypoints3d,
                keypoints3d_mask, smpl are added if available.

        """
        if enable_multi_human_data:
            # use MultiHumanData to store all data
            human_data = MultiHumanData()
        else:
            # use HumanData to store all data
            human_data = HumanData()

        image_path_, keypoints2d_, bbox_xywh_ = [], [], []

        if mode in self.mapping_dict.keys():
            seq_file = self.mapping_dict[mode]
            seq_path = os.path.join(dataset_path, seq_file)

        data = np.load(seq_path)

        keypoints2d_ = data['part']
        image_path_ = data['imgname']

        # center scale to bbox
        w = h = data['scale'] * 200
        x = data['center'][:, 0] - w / 2
        y = data['center'][:, 1] - h / 2

        bbox_xywh_ = np.column_stack((x, y, w, h))

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 24, 3))
        keypoints2d_, keypoints2d_mask = convert_kps(keypoints2d_, 'smpl_24',
                                                     'human_data')

        if 'S' in data:
            keypoints3d_ = data['S']
            keypoints3d_ = np.array(keypoints3d_).reshape((-1, 24, 4))
            keypoints3d_, keypoints3d_mask = convert_kps(
                keypoints3d_, 'smpl_24', 'human_data')
            human_data['keypoints3d_mask'] = keypoints3d_mask
            human_data['keypoints3d'] = keypoints3d_

        if 'has_smpl' in data:
            has_smpl = data['has_smpl']
            smpl = {}
            smpl['body_pose'] = np.array(data['pose'][:, 3:]).reshape(
                (-1, 23, 3))
            smpl['global_orient'] = np.array(data['pose'][:, :3]).reshape(
                (-1, 3))
            smpl['betas'] = np.array(data['shape']).reshape((-1, 10))
            human_data['smpl'] = smpl
            human_data['has_smpl'] = has_smpl

        human_data['image_path'] = image_path_.tolist()
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = keypoints2d_mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['config'] = mode
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, f'cliff_{mode}_train.npz')
        human_data.dump(out_file)
