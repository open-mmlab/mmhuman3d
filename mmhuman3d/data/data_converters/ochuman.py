import json
import os
from typing import List

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class OCHumanConverter(BaseModeConverter):
    """OCHuman dataset
    `Pose2Seg: Detection Free Human Instance Segmentation' CVPR'2019
    More details can be found in the `paper
    <https://arxiv.org/abs/1803.10683>`__ .

    Args:
        modes (list): 'val', 'train' and/or 'test' for
        accepted modes
    """
    ACCEPTED_MODES = ['val', 'train', 'test']

    def __init__(self, modes: List = []) -> None:
        super(OCHumanConverter, self).__init__(modes)
        self.json_mapping_dict = {
            'train': 'ochuman.json',
            'val': 'ochuman_coco_format_val_range_0.00_1.00.json',
            'test': 'ochuman_coco_format_test_range_0.00_1.00.json',
        }

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we need
        image_path_, keypoints2d_, bbox_xywh_ = [], [], []

        # json annotation file
        json_path = os.path.join(dataset_path, self.json_mapping_dict[mode])

        json_data = json.load(open(json_path, 'r'))

        for img in tqdm(json_data['images']):
            img_path = str(img['file_name'])
            img_path = os.path.join('images', img_path)

            if 'annotations' in img:
                for annot in img['annotations']:
                    # keypoints processing
                    keypoints2d = annot['keypoints']
                    if keypoints2d is not None:
                        keypoints2d = np.reshape(keypoints2d, (19, 3))
                        keypoints2d[keypoints2d[:, 2] > 0, 2] = 1

                        # scale and center
                        bbox_xywh = annot['bbox']

                        # store data
                        image_path_.append(img_path)
                        keypoints2d_.append(keypoints2d)
                        bbox_xywh_.append(bbox_xywh)

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 19, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'ochuman', 'human_data')

        human_data['image_path'] = image_path_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['config'] = 'ochuman'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'ochuman_{}.npz'.format(mode))
        human_data.dump(out_file)
