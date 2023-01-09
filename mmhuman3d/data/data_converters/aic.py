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
class AicConverter(BaseModeConverter):
    """AI Challenger dataset `Ai challenger: A large-scale dataset for going
    deeper in image understanding' arXiv'2017 More details can be found in the
    `paper.

    <https://arxiv.org/abs/1711.06475>`__ .

    Args:
        modes (list): 'validation' and/or 'train' for
        accepted modes
    """
    ACCEPTED_MODES = ['validation', 'train']

    def __init__(self, modes: List = []) -> None:
        super(AicConverter, self).__init__(modes)
        self.json_mapping_dict = {
            'train': ['20170909', '20170902'],
            'validation': ['20170911', '20170911'],
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
        iid = self.json_mapping_dict[mode][1]
        aid = self.json_mapping_dict[mode][0]
        root_dir = f'ai_challenger_keypoint_{mode}_{aid}'
        json_path = os.path.join(dataset_path, root_dir,
                                 f'keypoint_{mode}_annotations_{aid}.json')
        img_dir = f'{root_dir}/keypoint_{mode}_images_{iid}'

        json_data = json.load(open(json_path, 'r'))

        for annot in tqdm(json_data):

            # image name
            image_id = annot['image_id']
            img_path = os.path.join(img_dir, f'{image_id}.jpg')
            if not os.path.exists(os.path.join(dataset_path, img_path)):
                print('image path does not exist')
            keypoints_annot = annot['keypoint_annotations']
            bbox_annot = annot['human_annotations']

            for pid in list(keypoints_annot.keys()):
                # scale and center
                keypoints2d = np.array(keypoints_annot[pid]).reshape(14, 3)
                bbox_xywh = np.array(bbox_annot[pid]).reshape(-1)
                keypoints2d[keypoints2d[:, 2] < 0, 2] = 0
                # check if all keypoints are annotated
                if sum(keypoints2d[:, 2] > 0) == 14:
                    # store data
                    image_path_.append(img_path)
                    keypoints2d_.append(keypoints2d)
                    bbox_xywh_.append(bbox_xywh)

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 14, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'aic', 'human_data')

        human_data['image_path'] = image_path_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['config'] = 'aic'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'aic_{}.npz'.format(mode))
        human_data.dump(out_file)
