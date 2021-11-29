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
class CrowdposeConverter(BaseModeConverter):
    """CrowdPose dataset
    `CrowdPose: Efficient Crowded Scenes Pose Estimation and A New
    Benchmark' CVPR'2019
    More details can be found in the `paper
    <https://arxiv.org/pdf/1812.00324.pdf>`__ .

    Args:
        modes (list): 'val', 'train', 'trainval' and/or 'test' for
        accepted modes
    """
    ACCEPTED_MODES = ['val', 'train', 'trainval', 'test']

    def __init__(self, modes: List = []) -> None:
        super(CrowdposeConverter, self).__init__(modes)

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
        json_path = os.path.join(dataset_path,
                                 'crowdpose_{}.json'.format(mode))

        json_data = json.load(open(json_path, 'r'))

        imgs = {}
        for img in json_data['images']:
            imgs[img['id']] = img

        for annot in tqdm(json_data['annotations']):

            # image name
            image_id = annot['image_id']
            img_path = str(imgs[image_id]['file_name'])
            img_path = os.path.join('images', img_path)

            # scale and center
            bbox_xywh = np.array(annot['bbox'])

            # keypoints processing
            keypoints2d = np.array(annot['keypoints'])
            keypoints2d = np.reshape(keypoints2d, (14, 3))
            keypoints2d[keypoints2d[:, 2] > 0, 2] = 1
            # check if all keypoints are annotated
            if sum(keypoints2d[:, 2] > 0) < 14:
                continue

            # check that all joints are within image bounds
            height = imgs[image_id]['height']
            width = imgs[image_id]['width']
            x_in = np.logical_and(keypoints2d[:, 0] < width,
                                  keypoints2d[:, 0] >= 0)
            y_in = np.logical_and(keypoints2d[:, 1] < height,
                                  keypoints2d[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)
            if np.sum(ok_pts) < 14:
                continue

            # store data
            image_path_.append(img_path)
            keypoints2d_.append(keypoints2d)
            bbox_xywh_.append(bbox_xywh)

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 14, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'crowdpose',
                                         'human_data')

        human_data['image_path'] = image_path_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['config'] = 'crowdpose'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'crowdpose_{}.npz'.format(mode))
        human_data.dump(out_file)
