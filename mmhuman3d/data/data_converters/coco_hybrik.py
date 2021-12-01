import json
import os

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.datasets.pipelines.hybrik_transforms import get_bbox
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class CocoHybrIKConverter(BaseConverter):
    """CocoDataset dataset `Microsoft COCO: Common Objects in Context'
    ECCV'2014 More details can be found in the `paper.

    <https://arxiv.org/abs/1405.0312>`__ .
    """

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file

        Returns:
            dict:
                A dict containing keys image_path, image_height, image_width,
                depth_factor, bbox_xywh, keypoints2d, keypoints2d_mask
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we need
        image_path_, keypoints2d_, bbox_xywh_, image_height_, \
            image_width_, depth_factor_ = [], [], [], [], [], []

        # json annotation file
        json_path = os.path.join(dataset_path, 'annotations',
                                 'person_keypoints_train2017.json')

        json_data = json.load(open(json_path, 'r'))

        imgs = {}
        for img in json_data['images']:
            imgs[img['id']] = img

        for annot in tqdm(json_data['annotations']):
            # keypoints processing
            keypoints2d = annot['keypoints']
            keypoints2d = np.reshape(keypoints2d, (17, 3))
            keypoints2d[keypoints2d[:, 2] > 0, 2] = 1
            # check if all major body joints are annotated
            if sum(keypoints2d[5:, 2] > 0) < 12:
                continue

            # image name
            image_id = annot['image_id']
            img_path = str(imgs[image_id]['file_name'])
            img_height = imgs[image_id]['height']
            img_width = imgs[image_id]['width']
            img_path = os.path.join('train_2017', 'train2017', img_path)

            # scale and center
            bbox_xywh = annot['bbox']
            bbox_xywh = get_bbox(np.array(bbox_xywh), img_width, img_height)

            image_path_.append(img_path)
            depth_factor_.append(2000.)
            keypoints2d_.append(keypoints2d)
            bbox_xywh_.append(bbox_xywh)
            image_height_.append(img_height)
            image_width_.append(img_width)

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'coco', 'human_data')

        human_data['image_path'] = image_path_
        human_data['image_height'] = image_height_
        human_data['image_width'] = image_width_
        human_data['depth_factor'] = depth_factor_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['config'] = 'coco'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'hybrik_coco_2017_train.npz')
        human_data.dump(out_file)
