import json
import os

import numpy as np
import mmcv
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.multi_human_data import MultiHumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS

def sort_json(json):
    return int(json['image_id'])
@DATA_CONVERTERS.register_module()
class CocoConverter(BaseConverter):
    """CocoDataset dataset `Microsoft COCO: Common Objects in Context'
    ECCV'2014 More details can be found in the `paper.

    <https://arxiv.org/abs/1405.0312>`__ .
    """


    
    def convert(self,
                dataset_path: str,
                out_path: str,
                multi_human_data: bool = False,
                file_client_args: dict = None) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask stored in HumanData() format
        """
        if multi_human_data:
            # use MultiHumanData to store all data
            human_data = MultiHumanData()
        else:
            # use HumanData to store all data
            human_data = HumanData()

        # structs we need
        image_path_, keypoints2d_, bbox_xywh_ = [], [], []

        # json annotation file
        json_path = os.path.join(dataset_path, 'annotations',
                                 'person_keypoints_train2014.json')

        if file_client_args is not None:
            json_data = mmcv.load(
                json_path,
                file_client_args=file_client_args)
        else:         
            json_data = json.load(open(json_path, 'r'))

        imgs = {}
        for img in json_data['images']:
            imgs[img['id']] = img

        json_data['annotations'].sort(key=sort_json)
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
            img_path = os.path.join('train2014', img_path)

            # scale and center
            bbox_xywh = annot['bbox']

            # store data
            image_path_.append(img_path)
            keypoints2d_.append(keypoints2d)
            else:
                # 1. MultiHumanData and keypoints2d_num>=4
                # 2. HumanData and keypoints2d_num>=12
                keypoints2d_.append(keypoints2d)
            image_path_.append(img_path)
            bbox_xywh_.append(bbox_xywh)

        if multi_human_data:
            # optional
            optional = {}
            optional['frame_range'] = []
            frame_start = 0
            frame_end = 0
            for image_path in sorted(set(image_path_), key=image_path_.index):
                frame_end = frame_start + \
                    image_path_.count(image_path)
                optional['frame_range'].append([frame_start, frame_end])
                frame_start = frame_end
            optional['frame_range'] = np.array(optional['frame_range'])
            human_data['optional'] = optional

        
        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'coco', 'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['config'] = 'coco'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'coco_2014_train.npz')
        human_data.dump(out_file)
