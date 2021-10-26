import json
import os

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class CocoWholebodyConverter(BaseModeConverter):

    ACCEPTED_MODES = ['val', 'train']

    def __init__(self, modes=[]):
        super(CocoWholebodyConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path, out_path, mode):
        # total dictionary to store all data
        total_dict = {}

        # structs we need
        image_path_, keypoints2d_, bbox_xywh_ = [], [], []

        json_path = os.path.join(dataset_path, 'annotations',
                                 'coco_wholebody_{}_v1.0.json'.format(mode))
        img_dir = os.path.join(dataset_path, '{}2017'.format(mode))

        json_data = json.load(open(json_path, 'r'))

        imgs = {}
        for img in json_data['images']:
            imgs[img['id']] = img

        for annot in tqdm(json_data['annotations']):

            # keypoints processing
            keypoints2d = np.zeros((133, 3))

            body_kps = annot['keypoints']
            body_kps = np.reshape(body_kps, (17, 3))
            body_kps[body_kps[:, 2] > 0, 2] = 1
            # check if all major body joints are annotated
            if sum(body_kps[5:, 2] > 0) < 12:
                continue

            keypoints2d[:17] = body_kps
            if annot['foot_valid']:
                foot_kps = np.reshape(annot['foot_kpts'], (6, 3))
                keypoints2d[17:23] = foot_kps
            if annot['face_valid']:
                face_kps = np.reshape(annot['face_kpts'], (68, 3))
                keypoints2d[23:91] = face_kps
            if annot['righthand_valid']:
                righthand_kps = np.reshape(annot['righthand_kpts'], (21, 3))
                keypoints2d[112:] = righthand_kps
            if annot['lefthand_valid']:
                lefthand_kps = np.reshape(annot['lefthand_kpts'], (21, 3))
                keypoints2d[91:112] = lefthand_kps

            # image name
            image_id = annot['image_id']
            img_name = str(imgs[image_id]['file_name'])
            img_path = os.path.join(img_dir, img_name)

            # scale and center
            bbox_xywh = annot['bbox']

            # store data
            image_path_.append(img_path)
            keypoints2d_.append(keypoints2d)
            bbox_xywh_.append(bbox_xywh)

        # convert keypoints
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 133, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'coco_wholebody',
                                         'human_data')

        total_dict['image_path'] = image_path_
        total_dict['keypoints2d'] = keypoints2d_
        total_dict['bbox_xywh'] = bbox_xywh_
        total_dict['mask'] = mask
        total_dict['config'] = 'coco_wholebody'

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'coco_wholebody_{}.npz'.format(mode))
        np.savez_compressed(out_file, **total_dict)
