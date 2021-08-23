import json
import os

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps


def coco_extract(dataset_path, out_path):
    # total dictionary to store all data
    total_dict = {}

    # config name
    total_dict['config'] = 'coco'

    # structs we need
    image_path_, keypoints2d_, bbox_xywh_ = [], [], []

    # json annotation file
    json_path = os.path.join(dataset_path, 'annotations',
                             'person_keypoints_train2014.json')

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
        img_path = os.path.join('train2014', img_path)

        # scale and center
        bbox_xywh = annot['bbox']

        # store data
        image_path_.append(img_path)
        keypoints2d_.append(keypoints2d)
        bbox_xywh_.append(bbox_xywh)

    # convert keypoints
    keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
    keypoints2d_, mask = convert_kps(keypoints2d_, 'coco', 'smplx')

    total_dict['image_path'] = image_path_
    total_dict['keypoints2d'] = keypoints2d_
    total_dict['bbox_xywh'] = bbox_xywh_
    total_dict['mask'] = mask
    total_dict['config'] = 'coco'

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'coco_2014_train.npz')
    np.savez_compressed(out_file, **total_dict)
