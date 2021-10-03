import glob
import os

import mmcv
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps


def posetrack_extract(dataset_path, out_path, mode='train'):

    # total dictionary to store all data
    total_dict = {}

    # structs we use
    image_path_, bbox_xywh_, keypoints2d_ = [], [], []

    # training mode
    ann_folder = os.path.join(
        dataset_path, 'posetrack_data/annotations/{}/*.json'.format(mode))
    ann_files = sorted(glob.glob(ann_folder))

    for ann_file in tqdm(ann_files):
        json_data = mmcv.load(ann_file)

        counter = 0
        for im, ann in zip(json_data['images'], json_data['annotations']):
            # sample every 10 image and check image is labelled
            if counter % 10 != 0 and not im['is_labeled']:
                continue
            keypoints2d = np.array(ann['keypoints']).reshape(17, 3)
            keypoints2d[keypoints2d[:, 2] > 0, 2] = 1
            # check if all major body joints are annotated
            if sum(keypoints2d[5:, 2] > 0) < 12:
                continue

            image_path = im['file_name']
            image_abs_path = os.path.join(dataset_path, image_path)
            if not os.path.exists(image_abs_path):
                print('{} does not exist!'.format(image_abs_path))
                continue
            counter += 1
            bbox_xywh = np.array(ann['bbox'])

            # store data
            image_path_.append(image_path)
            keypoints2d_.append(keypoints2d)
            bbox_xywh_.append(bbox_xywh)

    # convert keypoints
    keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
    keypoints2d_, mask = convert_kps(keypoints2d_, 'posetrack', 'smplx')

    total_dict['image_path'] = image_path_
    total_dict['keypoints2d'] = keypoints2d_
    total_dict['bbox_xywh'] = bbox_xywh_
    total_dict['mask'] = mask
    total_dict['config'] = 'posetrack'

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'posetrack_{}.npz'.format(mode))
    np.savez_compressed(out_file, **total_dict)
