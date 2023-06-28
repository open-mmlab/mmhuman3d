# import ast
# import glob
import json
import os
import random
from typing import List

# import cv2
import numpy as np
# import pandas as pd
# import smplx
# import torch
from tqdm import tqdm

# from mmhuman3d.core.cameras import build_cameras
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
# import mmcv
# from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class HumanartConverter(BaseModeConverter):

    ACCEPTED_MODES = ['real_human', '2D_virtual_human', '3D_virtual_human']

    def __init__(self, modes: List[str] = None):

        super(HumanartConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:

        modes_dict = {
            'real_human': ['acrobatics', 'cosplay', 'dance', 'drama', 'movie'],
            '2D_virtual_human': [
                'cartoon', 'digital_art', 'ink_painting', 'kids_drawing',
                'mural', 'oil_painting', 'shadow_play', 'sketch',
                'stained_glass', 'ukiyoe', 'watercolor'
            ],
            '3D_virtual_human': ['garage_kits', 'relief', 'sculpture'],
        }

        seed, size = '230420', '99999'
        random.seed(int(seed))

        batch_part = ['training', 'validation']
        batch_name = modes_dict[mode]

        for part in batch_part:
            # use HumanData to store the data
            human_data = HumanData()

            # initialize
            image_path_, keypoints21_, bbox_xywh_ = [], [], []
            height_, width_ = [], []
            batch_name_, description_ = [], []

            size_n = 0

            for name in batch_name:

                # load annotations
                anno = json.load(
                    open(
                        os.path.join(dataset_path, 'annotations',
                                     f'{part}_humanart_{name}.json')))

                size_n += len(anno['annotations'])

                # pdb.set_trace()

                # create image reference dict
                image_ref = {}
                for idx, img_dict in enumerate(anno['images']):
                    image_ref[img_dict['id']] = idx

                # iterate over images
                for idx, annotation in enumerate(
                        tqdm(
                            anno['annotations'],
                            desc=f'Processing {mode} {part} {name}')):

                    # get image id
                    image_id = annotation['image_id']
                    image_idx = image_ref[image_id]

                    # get image path
                    imgp = anno['images'][image_idx]['file_name'].split(
                        os.path.sep, 1)[-1]
                    image_path_.append(imgp)

                    # get image size
                    height, width = anno['images'][image_idx]['height'], anno[
                        'images'][image_idx]['width']
                    height_.append(height)
                    width_.append(width)

                    # get text description
                    description = anno['images'][image_idx]['description']
                    description_.append(description)

                    # get keypoints
                    keypoints21 = np.array(annotation['keypoints_21']).reshape(
                        -1, 3)
                    # change [x, y, 2] to [x, y, 1] (conf == 1)
                    keypoints21[:,
                                -1] = (keypoints21[:,
                                                   -1] == 2).astype(np.float32)
                    keypoints21_.append(keypoints21)

                    # get bbox
                    bbox_xywh = annotation['bbox']
                    bbox_xywh.append(1)
                    # pdb.set_trace()
                    bbox_xywh_.append(np.array(bbox_xywh))

                    # get batch name
                    batch_name_.append(name)

            keypoints2d = np.array(keypoints21_).reshape(-1, 21, 3)
            keypoints2d, keypoints2d_mask = \
                convert_kps(keypoints2d, src='humanart', dst='human_data')
            human_data['keypoints2d_humanart'] = keypoints2d
            human_data['keypoints2d_humanart_mask'] = keypoints2d_mask

            human_data['image_path'] = image_path_
            human_data['config'] = f'humanart_{mode}'
            human_data['bbox_xywh'] = np.array(bbox_xywh_).reshape(-1, 5)

            # meta
            meta_ = {
                'image_height': height_,
                'image_width': width_,
                'image_description': description_,
                'batch_name': batch_name_
            }
            human_data['meta'] = meta_

            # misc
            human_data['misc'] = {'batch_names': modes_dict[mode]}

            human_data.compress_keypoints_by_mask()
            os.makedirs(out_path, exist_ok=True)
            size_i = min(size_n, int(size))
            out_file = os.path.join(
                out_path,
                f'humanart_{mode}_{part}_{seed}_{"{:05d}".format(size_i)}.npz')
            human_data.dump(out_file)
