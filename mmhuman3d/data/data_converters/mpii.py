import os

import h5py
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class MpiiConverter(BaseConverter):

    @staticmethod
    def center_scale_to_bbox(center, scale):
        w, h = scale * 200, scale * 200
        x, y = center[0] - w / 2, center[1] - h / 2
        return [x, y, w, h]

    def convert(self, dataset_path, out_path):
        # total dictionary to store all data
        total_dict = {}

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_ = [], [], []

        # annotation files
        annot_file = os.path.join(dataset_path, 'train.h5')

        # read annotations
        f = h5py.File(annot_file, 'r')
        centers, image_path, keypoints2d, scales = \
            f['center'], f['imgname'], f['part'], f['scale']

        # go over all annotated examples
        for center, imgname, keypoints2d16, scale in tqdm(
                zip(centers, image_path, keypoints2d, scales)):
            imgname = imgname.decode('utf-8')
            # check if all major body joints are annotated
            if (keypoints2d16 > 0).sum() < 2 * 16:
                continue

            # keypoints
            keypoints2d16 = np.hstack([keypoints2d16, np.ones([16, 1])])

            # bbox
            bbox_xywh = self.center_scale_to_bbox(center, scale)

            # store data
            image_path_.append(os.path.join('images', imgname))
            bbox_xywh_.append(bbox_xywh)
            keypoints2d_.append(keypoints2d16)

        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 16, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'mpii', 'human_data')
        total_dict['image_path'] = image_path_
        total_dict['bbox_xywh'] = bbox_xywh_
        total_dict['keypoints2d'] = keypoints2d_
        total_dict['mask'] = mask
        total_dict['config'] = 'mpii'

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        out_file = os.path.join(out_path, 'mpii_train.npz')
        np.savez_compressed(out_file, **total_dict)
