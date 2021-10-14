import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class LspExtendedConverter(BaseConverter):

    def convert(self, dataset_path, out_path):

        # total dictionary to store all data
        total_dict = {}

        # training mode
        png_path = os.path.join(dataset_path, '*.png')
        imgs = glob.glob(png_path)
        imgs.sort()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_ = [], [], []

        # annotation files
        annot_file = os.path.join(dataset_path, 'joints.mat')
        keypoints2d = sio.loadmat(annot_file)['joints']

        for i, imgname in enumerate(tqdm(imgs)):
            # image name
            imgname = imgname.split('/')[-1]
            image_path = os.path.join(dataset_path, imgname)
            im = cv2.imread(image_path)
            h, w, _ = im.shape

            # keypoints
            keypoints2d14 = keypoints2d[:, :2, i]
            keypoints2d14 = np.hstack([keypoints2d14, np.ones([14, 1])])

            # bbox
            bbox_xywh = [
                min(keypoints2d14[:, 0]),
                min(keypoints2d14[:, 1]),
                max(keypoints2d14[:, 0]),
                max(keypoints2d14[:, 1])
            ]

            if 0 <= bbox_xywh[0] <= w and 0 <= bbox_xywh[2] <= w and \
                    0 <= bbox_xywh[1] <= h and 0 <= bbox_xywh[3] <= h:
                bbox_xywh = self._bbox_expand(bbox_xywh, scale_factor=1.2)
            else:
                print('Bbox out of image bounds. Skipping image {}'.format(
                    imgname))
                continue

            # store data
            image_path_.append(image_path)
            bbox_xywh_.append(bbox_xywh)
            keypoints2d_.append(keypoints2d14)

        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 14, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'lsp', 'smplx')
        total_dict['image_path'] = image_path_
        total_dict['bbox_xywh'] = bbox_xywh_
        total_dict['keypoints2d'] = keypoints2d_
        total_dict['mask'] = mask
        total_dict['config'] = 'hr-lspet'

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        out_file = os.path.join(out_path, 'hr-lspet_train.npz')
        np.savez_compressed(out_file, **total_dict)
