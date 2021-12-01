import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class LspExtendedConverter(BaseConverter):
    """Leeds Sports Pose Extended Training Dataset `Learning Effective Human
    Pose Estimation from Inaccurate Annotation' CVPR'2011 More details can be
    found in the `paper.

    <http://sam.johnson.io/research/publications/johnson11cvpr.pdf>`__ .
    """

    def convert(self, dataset_path: str, out_path: str) -> dict:
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
        # use HumanData to store all data
        human_data = HumanData()

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
            image_path_.append(imgname)
            bbox_xywh_.append(bbox_xywh)
            keypoints2d_.append(keypoints2d14)

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 14, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'lsp', 'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['config'] = 'hr-lspet'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        out_file = os.path.join(out_path, 'lspet_train.npz')
        human_data.dump(out_file)
