import glob
import os

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class People3dConverter(BaseConverter):
    """3DPeople Dataset
    `3DPeople: Modeling the Geometry of Dressed Humans' ICCV'2019
    More details can be found in the `paper.

    <https://arxiv.org/abs/1904.04571>`__ .
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
        rgb_path = os.path.join(dataset_path, 'rgb', '*/*/*/*/*.jpg')
        imgs = glob.glob(rgb_path)
        imgs.sort()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []

        for i, imgname in enumerate(tqdm(imgs)):

            img_dir = imgname.split('/')[-2]
            skel_path = imgname.replace('rgb', 'skeleton').replace(
                f'{img_dir}/', '').replace('.jpg', '.txt')
            keypoints2d = np.loadtxt(skel_path)[..., :2]
            keypoints3d = np.loadtxt(skel_path)[..., :3]

            # keypoints
            keypoints2d = np.hstack([keypoints2d, np.ones([67, 1])])
            keypoints3d = np.hstack([keypoints3d, np.ones([67, 1])])
            keypoints3d -= keypoints3d[0]  # root-centered

            # bbox
            bbox_xyxy = [
                min(keypoints2d[:, 0]),
                min(keypoints2d[:, 1]),
                max(keypoints2d[:, 0]),
                max(keypoints2d[:, 1])
            ]

            bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
            bbox_xywh = self._xyxy2xywh(bbox_xyxy)

            # store data
            image_path_.append(imgname.replace(dataset_path + '/', ''))
            bbox_xywh_.append(bbox_xywh)
            keypoints2d_.append(keypoints2d)
            keypoints3d_.append(keypoints3d)

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 67, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'people3d',
                                         'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, 67, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, 'people3d', 'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d_mask'] = mask
        human_data['keypoints3d'] = keypoints3d_
        human_data['config'] = 'people3d'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        out_file = os.path.join(out_path, 'people3d_train.npz')
        human_data.dump(out_file)
