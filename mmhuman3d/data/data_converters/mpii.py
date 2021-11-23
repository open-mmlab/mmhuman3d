import os
from typing import List

import h5py
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class MpiiConverter(BaseConverter):
    """MPII Dataset `2D Human Pose Estimation: New Benchmark and State of the
    Art Analysis' CVPR'2014. More details can be found in the `paper.

    <http://human-pose.mpi-inf.mpg.de/contents/andriluka14cvpr.pdf>`__ .
    """

    @staticmethod
    def center_scale_to_bbox(center: float, scale: float) -> List[float]:
        """Obtain bbox given center and scale."""
        w, h = scale * 200, scale * 200
        x, y = center[0] - w / 2, center[1] - h / 2
        return [x, y, w, h]

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

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 16, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'mpii', 'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['config'] = 'mpii'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        out_file = os.path.join(out_path, 'mpii_train.npz')
        human_data.dump(out_file)
