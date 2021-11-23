import glob
import os

import mmcv
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class PosetrackConverter(BaseModeConverter):
    """PoseTrack18 dataset
    `Posetrack: A benchmark for human pose estimation and tracking' CVPR'2018
    More details can be found in the `paper
    <https://arxiv.org/abs/1710.10000>`_ .
    """

    ACCEPTED_MODES = ['val', 'train']

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

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
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'posetrack',
                                         'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['config'] = 'posetrack'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'posetrack_{}.npz'.format(mode))
        human_data.dump(out_file)
