import glob
import os
from typing import List, Tuple

import cv2
import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class Mupots3dConverter(BaseConverter):
    """MuPoTs-3D dataset `Single-Shot Multi-Person 3D Pose Estimation 
    From Monocular RGB' 3DV'2018
    More details can be found in the `paper.

    <https://arxiv.org/abs/1712.03453>`__ .
    """
    @staticmethod
    def load_annot(fname):
        def parse_pose(dt):
            res = {}
            annot2 = dt['annot2'][0,0]
            annot3 = dt['annot3'][0,0]
            annot3_univ = dt['univ_annot3'][0,0]
            is_valid = dt['isValidFrame'][0,0][0,0]
            res['annot2'] = annot2
            res['annot3'] = annot3
            res['annot3_univ'] = annot3_univ
            res['is_valid'] = is_valid
            return res 
        data = sio.loadmat(fname)['annotations']
        results = []
        num_frames, num_inst = data.shape[0], data.shape[1]
        for j in range(num_inst):
            buff = []
            for i in range(num_frames):
                buff.append(parse_pose(data[i,j]))
            results.append(buff)
        return results

    def extract_keypoints(
        self, keypoints2d: np.ndarray, keypoints3d: np.ndarray,
        num_keypoints: int
    ) -> Tuple[bool, np.ndarray, np.ndarray, List[float]]:
        """Check keypoints validiy and add confidence and bbox."""

        bbox_xyxy = [
            min(keypoints2d[:, 0]),
            min(keypoints2d[:, 1]),
            max(keypoints2d[:, 0]),
            max(keypoints2d[:, 1])
        ]
        bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
        bbox_xywh = self._xyxy2xywh(bbox_xyxy)

        # check that all joints are visible
        h, w = 2048, 2048
        x_in = np.logical_and(keypoints2d[:, 0] < w, keypoints2d[:, 0] >= 0)
        y_in = np.logical_and(keypoints2d[:, 1] < h, keypoints2d[:, 1] >= 0)
        ok_pts = np.logical_and(x_in, y_in)
        if np.sum(ok_pts) < num_keypoints:
            valid = False

        # add confidence column
        keypoints2d = np.hstack([keypoints2d, np.ones((num_keypoints, 1))])
        keypoints3d = np.hstack([keypoints3d, np.ones((num_keypoints, 1))])
        valid = True

        return valid, keypoints2d, keypoints3d, bbox_xywh

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask, keypoints3d, keypoints3d_mask stored in
                HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []


        # test data
        user_list = range(1, 21)

        for user_i in tqdm(user_list, desc='user'):
            seq_path = os.path.join(dataset_path, 'MultiPersonTestSet',
                                    'TS' + str(user_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')

            annot = self.load_annot(annot_file)
            for person in annot:
                for frame_i, ann in tqdm(enumerate(person), desc='frame'):
                    keypoints2d = ann['annot2'].transpose(1, 0)
                    keypoints3d = ann['annot3_univ'].transpose(1, 0) / 1000
                    valid = ann['is_valid']

                    if valid == 0:
                        continue
                    image_path = os.path.join(
                        'MultiPersonTestSet', 'TS' + str(user_i),
                        'img_' + str(frame_i).zfill(6) + '.jpg')
                    keypoints3d = keypoints3d - keypoints3d[14]  # 14 is pelvis

                    valid, keypoints2d, keypoints3d, bbox_xywh = \
                        self.extract_keypoints(keypoints2d, keypoints3d, 17)

                    if not valid:
                        continue

                    # store the data
                    image_path_.append(image_path)
                    bbox_xywh_.append(bbox_xywh)
                    keypoints2d_.append(keypoints2d)
                    keypoints3d_.append(keypoints3d)

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack(
            [bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'mpi_inf_3dhp_test',
                                            'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, 17, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, 'mpi_inf_3dhp_test',
                                        'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints3d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d'] = keypoints3d_
        human_data['config'] = 'mupots3d'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'mupots3d_test.npz')
        human_data.dump(out_file)
