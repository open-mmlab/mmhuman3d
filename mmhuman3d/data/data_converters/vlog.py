from .base_converter import BaseModeConverter
from typing import List
import json
import os
import glob

import cv2
import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class VlogConverter(BaseModeConverter):
    """VLOG-people dataset `Learning 3D Human Dynamics from Video' CVPR'2019
    More details can be found in the `paper.

    <https://arxiv.org/pdf/1812.01601.pdf>`__ .

    Args:
        modes (list): 'train' for accepted modes
    """
    ACCEPTED_MODES = ['train']

    def __init__(self,
                 modes: List = [],
                 extract_img: bool = False) -> None:
        super(VlogConverter, self).__init__(modes)
        self.extract_img = extract_img

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

        # structs we need
        image_path_, keypoints2d_, bbox_xywh_ = [], [], []

        filenames = glob.glob(os.path.join(dataset_path, f'{mode}/*.tfrecord'))
        raw_dataset = tf.data.TFRecordDataset(filenames)

        for raw_record in raw_dataset.take(-1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            # Now these are sequences.
            N = int(example.features.feature['meta/N'].int64_list.value[0])
            print(N)
            # This is a list of length N
            images_data = example.features.feature[
                'image/encoded'].bytes_list.value

            images_name = example.features.feature[
                'image/filenames'].bytes_list.value

            xys = example.features.feature['image/xys'].float_list.value
            xys = np.array(xys).reshape(-1, 2, 14)

            face_pts = example.features.feature[
                'image/face_pts'].float_list.value
            face_pts = np.array(face_pts).reshape(-1, 3, 5)

            toe_pts = example.features.feature[
                'image/toe_pts'].float_list.value

            if len(toe_pts) == 0:
                toe_pts = np.zeros(xys.shape[0], 3, 6)

            toe_pts = np.array(toe_pts).reshape(-1, 3, 6)

            visibles = example.features.feature[
                'image/visibilities'].int64_list.value
            visibles = np.array(visibles).reshape(-1, 1, 14)

            for i in tqdm(range(N)):
                image = tf.image.decode_jpeg(images_data[i], channels=3)
                kp = np.vstack((xys[i], visibles[i]))
                faces = face_pts[i]

                toes = toe_pts[i]
                kp = np.hstack((kp, faces, toes))
                if 'image/phis' in example.features.feature.keys():
                    # Preprocessed, so kps are in [-1, 1]
                    img_shape = image.shape[0]
                    vis = kp[2, :]
                    kp = ((kp[:2, :] + 1) * 0.5) * img_shape
                    kp = np.vstack((kp, vis))

                keypoints2d = kp.T

                # get bbox from visible keypoints
                vis_index = np.where(keypoints2d[:, 2] == 1)[0]
                keypoints2d_vis = keypoints2d[vis_index]
                bbox_xyxy = [
                    min(keypoints2d_vis[:, 0]),
                    min(keypoints2d_vis[:, 1]),
                    max(keypoints2d_vis[:, 0]),
                    max(keypoints2d_vis[:, 1])
                ]
                bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
                bbox_xywh = self._xyxy2xywh(bbox_xyxy)

                image_path = images_name[i].decode(
                    "utf-8").replace('/scratch1/storage/git_repos/', 'images/')

                if self.extract_img:
                    image_abs_path = os.path.join(dataset_path, image_path)
                    folder = os.path.dirname(image_abs_path)
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)
                    cv2.imwrite(image_abs_path, np.array(image))
                    
                image_path_.append(image_path)
                keypoints2d_.append(keypoints2d)
                bbox_xywh_.append(bbox_xywh)

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 25, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'instavariety_nop',
                                         'human_data')
        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['config'] = 'vlog'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, f'vlog_{mode}.npz')
        human_data.dump(out_file)

