import os

import h5py
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class InstaVibeConverter(BaseConverter):
    """Instavariety dataset `Learning 3D Human Dynamics from Video' CVPR'2019
    More details can be found in the `paper.

    <https://arxiv.org/pdf/1812.01601.pdf>`__ .
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
                keypoints2d_mask, video_path, frame_idx, features stored in
                HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we need
        image_path_, keypoints2d_, bbox_xywh_, vid_path_, frame_idx_, \
            features_ = [], [], [], [], [], []

        annot_path = os.path.join(dataset_path, 'insta_train_db.h5')

        data = h5py.File(annot_path, 'r')
        features_list = data['features']
        frame_list = data['frame_id']
        keypoints_list = data['joints2D']
        vid_list = data['vid_name']

        num_data = features_list.shape[0]

        for i in tqdm(range(num_data)):
            vid_id = vid_list[i].decode('utf-8')
            frame_id = frame_list[i]
            image_path = os.path.join(vid_id, str(frame_id))
            keypoints2d = keypoints_list[i]

            # get bbox from visible keypoints
            vis_index = np.where(keypoints2d[:, 2] == 1)[0]
            keypoints2d_vis = keypoints2d[vis_index]
            bbox_xyxy = [
                min(keypoints2d_vis[:, 0]),
                min(keypoints2d_vis[:, 1]),
                max(keypoints2d_vis[:, 0]),
                max(keypoints2d_vis[:, 1])
            ]
            bbox_xywh = self._bbox_expand(bbox_xyxy, scale_factor=1.2)

            vid_path_.append(vid_id)
            image_path_.append(image_path)
            frame_idx_.append(frame_id)
            keypoints2d_.append(keypoints2d)
            bbox_xywh_.append(bbox_xywh)
            features_.append(features_list[i])

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 25, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'instavariety',
                                         'human_data')
        features_ = np.array(features_).reshape((-1, 2048))

        human_data['image_path'] = image_path_
        human_data['video_path'] = vid_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['frame_idx'] = frame_idx_
        human_data['features'] = features_
        human_data['config'] = 'instavariety'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'insta_variety.npz')
        human_data.dump(out_file)
