import glob
import os
from typing import List

import cdflib
import h5py
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class H36mSMPLXConverter(BaseModeConverter):
    """Human3.6M dataset
    `Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments' TPAMI`2014
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    ExPose use Human3.6M dataset for smplx 3D supervision for the body.

    Args:
        modes (list): 'train' for accepted modes
        protocol (int): 1 or 2 for available protocols
    """
    ACCEPTED_MODES = ['train']

    def __init__(self, modes: List = [], protocol: int = 1) -> None:
        super(H36mSMPLXConverter, self).__init__(modes)
        accepted_protocol = [1, 2]
        if protocol not in accepted_protocol:
            raise ValueError('Input protocol not in accepted protocol. \
                Use either 1 or 2')
        self.protocol = protocol
        self.camera_name_to_idx = {
            '54138969': 0,
            '55011271': 1,
            '58860488': 2,
            '60457274': 3,
        }

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
                keypoints2d_mask, keypoints3d, keypoints3d_mask, cam_param
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # pick 17 joints from 32 (repeated) joints
        h36m_idx = [
            11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27
        ]

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []

        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []

        # choose users ids for different set
        if mode == 'train':
            user_list = [1, 5, 6, 7, 8]
        elif mode == 'valid':
            user_list = [9, 11]

        # go over each user
        for user_i in tqdm(user_list, desc='user id'):
            user_name = f'S{user_i}'
            # path with GT bounding boxes
            bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat',
                                     'ground_truth_bs')
            # path with GT 2D pose
            pose2d_path = os.path.join(dataset_path, user_name,
                                       'MyPoseFeatures', 'D2_Positions')
            # path with GT 3D pose
            pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                     'D3_Positions_mono')

            # go over all the sequences of each user
            seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
            seq_list.sort()

            for seq_i in tqdm(seq_list, desc='sequence id'):

                # sequence info
                seq_name = seq_i.split('/')[-1]
                action, camera, _ = seq_name.split('.')
                action = action.replace(' ', '_')
                # irrelevant sequences
                if action == '_ALL':
                    continue

                # 2D pose file
                pose2d_file = os.path.join(pose2d_path, seq_name)
                poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

                # 3D pose file
                poses_3d = cdflib.CDF(seq_i)['Pose'][0]

                # bbox file
                bbox_file = os.path.join(bbox_path,
                                         seq_name.replace('cdf', 'mat'))
                bbox_h5py = h5py.File(bbox_file)

                # go over each frame of the sequence
                for frame_i in tqdm(range(poses_3d.shape[0]), desc='frame id'):
                    # check if you can keep this frame
                    if frame_i % 5 == 0 and (self.protocol == 1
                                             or camera == '60457274'):
                        # image name
                        seq_id = f'{user_name}_{action}'
                        image_name = f'{seq_id}.{camera}_{frame_i + 1:06d}.jpg'
                        img_folder_name = f'{user_name}_{action}.{camera}'
                        image_path = os.path.join(user_name, 'images',
                                                  img_folder_name, image_name)

                        # get bbox from mask
                        mask = bbox_h5py[bbox_h5py['Masks'][frame_i, 0]][:].T
                        ys, xs = np.where(mask == 1)
                        bbox_xyxy = np.array([
                            np.min(xs),
                            np.min(ys),
                            np.max(xs) + 1,
                            np.max(ys) + 1
                        ])
                        bbox_xyxy = self._bbox_expand(
                            bbox_xyxy, scale_factor=0.9)
                        bbox_xywh = self._xyxy2xywh(bbox_xyxy)

                        # read GT 2D pose
                        keypoints2dall = np.reshape(poses_2d[frame_i, :],
                                                    [-1, 2])
                        keypoints2d17 = keypoints2dall[h36m_idx]
                        keypoints2d17 = np.concatenate(
                            [keypoints2d17, np.ones((17, 1))], axis=1)

                        # read GT 3D pose
                        keypoints3dall = np.reshape(poses_3d[frame_i, :],
                                                    [-1, 3]) / 1000.
                        keypoints3d17 = keypoints3dall[h36m_idx]
                        keypoints3d17 -= keypoints3d17[0]  # root-centered
                        keypoints3d17 = np.concatenate(
                            [keypoints3d17, np.ones((17, 1))], axis=1)

                        # store data
                        image_path_.append(image_path)
                        bbox_xywh_.append(bbox_xywh)
                        keypoints2d_.append(keypoints2d17)
                        keypoints3d_.append(keypoints3d17)

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'h36m_smplx',
                                         'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, 17, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, 'h36m_smplx', 'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        # human_data['keypoints2d_mask'] = mask
        human_data['keypoints3d_mask'] = mask
        # human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d'] = keypoints3d_
        human_data['config'] = 'h36m'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        out_file = os.path.join(out_path, f'h36m_smplx_{mode}.npz')
        human_data.dump(out_file)
