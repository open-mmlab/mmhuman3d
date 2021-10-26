import glob
import os

import cdflib
import cv2
import h5py
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class H36mConverter(BaseModeConverter):

    ACCEPTED_MODES = ['valid', 'train']

    def __init__(self, modes=[], protocol=1, extract_img=False):
        super(H36mConverter, self).__init__(modes)
        accepted_protocol = [1, 2]
        if protocol not in accepted_protocol:
            raise ValueError('Input protocol not in accepted protocol. \
                Use either 1 or 2')
        self.protocol = protocol
        self.extract_img = extract_img

    def convert_by_mode(self, dataset_path, out_path, mode):
        # total dict to store data
        total_dict = {}

        # pick 17 joints from 32 (repeated) joints
        h36m_idx = [
            11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27
        ]

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []

        # choose users ids for different set
        if mode == 'train':
            user_list = [1, 5, 6, 7, 8]
        elif mode == 'valid':
            user_list = [9, 11]

        # go over each user
        for user_i in tqdm(user_list, desc='user id'):
            user_name = 'S%d' % user_i
            # path with GT bounding boxes
            bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat',
                                     'ground_truth_bs')
            # path with GT 2D pose
            pose2d_path = os.path.join(dataset_path, user_name,
                                       'MyPoseFeatures', 'D2_Positions')
            # path with GT 3D pose
            pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                     'D3_Positions_mono')
            # path with videos
            vid_path = os.path.join(dataset_path, user_name, 'Videos')

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

                # video file
                if self.extract_img:
                    vid_file = os.path.join(vid_path,
                                            seq_name.replace('cdf', 'mp4'))
                    vidcap = cv2.VideoCapture(vid_file)

                # go over each frame of the sequence
                for frame_i in tqdm(range(poses_3d.shape[0]), desc='frame id'):
                    # read video frame
                    if self.extract_img:
                        success, image = vidcap.read()
                        if not success:
                            break

                    # check if you can keep this frame
                    if frame_i % 5 == 0 and (self.protocol == 1
                                             or camera == '60457274'):
                        # image name
                        image_name = '%s_%s.%s_%06d.jpg' % (
                            user_name, action, camera, frame_i + 1)
                        img_folder_name = '%s_%s.%s' % (user_name, action,
                                                        camera)
                        image_path = os.path.join(dataset_path, user_name,
                                                  'images', img_folder_name,
                                                  image_name)

                        # save image
                        if self.extract_img:
                            cv2.imwrite(image_path)

                        # read GT bounding box
                        mask = bbox_h5py[bbox_h5py['Masks'][frame_i, 0]][:].T
                        ys, xs = np.where(mask == 1)
                        bbox_xyxy = np.array([
                            np.min(xs),
                            np.min(ys),
                            np.max(xs) + 1,
                            np.max(ys) + 1
                        ])
                        bbox_xywh = self._bbox_expand(
                            bbox_xyxy, scale_factor=0.9)

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

        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 17, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'h36m', 'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, 17, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, 'h36m', 'human_data')

        total_dict['image_path'] = image_path_
        total_dict['bbox_xywh'] = bbox_xywh_
        total_dict['keypoints2d'] = keypoints2d_
        total_dict['keypoints3d'] = keypoints3d_
        total_dict['mask'] = mask
        total_dict['config'] = 'h36m'

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        if mode == 'train':
            out_file = os.path.join(out_path, 'h36m_train.npz')
        elif mode == 'valid':
            out_file = os.path.join(
                out_path, 'h36m_valid_protocol%d.npz' % self.protocol)
        np.savez_compressed(out_file, **total_dict)
