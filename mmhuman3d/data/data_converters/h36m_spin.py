import glob
import os
import pickle

import cdflib
import cv2
import h5py
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class H36mSpinConverter(BaseModeConverter):
    """Human3.6M dataset for SPIN
    `Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments' TPAMI`2014
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    Args:
        modes (list): 'test' or 'train' for accepted modes
        extract_img (bool): Store True to extract images into a separate
        folder. Default: False.
        mosh_dir (str, optional): Path to directory containing mosh files
        openpose_dir (str, optional): Path to directory containing openpose
        predictions
    """

    ACCEPTED_MODES = ['test', 'train']

    def __init__(self,
                 modes=[],
                 protocol=1,
                 extract_img=False,
                 mosh_dir=None,
                 openpose_dir=None):
        super(H36mSpinConverter, self).__init__(modes)
        accepted_protocol = [1, 2]
        if protocol not in accepted_protocol:
            raise ValueError('Input protocol not in accepted protocol. \
                Use either 1 or 2')
        self.protocol = protocol
        self.extract_img = extract_img
        self.mosh_dir = mosh_dir
        self.camera_name_to_idx = {
            '54138969': 0,
            '55011271': 1,
            '58860488': 2,
            '60457274': 3,
        }
        self.openpose_dir = openpose_dir

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
                keypoints2d_mask, keypoints3d, keypoints3d_mask, smpl
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # pick 17 joints from 32 (repeated) joints
        h36m_idx = [
            11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27
        ]
        global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

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

            # mosh path
            mosh_path = os.path.join(self.mosh_dir, user_name)
            if not os.path.exists(mosh_path):
                raise ValueError('not a valid mosh path')

            # go over all the sequences of each user
            seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
            seq_list.sort()

            for seq_i in tqdm(seq_list, desc='sequence id'):

                # sequence info
                seq_name = seq_i.split('/')[-1]
                action, camera, _ = seq_name.split('.')
                action_raw = action
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

                # 3D mosh file
                mosh_name = '%s_cam%s_aligned.pkl' % (
                    action_raw, self.camera_name_to_idx[camera])
                mosh_file = os.path.join(mosh_path, mosh_name)
                if os.path.exists(mosh_file):
                    with open(mosh_file, 'rb') as file:
                        mosh_data = pickle.load(file, encoding='latin1')
                else:
                    continue
                thetas = mosh_data['new_poses']
                betas = mosh_data['betas']

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
                        image_path = os.path.join(user_name, 'images',
                                                  img_folder_name, image_name)
                        image_abs_path = os.path.join(dataset_path, image_path)

                        # save image
                        if self.extract_img:
                            cv2.imwrite(image_abs_path, image)

                        # read GT bounding box
                        mask = bbox_h5py[bbox_h5py['Masks'][frame_i, 0]][:].T
                        ys, xs = np.where(mask == 1)
                        bbox_xyxy = np.array([
                            np.min(xs),
                            np.min(ys),
                            np.max(xs) + 1,
                            np.max(ys) + 1
                        ])
                        bbox = self._bbox_expand(bbox_xyxy, scale_factor=0.9)

                        # read GT 2D pose
                        keypoints2dall = np.reshape(poses_2d[frame_i, :],
                                                    [-1, 2])
                        keypoints2d17 = keypoints2dall[h36m_idx]
                        keypoints2d = np.zeros([24, 3])
                        keypoints2d[global_idx, :2] = keypoints2d17
                        keypoints2d[global_idx, 2] = 1

                        # read GT 3D pose
                        keypoints3dall = np.reshape(poses_3d[frame_i, :],
                                                    [-1, 3]) / 1000.
                        keypoints3d17 = keypoints3dall[h36m_idx]
                        root_coord = keypoints3dall[h36m_idx][0].copy()
                        keypoints3d17 -= root_coord  # root-centered
                        keypoints3d = np.zeros([24, 4])
                        keypoints3d[global_idx, :3] = keypoints3d17
                        keypoints3d[global_idx, 3] = 1

                        # # read openpose detections
                        # json_file = os.path.join(openpose_path, 'coco',
                        #     image_name.replace('.jpg', '_keypoints.json'))
                        # openpose = read_openpose(json_file, keypoints2d,
                        # 'h36m')
                        # keypoints2d = np.hstack([keypoints2d, openpose])

                        pose = thetas[frame_i // 5, :]
                        R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
                        R_root = cv2.Rodrigues(pose[:3])[0]
                        new_R_root = R_root.dot(R_mod)
                        pose[:3] = cv2.Rodrigues(new_R_root)[0].reshape(3)

                        image_path_.append(image_path)
                        bbox_xywh_.append(bbox)
                        keypoints2d_.append(keypoints2d.reshape(24, 3))
                        keypoints3d_.append(keypoints3d.reshape(24, 4))
                        smpl['betas'].append(betas.reshape((10)))
                        smpl['body_pose'].append(pose[3:].reshape((23, 3)))
                        smpl['global_orient'].append(pose[:3].reshape((3)))

        # change list to np array
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))

        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 24, 3))
        keypoints2d_, keypoints2d_mask = convert_kps(keypoints2d_, 'smpl_24',
                                                     'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, 24, 4))
        keypoints3d_, keypoints3d_mask = convert_kps(keypoints3d_, 'smpl_24',
                                                     'human_data')
        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = keypoints2d_mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d_mask'] = keypoints3d_mask
        human_data['keypoints3d'] = keypoints3d_
        human_data['smpl'] = smpl
        human_data['config'] = 'h36m'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        if mode == 'train':
            out_file = os.path.join(out_path, 'spin_h36m_train.npz')
        elif mode == 'test':
            out_file = os.path.join(out_path, 'spin_h36m_valid_protocol2.npz')
        human_data.dump(out_file)
