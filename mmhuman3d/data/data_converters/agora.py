import os
from typing import List, Tuple

import cv2
import numpy as np
import pickle5 as pickle
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class AgoraConverter(BaseModeConverter):
    """AGORA dataset
    `AGORA: Avatars in Geography Optimized for Regression Analysis' CVPR`2021
    More details can be found in the `paper
    <https://arxiv.org/pdf/2104.14643.pdf>`__.

    Args:
        modes (list): 'validation' or 'train' for accepted modes
        fit (str): 'smpl' or 'smplx for available body model fits
        res (tuple): (1280, 720) or (3840, 2160) for available image resolution
    """
    ACCEPTED_MODES = ['validation', 'train']

    def __init__(self, modes: List = [], fit: str = 'smpl',
                 res: Tuple[int, int] = (1280, 720)) -> None:  # yapf: disable
        super(AgoraConverter, self).__init__(modes)
        accepted_fits = ['smpl', 'smplx']
        if fit not in accepted_fits:
            raise ValueError('Input fit not in accepted fits. \
                Use either smpl or smplx')
        self.fit = fit

        accepted_res = [(1280, 720), (3840, 2160)]
        if res not in accepted_res:
            raise ValueError('Input resolution not in accepted resolution. \
                Use either (1280, 720) or (3840, 2160)')
        self.res = res

    def get_global_orient(self,
                          imgPath,
                          df,
                          i,
                          pNum,
                          globalOrient=None,
                          meanPose=False):
        """Modified from https://github.com/pixelite1201/agora_evaluation/blob/
        master/agora_evaluation/projection.py specific to AGORA.

        Args:
            imgPath: image path
            df: annotation dataframe
            i: frame index
            pNum: person index
            globalOrient: original global orientation
            meanPose: Store True for mean pose from vposer

        Returns:
            globalOrient: rotated global orientation
        """
        if 'hdri' in imgPath:
            camYaw = 0
            camPitch = 0

        elif 'cam00' in imgPath:
            camYaw = 135
            camPitch = 30
        elif 'cam01' in imgPath:
            camYaw = -135
            camPitch = 30
        elif 'cam02' in imgPath:
            camYaw = -45
            camPitch = 30
        elif 'cam03' in imgPath:
            camYaw = 45
            camPitch = 30
        elif 'ag2' in imgPath:
            camYaw = 0
            camPitch = 15
        else:
            camYaw = df.iloc[i]['camYaw']
            camPitch = 0

        if meanPose:
            yawSMPL = 0
        else:
            yawSMPL = df.iloc[i]['Yaw'][pNum]

        # scans have a 90deg rotation, but for mean pose from vposer there is
        # no such rotation
        if meanPose:
            rotMat, _ = cv2.Rodrigues(
                np.array([[0, (yawSMPL) / 180 * np.pi, 0]], dtype=float))
        else:
            rotMat, _ = cv2.Rodrigues(
                np.array([[0, ((yawSMPL - 90) / 180) * np.pi, 0]],
                         dtype=float))

        camera_rotationMatrix, _ = cv2.Rodrigues(
            np.array([0, ((-camYaw) / 180) * np.pi, 0]).reshape(3, 1))
        camera_rotationMatrix2, _ = cv2.Rodrigues(
            np.array([camPitch / 180 * np.pi, 0, 0]).reshape(3, 1))

        # flip pose
        R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
        R_root = cv2.Rodrigues(globalOrient.reshape(-1))[0]
        new_root = R_root.dot(R_mod)
        globalOrient = cv2.Rodrigues(new_root)[0].reshape(3)

        # apply camera matrices
        globalOrient = self.rotate_global_orient(rotMat, globalOrient)
        globalOrient = self.rotate_global_orient(camera_rotationMatrix,
                                                 globalOrient)
        globalOrient = self.rotate_global_orient(camera_rotationMatrix2,
                                                 globalOrient)

        return globalOrient

    @staticmethod
    def rotate_global_orient(rotMat, global_orient):
        """Transform global orientation given rotation matrix.

        Args:
            rotMat: rotation matrix
            global_orient: original global orientation

        Returns:
            new_global_orient: transformed global orientation
        """
        new_global_orient = cv2.Rodrigues(
            np.dot(rotMat,
                   cv2.Rodrigues(global_orient.reshape(-1))[0]))[0].T[0]
        return new_global_orient

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
                keypoints3d, keypoints2d_mask, keypoints3d_mask, meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []

        # get a list of .pkl files in the directory
        img_path = os.path.join(dataset_path, 'images', mode)

        body_model = {}
        body_model['body_pose'] = []
        body_model['global_orient'] = []
        body_model['betas'] = []
        body_model['transl'] = []

        if self.fit == 'smplx':
            annot_path = os.path.join(dataset_path, 'camera_dataframe')
            body_model['left_hand_pose'] = []
            body_model['right_hand_pose'] = []
            body_model['expression'] = []
            body_model['leye_pose'] = []
            body_model['reye_pose'] = []
            body_model['jaw_pose'] = []
            num_keypoints = 127
            keypoints_convention = 'agora'
            num_body_pose = 21
        else:
            annot_path = os.path.join(dataset_path, 'camera_dataframe_smpl')
            num_keypoints = 45
            keypoints_convention = 'smpl_45'
            num_body_pose = 23

        meta = {}
        meta['gender'] = []
        meta['age'] = []
        meta['ethnicity'] = []
        meta['kid'] = []
        meta['occlusion'] = []

        # go through all the .pkl files
        annot_dataframes = [
            os.path.join(annot_path, f) for f in os.listdir(annot_path)
            if f.endswith('.pkl') and '{}'.format(mode) in f
        ]

        for filename in tqdm(sorted(annot_dataframes)):
            df = pickle.load(open(filename, 'rb'))
            for idx in tqdm(range(len(df))):
                imgname = df.iloc[idx]['imgPath']
                if self.res == (1280, 720):
                    imgname = imgname.replace('.png', '_1280x720.png')
                img_path = os.path.join('images', mode, imgname)
                valid_pers_idx = np.where(df.iloc[idx].at['isValid'])[0]
                for pidx in valid_pers_idx:
                    # obtain meta data
                    gender = df.iloc[idx]['gender'][pidx]
                    age = df.iloc[idx]['age'][pidx]
                    kid = df.iloc[idx]['kid'][pidx]
                    occlusion = df.iloc[idx]['occlusion'][pidx]
                    ethnicity = df.iloc[idx]['ethnicity'][pidx]

                    # obtain keypoints
                    keypoints2d = df.iloc[idx]['gt_joints_2d'][pidx]
                    if self.res == (1280, 720):
                        keypoints2d *= (720 / 2160)
                    keypoints3d = df.iloc[idx]['gt_joints_3d'][pidx]

                    gt_bodymodel_path = os.path.join(
                        dataset_path,
                        df.iloc[idx][f'gt_path_{self.fit}'][pidx])
                    gt_bodymodel_path = gt_bodymodel_path.replace(
                        '.obj', '.pkl')
                    ann = pickle.load(open(gt_bodymodel_path, 'rb'))

                    if self.fit == 'smplx':
                        # obtain smplx data
                        body_model['body_pose'].append(ann['body_pose'])
                        global_orient = ann['global_orient']
                        body_model['betas'].append(
                            ann['betas'].reshape(-1)[:10])
                        body_model['transl'].append(ann['transl'])
                        body_model['left_hand_pose'].append(
                            ann['left_hand_pose'])
                        body_model['right_hand_pose'].append(
                            ann['right_hand_pose'])
                        body_model['jaw_pose'].append(ann['jaw_pose'])
                        body_model['leye_pose'].append(ann['leye_pose'])
                        body_model['reye_pose'].append(ann['reye_pose'])
                        body_model['expression'].append(ann['expression'])

                    else:
                        # obtain smpl data
                        body_model['body_pose'].append(
                            ann['body_pose'].cpu().detach().numpy())
                        global_orient = ann['root_pose'].cpu().detach().numpy()
                        body_model['betas'].append(
                            ann['betas'].cpu().detach().numpy().reshape(
                                -1)[:10])
                        body_model['transl'].append(
                            ann['translation'].cpu().detach().numpy())

                    global_orient = self.get_global_orient(
                        img_path, df, idx, pidx, global_orient.reshape(-1))

                    # add confidence column
                    keypoints2d = np.hstack(
                        [keypoints2d, np.ones((num_keypoints, 1))])
                    keypoints3d = np.hstack(
                        [keypoints3d, np.ones((num_keypoints, 1))])

                    bbox_xyxy = [
                        min(keypoints2d[:, 0]),
                        min(keypoints2d[:, 1]),
                        max(keypoints2d[:, 0]),
                        max(keypoints2d[:, 1])
                    ]
                    bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
                    bbox_xywh = self._xyxy2xywh(bbox_xyxy)

                    keypoints2d_.append(keypoints2d)
                    keypoints3d_.append(keypoints3d)
                    bbox_xywh_.append(bbox_xywh)
                    image_path_.append(img_path)
                    body_model['global_orient'].append(global_orient)
                    meta['gender'].append(gender)
                    meta['age'].append(age)
                    meta['kid'].append(kid)
                    meta['occlusion'].append(occlusion)
                    meta['ethnicity'].append(ethnicity)

        # change list to np array
        if self.fit == 'smplx':
            body_model['left_hand_pose'] = np.array(
                body_model['left_hand_pose']).reshape((-1, 15, 3))
            body_model['right_hand_pose'] = np.array(
                body_model['right_hand_pose']).reshape((-1, 15, 3))
            body_model['expression'] = np.array(
                body_model['expression']).reshape((-1, 10))
            body_model['leye_pose'] = np.array(
                body_model['leye_pose']).reshape((-1, 3))
            body_model['reye_pose'] = np.array(
                body_model['reye_pose']).reshape((-1, 3))
            body_model['jaw_pose'] = np.array(body_model['jaw_pose']).reshape(
                (-1, 3))

        body_model['body_pose'] = np.array(body_model['body_pose']).reshape(
            (-1, num_body_pose, 3))
        body_model['global_orient'] = np.array(
            body_model['global_orient']).reshape((-1, 3))
        body_model['betas'] = np.array(body_model['betas']).reshape((-1, 10))
        body_model['transl'] = np.array(body_model['transl']).reshape((-1, 3))

        meta['gender'] = np.array(meta['gender'])
        meta['age'] = np.array(meta['age'])
        meta['kid'] = np.array(meta['kid'])
        meta['occlusion'] = np.array(meta['occlusion'])
        meta['ethnicity'] = np.array(meta['ethnicity'])

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])

        # change list to np array
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, num_keypoints, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, keypoints_convention,
                                         'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, num_keypoints, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, keypoints_convention,
                                      'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints3d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d'] = keypoints3d_
        human_data['meta'] = meta
        human_data['config'] = 'agora'
        if self.fit == 'smplx':
            human_data['smplx'] = body_model
        else:
            human_data['smpl'] = body_model
        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = f'agora_{mode}_{self.fit}.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
