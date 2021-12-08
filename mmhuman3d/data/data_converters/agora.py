import os
from typing import List

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
        res (str): '1280x720' or '3840x2160' for available image resolution
    """
    ACCEPTED_MODES = ['validation', 'train']

    def __init__(self,
                 modes: List = [],
                 fit: str = 'smpl',
                 res='1280x720') -> None:
        super(AgoraConverter, self).__init__(modes)
        accepted_fits = ['smpl', 'smplx']
        if fit not in accepted_fits:
            raise ValueError('Input fit not in accepted fits. \
                Use either smpl or smplx')
        self.fit = fit

        accepted_res = ['1280x720', '3840x2160']
        if res not in accepted_res:
            raise ValueError('Input resolution not in accepted resolution. \
                Use either 1280x720 or 3840x2160')
        self.res = res

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

        bodymodel = {}
        bodymodel['body_pose'] = []
        bodymodel['global_orient'] = []
        bodymodel['betas'] = []
        bodymodel['transl'] = []

        if self.fit == 'smplx':
            annot_path = os.path.join(dataset_path, 'camera_dataframe')
            bodymodel['left_hand_pose'] = []
            bodymodel['right_hand_pose'] = []
            bodymodel['expression'] = []
            bodymodel['leye_pose'] = []
            bodymodel['reye_pose'] = []
            bodymodel['jaw_pose'] = []
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
                if self.res == '1280x720':
                    imgname.replace('.png', '_1280x720.png')
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
                    if self.res == '1280x720':
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
                        bodymodel['body_pose'].append(ann['body_pose'])
                        bodymodel['global_orient'].append(ann['global_orient'])
                        bodymodel['betas'].append(
                            ann['betas'].reshape(-1)[:10])
                        bodymodel['transl'].append(ann['transl'])
                        bodymodel['left_hand_pose'].append(
                            ann['left_hand_pose'])
                        bodymodel['right_hand_pose'].append(
                            ann['right_hand_pose'])
                        bodymodel['jaw_pose'].append(ann['jaw_pose'])
                        bodymodel['leye_pose'].append(ann['leye_pose'])
                        bodymodel['reye_pose'].append(ann['reye_pose'])
                        bodymodel['expression'].append(ann['expression'])

                    else:
                        # obtain smpl data
                        bodymodel['body_pose'].append(
                            ann['body_pose'].cpu().detach().numpy())
                        bodymodel['global_orient'].append(
                            ann['root_pose'].cpu().detach().numpy())
                        bodymodel['betas'].append(
                            ann['betas'].cpu().detach().numpy().reshape(
                                -1)[:10])
                        bodymodel['transl'].append(
                            ann['translation'].cpu().detach().numpy())

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
                    bbox_xywh = self._bbox_expand(bbox_xyxy, scale_factor=1.2)

                    keypoints2d_.append(keypoints2d)
                    keypoints3d_.append(keypoints3d)
                    bbox_xywh_.append(bbox_xywh)
                    image_path_.append(img_path)

                    meta['gender'].append(gender)
                    meta['age'].append(age)
                    meta['kid'].append(kid)
                    meta['occlusion'].append(occlusion)
                    meta['ethnicity'].append(ethnicity)

        # change list to np array
        if self.fit == 'smplx':
            bodymodel['left_hand_pose'] = np.array(
                bodymodel['left_hand_pose']).reshape((-1, 15, 3))
            bodymodel['right_hand_pose'] = np.array(
                bodymodel['right_hand_pose']).reshape((-1, 15, 3))
            bodymodel['expression'] = np.array(
                bodymodel['expression']).reshape((-1, 10))
            bodymodel['leye_pose'] = np.array(bodymodel['leye_pose']).reshape(
                (-1, 3))
            bodymodel['reye_pose'] = np.array(bodymodel['reye_pose']).reshape(
                (-1, 3))
            bodymodel['jaw_pose'] = np.array(bodymodel['jaw_pose']).reshape(
                (-1, 3))

        bodymodel['body_pose'] = np.array(bodymodel['body_pose']).reshape(
            (-1, num_body_pose, 3))
        bodymodel['global_orient'] = np.array(
            bodymodel['global_orient']).reshape((-1, 3))
        bodymodel['betas'] = np.array(bodymodel['betas']).reshape((-1, 10))
        bodymodel['transl'] = np.array(bodymodel['transl']).reshape((-1, 3))

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
            human_data['smplx'] = bodymodel
        else:
            human_data['smpl'] = bodymodel
        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = f'agora_{mode}_{self.fit}.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
