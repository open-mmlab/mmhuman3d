import os
from typing import List

import pdb
import time
import numpy as np
import pandas as pd
import json
import cv2
import glob
import random
from tqdm import tqdm
import torch
import smplx
import ast

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
# import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx, get_keypoint_idxs_by_part
from mmhuman3d.models.body_models.utils import transform_to_camera_frame


import pdb


@DATA_CONVERTERS.register_module()
class EgobodyConverter(BaseModeConverter):

    ACCEPTED_MODES = ['egocentric_train', 'egocentric_test', 'egocentric_val',
                      'kinect_train', 'kinect_test', 'kinect_val']

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super(EgobodyConverter, self).__init__(modes)


    def _keypoints_to_scaled_bbox_fh(self, keypoints, occ, scale=1.0, convention='smplx'):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale

        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format
        '''
        bboxs = []
        for body_part in ['head', 'left_hand', 'right_hand']:
            kp_id = get_keypoint_idxs_by_part(body_part, convention=convention)

            # keypoints_factory=smplx.SMPLX_KEYPOINTS)
            kps = keypoints[kp_id]
            occ_p = occ[kp_id]

            if np.sum(occ_p) / len(kp_id) >= 0.1:
                conf = 0
                # print(f'{body_part} occluded, occlusion: {np.sum(occ_p) / len(kp_id)}, skip')
            else:
                # print(f'{body_part} good, {np.sum(self_occ_p + occ_p) / len(kp_id)}')
                conf = 1

            xmin, ymin = np.amin(kps, axis=0)
            xmax, ymax = np.amax(kps, axis=0)

            width = (xmax - xmin) * scale
            height = (ymax - ymin) * scale

            x_center = 0.5 * (xmax + xmin)
            y_center = 0.5 * (ymax + ymin)
            xmin = x_center - 0.5 * width
            xmax = x_center + 0.5 * width
            ymin = y_center - 0.5 * height
            ymax = y_center + 0.5 * height

            bbox = np.stack([xmin, ymin, xmax, ymax, conf], axis=0).astype(np.float32)

            bboxs.append(bbox)
        return bboxs[0], bboxs[1], bboxs[2]
    

    def _revert_smplx_hands_pca(self, param_dict, num_pca_comps, gender):
        # egobody 12
        hl_pca = param_dict['left_hand_pose']
        hr_pca = param_dict['right_hand_pose']

        smplx_model = dict(np.load(f'data/body_models/smplx/SMPLX_{gender.upper()}.npz', allow_pickle=True))

        hl = smplx_model['hands_componentsl'] # 45, 45
        hr = smplx_model['hands_componentsr'] # 45, 45

        hl_pca = np.concatenate((hl_pca, np.zeros((len(hl_pca), 45 - num_pca_comps))), axis=1)
        hr_pca = np.concatenate((hr_pca, np.zeros((len(hr_pca), 45 - num_pca_comps))), axis=1)

        hl_reverted = np.einsum('ij, jk -> ik', hl_pca, hl).astype(np.float32)
        hr_reverted = np.einsum('ij, jk -> ik', hr_pca, hr).astype(np.float32)

        param_dict['left_hand_pose'] = hl_reverted
        param_dict['right_hand_pose'] = hr_reverted

        return param_dict


    def _get_seq_data(self, dataset_p, seq, frame_list, mode, batch_part):

        # get all valid rdb image path
        if mode == 'egocentric':
            imgp = glob.glob(os.path.join(dataset_p, f'{mode}_color', seq, '*', 'PV', '*.jpg'))
        elif mode == 'kinect':
            imgp = glob.glob(os.path.join(dataset_p, f'{mode}_color', seq, '*', '*.jpg'))
        else:
            raise ValueError(f'{mode} is not supported')
        
        valid_rgb_list = [int(os.path.basename(p)[-9:-4]) for p in imgp]

        # get all valid keypoint path & image path
        kp_npz_p = glob.glob(os.path.join(dataset_p, 'egocentric_color', seq, '*', 'keypoints.npz'))[0]
        kp_npz = dict(np.load(kp_npz_p, allow_pickle=True))
        valid_kps_annp_list = [int(os.path.basename(p)[-9:-4]) for p in kp_npz['imgname']]
        keypoints3d = kp_npz['keypoints']
        image_list = kp_npz['imgname']

        # get all valid smpl/smplx path
        smplx_interatcee_ps = glob.glob(os.path.join(dataset_p, f'smplx_interactee_{batch_part}', 
                seq, '*', 'results', f'frame_*','000.pkl'))
        smplx_camera_wearer_ps = glob.glob(os.path.join(dataset_p, f'smplx_camera_wearer_{batch_part}', 
                seq, '*', 'results', f'frame_*','000.pkl'))
        smpl_interactee_ps = glob.glob(os.path.join(dataset_p, f'smpl_interactee_{batch_part}',
                seq, '*', 'results', f'frame_*','000.pkl'))
        smpl_camera_wearer_ps = glob.glob(os.path.join(dataset_p, f'smpl_camera_wearer_{batch_part}',
                seq, '*', 'results', f'frame_*','000.pkl'))
        valid_smplx_list = [int(p[-13:-8]) for p in smplx_interatcee_ps]
        
        # get all valid depth path
        pass

        # get intercept of all valid list
        valid_list = sorted(list(set(valid_rgb_list) & set(valid_kps_annp_list) 
                          & set(valid_smplx_list) & set(frame_list)))
        valid_idx = [i for i in range(len(valid_list)) if valid_list[i] in frame_list]

        # build smplx model for whole seq
        smplx_model = build_body_model(
                        dict(
                            type='SMPLX',
                            keypoint_src='smplx',
                            keypoint_dst='smplx',
                            model_path='data/body_models/smplx',
                            gender='female',
                            num_betas=10,
                            use_face_contour=True,
                            flat_hand_mean=True,
                            use_pca=False,
                            batch_size=1)).to(self.device)
        
        # build camera for sequence
        pv_info_path = glob.glob(os.path.join(dataset_p, 'egocentric_color', seq, '202*', '*_pv.txt'))[0]
        with open(pv_info_path) as f:
            lines = f.readlines()
        holo_cx, holo_cy, holo_w, holo_h = ast.literal_eval(lines[0])
        ## for a sequence, the focal length is the same
        focal_length_x, focal_length_y = ast.literal_eval(lines[1])[1:3]

        camera = build_cameras(
            dict(
                type='PerspectiveCameras',
                convention='pyrender',
                in_ndc=False,
                focal_length=np.array([focal_length_x, focal_length_y]).reshape(-1, 2),
                image_size=(holo_h, holo_w),
                principal_point=np.array([holo_cx, holo_cy]).reshape(-1, 2))).to(self.device)

        # iterate through valid frames
        for idx in valid_idx:
            frame_idx = frame_list[idx]

            # get smplx data
            smplx_idx = valid_smplx_list.index(frame_idx)
            smplx_interactee = dict(np.load(smplx_interatcee_ps[smplx_idx], allow_pickle=True))

            ## revert smplx hands pca
            hand_pca_comps = smplx_interactee['left_hand_pose'].shape[1]
            smplx_interactee = self._revert_smplx_hands_pca(smplx_interactee, 
                            hand_pca_comps, gender=smplx_interactee['gender'])
            
            ## get smplx keypoints: pelvis for camera transform
            output = smplx_model(
                    global_orient=torch.tensor(smplx_interactee['global_orient'], device=self.device),
                    body_pose=torch.tensor(smplx_interactee['body_pose'], device=self.device),
                    betas=torch.tensor(smplx_interactee['betas'], device=self.device),
                    transl=torch.tensor(smplx_interactee['transl'], device=self.device),
                    left_hand_pose=torch.tensor(smplx_interactee['left_hand_pose'], device=self.device),
                    right_hand_pose=torch.tensor(smplx_interactee['right_hand_pose'], device=self.device),
                    return_joints=True)
            keypoints_3d = output['joints']
            keypoints_3d = keypoints_3d.detach().cpu().numpy()
            
            pelvis_world = keypoints_3d(get_keypoint_idx('pelvis', 'smplx'))


            keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
            keypoints_2d = keypoints_2d_xyd[..., :2]

            
            keypoints_2d = keypoints_2d.detach().cpu().numpy()







            pdb.set_trace()


        pass


    
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

        # get trageted sequence list
        batch_name, batch_part = mode.split('_')
        seqs = pd.read_csv(os.path.join(dataset_path, 'data_splits.csv'))[batch_part]
        meta_df = pd.read_csv(os.path.join(dataset_path, 'data_info_release.csv'))

        # seed, size = '230412', '00100'
        # random.seed(int(seed))
        # random.shuffle(npzs)
        # seqs = seqs[:int(size)]
        print(len(seqs))

    
        for seq in seqs:
            # for seq in tqdm(seqs, desc='Extracting Sequence Data'):
            # get all possible valid frame
            seq_meta = meta_df[meta_df['recording_name'] == seq]
            frame_idxs = np.arange(seq_meta['start_frame'].values[0], seq_meta['end_frame'].values[0] + 1, 1)

            seq_p = os.path.join(dataset_path, mode, seq)
            params = self._get_seq_data(dataset_path, seq, frame_list=frame_idxs,
                                              mode=batch_name, batch_part=batch_part)

    
                




        pass