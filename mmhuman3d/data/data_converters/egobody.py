import ast
import glob
import json
import os
import pdb
import random
import time
from typing import List

import cv2
import numpy as np
import pandas as pd
import smplx
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData
# import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import (
    batch_transform_to_camera_frame,
    transform_to_camera_frame,
)
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class EgobodyConverter(BaseModeConverter):

    ACCEPTED_MODES = [
        'egocentric_train', 'egocentric_test', 'egocentric_val',
        'kinect_train', 'kinect_test', 'kinect_val'
    ]

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.misc_config = dict(
            bbox_source='keypoints2d_smplx',
            smplx_source='original',
            flat_hand_mean=False,
            camera_param_type='perspective',
            kps3d_root_aligned=False,
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
        )
        self.smplx_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 21, 3),
            'left_hand_pose': (-1, 15, 3),
            'right_hand_pose': (-1, 15, 3),
            'leye_pose': (-1, 3),
            'reye_pose': (-1, 3),
            'jaw_pose': (-1, 3),
            'expression': (-1, 10)
        }

        super(EgobodyConverter, self).__init__(modes)

    def _keypoints_to_scaled_bbox_bfh(self,
                                      keypoints,
                                      occ=None,
                                      body_scale=1.0,
                                      fh_scale=1.0,
                                      convention='smplx'):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale
        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format
        '''
        bboxs = []

        # supported kps.shape: (1, n, k) or (n, k), k = 2 or 3
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        if keypoints.shape[-1] != 2:
            keypoints = keypoints[:, :2]

        for body_part in ['body', 'head', 'left_hand', 'right_hand']:
            if body_part == 'body':
                scale = body_scale
                kps = keypoints
            else:
                scale = fh_scale
                kp_id = get_keypoint_idxs_by_part(
                    body_part, convention=convention)
                kps = keypoints[kp_id]

            if not occ is None:
                occ_p = occ[kp_id]
                if np.sum(occ_p) / len(kp_id) >= 0.1:
                    conf = 0
                    # print(f'{body_part} occluded, occlusion: {np.sum(occ_p) / len(kp_id)}, skip')
                else:
                    # print(f'{body_part} good, {np.sum(self_occ_p + occ_p) / len(kp_id)}')
                    conf = 1
            else:
                conf = 1
            if body_part == 'body':
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

            bbox = np.stack([xmin, ymin, xmax, ymax, conf],
                            axis=0).astype(np.float32)
            bboxs.append(bbox)

        return bboxs

    def _revert_smplx_hands_pca(self,
                                param_dict,
                                num_pca_comps,
                                gender='neutral',
                                gendered_model=None):
        # egobody 12
        hl_pca = param_dict['left_hand_pose']
        hr_pca = param_dict['right_hand_pose']

        if gendered_model != None:
            smplx_model = gendered_model
        else:
            smplx_model = dict(
                np.load(
                    f'data/body_models/smplx/SMPLX_{gender.upper()}.npz',
                    allow_pickle=True))

        hl = smplx_model['hands_componentsl']  # 45, 45
        hr = smplx_model['hands_componentsr']  # 45, 45

        hl_pca = np.concatenate(
            (hl_pca, np.zeros((len(hl_pca), 45 - num_pca_comps))), axis=1)
        hr_pca = np.concatenate(
            (hr_pca, np.zeros((len(hr_pca), 45 - num_pca_comps))), axis=1)

        hl_reverted = np.einsum('ij, jk -> ik', hl_pca, hl).astype(np.float32)
        hr_reverted = np.einsum('ij, jk -> ik', hr_pca, hr).astype(np.float32)

        param_dict['left_hand_pose'] = hl_reverted
        param_dict['right_hand_pose'] = hr_reverted

        return param_dict

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
        seqs = pd.read_csv(os.path.join(
            dataset_path, 'data_splits.csv'))[batch_part].dropna().to_list()
        meta_df = pd.read_csv(
            os.path.join(dataset_path, 'data_info_release.csv'))

        seed, size = '230622', '999'
        size_i = min(int(size), len(seqs))
        random.seed(int(seed))
        # random.shuffle(npzs)
        seqs = seqs[:int(size_i)]

        # initialize output for human_data
        smplx_ = {}
        for keys in self.smplx_shape.keys():
            smplx_[keys] = []
        keypoints2d_, keypoints3d_ = [], []
        bboxs_ = {}
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            bboxs_[bbox_name] = []
        meta_ = {}
        for meta_key in [
                'gender', 'character', 'view', 'principal_point',
                'focal_length'
        ]:
            meta_[meta_key] = []
        image_path_ = []

        # for seq in seqs:
        if batch_name == 'egocentric':
            for seq in tqdm(
                    seqs, desc='Extracting Sequence Data', leave=False):
                try:
                    # get all possible valid frame
                    seq_meta = meta_df[meta_df['recording_name'] == seq]
                    frame_list = np.arange(seq_meta['start_frame'].values[0],
                                           seq_meta['end_frame'].values[0] + 1,
                                           1)

                    # get all valid rdb image path
                    imgp = glob.glob(
                        os.path.join(dataset_path, f'{batch_name}_color', seq,
                                     '*', 'PV', '*.jpg'))

                    # get image shape
                    height, width, _ = cv2.imread(imgp[0]).shape

                    # get valid rgb frame list
                    valid_rgb_list = [
                        int(os.path.basename(p)[-9:-4]) for p in imgp
                    ]
                    valid_rgb_timestamp_list = [
                        int(os.path.basename(p)[:18]) for p in imgp
                    ]
                    valid_rgb_dict = dict(
                        zip(valid_rgb_timestamp_list, valid_rgb_list))

                    # get all valid keypoint path & image path
                    kp_npz_p = glob.glob(
                        os.path.join(dataset_path, 'egocentric_color', seq,
                                     '*', 'keypoints.npz'))[0]
                    kp_npz = dict(np.load(kp_npz_p, allow_pickle=True))
                    valid_kps_anno_list = [
                        int(os.path.basename(p)[-9:-4])
                        for p in kp_npz['imgname']
                    ]
                    keypoints3d_ego = kp_npz['keypoints']
                    image_list = kp_npz['imgname']

                    # get all valid smpl/smplx path
                    smplx_interatcee_ps = glob.glob(
                        os.path.join(dataset_path,
                                     f'smplx_interactee_{batch_part}', seq,
                                     '*', 'results', f'frame_*', '000.pkl'))
                    smplx_camera_wearer_ps = glob.glob(
                        os.path.join(dataset_path,
                                     f'smplx_camera_wearer_{batch_part}', seq,
                                     '*', 'results', f'frame_*', '000.pkl'))
                    smpl_interactee_ps = glob.glob(
                        os.path.join(dataset_path,
                                     f'smpl_interactee_{batch_part}', seq, '*',
                                     'results', f'frame_*', '000.pkl'))
                    smpl_camera_wearer_ps = glob.glob(
                        os.path.join(dataset_path,
                                     f'smpl_camera_wearer_{batch_part}', seq,
                                     '*', 'results', f'frame_*', '000.pkl'))
                    valid_smplx_list = [
                        int(p[-13:-8]) for p in smplx_interatcee_ps
                    ]

                    # get all valid depth path
                    pass

                    # get intercept of all valid list
                    ## valid_frame_list: frames that has all data included
                    valid_frame_list = sorted(
                        list(
                            set(valid_rgb_list) & set(valid_kps_anno_list)
                            & set(valid_smplx_list) & set(frame_list)))

                    valid_rgb_idx = [
                        valid_rgb_list.index(i) for i in valid_frame_list
                        if i in valid_rgb_list
                    ]
                    valid_kps_anno_idx = [
                        valid_kps_anno_list.index(i) for i in valid_frame_list
                        if i in valid_kps_anno_list
                    ]

                    # build smplx model for whole seq
                    gender = dict(
                        np.load(smplx_interatcee_ps[0],
                                allow_pickle=True))['gender']
                    smplx_model = build_body_model(
                        dict(
                            type='SMPLX',
                            keypoint_src='smplx',
                            keypoint_dst='smplx',
                            model_path='data/body_models/smplx',
                            gender=gender,
                            num_betas=10,
                            use_face_contour=True,
                            flat_hand_mean=False,
                            use_pca=False,
                            batch_size=1)).to(self.device)

                    # prepare possible paths
                    cam_calib_path = os.path.join(dataset_path, 'calibrations',
                                                  seq, 'cal_trans')

                    ## holo camera to kinect camera
                    holo2kinect_path = os.path.join(cam_calib_path,
                                                    'holo_to_kinect12.json')

                    with open(holo2kinect_path, 'r') as f:
                        trans_holo2kinect = np.array(json.load(f)['trans'])
                    trans_kinect2holo = np.linalg.inv(trans_holo2kinect)

                    ## RGB (hololens pv camera) to world camera
                    pv_info_path = glob.glob(
                        os.path.join(dataset_path, 'egocentric_color', seq,
                                     '*', '*_pv.txt'))[0]
                    with open(pv_info_path) as f:
                        lines = f.readlines()
                    holo_cx, holo_cy, holo_w, holo_h = ast.literal_eval(
                        lines[0])

                    holo_pv2world_trans_dict = {}
                    for i, frame in enumerate(lines[1:]):
                        frame = frame.split((','))
                        cur_timestamp = int(frame[0])  # string
                        cur_pv2world_transform = np.array(
                            frame[3:20]).astype(float).reshape((4, 4))

                        if cur_timestamp in valid_rgb_dict.keys():
                            cur_frame_id = valid_rgb_dict[cur_timestamp]
                            holo_pv2world_trans_dict[
                                cur_frame_id] = cur_pv2world_transform

                    # build camera for sequence
                    ## for a sequence, the focal length is the same
                    focal_length_x, focal_length_y = ast.literal_eval(
                        lines[1])[1:3]

                    camera = build_cameras(
                        dict(
                            type='PerspectiveCameras',
                            convention='opencv',
                            in_ndc=False,
                            focal_length=np.array(
                                [focal_length_x,
                                 focal_length_y]).reshape(-1, 2),
                            image_size=(holo_h, holo_w),
                            principal_point=np.array(
                                [holo_cx,
                                 holo_cy]).reshape(-1, 2))).to(self.device)

                    # iterate through valid frames
                    for idx, frame_idx in enumerate(
                            tqdm(
                                valid_frame_list,
                                desc=f'Processing {seq}',
                                position=1,
                                leave=False)):

                        # get smplx data
                        smplx_idx = valid_smplx_list.index(frame_idx)
                        smplx_interactee = dict(
                            np.load(
                                smplx_interatcee_ps[smplx_idx],
                                allow_pickle=True))

                        ## revert smplx hands pca
                        hand_pca_comps = smplx_interactee[
                            'left_hand_pose'].shape[1]
                        smplx_interactee = self._revert_smplx_hands_pca(
                            smplx_interactee,
                            hand_pca_comps,
                            gender=smplx_interactee['gender'])

                        ## get smplx keypoints
                        ### get pelvis for camera transform, transform params to hololens world coordinate
                        output = smplx_model(
                            global_orient=torch.tensor(
                                smplx_interactee['global_orient'],
                                device=self.device),
                            body_pose=torch.tensor(
                                smplx_interactee['body_pose'],
                                device=self.device),
                            betas=torch.tensor(
                                smplx_interactee['betas'], device=self.device),
                            transl=torch.tensor(
                                smplx_interactee['transl'],
                                device=self.device),
                            left_hand_pose=torch.tensor(
                                smplx_interactee['left_hand_pose'],
                                device=self.device),
                            right_hand_pose=torch.tensor(
                                smplx_interactee['right_hand_pose'],
                                device=self.device),
                            leye_pose=torch.tensor(
                                smplx_interactee['leye_pose'],
                                device=self.device),
                            reye_pose=torch.tensor(
                                smplx_interactee['reye_pose'],
                                device=self.device),
                            jaw_pose=torch.tensor(
                                smplx_interactee['jaw_pose'],
                                device=self.device),
                            expression=torch.tensor(
                                smplx_interactee['expression'],
                                device=self.device),
                            return_joints=True)
                        keypoints_3d = output['joints'].detach().cpu().numpy()
                        pelvis_world = keypoints_3d[
                            0, get_keypoint_idx('pelvis', 'smplx')]

                        ### transform smplx to hololens world coordinate
                        global_orient_holoworld, transl_holoworld = transform_to_camera_frame(
                            global_orient=smplx_interactee['global_orient'],
                            transl=smplx_interactee['transl'],
                            pelvis=pelvis_world,
                            extrinsic=trans_kinect2holo)

                        ### get pelvis for camera transform, transform params to RBG camera coordinate (egocentric)
                        output = smplx_model(
                            global_orient=torch.tensor(
                                np.array([global_orient_holoworld]),
                                device=self.device),
                            body_pose=torch.tensor(
                                smplx_interactee['body_pose'],
                                device=self.device),
                            betas=torch.tensor(
                                smplx_interactee['betas'], device=self.device),
                            transl=torch.tensor(
                                np.array([transl_holoworld]),
                                device=self.device),
                            left_hand_pose=torch.tensor(
                                smplx_interactee['left_hand_pose'],
                                device=self.device),
                            right_hand_pose=torch.tensor(
                                smplx_interactee['right_hand_pose'],
                                device=self.device),
                            leye_pose=torch.tensor(
                                smplx_interactee['leye_pose'],
                                device=self.device),
                            reye_pose=torch.tensor(
                                smplx_interactee['reye_pose'],
                                device=self.device),
                            jaw_pose=torch.tensor(
                                smplx_interactee['jaw_pose'],
                                device=self.device),
                            expression=torch.tensor(
                                smplx_interactee['expression'],
                                device=self.device),
                            return_joints=True)
                        keypoints_3d = output['joints'].detach().cpu().numpy()
                        pelvis_holoworld = keypoints_3d[
                            0, get_keypoint_idx('pelvis', 'smplx')]

                        ### transform smplx to pyrender RGB camera coordinate (egocentric)
                        cur_pv2world_transform = holo_pv2world_trans_dict[
                            frame_idx]
                        cur_world2pv_transform = np.linalg.inv(
                            cur_pv2world_transform)
                        global_orient_pyrender, transl_pyrender = transform_to_camera_frame(
                            global_orient=global_orient_holoworld,
                            transl=transl_holoworld,
                            pelvis=pelvis_holoworld,
                            extrinsic=cur_world2pv_transform)

                        ### get pelvis for pyrender to opencv camera transform
                        output = smplx_model(
                            global_orient=torch.tensor(
                                np.array([global_orient_pyrender]),
                                device=self.device),
                            body_pose=torch.tensor(
                                smplx_interactee['body_pose'],
                                device=self.device),
                            betas=torch.tensor(
                                smplx_interactee['betas'], device=self.device),
                            transl=torch.tensor(
                                np.array([transl_pyrender]),
                                device=self.device),
                            left_hand_pose=torch.tensor(
                                smplx_interactee['left_hand_pose'],
                                device=self.device),
                            right_hand_pose=torch.tensor(
                                smplx_interactee['right_hand_pose'],
                                device=self.device),
                            leye_pose=torch.tensor(
                                smplx_interactee['leye_pose'],
                                device=self.device),
                            reye_pose=torch.tensor(
                                smplx_interactee['reye_pose'],
                                device=self.device),
                            jaw_pose=torch.tensor(
                                smplx_interactee['jaw_pose'],
                                device=self.device),
                            expression=torch.tensor(
                                smplx_interactee['expression'],
                                device=self.device),
                            return_joints=True)
                        keypoints_3d = output['joints'].detach().cpu().numpy()
                        pelvis_pyrender = keypoints_3d[
                            0, get_keypoint_idx('pelvis', 'smplx')]

                        ### transform pyrender to opencv camera coordinate
                        pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                                    [0, -1, 0,
                                                     0], [0, 0, -1, 0],
                                                    [0, 0, 0, 1]])
                        global_orient_opencv, transl_opencv = transform_to_camera_frame(
                            global_orient=global_orient_pyrender,
                            transl=transl_pyrender,
                            pelvis=pelvis_pyrender,
                            extrinsic=pyrender2opencv)

                        ### get smplx keypoints
                        output = smplx_model(
                            global_orient=torch.tensor(
                                np.array([global_orient_opencv]),
                                device=self.device),
                            body_pose=torch.tensor(
                                smplx_interactee['body_pose'],
                                device=self.device),
                            betas=torch.tensor(
                                smplx_interactee['betas'], device=self.device),
                            transl=torch.tensor(
                                np.array([transl_opencv]), device=self.device),
                            left_hand_pose=torch.tensor(
                                smplx_interactee['left_hand_pose'],
                                device=self.device),
                            right_hand_pose=torch.tensor(
                                smplx_interactee['right_hand_pose'],
                                device=self.device),
                            leye_pose=torch.tensor(
                                smplx_interactee['leye_pose'],
                                device=self.device),
                            reye_pose=torch.tensor(
                                smplx_interactee['reye_pose'],
                                device=self.device),
                            jaw_pose=torch.tensor(
                                smplx_interactee['jaw_pose'],
                                device=self.device),
                            expression=torch.tensor(
                                smplx_interactee['expression'],
                                device=self.device),
                            return_joints=True)
                        # smplx_param['global_orient'] = global_orient_opencv
                        # smplx_param['transl'] = transl_opencv
                        keypoints_3d = output['joints']

                        ### transform 3d keypoints to 2d keypoints
                        keypoints_2d_xyd = camera.transform_points_screen(
                            keypoints_3d)
                        keypoints_2d = keypoints_2d_xyd[
                            ..., :2].detach().cpu().numpy()
                        keypoints_3d = keypoints_3d.detach().cpu().numpy()

                        # get bbox from 2d keypoints
                        bboxs = self._keypoints_to_scaled_bbox_bfh(
                            keypoints_2d,
                            body_scale=self.misc_config['bbox_body_scale'],
                            fh_scale=self.misc_config['bbox_facehand_scale'])
                        ## convert xyxy to xywh
                        for i, bbox_name in enumerate([
                                'bbox_xywh', 'face_bbox_xywh',
                                'lhand_bbox_xywh', 'rhand_bbox_xywh'
                        ]):
                            xmin, ymin, xmax, ymax, conf = bboxs[i]
                            bbox = np.array([
                                max(0, xmin),
                                max(0, ymin),
                                min(width, xmax),
                                min(height, ymax)
                            ])
                            bbox_xywh = self._xyxy2xywh(bbox)  # list of len 4
                            bbox_xywh.append(conf)  # (5,)
                            bboxs_[bbox_name].append(bbox_xywh)

                        # append frame specific data
                        ## smplx params
                        for key in [
                                'body_pose', 'betas', 'left_hand_pose',
                                'right_hand_pose', 'leye_pose', 'reye_pose',
                                'jaw_pose', 'expression'
                        ]:
                            smplx_[key].append(smplx_interactee[key])
                        smplx_['global_orient'].append(global_orient_opencv)
                        smplx_['transl'].append(transl_opencv)
                        ## keypoints
                        keypoints2d_.append(keypoints_2d)
                        keypoints3d_.append(keypoints_3d)

                        # meta
                        meta_['principal_point'].append((holo_cx, holo_cy))
                        meta_['focal_length'].append(
                            (focal_length_x, focal_length_y))

                    # prepare for output
                    for p in image_list[valid_kps_anno_idx].tolist():
                        image_path_.append(p)
                        meta_['gender'].append(gender)
                    keypoints3d_ego_ = keypoints3d_ego[valid_kps_anno_idx]
                except FloatingPointError:
                    print('Error in seq', seq)
                    continue

            for key in smplx_.keys():
                smplx_[key] = np.array(smplx_[key]).reshape(
                    self.smplx_shape[key])
            human_data['smplx'] = smplx_
            print('Smpl and/or Smplx finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            for key in bboxs_.keys():
                bbox_ = np.array(bboxs_[key]).reshape((-1, 5))
                human_data[key] = bbox_
            print('BBox generation finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            # keypoints 2d
            keypoints2d = np.array(keypoints2d_).reshape(-1, 144, 2)
            keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
            keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf],
                                         axis=-1)
            keypoints2d, keypoints2d_mask = \
                    convert_kps(keypoints2d, src='smplx', dst='human_data')
            human_data['keypoints2d_smplx'] = keypoints2d
            human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

            # keypoints 3d
            keypoints3d = np.array(keypoints3d_).reshape(-1, 144, 3)
            keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
            keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf],
                                         axis=-1)
            keypoints3d, keypoints3d_mask = \
                    convert_kps(keypoints3d, src='smplx', dst='human_data')
            human_data['keypoints3d_smplx'] = keypoints3d
            human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

            # keypoints 3d ego
            keypoints3d_ego = np.array(keypoints3d_ego_).reshape(-1, 25, 3)
            keypoints3d_ego_conf = np.ones([keypoints3d_ego.shape[0], 25, 1])
            keypoints3d_ego = np.concatenate(
                [keypoints3d_ego, keypoints3d_ego_conf], axis=-1)
            keypoints3d_ego, keypoints3d_ego_mask = \
                    convert_kps(keypoints3d_ego, src='openpose_25', dst='human_data')
            human_data['keypoints3d_original'] = keypoints3d_ego
            human_data['keypoints3d_original_mask'] = keypoints3d_ego_mask
            print('Keypoint conversion finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            # image path
            human_data['image_path'] = image_path_
            print('Image path writting finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            # meta
            human_data['meta'] = meta_
            print('MetaData finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            # store
            human_data['config'] = f'egobody_{mode}'
            human_data['misc'] = self.misc_config

        if batch_name == 'kinect':

            # calibration from sub kinect to main kinect (between color cameras)
            kinect_calib_dict = {
                'sub_1': 'kinect_11to12_color.json',
                'sub_2': 'kinect_13to12_color.json',
                'sub_3': 'kinect_14to12_color.json',
                'sub_4': 'kinect_15to12_color.json',
                'master': '',
            }

            # load smplx gendered model
            gendered_model = {}
            gendered_model['male'] = dict(
                np.load(
                    f'data/body_models/smplx/SMPLX_MALE.npz',
                    allow_pickle=True))
            gendered_model['female'] = dict(
                np.load(
                    f'data/body_models/smplx/SMPLX_FEMALE.npz',
                    allow_pickle=True))

            # seqs = seqs[:4]

            for seq in tqdm(
                    seqs, desc='Extracting Sequences Data', leave=False):
                try:
                    # get all possible valid frame
                    seq_meta = meta_df[meta_df['recording_name'] == seq]
                    frame_list = np.arange(seq_meta['start_frame'].values[0],
                                           seq_meta['end_frame'].values[0] + 1,
                                           1)

                    # get all possible views
                    avaliable_views = os.listdir(
                        os.path.join(dataset_path, f'{batch_name}_color', seq))

                    for view in tqdm(
                            avaliable_views,
                            desc='Extracting Views Data',
                            leave=False,
                            position=1):

                        # print(f'Preparing {seq} {view} data paths', end='/r')
                        # get all valid rdb image path
                        imgp = glob.glob(
                            os.path.join(dataset_path, f'{batch_name}_color',
                                         seq, view, '*.jpg'))

                        # get image shape
                        height, width, _ = cv2.imread(imgp[0]).shape

                        # get valid rgb frame list
                        valid_rgb_list = [
                            int(os.path.basename(p)[-9:-4]) for p in imgp
                        ]

                        # get image path
                        root_folder_id = dataset_path.lower().split(
                            os.path.sep).index('egobody')
                        image_path = [
                            os.path.sep.join(
                                p.split('/')[root_folder_id + 1:])
                            for p in imgp
                        ]

                        # get all valid smpl/smplx path
                        smplx_interatcee_ps = glob.glob(
                            os.path.join(dataset_path,
                                         f'smplx_interactee_{batch_part}', seq,
                                         '*', 'results', f'frame_*',
                                         '000.pkl'))
                        smplx_camera_wearer_ps = glob.glob(
                            os.path.join(dataset_path,
                                         f'smplx_camera_wearer_{batch_part}',
                                         seq, '*', 'results', f'frame_*',
                                         '000.pkl'))
                        smpl_interactee_ps = glob.glob(
                            os.path.join(dataset_path,
                                         f'smpl_interactee_{batch_part}', seq,
                                         '*', 'results', f'frame_*',
                                         '000.pkl'))
                        smpl_camera_wearer_ps = glob.glob(
                            os.path.join(dataset_path,
                                         f'smpl_camera_wearer_{batch_part}',
                                         seq, '*', 'results', f'frame_*',
                                         '000.pkl'))
                        valid_smplx_ee_list = [
                            int(p[-13:-8]) for p in smplx_interatcee_ps
                        ]
                        valid_smplx_wr_list = [
                            int(p[-13:-8]) for p in smplx_camera_wearer_ps
                        ]

                        # get all valid depth path
                        pass

                        # print(f'Preparing {seq} {view} smplx model', end='\r')
                        # get intercept of all valid list
                        ## valid_frame_list: frames that has all data included
                        valid_frame_list = sorted(
                            list(
                                set(valid_smplx_ee_list) & set(valid_rgb_list)
                                & set(valid_smplx_wr_list) & set(frame_list)))

                        valid_rgb_idx = [
                            valid_rgb_list.index(i) for i in valid_frame_list
                            if i in valid_rgb_list
                        ]

                        # build smplx model for whole seq
                        gender_interactee = dict(
                            np.load(smplx_interatcee_ps[0],
                                    allow_pickle=True))['gender']
                        smplx_model_interactee = build_body_model(
                            dict(
                                type='SMPLX',
                                keypoint_src='smplx',
                                keypoint_dst='smplx',
                                model_path='data/body_models/smplx',
                                gender=gender_interactee,
                                num_betas=10,
                                use_face_contour=True,
                                flat_hand_mean=False,
                                use_pca=False,
                                batch_size=1)).to(self.device)

                        gender_wearer = dict(
                            np.load(
                                smplx_camera_wearer_ps[0],
                                allow_pickle=True))['gender']
                        smplx_model_wearer = build_body_model(
                            dict(
                                type='SMPLX',
                                keypoint_src='smplx',
                                keypoint_dst='smplx',
                                model_path='data/body_models/smplx',
                                gender=gender_wearer,
                                num_betas=10,
                                use_face_contour=True,
                                flat_hand_mean=False,
                                use_pca=False,
                                batch_size=1)).to(self.device)

                        # prepare possible paths
                        cam_calib_path = os.path.join(dataset_path,
                                                      'calibrations', seq,
                                                      'cal_trans')

                        ## sub kinect to main kinect and reverse
                        if view != 'master':
                            trans_subtomain_path = os.path.join(
                                cam_calib_path, kinect_calib_dict[view])
                            with open(trans_subtomain_path, 'r') as f:
                                trans_subtomain = np.array(
                                    json.load(f)['trans'])
                            trans_maintosub = np.linalg.inv(trans_subtomain)

                        ## read kinect color camera intrinsics
                        with open(
                                os.path.join(dataset_path, 'kinect_cam_params',
                                             f'kinect_{view}', 'Color.json'),
                                'r') as f:
                            color_cam = json.load(f)
                        [f_x, f_y] = color_cam['f']
                        [c_x, c_y] = color_cam['c']

                        # build camera for sequence
                        ## for a sequence, the focal length is the same
                        (holo_h, holo_w) = cv2.imread(imgp[0]).shape[:2]
                        camera = build_cameras(
                            dict(
                                type='PerspectiveCameras',
                                convention='opencv',
                                in_ndc=False,
                                focal_length=np.array([f_x,
                                                       f_y]).reshape(-1, 2),
                                image_size=(holo_h, holo_w),
                                principal_point=np.array([c_x, c_y]).reshape(
                                    -1, 2))).to(self.device)

                        # valid_frame_list = valid_frame_list[0:10]

                        # print(f'Preparing {seq} {view} frame list', end='\r')
                        # iterate through valid frames to load smplx
                        smplx_interactee = []
                        smplx_wearer = []
                        for frame_idx in tqdm(
                                valid_frame_list,
                                desc=f'Preparing {seq} {view} frame list',
                                leave=False,
                                position=2):

                            # get smplx data
                            smplx_ee_idx = valid_smplx_ee_list.index(frame_idx)
                            smplx_interactee_temp = dict(
                                np.load(
                                    smplx_interatcee_ps[smplx_ee_idx],
                                    allow_pickle=True))
                            smplx_wr_idx = valid_smplx_wr_list.index(frame_idx)
                            smplx_wearer_temp = dict(
                                np.load(
                                    smplx_camera_wearer_ps[smplx_wr_idx],
                                    allow_pickle=True))

                            ## revert smplx hands pca
                            hand_pca_comps_ee = smplx_interactee_temp[
                                'left_hand_pose'].shape[1]
                            smplx_interactee.append(
                                self._revert_smplx_hands_pca(
                                    smplx_interactee_temp,
                                    hand_pca_comps_ee,
                                    gendered_model=gendered_model[
                                        smplx_interactee_temp['gender']]))
                            hand_pca_comps_wr = smplx_wearer_temp[
                                'left_hand_pose'].shape[1]
                            smplx_wearer.append(
                                self._revert_smplx_hands_pca(
                                    smplx_wearer_temp,
                                    hand_pca_comps_wr,
                                    gendered_model=gendered_model[
                                        smplx_wearer_temp['gender']]))

                        ## get smplx keypoints for interactee and wearer
                        character_list = ['interactee', 'wearer']
                        for character in character_list:
                            # print(f'Preparing {seq} {view} character: {character}', end='\r')
                            if character == 'interactee':
                                smplx_model = smplx_model_interactee
                                smplx_data = smplx_interactee
                            if character == 'wearer':
                                smplx_model = smplx_model_wearer
                                smplx_data = smplx_wearer

                            smplx_param = {}
                            for key in self.smplx_shape.keys():
                                smplx_param[key] = np.concatenate(
                                    [data[key] for data in smplx_data], axis=0)

                            if view != 'master':
                                # transfrom from main to sub
                                output = smplx_model(
                                    global_orient=torch.tensor(
                                        smplx_param['global_orient'],
                                        device=self.device),
                                    body_pose=torch.tensor(
                                        smplx_param['body_pose'],
                                        device=self.device),
                                    betas=torch.tensor(
                                        smplx_param['betas'],
                                        device=self.device),
                                    transl=torch.tensor(
                                        smplx_param['transl'],
                                        device=self.device),
                                    left_hand_pose=torch.tensor(
                                        smplx_param['left_hand_pose'],
                                        device=self.device),
                                    right_hand_pose=torch.tensor(
                                        smplx_param['right_hand_pose'],
                                        device=self.device),
                                    leye_pose=torch.tensor(
                                        smplx_param['leye_pose'],
                                        device=self.device),
                                    reye_pose=torch.tensor(
                                        smplx_param['reye_pose'],
                                        device=self.device),
                                    jaw_pose=torch.tensor(
                                        smplx_param['jaw_pose'],
                                        device=self.device),
                                    expression=torch.tensor(
                                        smplx_param['expression'],
                                        device=self.device),
                                    return_joints=True)
                                keypoints_3d = output['joints'].detach().cpu(
                                ).numpy()
                                pelvis_world = keypoints_3d[:,
                                                            get_keypoint_idx(
                                                                'pelvis',
                                                                'smplx'), :]

                                global_orient_sub, transl_sub = batch_transform_to_camera_frame(
                                    global_orient=smplx_param['global_orient'],
                                    transl=smplx_param['transl'],
                                    pelvis=pelvis_world,
                                    extrinsic=trans_maintosub)
                                global_orient_sub = np.array(
                                    [global_orient_sub]).reshape(-1, 3)
                                transl_sub = np.array([transl_sub
                                                       ]).reshape(-1, 3)
                            else:
                                global_orient_sub = smplx_param[
                                    'global_orient']
                                transl_sub = smplx_param['transl']

                            ### get smplx keypoints
                            output = smplx_model(
                                global_orient=torch.tensor(
                                    global_orient_sub, device=self.device),
                                body_pose=torch.tensor(
                                    smplx_param['body_pose'],
                                    device=self.device),
                                betas=torch.tensor(
                                    smplx_param['betas'], device=self.device),
                                transl=torch.tensor(
                                    transl_sub, device=self.device),
                                left_hand_pose=torch.tensor(
                                    smplx_param['left_hand_pose'],
                                    device=self.device),
                                right_hand_pose=torch.tensor(
                                    smplx_param['right_hand_pose'],
                                    device=self.device),
                                leye_pose=torch.tensor(
                                    smplx_param['leye_pose'],
                                    device=self.device),
                                reye_pose=torch.tensor(
                                    smplx_param['reye_pose'],
                                    device=self.device),
                                jaw_pose=torch.tensor(
                                    smplx_param['jaw_pose'],
                                    device=self.device),
                                expression=torch.tensor(
                                    smplx_param['expression'],
                                    device=self.device),
                                return_joints=True)
                            smplx_param['global_orient'] = global_orient_sub
                            smplx_param['transl'] = transl_sub
                            keypoints_3d = output['joints']

                            ### transform 3d keypoints to 2d keypoints
                            keypoints_2d_xyd = camera.transform_points_screen(
                                keypoints_3d)
                            keypoints_2d = keypoints_2d_xyd[
                                ..., :2].detach().cpu().numpy()
                            keypoints_3d = keypoints_3d.detach().cpu().numpy()

                            # get bbox from 2d keypoints
                            for keypoints_2d_frame in keypoints_2d:
                                bboxs = self._keypoints_to_scaled_bbox_bfh(
                                    keypoints_2d_frame,
                                    body_scale=self.
                                    misc_config['bbox_body_scale'],
                                    fh_scale=self.
                                    misc_config['bbox_facehand_scale'])
                                ## convert xyxy to xywh
                                for i, bbox_name in enumerate([
                                        'bbox_xywh', 'face_bbox_xywh',
                                        'lhand_bbox_xywh', 'rhand_bbox_xywh'
                                ]):
                                    xmin, ymin, xmax, ymax, conf = bboxs[i]
                                    bbox = np.array([
                                        max(0, xmin),
                                        max(0, ymin),
                                        min(width, xmax),
                                        min(height, ymax)
                                    ])
                                    bbox_xywh = self._xyxy2xywh(
                                        bbox)  # list of len 4
                                    bbox_xywh.append(conf)  # (5,)
                                    bboxs_[bbox_name].append(bbox_xywh)

                            # append frame specific data
                            ## smplx params

                            for key in [
                                    'body_pose', 'betas', 'left_hand_pose',
                                    'right_hand_pose', 'leye_pose',
                                    'reye_pose', 'jaw_pose', 'expression'
                            ]:
                                smplx_[key].append(smplx_param[key])
                            smplx_['global_orient'].append(global_orient_sub)
                            smplx_['transl'].append(transl_sub)
                            ## keypoints
                            keypoints2d_.append(keypoints_2d)
                            keypoints3d_.append(keypoints_3d)

                            # append meta data
                            for frame_idx in valid_frame_list:
                                image_path_.append(image_path[
                                    valid_rgb_list.index(frame_idx)])
                                meta_['view'].append(view)
                                meta_['character'].append(character)
                                meta_['gender'].append(smplx_model.gender)
                                meta_['focal_length'].append([f_x, f_y])
                                meta_['principal_point'].append([c_x, c_y])

                except FloatingPointError:
                    print('Error in seq', seq)
                    continue

            for key in smplx_.keys():
                smplx_[key] = np.concatenate(
                    smplx_[key], axis=0).reshape(self.smplx_shape[key])
            human_data['smplx'] = smplx_
            print('Smpl and/or Smplx finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            for key in bboxs_.keys():
                bbox_ = np.array(bboxs_[key]).reshape((-1, 5))
                human_data[key] = bbox_
            print('BBox generation finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            # keypoints 2d
            keypoints2d = np.concatenate(
                keypoints2d_, axis=0).reshape(-1, 144, 2)
            keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
            keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf],
                                         axis=-1)
            keypoints2d, keypoints2d_mask = \
                    convert_kps(keypoints2d, src='smplx', dst='human_data')
            human_data['keypoints2d_smplx'] = keypoints2d
            human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

            # keypoints 3d
            keypoints3d = np.concatenate(
                keypoints3d_, axis=0).reshape(-1, 144, 3)
            keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
            keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf],
                                         axis=-1)
            keypoints3d, keypoints3d_mask = \
                    convert_kps(keypoints3d, src='smplx', dst='human_data')
            human_data['keypoints3d_smplx'] = keypoints3d
            human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

            print('Keypoint conversion finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            # image path
            human_data['image_path'] = image_path_
            print('Image path writting finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            # meta
            human_data['meta'] = meta_
            print('MetaData finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            # store
            human_data['config'] = f'egobody_{mode}'
            human_data['misc'] = self.misc_config

        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path, f'egobody_{mode}_{seed}_{"{:03d}".format(size_i)}.npz')
        human_data.dump(out_file)
