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
class HancoConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train', 'val']

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.misc_config = dict(
            bbox_source='keypoints2d_smplx',
            smplx_source='original',
            bbox_facehand_scale=1.2,
            flat_hand_mean=False,
            camera_param_type='perspective',
            kps3d_root_aligned=False,
            image_shape=(224, 224), # height, width
        )
        self.smplx_shape = {
            'left_hand_pose': (-1, 15, 3),
            'right_hand_pose': (-1, 15, 3),
        }
        self.mano_shape = {
            'pose': (-1, 15, 3),
            'betas': (-1, 10),
            'global_orient': (-1, 3),
            'transl': (-1, 3),
        }

        super(HancoConverter, self).__init__(modes)


    def _slice_mano_param(self, theta):
        # slice mano param list
        return theta[:3], theta[3:48], theta[48:58], theta[-3:]


    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        
        # all right hand
        avaliable_image_modes = ['rgb', 'rgb_color_auto', 'rgb_color_sample',
                                 'rgb_homo', 'rgb_merged']

        image_mode = 'rgb'

        # parse sequences
        seqs = glob.glob(os.path.join(dataset_path, image_mode, '*'))

        # get data split
        split_meta = json.load(open(os.path.join(dataset_path, 'meta.json')))

        # adjust by split
        if mode == 'train':
            seqs = [seq for idx, seq in enumerate(seqs) 
                    if np.sum(split_meta['is_train'][idx]) > 0]

        else:
            seqs = [seq for idx, seq in enumerate(seqs) 
                    if np.sum(split_meta['is_train'][idx]) == 0]
    
        # use HumanData to store the data
        human_data = HumanData()

        # initialize
        smplx_ = {}
        for hand_type in ['right']:
            smplx_[f'{hand_type}_hand_pose'] = []
        bboxs_ = {}
        for hand_type in ['right']:  
            bboxs_[f'{hand_type[0]}hand_bbox_xywh'] = []
        bboxs_['bbox_xywh'] = []
        image_path_, keypoints2d_smplx_ = [], []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length']:
            meta_[meta_key] = []
        # save mano params for vis purpose
        mano_ = []

        seed = '230828'
        size = 9999

        for seq in tqdm(seqs, desc=f'Convert {mode}', 
                                   total=len(seqs), leave=False, position=0):
            # camera views
            seq_idx = int(os.path.basename(seq))
            camera_views = os.listdir(seq)
            
            # pdb.set_trace()
            valid_info = split_meta['has_fit'][seq_idx]
            # for cam in camera_views:
            for cam in tqdm(camera_views, desc=f'Sequence ID {os.path.basename(seq)}, 8 views', 
                            total=len(camera_views), leave=False, position=1):
                cid = int(cam.replace('cam', ''))
                frames = os.listdir(os.path.join(seq, cam))
                if mode == 'train':
                    frames = [frame for i, frame in enumerate(frames) if split_meta['is_train'][seq_idx][i]]
                else:
                    frames = [frame for i, frame in enumerate(frames) if not split_meta['is_train'][seq_idx][i]]
                frames = [frame for i, frame in enumerate(frames) if valid_info[i]]

                # print(f'seq ID {seq_idx}',np.array(valid_info).shape, len(frames), 
                #       np.sum(split_meta['is_train'][seq_idx]))
                # break

                anno_bp = os.path.join(seq, cam).replace(image_mode, 'shape')
                calib_bp = seq.replace(image_mode, 'calib')
                kps_bp = seq.replace(image_mode, 'xyz')

                for frame in tqdm(frames, desc=f'Processing {len(frames)} frames',
                                    total=len(frames), leave=False, position=2):
                    fname = frame[:-4]

                    image_path = os.path.join(seq, cam, frame).replace(f'{dataset_path}{os.path.sep}', '')

                    # camera calib
                    calib = json.load(open(os.path.join(calib_bp, f'{fname}.json')))
                    
                    # camera param
                    K = np.array(calib['K'][cid], dtype=np.float32)
                    M_w2cam = np.array(calib['M'][cid], dtype=np.float32)

                    # mano param
                    mano_param = json.load(open(os.path.join(anno_bp, f'{fname}.json')))
                    global_orient, pose_cam, shape_cam, global_t_cam = self._slice_mano_param(np.array(mano_param))
                    
                    # keypoints (21, 3)
                    j3d_w = np.array(json.load(open(os.path.join(kps_bp, f'{fname}.json'))))
                    j3d_c = np.matmul(j3d_w, M_w2cam[:3, :3].T) + M_w2cam[:3, 3][None]  # in camera coordinates

                    # project to 2d
                    j3d_temp = j3d_c / j3d_c[:, -1:]
                    j2d = np.matmul(j3d_temp, K.T)
                    j2d = j2d[:, :2] / j2d[:, -1:]

                    # get focal length and principal point
                    focal_length = (K[0, 0], K[1, 1])
                    principal_point = (K[0, 2], K[1, 2])

                    # for visualization test
                    camera = build_cameras(dict(
                        type='PerspectiveCameras',
                        convention='opencv',
                        in_ndc=False,
                        focal_length=focal_length,
                        principal_point=principal_point,
                        image_size=self.misc_config['image_shape'])).to(self.device)
                    # pdb.set_trace()      
                    j2d = camera.transform_points_screen(torch.tensor(
                                j3d_c.reshape(1, -1, 3), device=self.device, dtype=torch.float32))
                    j2d_orig = j2d[0,:,:2].detach().cpu().numpy()
                    # j2d = j2d_orig

                    # # plot kps
                    # img = cv2.imread(os.path.join(dataset_path, image_path))
                    # for i in range(j2d.shape[0]):
                    #     cv2.circle(img, (int(j2d[i, 0]), int(j2d[i, 1])), 3, (0, 0, 255), -1)
                    # cv2.imwrite(f'{out_path}/{fname}.png', img)

                    # append keypoints
                    j2d_conf = np.ones((j2d_orig.shape[0], 1))
                    j2d_orig = np.concatenate([j2d_orig, j2d_conf.reshape(-1, 1)], axis=-1)
                    keypoints2d_smplx_.append(j2d_orig)

                    # append bbox
                    bbox_rh = self._keypoints_to_scaled_bbox(j2d_orig, scale=1.2)
                    bbox_rh = [max(bbox_rh[0], 0), max(bbox_rh[1], 0), 
                               min(bbox_rh[2], self.misc_config['image_shape'][1]), 
                               min(bbox_rh[3], self.misc_config['image_shape'][0])]
                    bbox_rh_xywh = self._xyxy2xywh(bbox_rh)
                    bbox_rh_xywh += [1] # add confidence
                    bboxs_['rhand_bbox_xywh'].append(bbox_rh_xywh)
                    bbox_xywh = [0, 0, 224, 224, 1]
                    bboxs_['bbox_xywh'].append(bbox_xywh)
                    # append smplx
                    smplx_['right_hand_pose'].append(pose_cam.reshape(1, -1, 3))

                    # append mano
                    mano_.append(mano_param)
                    
                    # append meta
                    meta_['principal_point'].append(principal_point)
                    meta_['focal_length'].append(focal_length) 

                    # append image path
                    image_path_.append(image_path)


        size_i = min(size, len(seqs))
        # pdb.set_trace()

        # meta
        human_data['meta'] = meta_

        # image path
        human_data['image_path'] = image_path_

        # save bbox
        for bbox_name in bboxs_.keys():
            bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 5)
            human_data[bbox_name] = bbox_

        # save mano
        human_data['mano'] = mano_

        # save smplx
        human_data.skip_keys_check = ['smplx']
        for key in smplx_.keys():
            smplx_[key] = np.concatenate(
                smplx_[key], axis=0).reshape(self.smplx_shape[key])
        human_data['smplx'] = smplx_

        # keypoints2d_smplx
        keypoints2d_smplx = np.concatenate(
            keypoints2d_smplx_, axis=0).reshape(-1, 21, 3)
        keypoints2d_smplx, keypoints2d_smplx_mask = \
                convert_kps(keypoints2d_smplx, src='mano_right', dst='human_data')
        human_data['keypoints2d_smplx'] = keypoints2d_smplx
        human_data['keypoints2d_smplx_mask'] = keypoints2d_smplx_mask

        # misc
        human_data['misc'] = self.misc_config
        human_data['config'] = f'hanco_{mode}_{image_mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path,
            f'hanco_{mode}_{image_mode}_{seed}_{"{:04d}".format(size_i)}.npz')
        human_data.dump(out_file)