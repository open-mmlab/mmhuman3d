import os
from typing import List

import time
import numpy as np
import pandas as pd
import json
import cv2
import glob
import random
from tqdm import tqdm
from multiprocessing import Pool
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
class ShapyConverter(BaseModeConverter):

    ACCEPTED_MODES = ['test', 'val']

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.misc_config = dict(
            face_bbox='by_dataset', hand_bbpx='by_dataset', bbox='by_dataset', keypoints2d='by_dataset',
        )
        self.smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
                'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
                'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}

        super(ShapyConverter, self).__init__(modes)

    
    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:
        
        # use HumanData to store data
        human_data = HumanData()

        # initialize 
        image_path_ = []
        bbox_xywh_, face_bbox_xywh_, lhand_bbox_xywh_, rhand_bbox_xywh_ = [], [], [], []
        kps2d_b_, kps2d_lh_, kps2d_rh_, kps2d_f_ = [], [], [], []
        meta_ = {}
        meta_['width'], meta_['height'] = [], []
        # height_, width_ = [], []
        # batch_name_, description_ = [], []

        # get keypoint annotations
        kps_anno_ps = glob.glob(os.path.join(dataset_path, 'HBW', 'keypoints', 
                                           f'{mode}', '**','*.json'), recursive=True)
        kps_anno_ps = [p for p in kps_anno_ps if mode in p]

        # init betas for val
        smplx_ = {}
        smplx_['betas'] = []
        betas_path = os.path.join(dataset_path, 'HBW', 'shapy_val_params')

        for kps_anno_p in tqdm(kps_anno_ps):
            with open(kps_anno_p, 'r') as f:
                kps_anno = json.load(f)
            # get image path
            image_p = kps_anno_p.replace('keypoints', 'images').replace('.json', '.png')
            if not os.path.exists(image_p):
                continue

            # check image size
            img = cv2.imread(image_p)
            height, width, _ = img.shape
            meta_['width'].append(width)
            meta_['height'].append(height)

            # get image path
            root_idx = image_p.split(os.path.sep).index('shapy')
            image_path = os.path.sep.join(image_p.split(os.path.sep)[root_idx+1:])
            image_path_.append(image_path)

            # get keypoints            
            kps2d_b = np.array(kps_anno['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
            kps2d_lh = np.array(kps_anno['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)
            kps2d_rh = np.array(kps_anno['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)
            kps2d_f = np.array(kps_anno['people'][0]['face_keypoints_2d']).reshape(-1, 3)

            kps2d_b_.append(kps2d_b)
            kps2d_lh_.append(kps2d_lh)
            kps2d_rh_.append(kps2d_rh)
            kps2d_f_.append(kps2d_f)


            if mode == 'val':
                person_id = image_p.split(os.path.sep)[-3][:3]
                betas_param = dict(np.load(os.path.join(betas_path, f'{person_id}_param.npz'), allow_pickle=True))['betas']
                smplx_['betas'].append(betas_param)

            # kps3d_f = np.array(kps_anno['people'][0]['face_keypoints_3d']).reshape(-1, 3)
            # kps3d_lh = np.array(kps_anno['people'][0]['hand_left_keypoints_3d']).reshape(-1, 3)
            # kps3d_rh = np.array(kps_anno['people'][0]['hand_right_keypoints_3d']).reshape(-1, 3)
            # kps3d_b = np.array(kps_anno['people'][0]['pose_keypoints_3d']).reshape(-1, 3)

            # import cv2
            # img = cv2.imread(image_p)
            # for kp in kps2d_f:
            #     # draw keypoints on image
            #     img = cv2.circle(img, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), -1)
            # for kp_idx, kp in enumerate(kps2d_lh):
            #     # draw keypoints on image
            #     img = cv2.circle(img, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
            #     # write keypoints index on image
            #     img = cv2.putText(img, str(kp_idx), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # for kp in kps2d_rh:
            #     # draw keypoints on image
            #     img = cv2.circle(img, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
            # for kp_idx, kp in enumerate(kps2d_b):
            #     # draw keypoints on image
            #     img = cv2.circle(img, (int(kp[0]), int(kp[1])), 5, (255, 255, 0), -1)
            #     # write keypoints index on image
            #     img = cv2.putText(img, str(kp_idx), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            # # save image
            # cv2.imwrite('/mnt/d/shapy/output/test.png', img)

            # get bbox
            bbox_xyxy = self._keypoints_to_scaled_bbox(kps2d_b[..., :2], scale=1.35)
            bbox_xywh = self._xyxy2xywh(bbox_xyxy)

            lhand_bbox_xyxy = self._keypoints_to_scaled_bbox(kps2d_lh[..., :2], scale=1.0)
            lhand_bbox_xywh = self._xyxy2xywh(lhand_bbox_xyxy)

            rhand_bbox_xyxy = self._keypoints_to_scaled_bbox(kps2d_rh[..., :2], scale=1.0)
            rhand_bbox_xywh = self._xyxy2xywh(rhand_bbox_xyxy)

            face_bbox_xyxy = self._keypoints_to_scaled_bbox(kps2d_f[..., :2], scale=1.0)
            face_bbox_xywh = self._xyxy2xywh(face_bbox_xyxy)

            for bbox in [bbox_xywh, lhand_bbox_xywh, rhand_bbox_xywh, face_bbox_xywh]:
                if bbox[2] * bbox[3] > 0:
                    bbox.append(1)
                else:
                    bbox.append(0)
            bbox_xywh_.append(bbox_xywh)
            lhand_bbox_xywh_.append(lhand_bbox_xywh)
            rhand_bbox_xywh_.append(rhand_bbox_xywh)
            face_bbox_xywh_.append(face_bbox_xywh)

        # smplx betas
        if mode == 'val':
            smplx_['betas'] = np.array(smplx_['betas']).reshape(-1, 10)
            human_data['smplx'] = smplx_

        # keypoints
        kps2d_b_ = np.array(kps2d_b_)
        kps2d_lh_ = np.array(kps2d_lh_)
        kps2d_rh_ = np.array(kps2d_rh_)
        kps2d_f_ = np.array(kps2d_f_)

        keypoints2d = np.concatenate([kps2d_b_, kps2d_lh_, kps2d_rh_, kps2d_f_], axis=1).reshape(-1, 137, 3)
        keypoints2d, keypoints2d_mask = \
            convert_kps(keypoints2d, src='openpose_137', dst='human_data')
        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask

        # image path
        human_data['image_path'] = image_path_

        # bbox
        human_data['bbox_xywh'] = np.array(bbox_xywh_).reshape(-1, 5)
        human_data['face_bbox_xywh'] = np.array(face_bbox_xywh_).reshape(-1, 5)
        human_data['lhand_bbox_xywh'] = np.array(lhand_bbox_xywh_).reshape(-1, 5)
        human_data['rhand_bbox_xywh'] = np.array(rhand_bbox_xywh_).reshape(-1, 5)

        # misc
        human_data['misc'] = self.misc_config

        # meta
        human_data['meta'] = meta_

        # save
        # human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        seed, size = '230512', len(human_data['image_path'])
        out_file = os.path.join(out_path, f'shapy_{mode}_{seed}_{str(size)}.npz')
        human_data.dump(out_file)
            










        
            