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
class MotionXConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train']

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
        self.bbox_mapping = {
            'bbox_xywh': 'bbox',
            'face_bbox_xywh': 'face_box',
            'lhand_bbox_xywh': 'lefthand_box',
            'rhand_bbox_xywh': 'righthand_box'
        }
        self.smplx_mapping = {
            'betas': 'shape',
            'transl': 'trans',
            'global_orient': 'root_pose',
            'body_pose': 'body_pose',
            'left_hand_pose': 'lhand_pose',
            'right_hand_pose': 'rhand_pose',
            'jaw_pose': 'jaw_pose',
            'expression': 'expr'
        }

        super(MotionXConverter, self).__init__(modes)


    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        
        # parse seqs
        seqs = glob.glob(os.path.join(dataset_path, 'motion*'))

        # use HumanData to store all data
        human_data = HumanData()

        # random seed and size
        seed, size = '230727', '999999'

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
        for meta_key in ['principal_point', 'focal_length']:
            meta_[meta_key] = []
        image_path_ = []

        # parse seqs
        for seq in seqs:
            
            # load data
            with open(os.path.join(seq, 'motions', 'smplx_label.json'), 'r') as f:
                smplx_label = json.load(f)
            smplx_param_322 = np.load(os.path.join(seq, 'motions', 'smplx_322.npy'))

            # seq_p = 
            img_ps = smplx_label['file_name']

            for key in smplx_label.keys():
                print(key, np.array(smplx_label[key]).shape)
                
            for idx, imgp in enumerate(img_ps):

                image_path = os.path.join(seq, 'images', imgp)

                smplx_instance = smplx_label['smplx_param'][idx]['smplx_param']
                cam_instance = smplx_label['smplx_param'][idx]['cam_param']

                smplx_temp = {}
                for key in self.smplx_mapping.keys():
                    smplx_temp[key] = np.array(smplx_instance[self.smplx_mapping[key]],
                                                dtype=np.float32).reshape(self.smplx_shape[key])
                focal_length = cam_instance['focal']
                principal_point = cam_instance['princpt']

                pdb.set_trace()

                # append
                meta_['principal_point'].append(principal_point)
                meta_['focal_length'].append(focal_length)

                
        for key in smplx_.keys():
            smplx_[key] = np.concatenate(
                smplx_[key], axis=0).reshape(self.smplx_shape[key])
        human_data['smplx'] = smplx_

        for key in bboxs_.keys():
            bbox_ = np.array(bboxs_[key]).reshape((-1, 5))
            human_data[key] = bbox_

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

        # image path
        human_data['image_path'] = image_path_

        # meta
        human_data['meta'] = meta_

        # store
        human_data['config'] = f'motionx_{mode}'
        human_data['misc'] = self.misc_config

        size_i = int(size)

        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        print('HumanData dumping starts at',
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        out_file = os.path.join(
            out_path, f'motionx_{mode}_{seed}_{"{:06d}".format(size_i)}.npz')
        human_data.dump(out_file)

