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
class SgnifyConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train']

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.misc_config = dict(
            # bbox_source='keypoints2d_smplx',
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
            # 'leye_pose': (-1, 3),
            # 'reye_pose': (-1, 3),
            'jaw_pose': (-1, 3),
            'expression': (-1, 10)
        }

        super(SgnifyConverter, self).__init__(modes)


    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:
        
        # get target seqs
        seqs = glob.glob(os.path.join(dataset_path, 'frames', '*'))

        # use HumanData to store all data
        human_data = HumanData()

        # random
        seed, size = '230717', '99'
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

        # parse each sequence
        for sid, seq in enumerate(seqs):
            
            img_ps = glob.glob(os.path.join(seq, '*.png'))

            for imgp in tqdm(img_ps, desc=f'Processing {sid + 1} / total {len(seqs)}',
                             position=0, leave=False):

                # image path
                image_path = imgp.replace(f'{dataset_path}{os.path.sep}', '')

                # annotation
                img_bn = os.path.basename(imgp)
                fid = int(img_bn[4:7])
                annp = os.path.join(seq, f'{"{:05d}".format(2*fid)}.npz') \
                    .replace('frames', 'fitted_params')
                anno = dict(np.load(annp, allow_pickle=True))
                # pdb.set_trace()

                # image path
                image_path_.append(image_path)

                # smplx
                for key in self.smplx_shape.keys():
                    smplx_[key].append(anno[key])

        # meta
        # human_data['meta'] = meta_

        # image path
        human_data['image_path'] = image_path_

        # save bbox
        # for bbox_name in bboxs_.keys():
        #     bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 5)
        #     human_data[bbox_name] = bbox_

        # save smplx
        # human_data.skip_keys_check = ['smplx']
        for key in smplx_.keys():
            smplx_[key] = np.concatenate(
                smplx_[key], axis=0).reshape(self.smplx_shape[key])
        human_data['smplx'] = smplx_

        # keypoints2d_smplx
        # keypoints2d_smplx = np.concatenate(
        #     keypoints2d_smplx_, axis=0).reshape(-1, 42, 3)
        # keypoints2d_smplx, keypoints2d_smplx_mask = \
        #         convert_kps(keypoints2d_smplx, src='mano_hands', dst='human_data')
        # human_data['keypoints2d_smplx'] = keypoints2d_smplx
        # human_data['keypoints2d_smplx_mask'] = keypoints2d_smplx_mask

        # misc
        human_data['misc'] = self.misc_config
        human_data['config'] = f'sgnify_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        size_i = min(len(seqs), int(size))
        out_file = os.path.join(out_path,
            f'sgnify_{mode}_{seed}_{"{:02d}".format(size_i)}.npz')
        human_data.dump(out_file)
