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
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS
# import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx, get_keypoint_idxs_by_part
from mmhuman3d.models.body_models.utils import transform_to_camera_frame

import pdb

@DATA_CONVERTERS.register_module()
class Ssp3dConverter(BaseConverter):
    
    def __init__(self):
        self.device = torch.device('cuda:0')
        super(Ssp3dConverter, self).__init__()

    def convert(self, dataset_path: str, out_path: str) -> dict:

        # use HumanData to store the data
        human_data = HumanData()

        # initialize 
        image_path_, keypoints21_, bbox_xywh_ = [], [], []


        seed, size = '230508', '999'
        random.seed(int(seed))


        anno = dict(np.load(os.path.join(dataset_path, 'labels.npz')))

        male_model = build_body_model(
                        dict(
                            type='SMPL',
                            keypoint_src='smpl_45',
                            keypoint_dst='smpl_45',
                            model_path='data/body_models/smpl',
                            gender='male',
                            num_betas=10,
                            use_face_contour=True,
                            flat_hand_mean=True,
                            use_pca=False,
                            batch_size=1)).to(self.device)

        female_model = build_body_model(
                        dict(
                            type='SMPL',
                            keypoint_src='smpl_45',
                            keypoint_dst='smpl_45',
                            model_path='data/body_models/smpl',
                            gender='female',
                            num_betas=10,
                            use_face_contour=True,
                            flat_hand_mean=True,
                            use_pca=False,
                            batch_size=1)).to(self.device)
        

        focal_length = 5000
        image_size = 512

        camera = build_cameras(
                    dict(type='PerspectiveCameras',
                        convention='opencv',
                        in_ndc=False,
                        focal_length=focal_length,
                        image_size=image_size,
                        principal_point=np.array([image_size/2, image_size/2]).reshape(-1, 2))).to(self.device)

                    


        for idx in tqdm(range(len(anno['fnames']))):

            # get image path
            image_path_.append(os.path.join('images', anno['fnames'][idx]))

            betas = anno['shapes'][idx].reshape(1, -1)
            body_pose = anno['poses'][idx][3:].reshape(1, -1)
            global_orient = anno['poses'][idx][:3].reshape(1, -1)

            cam_transl = anno['cam_trans'][idx].reshape(1, -1) 

            transl = 2*focal_length/ (image_size * cam_transl + 1e-9)

            # pdb.set_trace()

            smpl_params = dict(global_orient=torch.tensor(global_orient, device=self.device),
                                body_pose=torch.tensor(body_pose, device=self.device),
                                betas=torch.tensor(betas, device=self.device),
                                transl=torch.tensor(transl, device=self.device),
                                return_joints=True)

            gender = anno['genders'][idx]
            if gender == 'f':
                output = female_model(**smpl_params)
            elif gender == 'm':
                output = male_model(**smpl_params)

            keypoints_3d = output['joints']
            keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
            keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
            keypoints_3d = keypoints_3d.detach().cpu().numpy()

            pdb.set_trace()
            pass
            






