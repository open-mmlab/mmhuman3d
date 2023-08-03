import ast
import glob
import json
import os
import pdb
import random
import time
from multiprocessing import Pool
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
from mmhuman3d.core.conventions.segmentation.smpl import SMPL_SEGMENTATION_DICT
from mmhuman3d.data.data_structures.human_data import HumanData
# import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import transform_to_camera_frame
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class Ssp3dConverter(BaseConverter):

    def __init__(self):
        self.device = torch.device('cuda:0')
        self.misc = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            bbox_source='keypoints2d_smpl',
            cam_param_type='prespective',
            cam_param_source='estimated',
            smpl_source='original',
            focal_length=(5000, 5000),
            principal_point=(256, 256),
            image_size=512,
        )
        self.smpl_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 23, 3)
        }

        super(Ssp3dConverter, self).__init__()
        

    def _keypoints_to_scaled_bbox_bfh(self,
                                      keypoints,
                                      vertices=None,
                                      occ=None,
                                      body_scale=1.0,
                                      fh_scale=1.0,
                                      convention='smpl_45'):
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

        for body_part in ['body', 'head', 'leftHand', 'rightHand']:
            if body_part == 'body':
                scale = body_scale
                kps = keypoints
            else:
                scale = fh_scale
                if convention == 'smpl_vertices':
                    vert_id = []
                    for seg in SMPL_SEGMENTATION_DICT[body_part]:
                        if len(seg) == 1:
                            vert_id.append(seg[0])
                        else:
                            vert_id += [i for i in range(seg[0], seg[1] + 1)]
                    kps = vertices[0][vert_id]
                else:
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

    def convert(self, dataset_path: str, out_path: str) -> dict:

        # use HumanData to store the data
        human_data = HumanData()

        # initialize
        image_path_, keypoints2d_, bbox_xywh_ = [], [], []

        seed, size = '230803', '999'
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
        width, height = image_size, image_size

        camera = build_cameras(
            dict(
                type='PerspectiveCameras',
                convention='opencv',
                in_ndc=False,
                focal_length=focal_length,
                image_size=image_size,
                principal_point=np.array([image_size / 2, image_size / 2
                                          ]).reshape(-1, 2))).to(self.device)

        # init
        keypoints2d_, keypoints3d_, keypoints2d_original_ = [], [], []
        meta_ = {}
        meta_['gender'] = []
        smpl_ = {}
        for key in self.smpl_shape.keys():
            smpl_[key] = []
        bboxs_ = {}
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            bboxs_[bbox_name] = []

        for idx in tqdm(range(len(anno['fnames']))):

            # get image path
            image_path_.append(os.path.join('images', anno['fnames'][idx]))

            betas = anno['shapes'][idx].reshape(1, -1)
            body_pose = anno['poses'][idx][3:].reshape(1, -1)
            global_orient = anno['poses'][idx][:3].reshape(1, -1)
            keypoints2d_original = anno['joints2D'][idx]
            keypoints2d_original_.append(keypoints2d_original)

            cam_transl = anno['cam_trans'][idx].reshape(1, -1)

            smpl_params = dict(
                global_orient=torch.tensor(global_orient, device=self.device),
                body_pose=torch.tensor(body_pose, device=self.device),
                betas=torch.tensor(betas, device=self.device),
                transl=torch.tensor(cam_transl, device=self.device),
                return_joints=True)

            # get smpl
            gender = anno['genders'][idx]
            if gender == 'f':
                output = female_model(**smpl_params)
                gender = 'female'
            elif gender == 'm':
                output = male_model(**smpl_params)
                gender = 'male'
            keypoints_3d = output['joints']
            keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
            vertices = output['vertices']
            vertices_2d = camera.transform_points_screen(vertices)[
                ..., :2].detach().cpu().numpy()
            keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
            keypoints_3d = keypoints_3d.detach().cpu().numpy()

            keypoints2d_.append(keypoints_2d.reshape(1, -1, 2))
            keypoints3d_.append(keypoints_3d.reshape(1, -1, 3))
            for key in ['betas', 'body_pose', 'global_orient', 'transl']:
                smpl_[key].append(
                    smpl_params[key].detach().cpu().numpy().reshape(
                        self.smpl_shape[key]))

            # get bbox
            bboxs = self._keypoints_to_scaled_bbox_bfh(
                keypoints_2d,
                vertices=vertices_2d,
                body_scale=self.misc['bbox_body_scale'],
                fh_scale=self.misc['bbox_facehand_scale'],
                convention='smpl_vertices')
            ## convert xyxy to xywh
            for i, bbox_name in enumerate([
                    'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                    'rhand_bbox_xywh'
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

            # gender
            meta_['gender'].append(gender)

        # smpl
        for key in self.smpl_shape.keys():
            smpl_[key] = np.concatenate(
                smpl_[key], axis=0).reshape(self.smpl_shape[key])
        human_data['smpl'] = smpl_

        # bboxs
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            bboxs_[bbox_name] = np.concatenate(
                bboxs_[bbox_name], axis=0).reshape(-1, 5)
            human_data[bbox_name] = bboxs_[bbox_name]

        # keypoints 2d original
        keypoints2d_original = np.concatenate(
            keypoints2d_original_, axis=0).reshape(-1, 17, 3)
        keypoints2d_original, keypoints2d_original_mask = \
                convert_kps(keypoints2d_original, src='coco', dst='human_data')
        human_data['keypoints2d_original'] = keypoints2d_original
        human_data['keypoints2d_original_mask'] = keypoints2d_original_mask

        # keypoints 2d
        keypoints2d = np.concatenate(keypoints2d_, axis=0).reshape(-1, 45, 2)
        keypoints2d_conf = np.ones([keypoints2d.shape[0], 45, 1])
        keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
        keypoints2d, keypoints2d_mask = \
                convert_kps(keypoints2d, src='smpl_45', dst='human_data')
        human_data['keypoints2d_smpl'] = keypoints2d
        human_data['keypoints2d_smpl_mask'] = keypoints2d_mask

        # keypoints 3d
        keypoints3d = np.concatenate(keypoints3d_, axis=0).reshape(-1, 45, 3)
        keypoints3d_conf = np.ones([keypoints3d.shape[0], 45, 1])
        keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
        keypoints3d, keypoints3d_mask = \
                convert_kps(keypoints3d, src='smpl_45', dst='human_data')
        human_data['keypoints3d_smpl'] = keypoints3d
        human_data['keypoints3d_smpl_mask'] = keypoints3d_mask

        print('Keypoint conversion finished at',
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # image path
        human_data['image_path'] = image_path_
        print('Image path writing finished at',
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # meta
        human_data['meta'] = meta_
        human_data['misc'] = self.misc
        print('MetaData finished at',
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # store
        human_data['config'] = f'ssp3d'
        # human_data['misc'] = self.misc_config

        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        size = min(len(image_path_), int(size))
        out_file = os.path.join(out_path,
                                f'ssp3d_{seed}_{"{:03d}".format(size)}.npz')
        human_data.dump(out_file)
