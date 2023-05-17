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
import torch
import smplx
import ast
import copy

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
# import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idx, get_keypoint_idxs_by_part
from mmhuman3d.models.body_models.utils import transform_to_camera_frame, batch_transform_to_camera_frame
from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa

import pdb


@DATA_CONVERTERS.register_module()
class RenbodyConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train', 'train_highrescam', 'test', 'test_highrescam']

    def __init__(self, modes: List = []) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.misc_config = dict(bbox_body_scale=1.2, bbox_facehand_scale=1.0, flat_hand_mean=False,
                                bbox_source='keypoints2d_original', kps3d_root_aligned=False, smplx_source='original', fps=30)
        self.smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
                'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
                'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}
        self.slices = {'train': 10, 'train_highrescam': 2, 'test': 2, 'test_highrescam': 1}
       
        super(RenbodyConverter, self).__init__(modes)

    # def _keypoints_to_scaled_bbox_fh(self, keypoints, occ, self_occ, scale=1.0, convention='renbody'):
    def _keypoints_to_scaled_bbox_bfh(self, keypoints, occ=None, body_scale=1.0, fh_scale=1.0, convention='smplx'):
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
                kp_id = get_keypoint_idxs_by_part(body_part, convention=convention)
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

            bbox = np.stack([xmin, ymin, xmax, ymax, conf], axis=0).astype(np.float32)
            bboxs.append(bbox)

        return bboxs


    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:
        
        # get trageted sequence list
        seed, size = '230517', '01000'
        random.seed(int(seed))
        camera_num_per_seq = 3
        invalid_camera_list = ['09']

        # get npz list
        root_dir, prefix = os.path.split(dataset_path)
        seqs = glob.glob(os.path.join(dataset_path, 'smpl', '*', '*', '*'))

        # load data split from txt file
        if 'train' in mode:
            batch = 'train'
        if 'test' in mode:
            batch = 'test'
        if 'highrescam' in mode:
            high_res_cam = True
        else:
            high_res_cam = False

        with open(os.path.join(dataset_path, f'new_{batch}_230207.txt'), 'r') as f:
            split = f.read().splitlines()

        seqs_split = [s for s in seqs if s.replace(f'{dataset_path}{os.path.sep}smpl{os.path.sep}', '')
                      .replace('/', os.path.sep) in split]

        # init smplx gendered models
        gender_model = {}
        for gender in ['male', 'female']:
            gender_model[gender] = build_body_model(
                dict(type='SMPLX',
                    keypoint_src='smplx',
                    keypoint_dst='smplx',
                    model_path='data/body_models/smplx',
                    gender=gender,
                    num_betas=10,
                    use_face_contour=True,
                    flat_hand_mean=False,
                    use_pca=False,
                    batch_size=1)).to(self.device)
            

        slices = self.slices[mode]
        slice_vids = int(len(seqs_split)/slices) + 1

        for slice_idx in range(slices):
            
            # use HumanData to store all data 
            human_data = HumanData()

            # initialize storage
            smplx_ = {}
            for keys in self.smplx_shape.keys():
                smplx_[keys] = []
            keypoints2d_, keypoints3d_, keypoints2d_original_, keypoints3d_original_ = [], [], [], []
            keypoints_original_mask_ = []
            bboxs_ = {}
            for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
                bboxs_[bbox_name] = []
            meta_ = {}
            for meta_key in ['principal_point', 'focal_length', 'height', 'width', 'gender', 'cam_id']:
                meta_[meta_key] = []
            image_path_ = []

            # for seq in seqs_split:
            for seq in tqdm(seqs_split[slice_vids*slice_idx:slice_vids*(slice_idx+1)]
                            , desc=f'Renbody {mode}, {slice_idx} / {slices}', position=0, leave=False):

                # get smplx params
                smplx_path = os.path.join(dataset_path, seq, 'smplx_xrmocap', 'human_data_smplx.npz')
                smplx_params = dict(np.load(smplx_path, allow_pickle=True))

                kps_path = os.path.join(dataset_path, seq, 'smplx_xrmocap', 'human_data_optimized_keypoints3d.npz')
                kps_params = dict(np.load(kps_path, allow_pickle=True))

                gender = str(smplx_params['gender'])

                # image folder
                image_dir = seq.replace('smpl', 'image')
                image_annot_dir, seq_basename = os.path.split(image_dir)

                # camera details
                annot1 = np.load(os.path.join(image_annot_dir, 'annots.npy'), allow_pickle=True).item()
                # annot2 = np.load(os.path.join(image_annot_dir, 'annots2.npy'), allow_pickle=True).item()
                # annot3 = np.load(os.path.join(image_annot_dir, 'annots3.npy'), allow_pickle=True).item()

                # load keypoints
                keypoints3d_original_world = kps_params['keypoints3d']
                keypoints3d_original_mask = kps_params['keypoints3d_mask']

                # camera list
                cam_list = annot1['cams'].keys()

                # for cam_id in cam_list:
                if high_res_cam:
                    cam_list = [cam_id for cam_id in cam_list if int(cam_id) > 47]
                else:
                    cam_list = [cam_id for cam_id in cam_list if int(cam_id) <= 47]

                for cam_id in tqdm(cam_list, desc=f'Processing {seq} camera views ', position=1, leave=False):

                    # get image path
                    img_ps = sorted(glob.glob(os.path.join(image_dir, 'image', cam_id, '*.jpg')))
                    image_paths = [p.replace(f'{dataset_path}{os.path.sep}', '') for p in img_ps]

                    height, width, _ = cv2.imread(img_ps[0]).shape

                    # get camera params
                    cam_param = annot1['cams'][cam_id]

                    extrinsics = np.linalg.inv(cam_param['RT'])
                    intrinsics = cam_param['K'] 

                    # build camera
                    principal_point = (intrinsics[0, 2], intrinsics[1, 2])
                    focal_length = (intrinsics[0, 0], intrinsics[1, 1])
                    camera_opencv = build_cameras(
                        dict(type='PerspectiveCameras',
                            convention='opencv',
                            in_ndc=False,
                            principal_point=principal_point,
                            focal_length=focal_length,
                            image_size=(width, height))).to(self.device)
                    
                    # prepare smplx params and get world smplx
                    smplx_param = {key: torch.tensor(smplx_params[key], device=self.device) for key in self.smplx_shape.keys()}
                    output = gender_model[gender](**smplx_param)
                    keypoints_3d = output['joints'].detach().cpu().numpy()
                    pelvis_world = keypoints_3d[..., get_keypoint_idx('pelvis', 'smplx'),:]
                    
                    # transform smplx params to camera frame
                    global_orient_cam, transl_cam = batch_transform_to_camera_frame(
                            global_orient=smplx_params['global_orient'], transl=smplx_params['transl'],
                            pelvis=pelvis_world, extrinsic=extrinsics)
                    
                    smplx_cam = smplx_params.copy()
                    smplx_cam['global_orient'] = global_orient_cam
                    smplx_cam['transl'] = transl_cam

                    # get smplx keypoints 2d
                    smplx_param = {key: torch.tensor(smplx_cam[key], device=self.device) for key in self.smplx_shape.keys()}
                    output = gender_model[gender](**smplx_param)
                    keypoints_3d = output['joints']
                    keypoints_2d_xyd = camera_opencv.transform_points_screen(keypoints_3d)
                    keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
                    keypoints_3d = keypoints_3d.detach().cpu().numpy()

                    # transform keypoints_3d_original to camera space  
                    world2local_r = extrinsics[:3, :3]
                    world2local_t = extrinsics[:3, 3]

                    keypoints3d_original = np.matmul(keypoints3d_original_world, world2local_r.T) + world2local_t
                    keypoints2d_original = camera_opencv.transform_points_screen(
                            torch.tensor(keypoints3d_original, device=self.device, dtype=torch.float32))[..., :2].detach().cpu().numpy()

                    # save an image for debug
                    # img = cv2.imread(img_ps[0])
                    # for i in range(len(keypoints2d_original[0])):
                    #     cv2.circle(img, (int(keypoints2d_original[0][i][0]), int(keypoints2d_original[0][i][1])), 5, (0, 0, 255), -1)
                    # for i in range(len(keypoints_2d[0])):
                    #     cv2.circle(img, (int(keypoints_2d[0][i][0]), int(keypoints_2d[0][i][1])), 5, (0, 255, 0), -1)
                    # cv2.imwrite(os.path.join(out_path, f'{seq_basename}_{cam_id}.jpg'), img)

                    # get bbox
                    for keypoints_2d_frame in keypoints_2d:
                        bboxs = self._keypoints_to_scaled_bbox_bfh(keypoints_2d_frame, 
                                body_scale=self.misc_config['bbox_body_scale'], fh_scale=self.misc_config['bbox_facehand_scale'])
                        ## convert xyxy to xywh
                        for i, bbox_name in enumerate(['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']):
                            xmin, ymin, xmax, ymax, conf = bboxs[i]
                            bbox = np.array([max(0, xmin), max(0, ymin), min(width, xmax), min(height, ymax)])
                            bbox_xywh = self._xyxy2xywh(bbox)  # list of len 4
                            bbox_xywh.append(conf)  # (5,)
                            bboxs_[bbox_name].append(bbox_xywh)

                    # get meta
                    meta_['principal_point'] += [principal_point] * len(image_paths)
                    meta_['focal_length'] += [focal_length] * len(image_paths)
                    meta_['height'] += [height] * len(image_paths)
                    meta_['width'] += [width] * len(image_paths)
                    meta_['gender'] += [gender] * len(image_paths)
                    meta_['cam_id'] += [cam_id] * len(image_paths)

                    # image path
                    image_path_ += image_paths

                    # smplx
                    for key in self.smplx_shape.keys():
                        smplx_[key].append(smplx_cam[key])

                    # keypoints
                    # keypoints2d_.append(keypoints_2d)
                    # keypoints3d_.append(keypoints_3d)
                    keypoints2d_original_.append(keypoints2d_original)
                    keypoints3d_original_.append(keypoints3d_original)
                    keypoints_original_mask_.append(keypoints3d_original_mask)

            # save smplx
            for key in smplx_.keys():
                smplx_[key] = np.concatenate(smplx_[key], axis=0).reshape(self.smplx_shape[key])
            human_data['smplx'] = smplx_
            print('Smpl and/or Smplx finished at', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            # save bbox
            for bbox_name in bboxs_.keys():
                bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 5)
                human_data[bbox_name] = bbox_
            print('Bbox finished at', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

            # # keypoints 2d
            # keypoints2d = np.concatenate(keypoints2d_, axis=0).reshape(-1, 144, 2)
            # keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
            # keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
            # keypoints2d, keypoints2d_mask = \
            #         convert_kps(keypoints2d, src='smplx', dst='human_data')
            # human_data['keypoints2d_smplx'] = keypoints2d
            # human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

            # # keypoints 3d
            # keypoints3d = np.concatenate(keypoints3d_, axis=0).reshape(-1, 144, 3)
            # keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
            # keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
            # keypoints3d, keypoints3d_mask = \
            #         convert_kps(keypoints3d, src='smplx', dst='human_data')
            # human_data['keypoints3d_smplx'] = keypoints3d
            # human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

            # keypoints 2d original
            keypoints_2d_original = np.concatenate(keypoints2d_original_, axis=0).reshape(-1, 190, 2)
            keypoints_2d_original_conf = np.concatenate(keypoints_original_mask_, axis=0).reshape(-1, 190, 1)
            keypoints_2d_original = np.concatenate([keypoints_2d_original, keypoints_2d_original_conf], axis=-1)
            keypoints_2d_original, keypoints_2d_original_mask = \
                    convert_kps(keypoints_2d_original, src='human_data', dst='human_data')
            human_data['keypoints2d_original'] = keypoints_2d_original
            human_data['keypoints2d_original_mask'] = keypoints_2d_original_mask

            # keypoints 3d original
            keypoints_3d_original = np.concatenate(keypoints3d_original_, axis=0).reshape(-1, 190, 3)
            keypoints_3d_original_conf = np.concatenate(keypoints_original_mask_, axis=0).reshape(-1, 190, 1)
            keypoints_3d_original = np.concatenate([keypoints_3d_original, keypoints_3d_original_conf], axis=-1)
            keypoints_3d_original, keypoints_3d_original_mask = \
                    convert_kps(keypoints_3d_original, src='human_data', dst='human_data')
            human_data['keypoints3d_original'] = keypoints_3d_original
            human_data['keypoints3d_original_mask'] = keypoints_3d_original_mask

            # meta
            human_data['meta'] = meta_

            # image path
            human_data['image_path'] = image_path_

            # misc
            human_data['misc'] = self.misc_config
            human_data['config'] = f'renbody_{mode}'

            # save
            human_data.compress_keypoints_by_mask()
            os.makedirs(out_path, exist_ok=True)
            size_i = min(len(seqs_split), int(size))
            out_file = os.path.join(out_path, f'renbody_{mode}_{seed}_{str(size_i)}_{str(slice_idx)}.npz')
            human_data.dump(out_file)



