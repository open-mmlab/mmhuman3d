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
class ImarDatasetsConverter(BaseModeConverter):

    ACCEPTED_MODES = ['FIT3D', 'CHI3D', 'HumanSC3D']


    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.misc_config = dict(bbox_body_scale=1.2, bbox_facehand_scale=1.0, bbox_source='keypoints2d_smplx',
                         cam_param_source='original', smplx_source='original', image_size=900,)
        self.smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
                'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
                'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}
        self.train_slices = {'HumanSC3D': 3, 'CHI3D': 2, 'FIT3D': 6}

        super(ImarDatasetsConverter, self).__init__(modes)

    
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
    
    
    def _read_video(self, vid_path, write_frames=False):
        frames = []
        cap = cv2.VideoCapture(vid_path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            # frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) )
            frames.append(frame)
        cap.release()
        frames = np.array(frames)

        image_path = []
        # write frames
        out_path = vid_path.replace('videos', 'frames')[:-4]
        os.makedirs(out_path, exist_ok=True)
        for i, frame in enumerate(frames):
            if write_frames:
                cv2.imwrite(os.path.join(out_path, f'{i:04d}.jpg'), frame)
            image_path.append(os.path.join(out_path, f'{i:04d}.jpg'))

        return image_path
    
    def _read_cam_params(self, cam_path):
        with open(cam_path) as f:
            cam_params = json.load(f)
            for key1 in cam_params:
                for key2 in cam_params[key1]:
                    cam_params[key1][key2] = np.array(cam_params[key1][key2]) 
        return cam_params


    def _read_data(self, vid_p, batch, mode, dataset_path, subject="w_markers"):
            
        vid_info1 = vid_p.split('videos')[0].split(os.path.sep)
        vid_info2 = vid_p.split('videos')[1].split(os.path.sep)

        subj_name = vid_info1[-2]

        action_name = vid_info2[-1][:-4]
        camera_name = vid_info2[-2]

        cam_path = vid_p.replace('videos', 'camera_parameters').replace('.mp4', '.json')
            
        j3d_path = os.path.sep.join(vid_info1[:-1] + ['joints3d_25', action_name + '.json'])
        gpp_path = j3d_path.replace('joints3d_25', 'gpp')
        smplx_path = j3d_path.replace('joints3d_25', 'smplx')

        cam_params = self._read_cam_params(cam_path)

        with open(j3d_path) as f:
            j3ds = np.array(json.load(f)['joints3d_25'])
        seq_len = j3ds.shape[-3]
        with open(gpp_path) as f:
            gpps = json.load(f)
        with open(smplx_path) as f:
            smplx_params = json.load(f)
        image_path = self._read_video(vid_p)[:seq_len]
        
        dataset_to_ann_type = {'CHI3D': 'interaction_contact_signature', 
                            'FIT3D': 'rep_ann', 
                            'HumanSC3D': 'self_contact_signature'}
        ann_type = dataset_to_ann_type[mode]
        annotations = None
        if ann_type:
            ann_path = os.path.join(dataset_path, mode, batch, subj_name, ann_type + '.json')
            with open(ann_path) as f:
                annotations = json.load(f)
        
        if mode == 'CHI3D': # 2 people in each frame
            subj_id = 0 if subject == "w_markers" else 1
            j3ds = j3ds[subj_id, ...]
            for key in gpps:
                gpps[key] = gpps[key][subj_id]
            for key in smplx_params:
                smplx_params[key] = smplx_params[key][subj_id]
            
        return image_path, j3ds, cam_params, gpps, smplx_params, annotations


    def _project_3d_to_2d(self, points3d, intrinsics, intrinsics_type):
        if intrinsics_type == 'w_distortion':
            p = intrinsics['p'][:, [1, 0]]
            x = points3d[:, :, :2] / points3d[:, :, 2:3]
            r2 = np.sum(x**2, axis=2)
            radial = 1 + np.transpose(np.matmul(intrinsics['k'], np.array([r2, r2**2, r2**3])))
            tan = np.matmul(x, (np.transpose(p)).repeat(x.shape[0], axis=0))
            xx = x*(tan + radial) + r2[:, np.newaxis] * p
            proj = intrinsics['f'] * xx + intrinsics['c']
        elif intrinsics_type == 'wo_distortion':
            xx = points3d[:, :, :2] / points3d[:, :, 2:3]
            proj = intrinsics['f'] * xx + intrinsics['c']
        return proj

    
    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
            
        # get all video paths    
        vid_ps_all = glob.glob(os.path.join(dataset_path, mode, '**', '*.mp4'), recursive=True)

        vid_ps_test = [vid_p for vid_p in vid_ps_all if 'test' in vid_p]
        vid_ps_train = [vid_p for vid_p in vid_ps_all if 'train' in vid_p]

        seed, size = '230605', '9999'
        random.seed(int(seed))

        # build smplx model
        smplx_model = build_body_model(
                dict(type='SMPLX',
                    keypoint_src='smplx',
                    keypoint_dst='smplx',
                    model_path='data/body_models/smplx',
                    gender='neutral',
                    num_betas=10,
                    use_face_contour=True,
                    flat_hand_mean=True,
                    use_pca=False,
                    batch_size=1)).to(self.device)

        for batch, vid_ps in zip(['train', 'test'], [vid_ps_train, vid_ps_test]):
        
            # vid_ps = vid_ps[:100]
            size_i = min(len(vid_ps), int(size))

            if batch == 'train':

                slices = self.train_slices[mode]
                # slices = 1
                print(f'Seperate in to {slices} files')

                slice_vids = int(int(size_i)/slices) + 1

                for slice_idx in range(slices):

                    # use HumanData to store all data
                    human_data = HumanData() 

                    # initialize output for human_data
                    smplx_ = {}
                    for keys in self.smplx_shape.keys():
                        smplx_[keys] = []
                    keypoints2d_, keypoints3d_ = [], []
                    keypoints_2d_original_, keypoints_3d_original_ = [], []
                    bboxs_ = {}
                    for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
                        bboxs_[bbox_name] = []
                    meta_ = {}
                    for meta_key in ['focal_length', 'principal_point']:
                        meta_[meta_key] = []
                    image_path_ = []

                    for vid_p in tqdm(vid_ps[slice_vids*slice_idx:slice_vids*(slice_idx+1)], 
                                    desc=f'Processing {mode} {batch} slice {slice_idx} / {slices}'):

                        if mode == 'CHI3D':
                            subj_ids = [0, 1]
                        else:
                            subj_ids = [0]

                        for subj_id in subj_ids:

                            subject = 'w_markers' if subj_id == 0 else 'wo_markers'
                            info = self._read_data(vid_p, batch, mode, dataset_path, subject=subject)

                            image_path = [imgp.replace(dataset_path + os.path.sep, '') for imgp in info[0]]
                            j3ds = info[1]
                            cam_params = info[2]
                            gpps = info[3]
                            smplx_params = info[4]
                            annotations = info[5]

                            width, height = 900, 900

                            # build camera
                            camera = build_cameras(
                                dict(type='PerspectiveCameras',
                                    convention='opencv',
                                    in_ndc=False,
                                    focal_length=cam_params['intrinsics_w_distortion']['f'].reshape(-1, 2),
                                    principal_point=cam_params['intrinsics_w_distortion']['c'].reshape(-1, 2),
                                    image_size=(width, height))).to(self.device)

                            # reshape smplx params
                            smplx_param = {}
                            for key in ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose']:
                                smplx_param[key] = rotmat_to_aa(np.array(smplx_params[key], dtype=np.float32)
                                                                .reshape(list(self.smplx_shape[key]) + [3]))
                            for key in ['betas', 'transl', 'expression']:
                                smplx_param[key] = np.array(smplx_params[key], dtype=np.float32).reshape(list(self.smplx_shape[key]))

                            # smplx_param = {key: np.array(smplx_params[key]).reshape(self.smplx_shape[key])for key in smplx_params}
                            # for key in smplx_param.keys():
                            #     print(key, smplx_param[key].shape)

                            output = smplx_model(**{key: torch.tensor(smplx_param[key]).to(self.device) for key in smplx_params},
                                    return_joints=True)

                            keypoints_3d = output['joints'].detach().cpu().numpy()
                            pelvis_world = keypoints_3d[:, get_keypoint_idx('pelvis', 'smplx'), :]

                            # convert R, T to extrinsics
                            # extrinsics = np.eye(4)
                            # extrinsics[:3, :3] = np.transpose(cam_params['extrinsics']['R'])
                            # extrinsics[:3, 3] = cam_params['extrinsics']['T']
                            transform_1 = np.eye(4)
                            transform_1[:3, 3] = -cam_params['extrinsics']['T']
                            transform_2 = np.eye(4)
                            transform_2[:3, :3] = cam_params['extrinsics']['R']
                            extrinsics = transform_2 @ transform_1
                            
                            global_orient_cam, transl_cam = batch_transform_to_camera_frame(
                                    global_orient=smplx_param['global_orient'], transl=smplx_param['transl'], 
                                    pelvis=pelvis_world, extrinsic=extrinsics)
                            global_orient_cam = np.array([global_orient_cam]).reshape(-1, 3)

                            smplx_param['global_orient'] = global_orient_cam
                            smplx_param['transl'] = transl_cam

                            output = smplx_model(**{key: torch.tensor(smplx_param[key]).to(self.device) for key in smplx_params},
                                    return_joints=True)
                            
                            # get keypoints 2d and 3d
                            keypoints_3d = output['joints']
                            keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
                            keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
                            keypoints_3d = keypoints_3d.detach().cpu().numpy()

                            # project 3d to camera space and then to 2d
                            keypoints_3d_original = np.matmul(np.array(j3ds) - cam_params['extrinsics']['T'], np.transpose(cam_params['extrinsics']['R']))
                            keypoints_2d_original = self._project_3d_to_2d(keypoints_3d_original, cam_params['intrinsics_wo_distortion'], 'wo_distortion')


                            # for index in [12, 50]:
                            #     img = cv2.imread(os.path.join(dataset_path, image_path[index]))
                            #     for kp in keypoints_2d[index]:
                            #         img = cv2.circle(img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)
                            #     for kp in keypoints_2d_original[index]:
                            #         img = cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
                            #     cv2.imwrite(os.path.join(out_path, f'{index:04d}_vis.jpg'), img)

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

                            # write smplx params
                            for key in smplx_param.keys():
                                smplx_[key].append(smplx_param[key])

                            # write keypoints
                            keypoints2d_.append(keypoints_2d)
                            keypoints3d_.append(keypoints_3d)
                            keypoints_2d_original_.append(keypoints_2d_original)
                            keypoints_3d_original_.append(keypoints_3d_original)

                            # write meta
                            meta_['focal_length'] += cam_params['intrinsics_w_distortion']['f'].repeat(len(image_path)).tolist()
                            meta_['principal_point'] += cam_params['intrinsics_w_distortion']['c'].repeat(len(image_path)).tolist()

                            # write image path
                            image_path_ += image_path

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

                    # keypoints 2d
                    keypoints2d = np.concatenate(keypoints2d_, axis=0).reshape(-1, 144, 2)
                    keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
                    keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
                    keypoints2d, keypoints2d_mask = \
                            convert_kps(keypoints2d, src='smplx', dst='human_data')
                    human_data['keypoints2d_smplx'] = keypoints2d
                    human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

                    # keypoints 3d
                    keypoints3d = np.concatenate(keypoints3d_, axis=0).reshape(-1, 144, 3)
                    keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
                    keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
                    keypoints3d, keypoints3d_mask = \
                            convert_kps(keypoints3d, src='smplx', dst='human_data')
                    human_data['keypoints3d_smplx'] = keypoints3d
                    human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

                    # keypoints 2d original
                    keypoints_2d_original = np.concatenate(keypoints_2d_original_, axis=0).reshape(-1, 25, 2)
                    keypoints_2d_original_conf = np.ones([keypoints_2d_original.shape[0], 25, 1])
                    keypoints_2d_original = np.concatenate([keypoints_2d_original, keypoints_2d_original_conf], axis=-1)
                    keypoints_2d_original, keypoints_2d_original_mask = \
                            convert_kps(keypoints_2d_original, src='openpose_25', dst='human_data')
                    human_data['keypoints2d_original'] = keypoints_2d_original
                    human_data['keypoints2d_original_mask'] = keypoints_2d_original_mask

                    # keypoints 3d original
                    keypoints_3d_original = np.concatenate(keypoints_3d_original_, axis=0).reshape(-1, 25, 3)
                    keypoints_3d_original_conf = np.ones([keypoints_3d_original.shape[0], 25, 1])
                    keypoints_3d_original = np.concatenate([keypoints_3d_original, keypoints_3d_original_conf], axis=-1)
                    keypoints_3d_original, keypoints_3d_original_mask = \
                            convert_kps(keypoints_3d_original, src='openpose_25', dst='human_data')
                    human_data['keypoints3d_original'] = keypoints_3d_original
                    human_data['keypoints3d_original_mask'] = keypoints_3d_original_mask

                    # meta
                    human_data['meta'] = meta_

                    # image path
                    human_data['image_path'] = image_path_

                    # misc
                    human_data['misc'] = self.misc_config
                    human_data['config'] = f'{mode}_{batch}'

                    # save
                    human_data.compress_keypoints_by_mask()
                    os.makedirs(out_path, exist_ok=True)
                    out_file = os.path.join(out_path, f'{mode}_{batch}_{seed}_{size_i}_{slice_idx}.npz')
                    human_data.dump(out_file)

            if batch == 'test':
                for vid_p in tqdm(vid_ps, desc=f'Processing {mode} {batch}'):
                    return
                    if mode == 'FIT3D':
                        pass

















                    if mode == 'CHI3D':
                        subj_ids = [0, 1]
                    else:
                        subj_ids = [0]
                    for subj_id in subj_ids:
                        info = self._read_video(vid_p)
                        image_path = [imgp.replace(dataset_path + os.path.sep, '') for imgp in info[0]]

                        # write image path
                        image_path_ += image_path

                        # write meta
                        cam_path = vid_p.replace('videos', 'camera_parameters').replace('.mp4', '.json')
                        cam_params = self._read_cam_params(cam_path)
                        meta_['focal_length'] += cam_params['intrinsics_wo_distortion']['f'].repeat(len(image_path)).tolist()
                        meta_['principal_point'] += cam_params['intrinsics_wo_distortion']['c'].repeat(len(image_path)).tolist()

            

            
            




                    





                




                