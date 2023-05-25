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

def rotate_yaxis(R, t):
    "rotate the transformation matrix around z-axis by 180 degree ==>> let y-axis point up"
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    global_trans = np.eye(4)
    global_trans[0, 0] = global_trans[1, 1] = -1  # rotate around z-axis by 180
    rotated = np.matmul(global_trans, transform)
    return rotated[:3, :3], rotated[:3, 3]


def load_kinect_poses_back(rot, trans, rotate=False):
    """
    backward transform
    rotate: kinect y-axis pointing down, if rotate, then return a transform that make y-axis pointing up
    """
    rotations, translations = rot, trans
    rotations_back = []
    translations_back = []
    for r, t in zip(rotations, translations):
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t

        trans_back = np.linalg.inv(trans) # now the y-axis point down

        r_back = trans_back[:3, :3]
        t_back = trans_back[:3, 3]
        if rotate:
            r_back, t_back = rotate_yaxis(r_back, t_back)

        rotations_back.append(r_back)
        translations_back.append(t_back)
    return rotations_back, translations_back


@DATA_CONVERTERS.register_module()
class BehaveConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train', 'test']

    def __init__(self, modes: List = []) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.misc_config = dict(bbox_body_scale=1.2, bbox_facehand_scale=1.0, bbox_source='original',
                         cam_param_source='original', smplh_source='original', image_size=(1536, 2048), fps=30,
                         downsampled=True, downsampled_fps=1, keypoints3d='keypoints3d_original')
        self.smplh_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 21, 3),
                           'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3)}
        

        super(BehaveConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        # use HumanData to store data
        human_data = HumanData()

        # init seed
        seed, size = '230519', '999'
        random.seed(int(seed))

        # init output for HumanData
        keypoints2d_, keypoints3d_, keypoints2d_original_, keypoints3d_original_ = [], [], [], []
        image_path_ = []
        smplh_ = {}
        for key in ['body_pose', 'global_orient', 'betas', 'transl', 'left_hand_pose', 'right_hand_pose']:
            smplh_[key] = []
        bboxs_ = {}
        for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
            bboxs_[bbox_name] = []
        height, width = self.misc_config['image_size']
        meta_ = {}
        for key in ['gender', 'focal_length', 'principal_point']:
            meta_[key] = []
        
        # get data split from json file
        with open(os.path.join(dataset_path, 'split.json')) as f:
            split = json.load(f)

        # load camera intrinsics
        camera_intrinsics_dict = {}
        for cam_id in ['k1', 'k2', 'k3', 'k0']:
            with open(os.path.join(dataset_path, 'calibs', 'intrinsics', cam_id[-1], 'calibration.json')) as f:
                camera_intrinsics_dict[cam_id] = json.load(f)

        # build opencv cameras
        cameras_opencv, intrinsics, dist_coeffs, intrinsics_undistorted = {}, {}, {}, {}
        focal_length, principal_point = {}, {}
        for cam_id in ['k1', 'k2', 'k3', 'k0']:

            fx, fy = camera_intrinsics_dict[cam_id]['color']['fx'], camera_intrinsics_dict[cam_id]['color']['fy']
            cx, cy = camera_intrinsics_dict[cam_id]['color']['cx'], camera_intrinsics_dict[cam_id]['color']['cy']
            intrinsics[cam_id] = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
            dist_coeffs[cam_id]=np.array(camera_intrinsics_dict[cam_id]['color']['opencv'][4:])

            # get undistort camera_matrix
            intrinsics_undistorted[cam_id], _ = cv2.getOptimalNewCameraMatrix(intrinsics[cam_id], 
                                                                      dist_coeffs[cam_id], self.misc_config['image_size'], alpha=0)
            # use transpose
            # focal_length[cam_id] = np.array([intrinsics_undistorted[cam_id][0, 2], intrinsics_undistorted[cam_id][1, 2]]).reshape(-1, 2)
            # principal_point[cam_id] = np.array([intrinsics_undistorted[cam_id][0, 0], intrinsics_undistorted[cam_id][1, 1]]).reshape(-1, 2)
            focal_length[cam_id] = (fx, fy)
            principal_point[cam_id] = (cx, cy)

            cameras_opencv[cam_id] = build_cameras(
                                dict(type='PerspectiveCameras',
                                    convention='opencv',
                                    in_ndc=False,
                                    principal_point=(cx, cy),
                                    focal_length=(fx, fy),
                                    image_size=self.misc_config['image_size'])).to(self.device)
            
        # build smpl model
        gendered_model = {}
        for gender in ['male', 'female']:
            gendered_model[gender] = build_body_model(dict(type='SMPL',
                                keypoint_src='smpl_45',
                                keypoint_dst='smpl_45',
                                model_path='data/body_models/smpl',
                                gender=gender,
                                num_betas=10,
                                use_pca=False,
                                batch_size=1)).to(self.device)
        # import pickle
        # with open(os.path.join('data/body_models/smplh', 'SMPLH_MALE.pkl'), 'rb') as f:
        #     smplh_male = pickle.load(f, encoding='latin1')
        # pdb.set_trace()
        # build smplh model
        # gendered_smplh_model = {}
        # for gender in ['male', 'female']:
        #     gendered_smplh_model[gender] = smplx.create(
        #         model_path='data/body_models',
        #         model_type='smplh',
        #         gender=gender, 
        #         use_face_contour=True,
        #         num_betas=10,
        #         use_pca=False,
        #         num_pca_comps=15,
        #         ext='pkl')
        # pdb.set_trace()
        # # build mano hand model
        # mano_model = build_body_model(dict(type='MANO',
        #                     keypoint_src='mano',
        #                     keypoint_dst='mano',
        #                     model_path='data/body_models/mano',)).to(self.device)

        # for seq in split[mode]:
        for seq in tqdm(split[mode], desc=f'Processing {mode} seqs', position=0, leave=False):

            # smpl_params = dict(np.load(os.path.join(dataset_path, 'behave-30fps-params-v1', 
            #                                    seq, 'smpl_fit_all.npz'), allow_pickle=True))
            
            valid_timestamp = [f for f in os.listdir(os.path.join(dataset_path, 'sequences', seq))
                            if f.endswith('.000')]
            
            # get seq data
            info_file = os.path.join(dataset_path, 'sequences', seq, 'info.json')
            seq_data = json.load(open(info_file))
            gender = seq_data['gender']

                            # get kinect view ids
            kinect_view_ids =  [f'k{str(id)}' for id in seq_data['kinects']]

            # pdb.set_trace()

            # for frame in valid_timestamp:
            for frame in tqdm(valid_timestamp, desc=f'Processing {seq} frames', position=1, leave=False):
                frame_data_folder = os.path.join(dataset_path, 'sequences', seq, frame)

                # read smpl params
                smpl_param = np.load(os.path.join(frame_data_folder, 'person', 
                                        'fit02', 'person_fit.pkl'), allow_pickle=True)
                
                # read joints3d
                j3d_p = os.path.join(frame_data_folder, 'person', 'person_J3d.json')
                with open(j3d_p) as f:
                    joints3d = np.array(json.load(f)['body_joints3d']).reshape(-1, 4)[:, :3]

                for view_id in sorted(kinect_view_ids):

                    # write image path
                    # image_path_.append(os.path.join(frame_data_folder, view_id + '.color.jpg'))

                    # read gt kps2d
                    j2d_gt = json.load(open(os.path.join(frame_data_folder, view_id + '.color.json')))

                    for key in j2d_gt.keys():
                        j2d_gt[key] = np.array(j2d_gt[key]).reshape(-1, 3)[:, :2]

                    # deal with kinect transfrom
                    # get kinect camera params
                    with open(os.path.join(dataset_path, 'calibs', seq[:6], 'config', view_id[-1], 'config.json')) as f:
                        camera_param = json.load(f)
                    world2local_r = np.array(camera_param['rotation']).reshape(3, 3)
                    world2local_t = np.array(camera_param['translation']).reshape(3)

                    # with open(os.path.join(frame_data_folder, view_id + '.mocap.json')) as f:
                    #     smpl_param = json.load(f)
                    # j3d_p = os.path.join(frame_data_folder, f'{view_id}.color.json')
                    # with open(j3d_p) as f:
                    #     joints3d = np.array(json.load(f)['body_joints3d']).reshape(-1, 4)[:, :3]
                    # get smpl params
                    betas = np.array(smpl_param['betas'][:10]).reshape((-1, 10))
                    trans = np.array(smpl_param['trans']).reshape((-1, 3))
                    root_orient = np.array(smpl_param['pose'][:3]).reshape((-1, 3))
                    pose_body = np.array(smpl_param['pose'][3:66]).reshape((-1, 21, 3))
                    pose_hand = smpl_param['pose'][66:]
                    left_hand_pose = np.array(pose_hand[:45]).reshape(-1, 15, 3)
                    right_hand_pose = np.array(pose_hand[45:]).reshape(-1, 15, 3)

                    # get output kps3d
                    smpl_pose = np.concatenate([pose_body, np.array([0, 0, 0], dtype=np.float32).reshape(-1, 1, 3), 
                                               np.array([0, 0, 0], dtype=np.float32).reshape(-1, 1, 3)], axis=1)
                    output = gendered_model[gender](
                        betas=torch.tensor(betas, device=self.device),
                        body_pose=torch.tensor(smpl_pose.reshape(1, -1), device=self.device),
                        global_orient=torch.tensor(root_orient, device=self.device),
                        transl=torch.tensor(trans, device=self.device),
                        )
                    keypoints_3d = output['joints'].detach().cpu().numpy()
                    pelvis_world = keypoints_3d[0, get_keypoint_idx('pelvis', 'smpl_45')]

                    # build camera extrinsic matrix
                    [world2local_r], [world2local_t] = load_kinect_poses_back([world2local_r], [world2local_t])

                    world2local = np.eye(4)
                    world2local[:3, :3] = world2local_r
                    world2local[:3, 3] = world2local_t

                    # transform keypoints original to kinect camera space
                    j3d_cam = np.matmul(joints3d, world2local_r.T) + world2local_t
                    # pdb.set_trace()

                    # transform smpl to kinect camera space
                    global_orient_kinect, transl_kinect = transform_to_camera_frame(
                            global_orient=root_orient, transl=trans, 
                            pelvis=pelvis_world, extrinsic=world2local)
                    
                    # get keypoints2d
                    output = gendered_model[gender](
                        betas=torch.tensor(betas, device=self.device),
                        body_pose=torch.tensor(smpl_pose.reshape(1, -1), device=self.device),
                        global_orient=torch.tensor(global_orient_kinect.reshape(1, -1), device=self.device),
                        transl=torch.tensor(transl_kinect.reshape(1, -1), device=self.device),)
                    keypoints_3d = output['joints']
                    keypoints_2d_xyd = cameras_opencv[view_id].transform_points_screen(keypoints_3d)
                    keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
                    keypoints_3d = keypoints_3d.detach().cpu().numpy()

                    # get bbox
                    bbox_names = ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']
                    bbox_sacle_name = ['bbox_body_scale', 'bbox_facehand_scale', 'bbox_facehand_scale', 'bbox_facehand_scale']
                    for bbox_id, key in enumerate(sorted(j2d_gt.keys())):
                        bbox_xyxy = self._keypoints_to_scaled_bbox(j2d_gt[key], 
                                        scale=self.misc_config[bbox_sacle_name[bbox_id]])
                        xmin, ymin, xmax, ymax = bbox_xyxy
                        bbox = np.array([max(0, xmin), max(0, ymin), min(width, xmax), min(height, ymax)])
                        bbox_xywh = self._xyxy2xywh(bbox)  # list of len 4
                        if bbox_xywh[2] * bbox_xywh[3] > 0:
                            conf = 1
                        else:
                            conf = 0
                        bboxs_[bbox_names[bbox_id]].append(bbox_xywh + [conf])

                    # save undistorted img
                    img_undis_p = os.path.join(os.path.join(frame_data_folder, view_id + '.color_undistorted.jpg'))
                    img = cv2.imread(os.path.join(frame_data_folder, view_id + '.color.jpg'))
                    undistorted_img = cv2.undistort(img, intrinsics[view_id], dist_coeffs[view_id])
                    if not os.path.exists(img_undis_p):
                        cv2.imwrite(img_undis_p, undistorted_img)

                    # save image path
                    image_path = img_undis_p.replace(dataset_path + os.path.sep, '')
                    image_path_.append(image_path)

                    # save smpl params
                    smplh_['body_pose'].append(pose_body)
                    smplh_['global_orient'].append(global_orient_kinect)
                    smplh_['betas'].append(betas)
                    smplh_['transl'].append(transl_kinect)
                    smplh_['left_hand_pose'].append(left_hand_pose)
                    smplh_['right_hand_pose'].append(right_hand_pose)

                    # meta
                    meta_['gender'].append(gender)
                    meta_['focal_length'].append(focal_length[view_id])
                    meta_['principal_point'].append(principal_point[view_id])

                    # save smpl keypoints
                    # keypoints2d_.append(keypoints_2d)
                    keypoints3d_.append(keypoints_3d)

                    # save original keypoints
                    j2d = np.concatenate([j2d_gt['body_joints'],
                            j2d_gt['left_hand_joints'], j2d_gt['right_hand_joints'], j2d_gt['face_joints']], axis=0)
                    j2d_conf = np.zeros((137,1))
                    j2d_conf[np.any(j2d != 0, axis=1)] = 1
                    keypoints2d_original_.append(np.concatenate([j2d, j2d_conf], axis=1))
                    keypoints3d_original_.append(j3d_cam)

                    # kps2d = keypoints_2d[0,:,:]
                    # for i in range(len(kps2d)):
                    #     cv2.circle(undistorted_img, (int(kps2d[i][0]), int(kps2d[i][1])), 5, (0, 0, 255), -1)
                    # cv2.imwrite(os.path.join(out_path, f'{seq}_{frame}_{view_id}.jpg'), undistorted_img)

                    # draw j2d on image (test)
                    for i in range(len(j2d)):
                        cv2.circle(undistorted_img, (int(j2d[i][0]), int(j2d[i][1])), 3, (0, 255, 0), -1)
                    cv2.imwrite(os.path.join(out_path, f'{seq}_{frame}_{view_id}.jpg'), undistorted_img)

                    # for key in j2d_gt.keys():
                    #     for i in range(len(j2d_gt[key])):
                    #         cv2.circle(undistorted_img, (int(j2d_gt[key][i][0]), int(j2d_gt[key][i][1])), 5, (0, 255, 0), -1)
                    # cv2.imwrite(os.path.join(out_path, f'{seq}_{frame}_{view_id}.jpg'), undistorted_img)

                    # pdb.set_trace()

        # save smplh
        for key in smplh_.keys():
                smplh_[key] = np.array(smplh_[key]).reshape(self.smplh_shape[key])
        human_data['smplh'] = smplh_
        print('Smpl and/or Smplx finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # save bboxs
        for key in bboxs_.keys():
            bbox_ = np.array(bboxs_[key]).reshape((-1, 5))
            human_data[key] = bbox_
        print('BBox generation finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # keypoints_2d_original
        keypoints2d_original = np.concatenate(keypoints2d_original_).reshape(-1, 137, 3)
        keypoints2d_original, keypoints2d_original_mask = \
                convert_kps(keypoints2d_original, src='openpose_137', dst='human_data')
        human_data['keypoints2d_original'] = keypoints2d_original
        human_data['keypoints2d_original_mask'] = keypoints2d_original_mask

        # keypoints 3d
        keypoints3d = np.concatenate(keypoints3d_).reshape(-1, 45, 3)
        keypoints3d_conf = np.ones([keypoints3d.shape[0], 45, 1])
        keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
        keypoints3d, keypoints3d_mask = \
                convert_kps(keypoints3d, src='smpl_45', dst='human_data')
        human_data['keypoints3d'] = keypoints3d
        human_data['keypoints3d_mask'] = keypoints3d_mask

        # keypoints_3d_original
        keypoints3d_original = np.concatenate(keypoints3d_original_).reshape(-1, 25, 3)
        keypoints3d_ = np.ones([keypoints3d.shape[0], 25, 1])
        keypoints3d_original = np.concatenate([keypoints3d_original, keypoints3d_], axis=-1)
        keypoints3d_original, keypoints3d_original_mask = \
                convert_kps(keypoints3d_original, src='openpose_25', dst='human_data')
        human_data['keypoints3d_original'] = keypoints3d_original
        human_data['keypoints3d_original_mask'] = keypoints3d_original_mask
        print('Keypoint conversion finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # image path
        human_data['image_path'] = image_path_
        print('Image path writting finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # meta
        human_data['meta'] = meta_
        print('MetaData finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # store
        human_data['config'] = f'behave_downsampled_{mode}'
        human_data['misc'] = self.misc_config

        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        size = min(len(split[mode]), int(size))
        out_file = os.path.join(out_path, f'behave_{mode}_{seed}_{"{:03d}".format(size)}_downsampled.npz')
        human_data.dump(out_file)
