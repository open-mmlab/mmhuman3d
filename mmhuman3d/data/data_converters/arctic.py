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
from mmhuman3d.core.conventions.cameras.convert_convention import convert_camera_matrix
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
import trimesh


@DATA_CONVERTERS.register_module()
class ArcticConverter(BaseModeConverter):

    ACCEPTED_MODES = ['p1_test', 'p1_train', 
                      'p1_val', 'p2_test', 
                      'p2_train', 'p2_val']

    def __init__(self, modes: List = []) -> None:
        self.device = torch.device('cuda')
        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            bbox_source='keypoints2d_smplx',
            flat_hand_mean=True,
            cam_param_type='prespective',
            cam_param_source='original',
            smpl_source='original',
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
            # 'expression': (-1, 10)
        }

        super(ArcticConverter, self).__init__(modes)
        
    def _check_valid(self, data_2d, data_cam, vidx, view_idx):  
        assert (
            vidx < data_2d["joints.right"].shape[0]
            ), "The requested camera id does not exist in annotation"
        is_valid = data_cam["is_valid"][vidx, view_idx]
        right_valid = data_cam["right_valid"][vidx, view_idx]
        left_valid = data_cam["left_valid"][vidx, view_idx]
        return vidx, is_valid, right_valid, left_valid
    
    def _keypoints_to_scaled_bbox_fh(self,
                                     keypoints,
                                     occ=None,
                                     scale=1.0,
                                     convention='smplx'):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale

        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format
        '''
        bboxs = []
        for body_part in ['head', 'left_hand', 'right_hand']:
            kp_id = get_keypoint_idxs_by_part(body_part, convention=convention)

            # keypoints_factory=smplx.SMPLX_KEYPOINTS)
            kps = keypoints[kp_id]

            if occ == None:
                conf = 1
            else:
                occ_p = occ[kp_id]

                if np.sum(occ_p) / len(kp_id) >= 0.1:
                    conf = 0
                    # print(f'{body_part} occluded, occlusion: {np.sum(occ_p) / len(kp_id)}, skip')
                else:
                    # print(f'{body_part} good, {np.sum(self_occ_p + occ_p) / len(kp_id)}')
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
        return bboxs[0], bboxs[1], bboxs[2]

        
    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:
        
        
        # python tools/convert_datasets.py     
        # --datasets arctic     --root_path /mnt/e/datasets     
        # --output_path /mnt/e/datasets/arctic/output     
        # --modes p1_val
        
        # build gendered smplx
        # gendered_smplx = {}
        # for gender in ['male', 'female', 'neutral']:
        #     gendered_smplx[gender] = build_body_model(
        #         dict(
        #             type='SMPLX',
        #             keypoint_src='smplx',
        #             keypoint_dst='smplx',
        #             model_path='data/body_models/smplx',
        #             gender='neutral',
        #             num_betas=10,
        #             use_face_contour=False,
        #             flat_hand_mean=self.misc_config['flat_hand_mean'],
        #             use_pca=False,
        #             batch_size=1)).to(self.device)
        
        # ue2opencv = np.array([[-1.0, 0, 0, 0],
        #     [0, -1, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]])
        
        # use HumanData to store all data
        human_data = HumanData()
        
        # init seed and size
        seed, size = '241023', '999'
        random.seed(int(seed))
        np.set_printoptions(suppress=True)
        random_ids = np.random.RandomState(seed=int(seed)).permutation(999999)
        used_id_num = 0
        
        # load split
        split_path = os.path.join(dataset_path, 'splits', mode + '.npy')
        split_info = np.load(split_path, allow_pickle=True).item()
        
        image_names = split_info['imgnames']
        image_names = [img.replace('./arctic_data/data/', '') for img in image_names]
        image_names = [img.replace('./arctic_data/', '') for img in image_names]
        data_dict = split_info['data_dict']
        seq_names = list(data_dict.keys())
        
        # get size
        size_i = min(int(size), len(seq_names))
        
        # train 4 split, val 1 split
        if 'train' in mode:
            split_num = 4
        else:
            split_num = 1
        
        # for split_id in range(split_num):
        for split_id in [0]:
        
            # set cuda
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(split_id)
            
            # group seq names
            size_b = len(seq_names) // split_num
            seq_names_batch = seq_names[split_id * size_b: (split_id + 1) * size_b]
        
            # load meta
            meta_path = os.path.join(dataset_path, 'meta', 'misc.json')
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                
            cam_params = {}
            for sub in metadata.keys():
                cam_params[sub] = { # [8, 4, 4] and [8, 3, 3]
                    "world2cam": np.array(metadata[sub]["world2cam"]),
                    "intrinsics": np.array(metadata[sub]["intris_mat"]),} 
                
            smplx_ = {}
            for key in self.smplx_shape.keys():
                smplx_[key] = []
            keypoints2d_, keypoints3d_ = [], []
            bboxs_ = {}
            for bbox_name in [
                    'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                    'rhand_bbox_xywh'
            ]:
                bboxs_[bbox_name] = []
            meta_ = {}
            for meta_name in ['principal_point', 'focal_length', 'height', 'width', 'RT',
                            'sequence_name', 'track_id', 'gender', 'is_valid', 'right_hand_valid', 'left_hand_valid']:
                meta_[meta_name] = []
            image_path_ = []
            vtemplate_path_ = []
            verts3d_path_ = []
            
            # high level path
            raw_path = os.path.join(dataset_path, 'raw_seqs')
            
            # seq_names_batch = seq_names_batch[:5]
            
            # group by sequence
            for seq in tqdm(seq_names_batch, desc=f'Split: {split_id+1} / {split_num}', position=0, leave=False):
                
                seq_data = data_dict[seq]
                sub_id = seq.split('/')[0]
                seq_name = seq.split('/')[1]
                
                # smplx idx = image idx - ioi_offset
                ioi_offset = metadata[sub_id]['ioi_offset']
                gender_seq = metadata[sub_id]['gender']
                image_size_list = metadata[sub_id]['image_size']
                
                # betas_path = os.path.join(dataset_path, 'meta', 'subject_vtemplates', f'{sub_id}.npz')
                # betas = dict(np.load(betas_path, allow_pickle=True))['betas']
                betas = np.zeros((1, 10))
                # pdb.set_trace()            
                # build gendered smplx
                # smplx_path = os.path.join(raw_path, f'{seq}.smplx.npy')
                # smplx_params = np.load(smplx_path, allow_pickle=True).item()
                
                # prepare vtemplates path
                vtemplates_path = os.path.join(dataset_path, 'meta', 'subject_vtemplates', f'{sub_id}.obj')
                # vtemplates_path = '/mnt/e/datasets/arctic/meta/subject_vtemplates/s01.obj'
                vtemplates = trimesh.load(vtemplates_path, force='mesh').vertices.reshape(1, 10475, 3)
                # try:
                #     vtemplates = trimesh.load(vtemplates_path).vertices.reshape(1, 10475, 3)
                # except:
                #     pdb.set_trace()
                # continue
                # vtemplates = trimesh.load(file_obj=trimesh.util.wrap_as_stream(vtemplates_path)).vertices.reshape(1, 10475, 3)
                
                # build gendered smplx
                gendered_smplx = build_body_model(
                    dict(
                        type='SMPLX',
                        keypoint_src='smplx',
                        keypoint_dst='smplx',
                        model_path='data/body_models/smplx',
                        gender=gender_seq,
                        v_template=vtemplates,
                        num_betas=10,
                        use_face_contour=True,
                        flat_hand_mean=self.misc_config['flat_hand_mean'],
                        use_pca=False,
                        batch_size=1)).to(self.device)  
                
                # load params
                data_cam = seq_data["cam_coord"]
                data_2d = seq_data["2d"]
                data_bbox = seq_data["bbox"]
                data_params = seq_data["params"]
                
                data_len = data_params["K_ego"].shape[0]
                
                for cidx in range(1, 9):
                    
                    cid = str(cidx)
                    
                    # prepare intrinsics
                    if cidx == 0: # ego space
                        seq_extrx = data_params['world2ego'].copy()
                        seq_intrx = data_params["K_ego"].copy()
                    else:
                        seq_extrx = np.array(cam_params[sub_id]['world2cam'])[cidx-1]
                        seq_extrx = np.repeat(seq_extrx[np.newaxis, :], data_len, axis=0)
                        seq_intrx = np.array(cam_params[sub_id]['intrinsics'])[cidx-1]
                        seq_intrx = np.repeat(seq_intrx[np.newaxis, :], data_len, axis=0)
                    
                    # prepare images to load
                    imgp_pattern = f'{seq}/{cid}'
                    image_paths = [imgp for imgp in image_names if imgp_pattern in imgp]        
                
                    if len(image_paths) == 0:
                        continue
                    # if cidx == 0:
                    #     pdb.set_trace()
                
                    smplx_param_save = {}
                    for key in self.smplx_shape.keys():
                        smplx_param_save[key] = []
                
                    for image_path in tqdm(image_paths, desc=f'Cid: {cid} / {9}', position=1, leave=False):
                        
                        imgp = os.path.join(dataset_path, image_path)

                        # image idx and mocap idx (vidx)
                        image_idx = image_path.split("/")[-1]
                        image_id = image_idx.split(".")[0]
                        vidx = int(image_id) - ioi_offset

                        # check image and hand validity
                        vidx, is_valid, right_valid, left_valid = self._check_valid(
                            data_2d, data_cam, vidx, cidx)

                        # prepare intrinsics
                        extrinsics = seq_extrx[vidx]
                        intrinsics = seq_intrx[vidx]
                        
                        # distortion parameters for egocam rendering
                        # dist = data_params["dist"][vidx].copy()
                
                        # smplx params in world space
                        smplx_param = {}
                        for key in self.smplx_shape.keys():
                            if key != 'betas':
                                smplx_param[key] = data_params[f'smplx_{key}'][vidx].reshape(self.smplx_shape[key])
                        smplx_param['betas'] = betas
                        
                        # prepare smplx tensor
                        smplx_param_tensor = {}
                        for key in self.smplx_shape.keys():
                            smplx_param_tensor[key] = torch.tensor(smplx_param[key].reshape(self.smplx_shape[key]), 
                                                                dtype=torch.float).to(self.device)
                            
                        # get output
                        output_world = gendered_smplx(**smplx_param_tensor)
                        kps3d = output_world['joints'].detach().cpu().numpy()
                        pelvis_world = kps3d[:, get_keypoint_idx('pelvis', 'smplx'), :]
                        
                        # transfrom to camera space
                        global_orient_cam, transl_cam = transform_to_camera_frame(
                            global_orient=smplx_param['global_orient'],
                            transl=smplx_param['transl'],
                            pelvis=pelvis_world,
                            extrinsic=extrinsics)
                        
                        # transfrom camera to opencv
                        # smplx_param_tensor = {}
                        # for key in self.smplx_shape.keys():
                        #     smplx_param_tensor[key] = torch.tensor(smplx_param[key].reshape(self.smplx_shape[key]), 
                        #                                            dtype=torch.float).to(self.device)                
                        # output_cam = gendered_smplx[gender_seq](**smplx_param_tensor)
                        # kps3d = output_cam['joints'].detach().cpu().numpy()
                        # pelvis_cam = kps3d[:, get_keypoint_idx('pelvis', 'smplx'), :]                   
                        # global_orient_cam, transl_cam = transform_to_camera_frame(
                        #     global_orient=smplx_param['global_orient'],
                        #     transl=smplx_param['transl'],
                        #     pelvis=pelvis_world,
                        #     extrinsic=extrinsics)                  
                        
                        smplx_param['global_orient'] = global_orient_cam
                        smplx_param['transl'] = transl_cam
                        
                        # prepare smplx tensor
                        smplx_param_tensor = {}
                        for key in self.smplx_shape.keys():
                            smplx_param_tensor[key] = torch.tensor(smplx_param[key].reshape(self.smplx_shape[key]), 
                                                                dtype=torch.float).to(self.device)
                    
                        # get output
                        output = gendered_smplx(**smplx_param_tensor, return_verts=True)
                    
                        width, height = image_size_list[cidx]
                        focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
                        principal_point = [intrinsics[0, 2], intrinsics[1, 2]]
                        
                        camera = build_cameras(
                            dict(
                                type='PerspectiveCameras',
                                convention='opencv',
                                in_ndc=False,
                                focal_length=focal_length,
                                image_size=(width, height),
                                principal_point=principal_point)).to(self.device)
                        
                        # project 3d to 2d
                        kps3d = output['joints']
                        verts3d = output['vertices']

                        # kps3d_r = data_cam['joints.smplx'][vidx, cidx]
                        # kps3d_c = torch.tensor(kps3d_c).reshape(1, -1, 3).to(self.device)
                        # kps2d = [width, height] - kps2d
                        
                        # visualize 
                        # verts2d = camera.transform_points_screen(verts3d).detach().cpu().numpy().squeeze()[:, :2]
                        # kps2d_a = data_2d['joints.smplx'][vidx, cidx]
                        # img = cv2.imread(imgp)
                        # for kps in verts2d:
                        #     cv2.circle(img, (int(kps[0]), int(kps[1])), 3, (0, 0, 255), -1)
                        # for kps in kps2d_a:
                        #     cv2.circle(img, (int(kps[0]), int(kps[1])), 3, (255, 0, 0), -1)
                        # os.makedirs(f'{dataset_path}/output', exist_ok=True)
                        # cv2.imwrite(f'{dataset_path}/output/{sub_id}_{seq_name}_{cid}_{str(image_id)}.png', img)
                        # break
                    
                        # kps3d_c_ra = kps3d_c - kps3d_c[0]
                        # kps3d_r_ra = kps3d_r - kps3d_r[0]
                    
                        # for key in self.smplx_shape.keys():
                        #     smplx_param_save[key].append(smplx_param[key])
                        
                        kps3d_c = kps3d.detach().cpu().numpy().squeeze()
                        kps2d = camera.transform_points_screen(kps3d).detach().cpu().numpy().squeeze()[:, :2]
                        verts3d = verts3d.detach().cpu().numpy().squeeze()
                        
                        # get bbox from 2d keypoints
                        bboxs = self._keypoints_to_scaled_bbox_bfh(
                            kps2d,
                            body_scale=self.misc_config['bbox_body_scale'],
                            fh_scale=self.misc_config['bbox_facehand_scale'])
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
                        
                        # save verts3d
                        verts3d_path = image_path.replace('jpg', 'npy').replace('images', 'vertices3d')
                        verts3dp = imgp.replace('jpg', 'npy').replace('images', 'vertices3d')
                        os.makedirs(os.path.dirname(verts3dp), exist_ok=True)
                        if not os.path.exists(verts3dp):
                            np.save(verts3dp, verts3d)
                        
                        # append image path
                        image_path_.append(image_path)
                        verts3d_path_.append(verts3d_path)
                        vtemplate_path_.append(vtemplates_path.replace(f'{dataset_path}/', ''))
                        
                        # append keypoints2d and 3d
                        keypoints2d_.append(kps2d)
                        keypoints3d_.append(kps3d_c)
                        
                        # add smplx params
                        for key in smplx_param.keys():
                            smplx_[key].append(smplx_param[key])
                        
                        sequence_name = f'{sub_id}_{seq_name}_{cid}'
                        # append meta
                        meta_['principal_point'].append(principal_point)
                        meta_['focal_length'].append(focal_length)
                        meta_['height'].append(height)
                        meta_['width'].append(width)
                        meta_['sequence_name'].append(sequence_name)
                        meta_['RT'].append(extrinsics)
                        meta_['track_id'].append(random_ids[int(sub_id[1:])])
                        meta_['gender'].append(gender_seq)
                        meta_['is_valid'].append(is_valid)
                        meta_['right_hand_valid'].append(right_valid)
                        meta_['left_hand_valid'].append(left_valid)
                    
                    shape = np.concatenate(keypoints2d_, axis=0).reshape(-1, 144, 2).shape[0]
                    if len(image_path_) != shape:
                        pdb.set_trace()
                    if len(meta_['focal_length']) != shape:
                        pdb.set_trace()
                    if len(bboxs_['bbox_xywh']) != shape:
                        pdb.set_trace()

            pdb.set_trace()
            # save keypoints 2d smplx
            keypoints2d = np.concatenate(keypoints2d_, axis=0).reshape(-1, 144, 2)
            keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
            keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
            keypoints2d, keypoints2d_mask = convert_kps(
                keypoints2d, src='smplx', dst='human_data')
            human_data['keypoints2d_smplx'] = keypoints2d
            human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

            # save keypoints 3d smplx
            keypoints3d = np.concatenate(keypoints3d_, axis=0).reshape(-1, 144, 3)
            keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
            keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
            keypoints3d, keypoints3d_mask = convert_kps(
                keypoints3d, src='smplx', dst='human_data')
            human_data['keypoints3d_smplx'] = keypoints3d
            human_data['keypoints3d_smplx_mask'] = keypoints3d_mask
                        
            # save bbox        
            for bbox_name in [
                    'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                    'rhand_bbox_xywh'
            ]:
                bbox_xywh_ = np.array(bboxs_[bbox_name]).reshape((-1, 5))
                human_data[bbox_name] = bbox_xywh_       
                
            # save smplx
            for key in smplx_.keys():
                smplx_[key] = np.concatenate(
                    smplx_[key], axis=0).reshape(self.smplx_shape[key])
            human_data['smplx'] = smplx_      
                    
            # save image path
            human_data['image_path'] = image_path_
            human_data['vtemplate_path'] = vtemplate_path_
            human_data['vertices3d_path'] = verts3d_path_
                        
            # save meta and misc
            human_data['config'] = 'arctic'
            human_data['misc'] = self.misc_config
            human_data['meta'] = meta_

            os.makedirs(out_path, exist_ok=True)
            out_file = os.path.join(
                # out_path, f'moyo_{self.misc_config["flat_hand_mean"]}.npz')
                out_path, f'arctic_{mode}_{seed}_{"{:03d}".format(size_i)}_{split_id}.npz')

            human_data.dump(out_file)        
                    
    # def process_image(args):
    #     # Unpack arguments
    #     (image_path, dataset_path, data_2d, data_cam, data_params, seq_extrx, seq_intrx,
    #     ioi_offset, cidx, image_size_list, betas, sub_id, seq_name, random_ids, gender_seq,
    #     vtemplates_path, smplx_shape, misc_config) = args
        
    #     def _check_valid(self, data_2d, data_cam, vidx, view_idx):  
    #         assert (
    #             vidx < data_2d["joints.right"].shape[0]
    #             ), "The requested camera id does not exist in annotation"
    #         is_valid = data_cam["is_valid"][vidx, view_idx]
    #         right_valid = data_cam["right_valid"][vidx, view_idx]
    #         left_valid = data_cam["left_valid"][vidx, view_idx]
    #         return vidx, is_valid, right_valid, left_valid

    #     # Initialize device within subprocess
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #     # Load or initialize your model within subprocess
    #     vtemplates = trimesh.load(vtemplates_path, force='mesh').vertices.reshape(1, 10475, 3)
    #     gendered_smplx = build_body_model(
    #         dict(
    #             type='SMPLX',
    #             keypoint_src='smplx',
    #             keypoint_dst='smplx',
    #             model_path='data/body_models/smplx',
    #             gender=gender_seq,
    #             v_template=vtemplates,
    #             num_betas=10,
    #             use_face_contour=True,
    #             flat_hand_mean=misc_config['flat_hand_mean'],
    #             use_pca=False,
    #             batch_size=1)).to(device)

    #     # 局部变量用于存储结果
    #     local_smplx_ = {}
    #     for key in smplx_shape.keys():
    #         local_smplx_[key] = []
    #     local_keypoints2d_, local_keypoints3d_ = [], []
    #     local_bboxs_ = {}
    #     for bbox_name in [
    #             'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
    #             'rhand_bbox_xywh'
    #     ]:
    #         local_bboxs_[bbox_name] = []
    #     local_meta_ = {}
    #     for meta_name in ['principal_point', 'focal_length', 'height', 'width', 'RT',
    #                     'sequence_name', 'track_id', 'gender', 'is_valid', 'right_hand_valid', 'left_hand_valid']:
    #         local_meta_[meta_name] = []
    #     local_image_path_ = []
    #     local_vtemplate_path_ = []
    #     local_verts3d_path_ = []

    #     cid = str(cidx)
    #     try:
    #         imgp = os.path.join(dataset_path, image_path)

    #         # image idx and mocap idx (vidx)
    #         image_idx = image_path.split("/")[-1]
    #         image_id = image_idx.split(".")[0]
    #         vidx = int(image_id) - ioi_offset

    #         # check image and hand validity
    #         vidx, is_valid, right_valid, left_valid = _check_valid(
    #             data_2d, data_cam, vidx, cidx)

    #         # prepare intrinsics
    #         extrinsics = seq_extrx[vidx]
    #         intrinsics = seq_intrx[vidx]

    #         # smplx params in world space
    #         smplx_param = {}
    #         for key in smplx_shape.keys():
    #             if key != 'betas':
    #                 smplx_param[key] = data_params[f'smplx_{key}'][vidx].reshape(smplx_shape[key])
    #         smplx_param['betas'] = betas

    #         # prepare smplx tensor
    #         smplx_param_tensor = {}
    #         for key in smplx_shape.keys():
    #             smplx_param_tensor[key] = torch.tensor(smplx_param[key].reshape(smplx_shape[key]),
    #                                                 dtype=torch.float).to(device)

    #         # get output
    #         output_world = gendered_smplx(**smplx_param_tensor)
    #         kps3d = output_world['joints'].detach().cpu().numpy()
    #         pelvis_world = kps3d[:, get_keypoint_idx('pelvis', 'smplx'), :]

    #         # transfrom to camera space
    #         global_orient_cam, transl_cam = transform_to_camera_frame(
    #             global_orient=smplx_param['global_orient'],
    #             transl=smplx_param['transl'],
    #             pelvis=pelvis_world,
    #             extrinsic=extrinsics)

    #         smplx_param['global_orient'] = global_orient_cam
    #         smplx_param['transl'] = transl_cam

    #         # prepare smplx tensor
    #         smplx_param_tensor = {}
    #         for key in smplx_shape.keys():
    #             smplx_param_tensor[key] = torch.tensor(smplx_param[key].reshape(smplx_shape[key]),
    #                                                 dtype=torch.float).to(device)

    #         # get output
    #         output = gendered_smplx(**smplx_param_tensor, return_verts=True)

    #         width, height = image_size_list[cidx]
    #         focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
    #         principal_point = [intrinsics[0, 2], intrinsics[1, 2]]

    #         camera = build_cameras(
    #             dict(
    #                 type='PerspectiveCameras',
    #                 convention='opencv',
    #                 in_ndc=False,
    #                 focal_length=focal_length,
    #                 image_size=(width, height),
    #                 principal_point=principal_point)).to(device)

    #         # project 3d to 2d
    #         kps3d = output['joints']
    #         verts3d = output['vertices']

    #         kps3d_c = kps3d.detach().cpu().numpy().squeeze()
    #         kps2d = camera.transform_points_screen(kps3d).detach().cpu().numpy().squeeze()[:, :2]
    #         verts3d = verts3d.detach().cpu().numpy().squeeze()

    #         # get bbox from 2d keypoints
    #         bboxs = self._keypoints_to_scaled_bbox_bfh(
    #             kps2d,
    #             body_scale=self.misc_config['bbox_body_scale'],
    #             fh_scale=self.misc_config['bbox_facehand_scale'])
    #         for i, bbox_name in enumerate([
    #                 'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
    #                 'rhand_bbox_xywh'
    #         ]):
    #             xmin, ymin, xmax, ymax, conf = bboxs[i]
    #             bbox = np.array([
    #                 max(0, xmin),
    #                 max(0, ymin),
    #                 min(width, xmax),
    #                 min(height, ymax)
    #             ])
    #             bbox_xywh = _xyxy2xywh(bbox)  # list of len 4
    #             bbox_xywh.append(conf)  # (5,)
    #             local_bboxs_[bbox_name].append(bbox_xywh)

    #         # save verts3d
    #         verts3d_path = image_path.replace('jpg', 'npy').replace('images', 'vertices3d')
    #         verts3dp = imgp.replace('jpg', 'npy').replace('images', 'vertices3d')
    #         os.makedirs(os.path.dirname(verts3dp), exist_ok=True)
    #         np.save(verts3dp, verts3d)

    #         # append image path
    #         local_image_path_.append(image_path)
    #         local_verts3d_path_.append(verts3d_path)
    #         local_vtemplate_path_.append(vtemplates_path.replace(f'{dataset_path}/', ''))

    #         # append keypoints2d and 3d
    #         local_keypoints2d_.append(kps2d)
    #         local_keypoints3d_.append(kps3d_c)

    #         # add smplx params
    #         for key in smplx_param.keys():
    #             local_smplx_[key].append(smplx_param[key])

    #         sequence_name = f'{sub_id}_{seq_name}_{cid}'
    #         # append meta
    #         local_meta_['principal_point'].append(principal_point)
    #         local_meta_['focal_length'].append(focal_length)
    #         local_meta_['height'].append(height)
    #         local_meta_['width'].append(width)
    #         local_meta_['sequence_name'].append(sequence_name)
    #         local_meta_['RT'].append(extrinsics)
    #         local_meta_['track_id'].append(random_ids[int(sub_id[1:])])
    #         local_meta_['gender'].append(gender_seq)
    #         local_meta_['is_valid'].append(is_valid)
    #         local_meta_['right_hand_valid'].append(right_valid)
    #         local_meta_['left_hand_valid'].append(left_valid)

    #         # 返回结果
    #         return {
    #             'smplx_': local_smplx_,
    #             'keypoints2d_': local_keypoints2d_,
    #             'keypoints3d_': local_keypoints3d_,
    #             'bboxs_': local_bboxs_,
    #             'meta_': local_meta_,
    #             'image_path_': local_image_path_,
    #             'vtemplate_path_': local_vtemplate_path_,
    #             'verts3d_path_': local_verts3d_path_
    #         }
    #     except Exception as e:
    #         print(f"Error processing image {image_path}: {e}")
    #         return None
  
    # def convert_by_mode(self, dataset_path: str, out_path: str,
    #                 mode: str) -> dict:
        
    #     import multiprocessing
    #     multiprocessing.set_start_method('spawn')
    #     # use HumanData to store all data
    #     human_data = HumanData()
        
    #     # init seed and size
    #     seed, size = '241023', '999'
    #     random.seed(int(seed))
    #     np.set_printoptions(suppress=True)
    #     random_ids = np.random.RandomState(seed=int(seed)).permutation(999999)
    #     used_id_num = 0
        
    #     # load split
    #     split_path = os.path.join(dataset_path, 'splits', mode + '.npy')
    #     split_info = np.load(split_path, allow_pickle=True).item()
        
    #     image_names = split_info['imgnames']
    #     image_names = [img.replace('./arctic_data/data/', '') for img in image_names]
    #     image_names = [img.replace('./arctic_data/', '') for img in image_names]
    #     data_dict = split_info['data_dict']
    #     seq_names = list(data_dict.keys())
        
    #     # get size
    #     size_i = min(int(size), len(seq_names))
        
    #     # train 4 split, val 1 split
    #     if 'train' in mode:
    #         split_num = 4
    #     else:
    #         split_num = 1
            
            
    #     for split_id in range(split_num):
            
    #         # group seq names
    #         size_b = len(seq_names) // split_num
    #         seq_names_batch = seq_names[split_id * size_b: (split_id + 1) * size_b]
        
    #         # load meta
    #         meta_path = os.path.join(dataset_path, 'meta', 'misc.json')
    #         with open(meta_path, 'r') as f:
    #             metadata = json.load(f)
                
    #         cam_params = {}
    #         for sub in metadata.keys():
    #             cam_params[sub] = { # [8, 4, 4] and [8, 3, 3]
    #                 "world2cam": np.array(metadata[sub]["world2cam"]),
    #                 "intrinsics": np.array(metadata[sub]["intris_mat"]),} 
                
    #         smplx_ = {}
    #         for key in self.smplx_shape.keys():
    #             smplx_[key] = []
    #         keypoints2d_, keypoints3d_ = [], []
    #         bboxs_ = {}
    #         for bbox_name in [
    #                 'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
    #                 'rhand_bbox_xywh'
    #         ]:
    #             bboxs_[bbox_name] = []
    #         meta_ = {}
    #         for meta_name in ['principal_point', 'focal_length', 'height', 'width', 'RT',
    #                         'sequence_name', 'track_id', 'gender', 'is_valid', 'right_hand_valid', 'left_hand_valid']:
    #             meta_[meta_name] = []
    #         image_path_ = []
    #         vtemplate_path_ = []
    #         verts3d_path_ = []
            
    #         # high level path
    #         raw_path = os.path.join(dataset_path, 'raw_seqs')
            
    #         # seq_names_batch = seq_names_batch[:5]
            
    #         # group by sequence
    #         for seq in tqdm(seq_names_batch, desc=f'Split: {split_id+1} / {split_num}', position=0, leave=False):
                
    #             seq_data = data_dict[seq]
    #             sub_id = seq.split('/')[0]
    #             seq_name = seq.split('/')[1]
                
    #             # smplx idx = image idx - ioi_offset
    #             ioi_offset = metadata[sub_id]['ioi_offset']
    #             gender_seq = metadata[sub_id]['gender']
    #             image_size_list = metadata[sub_id]['image_size']
                
    #             betas = np.zeros((1, 10))

    #             # prepare vtemplates path
    #             vtemplates_path = os.path.join(dataset_path, 'meta', 'subject_vtemplates', f'{sub_id}.obj')
    #             vtemplates = trimesh.load(vtemplates_path, force='mesh').vertices.reshape(1, 10475, 3)

    #             # build gendered smplx
    #             gendered_smplx = build_body_model(
    #                 dict(
    #                     type='SMPLX',
    #                     keypoint_src='smplx',
    #                     keypoint_dst='smplx',
    #                     model_path='data/body_models/smplx',
    #                     gender=gender_seq,
    #                     v_template=vtemplates,
    #                     num_betas=10,
    #                     use_face_contour=True,
    #                     flat_hand_mean=self.misc_config['flat_hand_mean'],
    #                     use_pca=False,
    #                     batch_size=1)).to(self.device)  
                
    #             # load params
    #             data_cam = seq_data["cam_coord"]
    #             data_2d = seq_data["2d"]
    #             data_bbox = seq_data["bbox"]
    #             data_params = seq_data["params"]
                
    #             data_len = data_params["K_ego"].shape[0]
                
    #             # 准备并行处理的参数列表
    #             args_list = []
    #             for cidx in range(1, 9):
    #                 cid = str(cidx)
    #                 # prepare intrinsics
    #                 if cidx == 0: # ego space
    #                     seq_extrx = data_params['world2ego'].copy()
    #                     seq_intrx = data_params["K_ego"].copy()
    #                 else:
    #                     seq_extrx = np.array(cam_params[sub_id]['world2cam'])[cidx-1]
    #                     seq_extrx = np.repeat(seq_extrx[np.newaxis, :], data_len, axis=0)
    #                     seq_intrx = np.array(cam_params[sub_id]['intrinsics'])[cidx-1]
    #                     seq_intrx = np.repeat(seq_intrx[np.newaxis, :], data_len, axis=0)
                    
    #                 # prepare images to load
    #                 imgp_pattern = f'{seq}/{cid}'
    #                 image_paths = [imgp for imgp in image_names if imgp_pattern in imgp]        

    #                 if len(image_paths) == 0:
    #                     continue

    #                 for image_path in image_paths:
    #                     args = (
    #                         image_path,
    #                         dataset_path,
    #                         data_2d,
    #                         data_cam,
    #                         data_params,
    #                         seq_extrx,
    #                         seq_intrx,
    #                         ioi_offset,
    #                         cidx,
    #                         image_size_list,
    #                         betas,
    #                         sub_id,
    #                         seq_name,
    #                         random_ids,
    #                         gender_seq,
    #                         vtemplates_path,
    #                         self.smplx_shape,
    #                         self.misc_config  # Pass misc_config instead of self
    #                     )
    #                     args_list.append(args)

    #             import concurrent
    #             # Start multiprocessing within __main__
    #             results = []
    #             with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    #                 futures = [executor.submit(self.process_image, args) for args in args_list]
    #                 for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f'Processing images', position=1):
    #                     result = future.result()
    #                     if result is not None:
    #                         results.append(result)

    #             # 合并结果
    #             for res in results:
    #                 # 合并 smplx_
    #                 for key in res['smplx_'].keys():
    #                     smplx_[key].extend(res['smplx_'][key])
    #                 # 合并 keypoints2d_ 和 keypoints3d_
    #                 keypoints2d_.extend(res['keypoints2d_'])
    #                 keypoints3d_.extend(res['keypoints3d_'])
    #                 # 合并 bboxs_
    #                 for bbox_name in res['bboxs_'].keys():
    #                     bboxs_[bbox_name].extend(res['bboxs_'][bbox_name])
    #                 # 合并 meta_
    #                 for meta_name in res['meta_'].keys():
    #                     meta_[meta_name].extend(res['meta_'][meta_name])
    #                 # 合并 image_path_, vtemplate_path_, verts3d_path_
    #                 image_path_.extend(res['image_path_'])
    #                 vtemplate_path_.extend(res['vtemplate_path_'])
    #                 verts3d_path_.extend(res['verts3d_path_'])
                        
    #         # save keypoints 2d smplx
    #         keypoints2d = np.concatenate(keypoints2d_, axis=0).reshape(-1, 144, 2)
    #         keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
    #         keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
    #         keypoints2d, keypoints2d_mask = convert_kps(
    #             keypoints2d, src='smplx', dst='human_data')
    #         human_data['keypoints2d_smplx'] = keypoints2d
    #         human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

    #         # save keypoints 3d smplx
    #         keypoints3d = np.concatenate(keypoints3d_, axis=0).reshape(-1, 144, 3)
    #         keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
    #         keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
    #         keypoints3d, keypoints3d_mask = convert_kps(
    #             keypoints3d, src='smplx', dst='human_data')
    #         human_data['keypoints3d_smplx'] = keypoints3d
    #         human_data['keypoints3d_smplx_mask'] = keypoints3d_mask
                        
    #         # save bbox        
    #         for bbox_name in [
    #                 'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
    #                 'rhand_bbox_xywh'
    #         ]:
    #             bbox_xywh_ = np.array(bboxs_[bbox_name]).reshape((-1, 5))
    #             human_data[bbox_name] = bbox_xywh_       
                
    #         # save smplx
    #         for key in smplx_.keys():
    #             smplx_[key] = np.concatenate(
    #                 smplx_[key], axis=0).reshape(self.smplx_shape[key])
    #         human_data['smplx'] = smplx_      
                    
    #         # save image path
    #         human_data['image_path'] = image_path_
    #         human_data['vtemplate_path'] = vtemplate_path_
    #         human_data['vertices3d_path'] = verts3d_path_
                        
    #         # save meta and misc
    #         human_data['config'] = 'arctic'
    #         human_data['misc'] = self.misc_config
    #         human_data['meta'] = meta_

    #         os.makedirs(out_path, exist_ok=True)
    #         out_file = os.path.join(
    #             # out_path, f'moyo_{self.misc_config["flat_hand_mean"]}.npz')
    #             out_path, f'arctic_{mode}_{seed}_{"{:03d}".format(size_i)}_{split_id}.npz')

    #         human_data.dump(out_file)   
                

            
            
      