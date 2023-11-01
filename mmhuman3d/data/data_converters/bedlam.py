import os
import pickle
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

import logging
import json
import mmcv
import cv2
import torch
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.multi_human_data import MultiHumanData
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.utils.transforms import rotmat_to_aa, aa_to_rotmat
from mmhuman3d.core.visualization.visualize_smpl import render_smpl,visualize_smpl_vibe
from mmhuman3d.core.visualization.visualize_keypoints2d import visualize_kp2d
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idxs_by_part
from mmhuman3d.utils.demo_utils import convert_verts_to_cam_coord, xywh2xyxy, xyxy2xywh,keypoints_to_scaled_bbox_bfh
from mmhuman3d.models.body_models.utils import transform_to_camera_frame, batch_transform_to_camera_frame
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

# def cam2pixel(cam_coord, f, c):
#     x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
#     y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
#     z = cam_coord[:,2]
#     return np.stack((x,y,z),1)

def cam2pixel(cam_coord, f, c):
    x = cam_coord[...,0] / cam_coord[...,2] * f[:, [0]] + c[:, [0]]
    y = cam_coord[...,1] / cam_coord[...,2] * f[:, [1]] + c[:, [1]]
    z = cam_coord[...,2]
    return np.stack((x,y,z),-1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord


@DATA_CONVERTERS.register_module()
class BedlamConverter(BaseModeConverter):
    ACCEPTED_MODES = ['train', 'test']
    def __init__(self, modes: List = []) -> None:
        super(BedlamConverter, self).__init__(modes)

        smplx_config = dict(
            type='SMPLX',
            keypoint_src='smplx',
            keypoint_dst='smplx',
            model_path='data/body_models/smplx',
            gender='male',
            num_betas=11,
            use_face_contour=True,
            flat_hand_mean=True,
            # create_expression=False,
            # create_jaw_pose=True,
            use_pca=False,
            device="cuda",
            batch_size=1)        
        self.smplx_male = build_body_model(smplx_config)
        smplx_config['gender'] = 'female'
        self.smplx_female = build_body_model(smplx_config)
        smplx_config['gender'] = 'neutral'
        self.smplx_neutral = build_body_model(smplx_config)
        
    
    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        
        human_data = HumanData()
        
        instance_folder = os.path.join(dataset_path, 'train_images')
        data_folder = os.path.join(dataset_path, 'all_npz_12_training')
        instance_names = os.listdir(instance_folder)
        
        image_path_, bbox_xywh_, keypoints2d_smplx_, keypoints3d_smplx_= [], [], [], []

        face_bbox_xywh_, lhand_bbox_xywh_, rhand_bbox_xywh_ = [], [], []
        
        # smplx
        smplx = {}
        smplx_key = [
            'betas', 'betas_neutral', 'transl', 'global_orient', 'body_pose', 'left_hand_pose', 
            'right_hand_pose', 'leye_pose', 'reye_pose', 'jaw_pose', 'expression', 'flat_hand_mean'
            ]
        for key in smplx_key:
            smplx[key] = []

        
        meta = {}
        meta['focal_length'] = []
        meta['principal_point'] = []
        meta['height'] = []
        meta['width'] = []
        meta['gender'] = []
        # 

        for j, instance_name in tqdm( enumerate(instance_names)):

            seq_folder = os.path.join(instance_folder, instance_name, 'png')
            # ['imgname', 'center', 'scale', 'pose_cam', 'pose_world', 'shape', 'trans_cam', 'trans_world', 'gtkps', 'cam_int', 'cam_ext', 'gender', 'proj_verts']
            if not os.path.exists(seq_folder):
                continue
            try:
                body_data = np.load(os.path.join(data_folder, instance_name+'.npz'))
            except:
                continue
            img_names_ = body_data['imgname']
            transl_ = body_data['trans_cam']
            betas_ = body_data['shape']
            gender_ = body_data['gender']
            
            cam_int_ = body_data['cam_int']
            focal_ = np.stack([cam_int_[:, 0, 0], cam_int_[:, 1, 1]],-1)
            princpt_ = np.stack([cam_int_[:, 0, 2], cam_int_[:, 1, 2]],-1)
            
            cam_ext_ = body_data['cam_ext']
            R_ = cam_ext_[:, :3, :3]
            T_ = cam_ext_[:, :, 3][:,:3]
            
            transl_ = transl_ + T_
            
            gtkps_ = body_data['gtkps']
            center_ = body_data['center']
            scale_ = body_data['scale']
            
            pose_ = body_data['pose_cam'].reshape(-1, 55, 3)
            global_orient_ = pose_[:,:1,:]
            
            # global_orient_ = aa_to_rotmat(global_orient_)
            # global_orient_ = rotmat_to_aa(np.dot(R_, global_orient_))
            # global_orient_ = torch.from_numpy(global_orient_)

            body_pose_ = pose_[:,1:22,:] 
            
            jaw_pose_ = pose_[:,22:23,:]
            leye_pose_ = pose_[:,23:24,:]
            reye_pose_ = pose_[:,24:25,:]
            
            left_hand_pose_ = pose_[:,25:40,:]
            right_hand_pose_ = pose_[:,40:55,:]
            


            batch_size = global_orient_.shape[0]
            global_orient = torch.FloatTensor(global_orient_).view(-1, 3) # (1,3)
            body_pose = torch.FloatTensor(body_pose_).view(-1,21*3) # (21,3)
            betas = torch.FloatTensor(betas_).view(-1, 11) # SMPLX shape parameter
            transl = torch.FloatTensor(transl_).view(-1, 3) # translation vector 
            left_hand_pose = torch.FloatTensor(left_hand_pose_).view(-1, 15*3)
            right_hand_pose = torch.FloatTensor(right_hand_pose_).view(-1, 15*3)
            leye_pose = torch.FloatTensor(leye_pose_).view(-1, 3)
            reye_pose = torch.FloatTensor(reye_pose_).view(-1, 3)
            jaw_pose = torch.FloatTensor(jaw_pose_).view(-1, 3)
            expression = torch.zeros([batch_size, 10])
            
            smplx_config = dict(
                type='SMPLX',
                keypoint_src='smplx',
                keypoint_dst='smplx',
                model_path='data/body_models/smplx',
                gender='neutral',
                num_betas=11,
                use_face_contour=True,
                flat_hand_mean=True,
                # create_expression=False,
                # create_jaw_pose=True,
                use_pca=False,
                device="cuda",
                batch_size=batch_size)    
            self.smplx_neutral = build_body_model(smplx_config)

            smplx_res = self.smplx_neutral(
                betas=betas,
                global_orient=global_orient,
                transl=transl,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                body_pose=body_pose,
                expression=expression,
                pose2rot=True
                )
            keypoints3d_smplx_cam=smplx_res['joints'].detach().cpu().numpy()
            keypoints3d_smplx_img = cam2pixel(keypoints3d_smplx_cam, focal_, princpt_)
            keypoints2d_smplx = keypoints3d_smplx_img[...,:2]
            
            bbox_xyxy_smplx = []
            for kp2d in keypoints2d_smplx:
                bbox_xyxy = keypoints_to_scaled_bbox_bfh(
                    kp2d, body_scale=1.2, fh_scale=1, convention='smplx')
                bbox_xyxy_smplx.append(bbox_xyxy)
            bbox_xyxy_smplx = np.array(bbox_xyxy_smplx)
            bbox_xywh_smplx = xyxy2xywh(bbox_xyxy_smplx)            

            
            for i, img_name in enumerate(img_names_):
                img_path = os.path.join(seq_folder, img_name)
                img = cv2.imread(img_path)

                if 'closeup' in seq_folder:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
                    
                    p_folder = os.path.dirname(img_path.replace('_closeup', ''))
                    if not os.path.exists(p_folder):
                        os.makedirs(p_folder)

                    cv2.imwrite(os.path.join(seq_folder.replace('_closeup', ''), img_name), img)
                
                height = img.shape[0]
                width = img.shape[1] 

                                
                if False:
                    keypoints3d_smplx_cam=smplx_res['joints'].detach().cpu().numpy()[0]
                    keypoints3d_smplx_img = cam2pixel(keypoints3d_smplx_cam, focal_[i], princpt_[i])
                    keypoints2d_smplx = keypoints3d_smplx_img[:,:2]
                    bbox_xyxy_smplx = keypoints_to_scaled_bbox_bfh(
                        keypoints2d_smplx, body_scale=1, fh_scale=1, convention='smplx')
                    bbox_xywh_smplx = xyxy2xywh(np.array(bbox_xyxy_smplx))
                    img = visualize_kp2d(keypoints2d_smplx[None], output_path='.', image_array=img.copy()[None], data_source='smplx',overwrite=True)
                    img = mmcv.imshow_bboxes(img[0].copy(),np.array(bbox_xyxy_smplx), show=False)
                    cv2.imwrite('test_smplx.png',img)
                    
                    gtkps = gtkps_[i]

                    keypoints2d_smplx = gtkps[:,:2]
                    keypoints2d_smplx = np.vstack([keypoints2d_smplx,np.zeros([22,2])])
                    bbox_xyxy_smplx = keypoints_to_scaled_bbox_bfh(keypoints2d_smplx, body_scale=1.2, fh_scale=1.2,convention='smpl')
                    bbox_xywh_smplx = xyxy2xywh(np.array(bbox_xyxy_smplx)) 
                    img = visualize_kp2d(keypoints2d_smplx[None], output_path='.', image_array=img.copy()[None], data_source='smplx',overwrite=True)
                    img = mmcv.imshow_bboxes(img[0].copy(),np.array(bbox_xyxy_smplx), show=False)
                    cv2.imwrite('test_smplx.png',img)    

                image_path_.append(os.path.join(instance_name.replace('_closeup', ''),'png',img_name))
                bbox_xywh_.append(bbox_xywh_smplx[i][0])
                face_bbox_xywh_.append(bbox_xywh_smplx[i][1]) 
                lhand_bbox_xywh_.append(bbox_xywh_smplx[i][2])
                rhand_bbox_xywh_.append(bbox_xywh_smplx[i][3])
                keypoints2d_smplx_.append(keypoints2d_smplx[i])
                keypoints3d_smplx_.append(keypoints3d_smplx_cam[i])                

                # smplx
                smplx['flat_hand_mean'].append(True)
                smplx['betas_neutral'].append(np.array(betas[i]).reshape(11))
                # smplx['betas'].append(np.array(betas).reshape(-1))
                smplx['global_orient'].append(np.array(global_orient[i]).reshape(3))
                smplx['body_pose'].append(np.array(body_pose[i]).reshape(21,3))
                smplx['transl'].append(np.array(transl[i]).reshape(3))
                smplx['left_hand_pose'].append(np.array(left_hand_pose[i]).reshape(15,3))
                smplx['right_hand_pose'].append(np.array(right_hand_pose[i]).reshape(15,3))
                smplx['leye_pose'].append(np.array(leye_pose[i]).reshape(3))
                smplx['reye_pose'].append(np.array(reye_pose[i]).reshape(3))
                smplx['jaw_pose'].append(np.array(jaw_pose[i]).reshape(3))
                smplx['expression'].append(np.array(expression[i]).reshape(10))

                # meta
                meta['height'].append(height)
                meta['width'].append(width)
                meta['focal_length'].append(focal_[i])
                meta['principal_point'].append(princpt_[i])
                meta['gender'].append(gender_[i])                    
                
            
            
        # smplx
        smplx['betas'] = smplx['betas_neutral']
        for k, v in smplx.items():
            smplx[k] = np.array(v)
        human_data['smplx'] = smplx
        
        # kp2d_smplx
        keypoints2d_smplx = np.array(keypoints2d_smplx_).reshape(-1, 144, 2)
        keypoints2d_smplx_conf = np.ones([keypoints2d_smplx.shape[0], 144, 1])
        keypoints2d_smplx = np.concatenate([keypoints2d_smplx, keypoints2d_smplx_conf], axis=-1)
        keypoints2d_smplx, keypoints2d_smplx_mask = \
            convert_kps(keypoints2d_smplx, src='smplx', dst='human_data')
        human_data['keypoints2d_smplx'] = keypoints2d_smplx
        human_data['keypoints2d_smplx_mask'] = keypoints2d_smplx_mask
        
        # kp3d_smplx
        keypoints3d_smplx = np.array(keypoints3d_smplx_).reshape(-1, 144, 3)
        keypoints3d_smplx_conf = np.ones([keypoints3d_smplx.shape[0], 144, 1])
        keypoints3d_smplx = np.concatenate([keypoints3d_smplx, keypoints3d_smplx_conf], axis=-1)
        keypoints3d_smplx, keypoints3d_smplx_mask = \
                convert_kps(keypoints3d_smplx, src='smplx', dst='human_data')
        human_data['keypoints3d_smplx'] = keypoints3d_smplx
        human_data['keypoints3d_smplx_mask'] = keypoints3d_smplx_mask

        human_data['image_path'] = image_path_
        # bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 5))
        human_data['bbox_xywh'] = np.array(bbox_xywh_)
        human_data['face_bbox_xywh'] = np.array(face_bbox_xywh_)
        human_data['lhand_bbox_xywh'] = np.array(lhand_bbox_xywh_)
        human_data['rhand_bbox_xywh'] = np.array(rhand_bbox_xywh_)
        human_data['config'] = 'bedlam'


        for k, v in meta.items():
            meta[k] = np.array(v)
        human_data['meta'] = meta
        
        human_data.compress_keypoints_by_mask()
        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, f'bedlam_{mode}.npz')
        human_data.dump('/mnt/lustre/share_data/sunqingping/edpose/data/preprocessed_npz/bedlam_train_fix_img_shape.npz')
