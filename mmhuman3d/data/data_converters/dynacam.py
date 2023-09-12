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


@DATA_CONVERTERS.register_module()
class DynacamConverter(BaseModeConverter):

    ACCEPTED_MODES = ['panorama_test', 'panorama_train', 
                      'panorama_val', 'translation_test', 
                      'translation_train', 'translation_val']

    def __init__(self, modes: List = []) -> None:
        self.device = torch.device('cuda:0')
        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_source='keypoints2d_smpl',
            cam_param_type='prespective',
            smpl_source='original',
        )
        self.smpl_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 23, 3),
        }

        super(DynacamConverter, self).__init__(modes)


    def convert_by_mode(self, dataset_path: str, out_path: str,
                mode: str) -> dict:
        
        # use HumanData to store all data
        human_data = HumanData()

        # initailize
        smpl_ = {}
        for key in self.smpl_shape.keys():
            smpl_[key] = []
        bboxs_ = {}
        for key in ['bbox_xywh']:
            bboxs_[key] = []
        image_path_, keypoints2d_smpl_ = [], []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length', 'height', 'width']:
            meta_[meta_key] = []

        seed = 230720
        size = 999

        # prepare batch
        batch_name, batch_mode = mode.split('_')

        # build body model
        smpl_model = build_body_model(dict(
            type='SMPL',
            keypoint_src='smpl_45',
            keypoint_dst='smpl_45',
            model_path='data/body_models/smpl',
            gender='neutral',
            num_betas=10,
            use_pca=False,
            batch_size=1)).to(self.device)


        # read annotation
        annot_path = os.path.join(dataset_path, 'annotations', f'{mode}.npz')
        annots = dict(np.load(annot_path, allow_pickle=True))['annots'].item()
        img_ext = 'jpg' if 'panorama' in mode else 'png'
        
        # parse seqs
        seqs = list(annots.keys())
        seqs.remove('sequence_dict')
        seqs.remove('ID_num')

        # parse seqs
        # for seq in seqs:
        for seq in tqdm(seqs, desc=f'convert {mode}', total=len(seqs), 
                        position=0, leave=False):    
            
            # image base path
            img_bp = os.path.join(dataset_path, f'video_frames_{batch_mode}', 
                                  mode, seq)


            annot = annots[seq]

            if 'world_grots_aligned' in annots:
                global_orient_ws = annot['world_grots_aligned']
                transl_ws = annot['world_trans_aligned']
                camera_intrinsics = annot['camera_intrinsics']
                camera_extrinsics = annot['camera_extrinsics_aligned']
            else:
                global_orient_ws = annot['world_grots']
                transl_ws = annot['world_trans']
                camera_intrinsics = annot['camera_intrinsics']
                camera_extrinsics = annot['camera_extrinsics']
                camera_extrinsics = np.concatenate([camera_extrinsics, 
                                                    np.repeat(np.array([[[0,0,0,1]]]), 
                                                            len(camera_extrinsics), axis=0)], axis=1)
       
            # get R T from extrinsics
            R = camera_extrinsics[:, :3, :3]
            T = camera_extrinsics[:, :3, 3]
            _, camera_extrinsics, _ = convert_camera_matrix(R=R, T=T, 
                                                convention_src='open3d', 
                                                convention_dst='opencv')
            
            
            
            smpl_thetas = annot['poses']
            subject_num, frame_num = smpl_thetas.shape[:2]
            smpl_poses = smpl_thetas[:, :, 1:].reshape(subject_num, frame_num, 23*3)
            smpl_betas = annot['betas']

            frame_names = annot['frame_ids']
            
            # get image size
            img_path = glob.glob(os.path.join(img_bp, f'00*.{img_ext}'))[0]
            img = cv2.imread(img_path)
            height, width = img.shape[:2]

            for sid in range(subject_num):

                for fid, fname in tqdm(enumerate(frame_names), desc=f'Subject {sid+1}/{subject_num}',
                                        total=len(frame_names), position=1, leave=False):

                    # image path
                    img_path = os.path.join(img_bp, f'{"{:06d}".format(fname)}.{img_ext}')
                    if not os.path.exists(img_path):
                        print(f'Not exist: {img_path}')
                        continue
                    image_path = img_path.replace(f'{dataset_path}{os.path.sep}', '')

                    betas = smpl_betas[sid, fid].reshape(1, -1)
                    body_pose = smpl_poses[sid, fid].reshape(1, -1)
                    global_orient_w = global_orient_ws[sid, fid].reshape(1, -1)
                    transl_w = transl_ws[sid, fid].reshape(1, -1)

                    # build camera
                    extrinsic = camera_extrinsics[fid]
                    intrinsic = camera_intrinsics[fid]

                    eye4 = np.eye(4)
                    eye4[:3, :3] = camera_extrinsics[fid]
                    extrinsic = eye4


                    # pdb.set_trace()

                    camera = build_cameras(
                        dict(
                            type='PerspectiveCameras',
                            convention='opencv',
                            in_ndc=False,
                            image_size=(width, height),
                            focal_length=(intrinsic[0, 0], intrinsic[1, 1]),
                            principal_point=(intrinsic[0, 2], intrinsic[1, 2]))
                            ).to(self.device)
          
                    # build smpl
                    output = smpl_model(
                        global_orient=torch.tensor(global_orient_w, device=self.device),
                        transl=torch.tensor(transl_w, device=self.device),
                        body_pose=torch.tensor(body_pose, device=self.device),
                        betas=torch.tensor(betas, device=self.device),
                        )
                    keypoints_3d = output['joints'].detach().cpu().numpy()
                    pelvis_world = keypoints_3d[0, get_keypoint_idx('pelvis', 'smpl')]

                    # convert to camera frame
                    global_orient, transl = transform_to_camera_frame(
                        global_orient=global_orient_w,
                        transl=transl_w,
                        pelvis=pelvis_world,
                        extrinsic=extrinsic)  
                    
                    output = smpl_model(
                        global_orient=torch.tensor(global_orient.reshape(1, -1), device=self.device),
                        transl=torch.tensor(transl.reshape(1, -1), device=self.device),
                        body_pose=torch.tensor(body_pose, device=self.device),
                        betas=torch.tensor(betas, device=self.device),
                        )
                    keypoints_3d = output['joints']
                    keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
                    keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()

                    # get bbox_xywh
                    bbox_xyxy = self._keypoints_to_scaled_bbox(keypoints_2d, scale=1.2)
                    xmin, ymin, xmax, ymax = bbox_xyxy
                    bbox = np.array([max(0, xmin), max(0, ymin),
                        min(width, xmax), min(height, ymax), 1])

                    # append image path
                    image_path_.append(image_path)

                    # append keypoints2d
                    keypoints2d_smpl_.append(keypoints_2d)

                    # append bbox
                    bboxs_['bbox_xywh'].append(bbox)

                    # append smpl
                    smpl_['global_orient'].append(global_orient.reshape(1, -1))
                    smpl_['betas'].append(betas.reshape(1, -1))
                    smpl_['body_pose'].append(body_pose.reshape(1, -1))
                    smpl_['transl'].append(transl.reshape(1, -1))

                    # append meta
                    meta_['principal_point'].append((intrinsic[0, 2], intrinsic[1, 2]))
                    meta_['focal_length'].append((intrinsic[0, 0], intrinsic[1, 1]))
                    meta_['height'].append(height)
                    meta_['width'].append(width)

                # visulize
                # img = cv2.imread(img_path)
                # kps2d = keypoints_2d[0]
                # for kp in kps2d:
                #     cv2.circle(img, (int(kp[0]), int(kp[1])), 2, (0,0,255), -1)
                # os.makedirs(f'{out_path}', exist_ok=True)
                # cv2.imwrite(f'{out_path}/{seq}_{fname}.jpg', img)

        size_i = min(size, len(seqs))

        # meta
        human_data['meta'] = meta_

        # image path
        human_data['image_path'] = image_path_

        # save bbox
        for bbox_name in bboxs_.keys():
            bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 5)
            human_data[bbox_name] = bbox_

        # save smplx
        # human_data.skip_keys_check = ['smplx']
        for key in smpl_.keys():
            smpl_[key] = np.concatenate(
                smpl_[key], axis=0).reshape(self.smpl_shape[key])
        human_data['smpl'] = smpl_

        # keypoints2d_smplx
        keypoints2d_smpl = np.concatenate(
            keypoints2d_smpl_, axis=0).reshape(-1, 45, 2)
        keypoints2d_smpl_conf = np.ones([keypoints2d_smpl.shape[0], 45, 1])
        keypoints2d_smpl = np.concatenate(
            [keypoints2d_smpl, keypoints2d_smpl_conf], axis=-1)
        keypoints2d_smpl, keypoints2d_smpl_mask = \
                convert_kps(keypoints2d_smpl, src='smpl_45', dst='human_data')
        human_data['keypoints2d_smpl'] = keypoints2d_smpl
        human_data['keypoints2d_smpl_mask'] = keypoints2d_smpl_mask

        # misc
        human_data['misc'] = self.misc_config
        human_data['config'] = f'dynacam_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path,
            f'dynacam_{mode}_{seed}_{"{:03d}".format(size_i)}.npz')
        human_data.dump(out_file)