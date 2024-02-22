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
class Idea400Converter(BaseModeConverter):

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

        super(Idea400Converter, self).__init__(modes)

    def _split_motion_param(self, anno):

        # motion_params = {
        #     'global_orient': motion[:, :3],  # controls the global root orientation
        #     'body_pose': motion[:, 3:3+63],  # controls the body
        #     'left_hand_pose': motion[:, 66:66+45],  # controls the finger articulation
        #     'right_hand_pose': motion[:, 66+45:66+90],  # controls the finger articulation
        #     'jaw_pose': motion[:, 66+90:66+93],  # controls the yaw pose
        #     'face_expr': motion[:, 159:159+50],  # controls the face expression
        #     'expression': motion[:, 159:159+10],  # controls the face expression
        #     'face_shape': motion[:, 209:209+100],  # controls the face shape
        #     'transl': motion[:, 309:309+3],  # controls the global body position
        #     'betas': motion[:, 312:],  # controls the body shape. Body shape is static
        #     'leye_pose': np.zeros((motion.shape[0], 3)),
        #     'reye_pose': np.zeros((motion.shape[0], 3)),
        # }

        ped_num, frame_num, _ = anno['trans'].shape

        motion_params = {
            'betas': anno['betas'].reshape(ped_num, 1, -1).repeat(frame_num, axis=1),
            'transl': anno['trans'].reshape(ped_num, frame_num, 3),
            'global_orient': anno['root_orient'].reshape(ped_num, frame_num, 3),
            'body_pose': anno['pose_body'].reshape(ped_num, frame_num, 21, 3),
            'left_hand_pose': anno['pose_lhand'].reshape(ped_num, frame_num, 15, 3),
            'right_hand_pose': anno['pose_rhand'].reshape(ped_num, frame_num, 15, 3),
            'leye_pose': np.zeros((ped_num, frame_num, 3)),
            'reye_pose': np.zeros((ped_num, frame_num, 3)),
            'expression': anno['expr'].reshape(ped_num, frame_num, -1)[:, :, :10],
            'jaw_pose': anno['pose_jaw'].reshape(ped_num, frame_num, 3),       
        }

        return motion_params, ped_num, frame_num

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        
        print('Converting Idea400 dataset...')



        # random seed and size
        seed, size = '240123', '99999'
        np.set_printoptions(suppress=True)
        np.random.seed(int(seed))

        # init track id list
        random_ids = np.random.RandomState(seed=int(seed)).permutation(999999)
        used_id_num = 0

        # build smplx model
        smplx_model = build_body_model(
            dict(
                type='SMPLX',
                keypoint_src='smplx',
                keypoint_dst='smplx',
                model_path='data/body_models/smplx',
                gender='neutral',
                num_betas=10,
                use_face_contour=True,
                flat_hand_mean=True,
                use_pca=False,
                batch_size=1)).to(self.device)

        # Find all seqs annotation
        # anno_ps = sorted(glob.glob(os.path.join(dataset_path, 'motion', '*', '*.npy')))
        anno_ps = glob.glob(os.path.join(dataset_path, 'idea400_smplx_HPE', '*', 'subset_*', '*.npz'))
        anno_ps = sorted(anno_ps)[:int(size)]

        # image_folder_ps = sorted(glob.glob(os.path.join(dataset_path, 'image', 'subset_0034', '*')))
        # anno_ps = anno_ps[3:4]

        # get slices and slice length
        slices = 5
        slice_len = len(anno_ps) // slices

        for sl_id in range(slices):
            # use HumanData to store all data
            human_data = HumanData()

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
            for meta_key in ['principal_point', 'focal_length', 'track_id', 'sequence_name', 
                            'RT', 'width', 'height']:
                meta_[meta_key] = []
            image_path_ = []

            for aid, anno_p in enumerate(tqdm(anno_ps[slice_len*sl_id:slice_len*(sl_id+1)],
                                                desc=f'Processing {mode}, slice {sl_id+1} / {slices}',
                                                position=0, leave=False)):

                # load annotation
                anno = dict(np.load(anno_p, allow_pickle=True))
                smplx_param, ped_num, frame_num = self._split_motion_param(anno)

                # print(anno['world_scale'])

                # sequence name & image folder
                seq_name = os.path.basename(anno_p).split('.')[0]
                image_folder = anno_p.replace('idea400_smplx_HPE', 'image').replace('.npz', '')

                # image_folder = ''
                # for i in range(len(image_folder_ps)):
                #     img_num = len(os.listdir(image_folder_ps[i]))
                #     if frame_num != len(os.listdir(image_folder_ps[i])):
                #         continue
                #     image_folder = image_folder_ps[i]
                if not os.path.exists(image_folder):
                    print(f'Image folder {image_folder} does not exist!')
                    continue

                # get height and width
                img_n = os.listdir(image_folder)[0]
                imgp_sample = os.path.join(image_folder, img_n)
                img_sample = cv2.imread(imgp_sample)
                height, width = img_sample.shape[:2]

                # load camera intrinics
                focal_length = [anno['intrins'][0], anno['intrins'][1]]
                principal_point = [anno['intrins'][2], anno['intrins'][3]]

                # build intrinsics camera
                camera = build_cameras(
                    dict(
                        type='PerspectiveCameras',
                        convention='opencv',
                        in_ndc=False,
                        focal_length=focal_length,
                        image_size=(width, height),
                        principal_point=principal_point)).to(self.device)
                
                # suppose all extrinsics are same
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = anno['cam_R'][0][0]
                extrinsic[:3, 3] = anno['cam_t'][0][0]

                for pid in range(ped_num):

                    # get track id
                    track_id = random_ids[used_id_num]
                    used_id_num += 1

                    # get output
                    smplx_param_tensor = {}
                    for key in self.smplx_shape.keys():
                        smplx_param_tensor[key] = torch.tensor(smplx_param[key][pid].reshape(self.smplx_shape[key])
                                                            ).float().to(self.device)
                    output = smplx_model(**smplx_param_tensor)

                    # get keypoints3d
                    keypoints3d = output['joints'].detach().cpu().numpy()
                    pelvis_world = keypoints3d[:, get_keypoint_idx('pelvis', 'smplx')]
                    # print(pelvis_world.shape)

                    # batch transform to camera frame
                    global_orient_c, transl_c = batch_transform_to_camera_frame(
                        global_orient=smplx_param['global_orient'][pid],
                        transl=smplx_param['transl'][pid],
                        pelvis=pelvis_world,
                        extrinsic=extrinsic)
                    
                    smplx_param_tensor['global_orient'] = torch.tensor(global_orient_c).float().to(self.device)
                    smplx_param_tensor['transl'] = torch.tensor(transl_c).float().to(self.device)

                    # get keypoints3d
                    output = smplx_model(**smplx_param_tensor)
                    keypoints3d = output['joints'].detach().cpu().numpy()

                    frame_len = keypoints3d.shape[0]

                    # pdb.set_trace()
                    for fid in tqdm(range(frame_len), desc=f'Processing {seq_name}', 
                                    position=1, leave=False):
                        
                        # if fid % 2 != 0:
                        #     continue
                        # iid = fid / 2 + 1

                        # get image path
                        imgp = os.path.join(image_folder, f'{fid+1:06d}.png')
                        image_path = imgp.replace(f'{dataset_path}/', '')
                        if not os.path.exists(imgp):
                            print(f'Image {imgp} does not exist!')
                            continue
                        
                        # filter out T pose
                        # pdb.set_trace()
                        if np.sum(np.abs(smplx_param_tensor['body_pose'][fid].detach().cpu().numpy())) < 1:
                            print(f'Image {imgp} is T pose!')
                            continue

                        # project kps3d
                        kps3d = keypoints3d[fid]
                        kps3d_tensor = torch.tensor(kps3d).float().to(self.device)
                        kps2d = camera.transform_points_screen(kps3d_tensor)[..., :2].detach().cpu().numpy()

                        # test overlay
                        # img = cv2.imread(imgp)
                        # for kps in kps2d:
                        #     cv2.circle(img, (int(kps[0]), int(kps[1])), 3, (0, 0, 255), -1)
                        # cv2.imwrite(f'{dataset_path}/output/{int(iid):06d}.png', img)


                        # get bbox from 2d keypoints
                        bboxs = self._keypoints_to_scaled_bbox_bfh(
                            kps2d,
                            body_scale=self.misc_config['bbox_body_scale'],
                            fh_scale=self.misc_config['bbox_facehand_scale'])
                        ## convert xyxy to xywh
                        for i, bbox_name in enumerate([
                                'bbox_xywh', 'face_bbox_xywh',
                                'lhand_bbox_xywh', 'rhand_bbox_xywh'
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

                        # append keypoints
                        keypoints2d_.append(kps2d)
                        keypoints3d_.append(kps3d)

                        # append smplx
                        for key in smplx_param.keys():
                            smplx_[key].append(smplx_param_tensor[key][fid].detach().cpu().numpy())

                        # append image path
                        image_path_.append(image_path)

                        # append meta
                        meta_['principal_point'].append(principal_point)
                        meta_['focal_length'].append(focal_length)
                        meta_['track_id'].append(track_id)
                        meta_['sequence_name'].append(seq_name)
                        meta_['RT'].append(extrinsic)
                        meta_['width'].append(width)
                        meta_['height'].append(height)

            # get size
            size_i = min(int(size), len(anno_ps))

            for key in smplx_.keys():
                smplx_[key] = np.concatenate(
                    smplx_[key], axis=0).reshape(self.smplx_shape[key])
            human_data['smplx'] = smplx_.copy()
            del smplx_


            for key in bboxs_.keys():
                bbox_ = np.array(bboxs_[key]).reshape((-1, 5))
                human_data[key] = bbox_.copy()
            del bboxs_

            # keypoints 2d
            keypoints2d = np.concatenate(
                keypoints2d_, axis=0).reshape(-1, 144, 2)
            keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
            keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf],
                                            axis=-1)
            keypoints2d, keypoints2d_mask = \
                    convert_kps(keypoints2d, src='smplx', dst='human_data')
            human_data['keypoints2d_smplx'] = keypoints2d.copy()
            human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

            del keypoints2d_, keypoints2d, keypoints2d_mask

            # keypoints 3d
            keypoints3d = np.concatenate(
                keypoints3d_, axis=0).reshape(-1, 144, 3)
            keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
            keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf],
                                            axis=-1)
            keypoints3d, keypoints3d_mask = \
                    convert_kps(keypoints3d, src='smplx', dst='human_data')
            human_data['keypoints3d_smplx'] = keypoints3d.copy()
            human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

            del keypoints3d_, keypoints3d, keypoints3d_mask

            # image path
            human_data['image_path'] = image_path_

            # meta
            human_data['meta'] = meta_.copy()
            del meta_

            # store
            human_data['config'] = f'idea400_{mode}'
            human_data['misc'] = self.misc_config

            human_data.compress_keypoints_by_mask()
            os.makedirs(out_path, exist_ok=True)
            print('HumanData dumping starts at',
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            out_file = os.path.join(
                out_path, f'idea400_{mode}_{seed}_{"{:05d}".format(size_i)}_{sl_id}.npz')
            human_data.dump(out_file)


