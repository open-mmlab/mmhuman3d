import glob
import os
import pdb
import random
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm
import joblib

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

from scipy.spatial.transform import Rotation

@DATA_CONVERTERS.register_module()
class Flag3dConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train', 'val']

    def __init__(self, modes=[], *args, **kwargs):

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.misc_config = dict(
            bbox_source='keypoints2d_original',
            smpl_source='original',
            cam_param_type='prespective',
            kps3d_root_aligned=False,
            image_size=(854, 480),
            fps=30,
        )

        self.smpl_shape = {
            'body_pose': (-1, 69),
            'betas': (-1, 10),
            'global_orient': (-1, 3),
            'transl': (-1, 3),}
        
        super(Flag3dConverter, self).__init__(modes)

    def _inverse_symbol(self, name):
        m = int(name[3])
        p = int(name[6:8])
        a = int(name[10:12])
        r = int(name[15])
        n = (m - 1) * 1800 + (p - 1) * 180 + (a - 1) * 3 + r
        if n % 1800 == 0:
            s = n // 1800
        else:
            s = n // 1800 + 1
        if (n - (s - 1) * 1800) % 300 == 0:
            c = (n - (s - 1) * 1800) // 300
        else:
            c = (n - (s - 1) * 1800) // 300 + 1
        return s, c
    
    def _load_camera_params(self, cam_p):
        with open(cam_p, 'r') as f:
            cam_info = f.readlines()
        extrinsic = np.eye(4)
        extrinsic[0] = np.array(cam_info[5].strip().split(' '), np.float32)
        extrinsic[1] = np.array(cam_info[6].strip().split(' '), np.float32)
        extrinsic[2] = np.array(cam_info[7].strip().split(' '), np.float32)

        T = np.array(cam_info[1].strip().split(' '), np.float32)
        R = np.array(cam_info[3].strip().split(' '), np.float32)
        
        fov = np.array(cam_info[12].strip().split(' '), dtype=np.float32)
        # focal_length = np.array(cam_info[14].strip().split(' '))

        return extrinsic, fov, R, T
    

    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:
        print('Converting Flag3d dataset...')

        # load keypoints  'flag3d_keypoint.pkl'
        kp_p = os.path.join(dataset_path, 'flag3d_keypoint_reformat.pkl')
        with open(kp_p, 'rb') as f:
            kp_param = joblib.load(f)

        # parse sequences
        seq_names = kp_param['split'][mode]
        seqs = [os.path.join(dataset_path, 'subset_video', f'{sn[6:]}.mp4')
                for sn in seq_names]

        # build smpl model
        smpl_model = build_body_model(
            dict(
                type='SMPL',
                keypoint_src='smpl_45',
                keypoint_dst='smpl_45',
                model_path='data/body_models/smpl',
                gender='neutral',
                num_betas=10,
                use_pca=False,
                batch_size=1)).to(self.device)
        
        # use HumanData to store the data
        human_data = HumanData()

        # initialize
        smpl_ = {}
        for key in self.smpl_shape.keys():
            smpl_[key] = []
        bboxs_ = {}
        for key in ['bbox_xywh']:  
            bboxs_[key] = []
        image_path_, keypoints2d_original_ = [], []
        keypoints2d_smpl_, keypoints3d_smpl_ = [], []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length', 'height', 'width', 
                         'gender']:
            meta_[meta_key] = []

        seed = '230922'
        size = 9999

        # parse seqs
        for seq in seqs:

            # load smpl params
            seq_name = os.path.basename(seq).replace('.mp4', '')
            seq_name = seq_name[2:]

            # get camera id
            scene_id, camera_id = self._inverse_symbol(seq_name)

            # load camera params
            cam_p = os.path.join(dataset_path, 'camera', 
                                 f's{scene_id}camera', f'camera{camera_id}.txt')

            # load camera params
            extrinsics, fov, R, T = self._load_camera_params(cam_p)

            # parpare camera
            height, width = 480, 854
            f = width / (2 * np.tan(fov * np.pi / 180 / 2))
            f = 2300
            focal_length = np.array((f, f)).reshape(-1, 2)
            principal_point = (width / 2, height / 2)

            principal_point = (0, 0)

            camera = build_cameras(
                dict(
                    type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=focal_length,
                    image_size=(width, height),
                    principal_point=principal_point)).to(self.device)
            
            # load keypoints3d guess openpose_25
            j3d = kp_param['annotations'][seq_name]['keypoint'].reshape(-1, 25, 3)


            # pickle is saved by joblib
            pkl_p = os.path.join(dataset_path, f'subset_smpl_param', seq_name + '.pkl')
            with open(pkl_p, 'rb') as f:
                smpl_pkl = joblib.load(f)

            # prepare smpl params
            frame_num = smpl_pkl['poses'].shape[0]
            smpl_param = {}
            smpl_param['body_pose'] = smpl_pkl['poses'][:, 3:].reshape(-1, 23, 3)
            smpl_param['betas'] = smpl_pkl['shapes'].repeat(frame_num).reshape(-1, 10)
            smpl_param['global_orient'] = smpl_pkl['Rh']
            smpl_param['transl'] = smpl_pkl['Th']

            # prepare image path
            image_folder = os.path.join(dataset_path, f'subset_image', seq_name)
            image_paths = glob.glob(os.path.join(image_folder, '*.png'))

            assert len(image_paths) == frame_num, \
                f'frame num not match: {len(image_paths)} vs {frame_num} in {seq_name}'

            # prepare smpl
            body_model_param_tensor = {key: torch.tensor(
                np.array(smpl_param[key]).reshape(self.smpl_shape[key]),
                        device=self.device, dtype=torch.float32)
                        for key in self.smpl_shape.keys()
                        if len(smpl_param[key]) > 0}
            output = smpl_model(**body_model_param_tensor, return_verts=True)
            smpl_joints = output['joints']
            kps3d_w = smpl_joints.detach().cpu().numpy()

            # get pelvis in world frame
            pelvis_world = kps3d_w[:, get_keypoint_idx('pelvis', 'smpl')]

            global_orient_c, transl_c = batch_transform_to_camera_frame(
                global_orient=smpl_param['global_orient'],
                transl=smpl_param['transl'],
                pelvis=pelvis_world,
                extrinsic=extrinsics,)
            
            # update smpl param
            smpl_param['global_orient'] = global_orient_c
            smpl_param['transl'] = transl_c

            # update smpl
            body_model_param_tensor = {key: torch.tensor(
                np.array(smpl_param[key]).reshape(self.smpl_shape[key]),
                        device=self.device, dtype=torch.float32)
                        for key in self.smpl_shape.keys()
                        if len(smpl_param[key]) > 0}
            output = smpl_model(**body_model_param_tensor, return_verts=True)
            smpl_joints = output['joints']
            kps3d = smpl_joints.detach().cpu().numpy()
            kps2d = camera.transform_points_screen(smpl_joints)[..., :2].detach().cpu().numpy()

            # prepare bbox
            bboxs = []
            for kp2d in kps2d:
                bbox_xyxy = self._keypoints_to_scaled_bbox(kp2d, scale=1.2)
                bbox_xywh = self._xyxy_to_xywh(bbox_xyxy)
                bbox_xywh.append(1)
                bboxs.append(bbox_xywh)

            # image_path
            img_ps = [p.replace(f'{dataset_path}{os.path.sep}', '') for p in image_paths]
            image_path_ += img_ps

            # bbox 
            bboxs_['bbox_xywh'] += bboxs

            # smpl parameters
            for key in self.smpl_shape.keys():
                smpl_[key].append(smpl_param[key])

            # camera parameters
            meta_['focal_length'].append(focal_length)
            meta_['principal_point'].append(principal_point)
            meta_['width'].append(width)
            meta_['height'].append(height)

            # need fix projection
            # test projection using R, T
            
            j3d_c = np.dot(extrinsics[:3, :3], j3d[0].transpose(1,0)).transpose(1,0) + T.reshape(1,3)
            j2d = camera.transform_points_screen(torch.tensor(j3d_c, device=self.device, dtype=torch.float32))
            j2d = j2d.detach().cpu().numpy()[..., :2]

            pdb.set_trace()

            # keypoints
            keypoints2d_smpl_.append(kps2d)
            keypoints3d_smpl_.append(kps3d)
            keypoints2d_original_.append(j2d)

        size_i = min(size, len(seqs))

        # image_path
        human_data['image_path'] = image_path_

        # meta
        human_data['meta'] = meta_

        # bbox
        for bbox_name in bboxs_.keys():
            bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 5)
            human_data[bbox_name] = bbox_      

        # smpl
        for smpl_name in smpl_.keys():
            smpl_[key] = np.concatenate(
                smpl_[key], axis=0).reshape(self.smpl_shape[key])
            
        # keypoints2d_smpl
        keypoints2d_smpl = np.concatenate(
            keypoints2d_smpl_, axis=0).reshape(-1, 45, 2)
        keypoints2d_smpl_conf = np.ones([keypoints2d_smpl.shape[0], 45, 1])
        keypoints2d_smpl = np.concatenate(
            [keypoints2d_smpl, keypoints2d_smpl_conf], axis=-1)
        keypoints2d_smpl, keypoints2d_smpl_mask = \
                convert_kps(keypoints2d_smpl, src='smpl', dst='human_data')
        human_data['keypoints2d_smpl'] = keypoints2d_smpl
        human_data['keypoints2d_smpl_mask'] = keypoints2d_smpl_mask
        
        # keypoints3d_smpl
        keypoints3d_smpl = np.concatenate(
            keypoints3d_smpl_, axis=0).reshape(-1, 45, 3)
        keypoints3d_smpl_conf = np.ones([keypoints3d_smpl.shape[0], 45, 1])
        keypoints3d_smpl = np.concatenate(
            [keypoints3d_smpl, keypoints3d_smpl_conf], axis=-1)
        keypoints3d_smpl, keypoints3d_smpl_mask = \
                convert_kps(keypoints3d_smpl, src='smpl', dst='human_data')
        human_data['keypoints3d_smpl'] = keypoints3d_smpl
        human_data['keypoints3d_smpl_mask'] = keypoints3d_smpl_mask

        # keypoints2d_original
        keypoints2d_original = np.concatenate(
            keypoints2d_original_, axis=0).reshape(-1, 25, 2)
        keypoints2d_original_conf = np.ones([keypoints2d_original.shape[0], 25, 1])
        keypoints2d_original = np.concatenate(
            [keypoints2d_original, keypoints2d_original_conf], axis=-1)
        keypoints2d_original, keypoints2d_original_mask = \
                convert_kps(keypoints2d_original, src='openpose_25', dst='human_data')
        human_data['keypoints2d_original'] = keypoints2d_original
        human_data['keypoints2d_original_mask'] = keypoints2d_original_mask

        # misc
        human_data['misc'] = self.misc_config
        human_data['config'] = f'flag3d_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, 
            f'flag3d_{mode}_{seed}_{"{:04d}".format(size_i)}.npz')
        # human_data.dump(out_file)
        

