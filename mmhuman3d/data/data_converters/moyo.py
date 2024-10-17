import ast
import glob
import json
import os
import pdb
import pickle
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

import ezc3d
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
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame
from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class MoyoConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train', 'val']

    def __init__(self, modes: List = []) -> None:

        self.device = torch.device('cuda:0')
        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            bbox_source='keypoints2d_original',
            flat_hand_mean=True,
            cam_param_type='prespective',
            cam_param_source='original',
            smplx_source='original',
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
        self.kps_body_part = {
            'body': (0, 73),
            'head': (0, 6),
            'left_hand': (16, 29),
            'right_hand': (36, 49),
        }

        super(MoyoConverter, self).__init__(modes)
        
    
    def _keypoints_to_scaled_bbox_bfh(self,
                                    keypoints,
                                    occ=None,
                                    body_scale=1.0,
                                    fh_scale=1.0,
                                    convention='smplx'):
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
            else:
                scale = fh_scale
            bp = get_keypoint_idxs_by_part[body_part]
            kp_id = list(range(bp[0], bp[1]))
            kps = keypoints[kp_id]

            if occ is not None:
                occ_p = occ[kp_id]
                if np.sum(occ_p) / len(kp_id) >= 0.1:
                    conf = 0
                else:
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


    def _load_camera_param(self, cam_params, downsample_factor=1.0):
        """
        Project 3D points to 2D
        Args:
            j3d: (N, 3) 3D joints
            cam_params: dict
            downsample_factor: resize factor

        Returns:
            j2d : 2D joint locations
        """
        mm2m = 1000
        # cam intrinsics
        f = cam_params['focal'] * downsample_factor
        cx = cam_params['princpt'][0] * downsample_factor
        cy = cam_params['princpt'][1] * downsample_factor

        # cam extrinsics
        R = torch.tensor(cam_params['rotation'])
        t = -torch.mm(R, torch.tensor(cam_params['position'])[:, None]).squeeze()  # t= -RC

        # cam matrix
        K = np.array([[f, 0, cx],
                        [0, f, cy],
                        [0, 0, 1]])

        Rt = np.zeros((3, 4))
        Rt[:, :3] = R
        Rt[:, 3] = t
        return f, cx, cy, K, Rt
    

    def _project2d(self, j3d, Rt, K):

        # apply extrinsics
        bs = j3d.shape[0]
        j3d = torch.tensor(j3d, dtype=torch.float32, device=self.device)
        Rt = torch.tensor(Rt, dtype=torch.float32, device=self.device)
        K = torch.tensor(K, dtype=torch.float32, device=self.device)
        j3d_cam = torch.bmm(Rt[None, :, :].expand(bs, -1, -1), j3d[:, :, None])
        j2d = torch.bmm(K[None, :].expand(bs, -1, -1), j3d_cam)
        j2d = j2d / j2d[:, [-1]]

        j2d = j2d[:, :2, :].squeeze().detach().cpu().numpy()
        j3d_cam = j3d_cam.squeeze().detach().cpu().numpy()
        return j2d, j3d_cam
    

    def _fullpose_to_params(self, fullpose):
        
        fullpose = fullpose.reshape(-1, 55, 3)
        params = {}
        params['global_orient'] = fullpose[:, 0].reshape(-1, 3)
        params['body_pose'] = fullpose[:, 1:22].reshape(-1, 63)
        params['jaw_pose'] = fullpose[:, 22].reshape(-1, 3)
        params['leye_pose'] = fullpose[:, 23].reshape(-1, 3)
        params['reye_pose'] = fullpose[:, 24].reshape(-1, 3)
        params['left_hand_pose'] = fullpose[:, 25:40].reshape(-1, 45)
        params['right_hand_pose'] = fullpose[:, 40:55].reshape(-1, 45)

        return params


    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        print('Converting MOYO dataset...')

        # use HumanData to store all data
        human_data = HumanData()

        # load splits
        split_path = os.path.join(dataset_path, 'split.json')

        # find seqs split
        # sd = {}
        # for mode in ['train', 'val']:
        #     zip_path = f'/mnt/d/datasets/moyo-zip/*/images/{mode}/*.zip'
        #     zps = glob.glob(zip_path)

        #     zns = [os.path.basename(zp).replace('.zip', '') for zp in zps]
        #     sd[mode] = zns
        # with open(split_path, 'w') as f:
        #     json.dump(sd, f)

        # init smplx model
        gender = 'Neutral'
        smplx_model = build_body_model(
            dict(
                type='SMPLX',
                keypoint_src='smplx',
                keypoint_dst='smplx',
                model_path='data/body_models/smplx',
                gender=gender,
                num_betas=10,
                use_face_contour=True,
                flat_hand_mean=self.misc_config['flat_hand_mean'],
                use_pca=False,
                batch_size=1)).to(self.device)
        
        with open(split_path, 'r') as f:
            split = json.load(f)
        seqs = split[mode]
        seq_base = os.path.join(dataset_path, 'images')

        # init seed and size
        seed, size = '240108', '999'
        size_i = min(int(size), len(seqs))
        random.seed(int(seed))
        np.set_printoptions(suppress=True)

        if mode == 'train':
            split = 2
        else:
            split = 1

        for spid in [1]:# range(split):
            
            slice_num = len(seqs) // split
            seqs_spilt = seqs[spid * slice_num: (spid + 1) * slice_num]
        
            # initialize output for human_data
            smplx_ = {}
            for key in self.smplx_shape.keys():
                smplx_[key] = []
            keypoints2d_, keypoints3d_, kps2d_orig_ = [], [], []
            bboxs_ = {}
            for bbox_name in [
                    'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                    'rhand_bbox_xywh'
            ]:
                bboxs_[bbox_name] = []
            meta_ = {}
            for meta_name in ['principal_point', 'focal_length', 'height', 'width', 'R']:
                meta_[meta_name] = []
            image_path_ = []

            frame_offset = 1

            # seqs_spilt = seqs_spilt[-10:]

            for sid, seq in enumerate(tqdm(seqs_spilt, desc=f'MOYO {mode}, Slice {spid+1}/{split}', position=0, leave=False)):
            # for sid, seq in enumerate(seqs):

                date = seq[:6]
                sub_folders = glob.glob(os.path.join(seq_base, seq, '*Cam*'))

                # try:
                # load all camera params
                cam_param_bp = os.path.join(dataset_path, 'cameras', f'20{date}')
                cam_param_p = glob.glob(os.path.join(cam_param_bp, '*', 'cameras_param.json'))[0]
                with open(cam_param_p, 'r') as f:
                    cam_dict = json.load(f)

                # load c3d
                c3d_p1 = os.path.join(dataset_path, 'vicon', mode, 'c3d', f'{seq}*.c3d')
                c3d_p2 = os.path.join(dataset_path, 'com', mode, f'{seq}*.c3d')
                if len(glob.glob(c3d_p1)) > 0:
                    c3d = ezc3d.c3d(glob.glob(c3d_p1)[0])
                elif len(glob.glob(c3d_p2)) > 0:
                    c3d = ezc3d.c3d(glob.glob(c3d_p2)[0])
                else:
                    # raise ValueError(f'No c3d found for {seq}')
                    print(f'No c3d found for {seq}')
                    # pdb.set_trace()
                    continue
                markers3d = (c3d['data']['points'] / 1000).transpose(2, 1, 0) 
                anno_len = markers3d.shape[0]
                
                # load seq mosh data
                mosh_p = os.path.join(dataset_path, 'mosh', mode, f'{seq}_stageii.pkl')
                if not os.path.exists(mosh_p):
                    print(f'No mosh data found for {seq}')
                    continue
                try:
                    with open(mosh_p, 'rb') as f:
                        anno = pickle.load(f)
                except Exception as e:
                        print('Pickled file corrupted: ', mosh_p)
                        continue

                # reformat anno
                params = self._fullpose_to_params(anno['fullpose'])
                params['transl'] = anno['trans']
                params['betas'] = anno['betas'][:10]
                params['betas'] = params['betas'].repeat(anno_len, 0).reshape(-1, 10)
                params['expression'] = np.zeros((anno_len, 10))

                # prepare smplx
                smplx_param = params

                # get pelvis world
                intersect_keys = list(
                    set(smplx_param.keys()) & set(self.smplx_shape.keys()))
                body_model_param_tensor = {
                    key: torch.tensor(
                        np.array(smplx_param[key]).reshape(self.smplx_shape[key]),
                        device=self.device,
                        dtype=torch.float32)
                    for key in intersect_keys
                }
                output = smplx_model(**body_model_param_tensor, return_joints=True)

                kps3d = output['joints'].detach().cpu().numpy()

                pelvis_world = kps3d[:, get_keypoint_idx('pelvis', 'smplx'), :]

                kps3d_conf = np.ones([kps3d.shape[0], 144, 1])
                kps3dw_conf = np.concatenate([kps3d, kps3d_conf], axis=-1)
                
                for sub_folder in sub_folders:
                    
                    # load camera params
                    cid = os.path.basename(sub_folder)[-1]
                    cam_param = cam_dict[f'cam_{cid}']

                    # reformat camera params
                    downsample_factor = 0.5
                    f, cx, cy, K, Rt_marker = self._load_camera_param(cam_param,
                                                            downsample_factor)

                    Rt = Rt_marker.copy()
                    Rt[:, 3] = Rt[:, 3] / 1000  # convert to meter
                    
                    # R1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                    # R1r = np.linalg.inv(R1)

                    # Rt_tmp = np.dot(R1r, Rt[:3, :3])
                    # Rt[:3, :3] = np.dot(R1, np.linalg.inv(Rt_tmp))
                    # cam 1 [1, 0, 0]
                    # rotmat = aa_to_rotmat(np.array([1, 0, 1]) * 0.5 * np.pi)
                    # Rt[:3, :3] = rotmat

                    # if cid != '1':
                    #     continue

                    # parse image
                    img_ps = glob.glob(os.path.join(sub_folder, '*.jpg'))  
                    height, width = cv2.imread(img_ps[0]).shape[:2]

                    # prepare extrinsics        
                    extrinsics = np.eye(4)
                    extrinsics[:3, :] = Rt
                    # extrinsics[:3, :3] = np.eye(3)

                    # build camera
                    camera = build_cameras(
                        dict(
                            type='PerspectiveCameras',
                            convention='opencv',
                            in_ndc=False,
                            focal_length=f,
                            image_size=(width, height),
                            principal_point=(cx, cy))).to(self.device)
                    
                    smplx_param_copy = smplx_param.copy()

                    # transform smplx to camera space
                    global_orient, transl = batch_transform_to_camera_frame(
                        global_orient=smplx_param_copy['global_orient'],
                        transl=smplx_param_copy['transl'],
                        pelvis=pelvis_world,
                        extrinsic=extrinsics)

                    # update smplx param
                    smplx_param_copy['global_orient'] = global_orient
                    smplx_param_copy['transl'] = transl

                    # update smplx
                    for update_key in ['global_orient', 'transl']:
                        body_model_param_tensor[update_key] = torch.tensor(
                            np.array(smplx_param_copy[update_key]).reshape(
                                self.smplx_shape[update_key]),
                            device=self.device,
                            dtype=torch.float32)
                    output = smplx_model(**body_model_param_tensor, return_joints=True)
                    kps3d_c = output['joints']

                    # get kps2d
                    keypoints_2d_xyd = camera.transform_points_screen(kps3d_c)
                    kps2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
                    kps3d_c = kps3d_c.detach().cpu().numpy()

                    # pdb.set_trace()

                    for imgp in tqdm(img_ps, desc=f'Seq ID: {sid+1}/{len(seqs)}, Sub ID: {cid}/{len(sub_folders)}',
                                    position=1, leave=False):
                    # for imgp in img_ps:

                        # prepare image path
                        image_path = imgp.replace(f'{dataset_path}{os.path.sep}', '')
                        fid = os.path.splitext(imgp)[0].split('_')[-1]
                        aid = (int(fid) - frame_offset) * 2 - 1

                        if aid < 0 or aid > anno_len - 1:
                            continue

                        # load keypoints3d
                        j3d = markers3d[aid]
                        j2d, j3d_cam = self._project2d(j3d, Rt_marker, K)
                    
                        image_path_.append(image_path)

                        # append kps
                        # keypoints2d_ += kps2d.tolist()
                        # keypoints3d_ += kps3d_c.tolist()

                        # append kps
                        kp2d_f = kps2d[aid:aid+1]
                        kp3d_f = kps3d_c[aid:aid+1]

                        if kp3d_f.shape[0] == 0:
                            continue
                        # kp3d = kps3d[aid]
                        keypoints3d_.append(kp3d_f)
                        keypoints2d_.append(kp2d_f)
                        kps2d_orig_.append(j2d)

                        # append meta
                        meta_['height'] += [height]
                        meta_['width'] += [width]
                        meta_['principal_point'] += [(cx, cy)]
                        meta_['focal_length'] += [(f, f)]
                        meta_['R'] += [Rt]

                        bbo_ = []
                        # get bbox from 2d keypoints
                        bboxs = self._keypoints_to_scaled_bbox_bfh(
                            kp2d_f, # kps2d[aid],
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
                            bbo_.append(bbox_xywh)

                        # save smplx params
                        for key in smplx_.keys():
                            smplx_[key].append(smplx_param_copy[key][aid:aid+1])
                        # pdb.set_trace()
                        

                        # kps3d_s = kps3d[aid]
                        # kps3d_s = np.concatenate([kps3d_s, np.ones([kps3d_s.shape[0], 1])], axis=-1)
                        # K0 = K.copy()
                        # # K0[0, 0] = f / 1000
                        # # K0[1, 1] = f / 1000
                        # kps2d, kps3d_c = self._project2d(kps3d_s, Rt, K0)

                        # pdb.set_trace()
                            
                        # if imgp == img_ps[0]:
                        #     j2d0 = j2d

                    # if cid == '3':
                    # test 2d overlay: success
                    # kpsm, j3d_cam = self._project2d(kps3dw_conf[0], Rt, K)  
                    # img = cv2.imread(img_ps[0])
                    # kpsm = kps2d[0]
                    # kpsm = j2d0

                    # kpsm += [-1000, 0]
                    # kpsm /= 3

                    # kps_m = kps2d[0] - np.array([kps2d[0, 0].min(), kps2d[0, 1].min()])
                    # kps_m = kps_m / np.array([100, 10])
                    # for jid, j in enumerate(kpsm):
                    #     cv2.circle(img, (int(j[0]), int(j[1])), 5, (0, 0, 255), -1)
                    #     # cv2.putText(img, str(jid), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # # for bbox in bbo_:
                    # #     cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2) 

                    # os.makedirs(f'{out_path}', exist_ok=True)
                    # cv2.imwrite(f'{out_path}/{os.path.basename(seq)}_{cid}.jpg', img)

            # convention
            # # save keypoints 2d original
            # keypoints2d_original = np.array(kps2d_orig_)
            # keypoints2d_original, keypoints2d_original_mask = convert_kps(
            #     keypoints2d_original, src='openpose_137', dst='human_data')
            # human_data['keypoints2d_original'] = keypoints2d_original
            # human_data['keypoints2d_original_mask'] = keypoints2d_original_mask

            # save keypoints 2d original
            # keypoints2d_original = np.array(kps2d_orig_)
            # keypoints2d_original, keypoints2d_original_mask = convert_kps(
            #     keypoints2d_original, src='openpose_137', dst='human_data')
            # human_data['keypoints2d_original'] = keypoints2d_original
            # human_data['keypoints2d_original_mask'] = keypoints2d_original_mask

            # save keypoints 2d smplx
            keypoints2d = np.concatenate(keypoints2d_, axis=0)
            keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
            keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
            keypoints2d, keypoints2d_mask = convert_kps(
                keypoints2d, src='smplx', dst='human_data')
            human_data['keypoints2d_smplx'] = keypoints2d
            human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

            # save keypoints 3d smplx
            keypoints3d = np.concatenate(keypoints3d_, axis=0)
            keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
            keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
            keypoints3d, keypoints3d_mask = convert_kps(
                keypoints3d, src='smplx', dst='human_data')
            human_data['keypoints3d_smplx'] = keypoints3d
            human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

            # pdb.set_trace()
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

            # save meta and misc
            human_data['config'] = 'moyo'
            human_data['misc'] = self.misc_config
            human_data['meta'] = meta_

            os.makedirs(out_path, exist_ok=True)
            out_file = os.path.join(
                # out_path, f'moyo_{self.misc_config["flat_hand_mean"]}.npz')
                out_path, f'moyo_{mode}_{seed}_{"{:03d}".format(size_i)}_{spid}.npz')
            human_data.dump(out_file)


