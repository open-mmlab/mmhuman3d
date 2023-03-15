import os
from typing import List

import pdb
import time
import numpy as np
import json
import cv2
import glob
import random
from tqdm import tqdm
import pdb

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
# import mmcv
# from mmhuman3d.models.body_models.builder import build_body_model
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idxs_by_part


@DATA_CONVERTERS.register_module()
class RenbodyConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train']

    def __init__(self, modes: List = []) -> None:
        super(RenbodyConverter, self).__init__(modes)


    def _get_imgname(self, v):
        root_folder_id = v.split('/').index('renbody')
        imglist = '/'.join(v.split('/')[root_folder_id+1:])
        images = [os.path.join(imglist, img) for img in os.listdir(v) if img.endswith('.jpg')]
        return images


    # def _keypoints_to_scaled_bbox_fh(self, keypoints, occ, self_occ, scale=1.0, convention='renbody'):
    def _keypoints_to_scaled_bbox_fh(self, keypoints, scale=1.0, convention='human_data'):
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
            # occ_p = occ[kp_id]
            # self_occ_p = self_occ[kp_id]

            # if np.sum(self_occ_p) / len(kp_id) >= 0.5 or np.sum(occ_p) / len(kp_id) >= 0.5:
            #     conf = 0
            #     # print(f'{body_part} occluded, occlusion: {np.sum(self_occ_p + occ_p) / len(kp_id)}, skip')
            # else:
            #     # print(f'{body_part} good, {np.sum(self_occ_p + occ_p) / len(kp_id)}')
            #     conf = 1
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
        return bboxs[0], bboxs[1], bboxs[2]

    
    def convert_by_mode(self, dataset_path: str, out_path: str,
                    mode: str) -> dict:

        # use HumanData to store all data 
        human_data = HumanData()

        # get trageted sequence list
        seed, size = '230228', '01000'
        random.seed(int(seed))
        camera_num_per_seq = 3
        invalid_camera_list = ['09']
        root_dir, prefix = os.path.split(dataset_path)
        npzs = glob.glob(os.path.join(dataset_path, 'Renbody_smplx', '*', '*', '*', 'smplx_*', 'human_data_smplx.npz'))
        # random.shuffle(npzs)
        # npzs = npzs[:4000]
        # print(npzs[:10])

        # initialize storage
        _bboxs = {}
        _meta = {}
        _meta['gender'] = []
        for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
            _bboxs[bbox_name] = []
        _image_path = []
        _keypoints2d_list, _keypoints3d_list = [], []
        _keypoints3d_mask_list = []

        # get data shape
        # npfile = dict(np.load(npzs[0], allow_pickle=True))
        # kp_shape = npfile['keypoints2d'].shape[1]
        # _keypoints2d = np.array([]).reshape(0, kp_shape, 3)
        # _keypoints3d = np.array([]).reshape(0, kp_shape, 4)
        # _keypoints2d_list, _keypoints3d_list = [], []

        # initialize smpl and smplx
        _smplx = {}
        smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 21, 3), \
                       'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 'leye_pose': (-1, 3),
                       'reye_pose': (-1, 3), \
                       'jaw_pose': (-1, 3), 'expression': (-1, 10)}
        for key in smplx_shape:
            _smplx[key] = np.array([]).reshape(smplx_shape[key])
        _smplx_l = {}
        for key in smplx_shape:
            _smplx_l[key] = []

        root_folder_id = dataset_path.split('/').index('renbody')

        for idx, npzf in enumerate(tqdm(npzs, desc='Npzfiles concating')):
        # for idx, npzf in enumerate(npzs):
            npfile = dict(np.load(npzf, allow_pickle=True))
            frame_num = len(npfile['transl'])
            # ['renbody', 'Renbody_smplx', '20220417', 'jiangfeiya_f', 
            #       'jiangfeiya_yf1_dz1', 'smplx_xrmocap', 'human_data_smplx.npz']
            batch_name, name_gender, cloth_motion, _, _= npzf.split('/')[root_folder_id+2:]

            # get camera list
            seq_2d_dir = os.path.join(dataset_path, 'Renbody_image', batch_name, name_gender, cloth_motion)
            camera_dir = os.path.join(seq_2d_dir, 'image')
            camera_list = [cid for cid in os.listdir(camera_dir) if cid.isnumeric()]
            for c in invalid_camera_list:
                camera_list.remove(c)
            random.seed(int(seed) + idx)
            random.shuffle(camera_list)
            camera_list = camera_list[:camera_num_per_seq]

            for c in camera_list:
                image_dir = os.path.join(camera_dir, c)

                # get image path
                _image_path.append(self._get_imgname(image_dir)[:frame_num])

                # get kps2d, kps3d
                params_2d_path = os.path.join(seq_2d_dir, 'pose_2d', f'human_data_{c}.npz')
                # params_2d = dict(np.load(params_2d_path, allow_pickle=True))
                # kp2d = params_2d['keypoints2d'][:frame_num, :, :]
                human_data2d = HumanData()
                human_data2d.load(params_2d_path)
                human_data2d.decompress_keypoints()
                _mask = human_data2d['keypoints2d_mask']
                kp2d = human_data2d['keypoints2d'][:frame_num, :, :]
                _keypoints2d_list.append(kp2d)

                params_3d_path = os.path.join(os.path.dirname(npzf), 'human_data_optimized_keypoints3d.npz')
                params_3d = dict(np.load(params_3d_path, allow_pickle=True))
                kp3d = params_3d['keypoints3d'][:frame_num, :, :]
                _keypoints3d_list.append(kp3d)
                _keypoints3d_mask_list.append(params_3d['keypoints3d_mask'][:frame_num, :])
                # pdb.set_trace()
                
                for frame_idx in range(frame_num):
                    kp = kp2d[frame_idx][:, :2]
                    bbox_tmp_ = {}
                    bbox_tmp_['bbox_xywh'] = human_data2d['bbox_xywh'][frame_idx][:4]
                    bbox_tmp_['face_bbox_xywh'], bbox_tmp_['lhand_bbox_xywh'], bbox_tmp_[
                        'rhand_bbox_xywh'] = self._keypoints_to_scaled_bbox_fh(kp, 1.25)
                    for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:

                        if bbox_name != 'bbox_xywh':
                            bbox = bbox_tmp_[bbox_name]
                            xmin, ymin, xmax, ymax, conf = bbox
                            bbox = np.array([max(0, xmin), max(0, ymin), min(1920, xmax), min(1080, ymax)])
                        
                            bbox_xywh = self._xyxy2xywh(bbox)
                        else:
                            bbox_xywh = bbox_tmp_[bbox_name].tolist()
                            xmin, ymin, w, h = bbox_xywh
                            xmax = w + xmin
                            ymax = h + ymin
                            bbox = np.array([max(0, xmin), max(0, ymin), min(1920, xmax), min(1080, ymax)])
                            conf = human_data2d['bbox_xywh'][frame_idx][-1]

                        if bool(set(bbox).intersection([0, 1280, 1920])):
                            bbox_xywh.append(0)
                        else: 
                            bbox_xywh.append(conf)
                        _bboxs[bbox_name].append(bbox_xywh)

                for k in smplx_shape.keys():
                    # _smplx[k] = np.concatenate((_smplx[k], npfile['smplx'].item()[k][valid_id].reshape(smplx_shape[k])), axis=0)
                    _smplx_l[k].append(npfile[k].reshape(smplx_shape[k]))
                

                s = npfile['gender']
                if s == 'male':
                    _meta['gender'].extend(['m'] * frame_num)
                elif s == 'female':
                    _meta['gender'].extend(['f'] * frame_num)
                else:
                    _meta['gender'].extend(['n'] * frame_num)

        print('Concating finished, converting to HumanData')

        human_data['image_path'] = _image_path
        print('Image path writting finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        for key in _bboxs.keys():
            bbox_ = np.array(_bboxs[key]).reshape((-1, 5))
            # bbox_ = np.hstack([bbox_, np.zeros([bbox_.shape[0], 1])])
            # import pdb; pdb.set_trace()
            human_data[key] = bbox_
        print('BBox generation finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # for kp_set in _keypoints2d_list:
        #     _keypoints2d = np.concatenate((_keypoints2d, kp_set), axis=0)
        _keypoints2d = np.concatenate(_keypoints2d_list, axis=0)
        _keypoints3d = np.concatenate(_keypoints3d_list, axis=0)
        _keypoints3d_mask = np.concatenate(_keypoints3d_mask_list, axis=0)
        _keypoints3d = np.concatenate((_keypoints3d, _keypoints3d_mask.reshape(-1, 190, 1)), axis=-1)

        # _keypoints2d, mask = convert_kps(_keypoints2d, 'smplx', 'human_data')
        # _keypoints3d, mask = convert_kps(_keypoints3d, 'smplx', 'human_data')

        human_data['keypoints2d'] = _keypoints2d
        human_data['keypoints3d'] = _keypoints3d
        human_data['keypoints2d_mask'] = _mask
        human_data['keypoints3d_mask'] = _mask
        print('Keypoint conversion finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        for key in _smplx.keys():
            # for model_param in _smplx_l[key]:
            _smplx[key] = np.concatenate(_smplx_l[key], axis=0)

        human_data['smplx'] = _smplx
        print('Smpl and/or Smplx finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        human_data['config'] = 'renbody_train'
        human_data['meta'] = _meta
        print('MetaData finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        human_data.compress_keypoints_by_mask()
        # store the data struct
        os.makedirs(out_path, exist_ok=True)

        out_file = os.path.join(out_path, f'renbody_train_{str(seed)}_{str(size)}.npz')
        human_data.dump(out_file)


