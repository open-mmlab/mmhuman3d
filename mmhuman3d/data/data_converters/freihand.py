import os
import os.path as osp
import random
import torch
from typing import List
import cv2
import numpy as np
from tqdm import tqdm
from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idxs_by_part
)
from mmhuman3d.data.datasets.pipelines.expose_transforms import keyps_to_bbox
from mmhuman3d.models.builder import build_body_model
import json
import pickle as pk

def project_points(K, xyz):
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

@DATA_CONVERTERS.register_module()
class FreihandConverter(BaseModeConverter):
    """Freihand dataset
    'A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images'
    More details can be found on the website:
    https://lmb.informatik.uni-freiburg.de/projects/freihand/
    Args:
        modes (list): 'train / val' for accepted modes
    """
    NUM_BETAS = 10
    NUM_EXPRESSION = 10


    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str, mean_pose_path: str = None) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, smplx, meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()
        # structs we use
        image_path_, bbox_xywh_  = [], []
        smplx = {}
        smplx['global_orient'] = []
        smplx['betas'] = []
        smplx['right_hand_pose'] = []
        keypoints2d = []
        key = 'training'


        intrinsics_path = osp.join(dataset_path, f'{key}_K.json')
        param_path = osp.join(dataset_path, f'{key}_mano.json')
        xyz_path = osp.join(dataset_path, f'{key}_xyz.json')
        with open(param_path,'r') as f:
            param = json.load(f)
        with open(intrinsics_path,'r') as f:
            intrinsics = json.load(f)
        with open(xyz_path, 'r') as f:
            xyz = json.load(f)


        num_green_bg = len(param)
        xyz = np.asarray(xyz, dtype=np.float32)
        param = np.asarray(param, dtype=np.float32).reshape(len(intrinsics), -1)
        intrinsics = np.asarray(intrinsics)
        pose = param[:,:48].reshape(num_green_bg,-1,3)
        betas = param[:,48:58]
        uv_root = param[:,58:60]
        scale = param[:,60:]
        global_pose = pose[:,0:1]
        right_hand_pose = pose[:,1:]

        # # load mean hand pose
        right_hand_mean = None
        if mean_pose_path is not None and osp.exists(mean_pose_path):
            with open(mean_pose_path, 'rb') as f:
                mean_poses_dict = pk.load(f)
                right_hand_mean = mean_poses_dict['right_hand_pose']['aa'].squeeze()
        if right_hand_mean is not None:
            right_hand_pose += right_hand_mean[np.newaxis]
        
        for index in range(4*num_green_bg):
            smplx['global_orient'].append(global_pose[index%num_green_bg].copy())
            smplx['betas'].append(betas[index%num_green_bg].copy())
            smplx['right_hand_pose'].append(right_hand_pose[index%num_green_bg].copy())
            img_path = osp.join(key, 'rgb',f'{index:08d}.jpg')
            image_path_.append(img_path)
            bbox_xywh_.append(np.array([0,0,224,224],dtype=np.float32))
            kps2d = project_points(intrinsics[index%num_green_bg],xyz[index%num_green_bg])
            kps2d = np.array(kps2d)
            kps2d = np.hstack([kps2d, np.ones([kps2d.shape[0], 1])])
            keypoints2d.append(kps2d)

        keypoints2d = np.array(keypoints2d)
        keypoints2d, keypoints2d_mask = convert_kps(keypoints2d, src='mano', dst='human_data')

        smplx['global_orient'] = np.array(smplx['global_orient'])
        smplx['betas'] = np.array(smplx['betas'])
        smplx['right_hand_pose'] = np.array(smplx['right_hand_pose'])
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask
        
        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smplx'] = smplx
        human_data['config'] = 'freihand'

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        if mode == 'train':
            human_data = human_data.get_slice(0,int(0.8*len(image_path_)))
        else:
            human_data = human_data.get_slice(int(0.8*len(image_path_)),len(image_path_)) 


        file_name = 'freihand_{}.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)


