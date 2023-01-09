import os
import pickle
import xml.etree.ElementTree as ET
from typing import List

import numpy as np
from tqdm import tqdm
import json
import pickle

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class MtpConverter(BaseModeConverter):
    """MTP dataset
    `On Self-Contact and Human Pose' CVPR`2021
    More details can be found in the `paper
    <https://arxiv.org/pdf/2104.03176.pdf>`__.

    Args:
        modes (list): 'valid' or 'train' for accepted modes
    """
    ACCEPTED_MODES = ['train', 'val']

    def __init__(self,
                 modes: List = []) -> None:
        super(MtpConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask, keypoints3d, keypoints3d_mask, cam_param
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_ = [], [], []

        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []

        json_path = os.path.join(dataset_path, 'train_val_split.json')

        json_data = json.load(open(json_path, 'r'))[f'{mode}']

        for img_id in tqdm(json_data):
            part_name = img_id.split('_')[1]
            image_path = f'images/{part_name}/{img_id}.png'
            keypoints_file = os.path.join(dataset_path, f'keypoints/openpose/{part_name}/{img_id}.json')
            smpl_file = os.path.join(dataset_path, f'smplify-xmc/smpl/params/{part_name}/{img_id}.pkl')

            with open(keypoints_file) as f:
                data = json.load(f)
                if len(data['people']) <1 :
                    continue
                keypoints2d = np.array(data['people'][0]['pose_keypoints_2d']).reshape(25, 3)
                keypoints2d[keypoints2d[:, 2] > 0.15, 2] = 1 # set based on keypoints confidence 
                
                vis_keypoints2d = keypoints2d[np.where(keypoints2d[:, 2]>0)[0]] 
                # bbox
                bbox_xyxy = [
                    min(vis_keypoints2d[:, 0]),
                    min(vis_keypoints2d[:, 1]),
                    max(vis_keypoints2d[:, 0]),
                    max(vis_keypoints2d[:, 1])
                ]
                bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
                bbox_xywh = self._xyxy2xywh(bbox_xyxy)

            with open(smpl_file, 'rb') as f:
                ann = pickle.load(f, encoding='latin1')
                pose = ann['pose']
                smpl['body_pose'].append(pose[3:].reshape(23, 3))
                smpl['global_orient'].append(pose[:3].reshape(3))
                smpl['betas'].append(ann['betas'])

            image_path_.append(image_path)
            keypoints2d_.append(keypoints2d)
            bbox_xywh_.append(bbox_xywh)

        smpl['body_pose'] = np.array(smpl['body_pose']).reshape(
            (-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))
        
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 25, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'prox', 'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['smpl'] = smpl
        human_data['config'] = 'mtp'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'mtp_{}.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)

