import json
import os
import cv2
from typing import List

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class OH50k3DConverter(BaseModeConverter):
    """3DOH50K dataset
    `Object-Occluded Human Shape and Pose Estimation from a Single Color 
    Image' CVPR'2020
    More details can be found in the `paper
    <https://www.yangangwang.com/papers/ZHANG-OOH-2020-03.pdf>`__ .

    Args:
        modes (list): 'train' and/or 'test' for
        accepted modes
    """
    ACCEPTED_MODES = ['train', 'test']

    def __init__(self, modes: List = []) -> None:
        super(OH50k3DConverter, self).__init__(modes)

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
                keypoints2d_mask stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_, cam_param_ = [], [], [], [], []

        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []
        smpl['transl'] = []

        # json annotation file
        json_path = os.path.join(dataset_path, f'{mode}set', 'annots.json')

        json_data = json.load(open(json_path, 'r'))

        for fid in tqdm(json_data.keys()):
            annot = json_data[fid]
            # image name
            extrinsic = np.array(annot['extri'])
            K = np.array(annot['intri'])
            img_path = f'{mode}set/' + annot['img_path']
            betas = np.array(annot['betas']).reshape(-1)
            pose = np.array(annot['pose']).reshape(-1)
            trans = np.array(annot['trans']).reshape(-1)
            scale = np.array(annot['scale'])
            smpl_joints_2d = np.array(annot['smpl_joints_2d']) # 24x2
            smpl_joints_3d = np.array(annot['smpl_joints_3d']) # 24x3
            lsp_joints_2d = np.array(annot['lsp_joints_2d']) # 14x2
            lsp_joints_3d = np.array(annot['lsp_joints_3d']) # 14x3

            # fix keypoints3d
            smpl_joints_3d = smpl_joints_3d - smpl_joints_3d[0]

            img_name = annot['img_path'].replace('\\', '/')
            img_path = f'{mode}set/{img_name}'
            h, w, _ = cv2.imread(f'{dataset_path}/{img_path}').shape

            # scale and center
            bbox_xyxy = np.array(annot['bbox']).reshape(-1) # 2x2 - check foramt
            bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
            bbox_xywh = self._xyxy2xywh(bbox_xyxy)
            smpl_joints_2d = np.hstack([smpl_joints_2d, np.ones([24, 1])])
            smpl_joints_3d = np.hstack([smpl_joints_3d, np.ones([24, 1])])
            lsp_joints_2d = np.hstack([lsp_joints_2d, np.ones([14, 1])])
            lsp_joints_3d = np.hstack([lsp_joints_3d, np.ones([14, 1])])

            R = extrinsic[:3, :3]
            T = extrinsic[:3, 3]

            camera = CameraParameter(H=h, W=w)
            camera.set_KRT(K, R, T)
            parameter_dict = camera.to_dict()
            pose[:3] = cv2.Rodrigues(
                np.dot(R,
                        cv2.Rodrigues(pose[:3])[0]))[0].T[0]

            # store data
            image_path_.append(img_path)
            keypoints2d_.append(smpl_joints_2d)
            keypoints3d_.append(smpl_joints_3d)
            bbox_xywh_.append(bbox_xywh)
            smpl['body_pose'].append(pose[3:].reshape((23, 3)))
            smpl['global_orient'].append(pose[:3])
            smpl['betas'].append(betas)
            smpl['transl'].append(trans)
            cam_param_.append(parameter_dict)

        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))
        smpl['transl'] = np.array(smpl['transl']).reshape((-1, 3))


        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 24, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'smpl',
                                         'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, 24, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, 'smpl', 'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints3d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d'] = keypoints3d_
        human_data['smpl'] = smpl
        human_data['cam_param'] = cam_param_
        human_data['config'] = 'oh30k3d'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'oh50k3d_{}.npz'.format(mode))
        human_data.dump(out_file)
