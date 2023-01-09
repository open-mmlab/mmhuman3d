import json
import os
from typing import List

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class Muco3dhpConverter(BaseConverter):
    """MuCo-3DHP dataset `Single-Shot Multi-Person 3D Pose Estimation 
    From Monocular RGB' 3DV'2018
    More details can be found in the `paper.

    <https://arxiv.org/abs/1712.03453>`__ .
    """
    @staticmethod
    def get_intrinsic_matrix(f: List[float],
                             c: List[float],
                             inv: bool = False) -> np.ndarray:
        """Get intrisic matrix (or its inverse) given f and c."""
        intrinsic_matrix = np.zeros((3, 3)).astype(np.float32)
        intrinsic_matrix[0, 0] = f[0]
        intrinsic_matrix[0, 2] = c[0]
        intrinsic_matrix[1, 1] = f[1]
        intrinsic_matrix[1, 2] = c[1]
        intrinsic_matrix[2, 2] = 1

        if inv:
            intrinsic_matrix = np.linalg.inv(intrinsic_matrix).astype(
                np.float32)
        return intrinsic_matrix

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask, keypoints3d, keypoints3d_mask, video_path,
                smpl, meta, cam_param stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, \
            cam_param_ = [], [], [], [], []

        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []

        annot_path = dataset_path.replace('muco', 'extras/MuCo-3DHP.json')
        smpl_param_path = os.path.join(dataset_path, 'SMPLX/smpl_param.json')

        db = COCO(annot_path)
        with open(smpl_param_path) as f:
            smpl_params = json.load(f)

        for iid in tqdm(db.imgs.keys()):
            img = db.imgs[iid]
            w, h = img['width'], img['height']
            imgname = img['file_name']

            R = np.array(img['R']).reshape(3, 3)
            T = np.array(img['T']).reshape(3, )
            K = self.get_intrinsic_matrix(img['f'], img['c'])

            ann_ids = db.getAnnIds(img['id'])
            anns = db.loadAnns(ann_ids)

            camera = CameraParameter(H=h, W=w)
            camera.set_KRT(K, R, T)
            parameter_dict = camera.to_dict()

            for i, pid in enumerate(ann_ids):
                try:
                    smpl_param = smpl_params[str(pid)]
                    pose, shape, trans = np.array(
                        smpl_param['pose']), np.array(
                            smpl_param['shape']), np.array(
                                smpl_param['trans'])
                    sum = pose.sum() + shape.sum() + trans.sum()
                    if np.isnan(sum):
                        continue
                except KeyError:
                    continue

                joint_img = np.array(anns[i]['keypoints_img'])

                bbox = np.array(anns[i]['bbox'])
                keypoints_vis = np.array(
                    anns[i]['keypoints_vis']).astype('int').reshape(-1, 1)
                if not int(keypoints_vis[14]) == 1:
                    continue
                joint_img = np.hstack([joint_img, keypoints_vis])

                keypoints2d_.append(joint_img)
                bbox_xywh_.append(bbox)
                smpl['body_pose'].append(pose[3:].reshape((23, 3)))
                smpl['global_orient'].append(pose[:3])
                smpl['betas'].append(shape)
                cam_param_.append(parameter_dict)
                image_path_.append(f'images/{imgname}')

        # change list to np array
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 21, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'muco', 'human_data')

        human_data['image_path'] = image_path_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smpl'] = smpl
        human_data['cam_param'] = cam_param_
        human_data['config'] = 'muco3dhp'
        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'muco3dhp_train.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)