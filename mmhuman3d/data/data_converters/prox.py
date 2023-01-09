import json
import os
import pickle
from typing import List

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class ProxConverter(BaseConverter):
    """PROX dataset `Resolving 3D Human Pose Ambiguities with 3D Scene
    Constraints' ICCV'2019 More details can be found in the `paper.

    <https://ps.is.tuebingen.mpg.de/uploads_file/attachment/
    attachment/530/ICCV_2019___PROX.pdf>`__ .
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
                keypoints2d_mask stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we need
        image_path_, keypoints2d_, bbox_xywh_, cam_param_ = [], [], [], []

        smplx = {}
        smplx['body_pose'] = []
        smplx['global_orient'] = []
        smplx['betas'] = []
        smplx['transl'] = []
        smplx['expression'] = []
        smplx['leye_pose'] = []
        smplx['reye_pose'] = []
        smplx['jaw_pose'] = []
        smplx['pose_embedding'] = []

        fx, fy, cx, cy = 1060.53, 1060.38, 951.30, 536.77
        K = self.get_intrinsic_matrix([fx, fy], [cx, cy])
        h, w = 1080, 1920

        # img_dir
        recordings_dir = os.path.join(dataset_path, 'recordings')
        keypoints_dir = os.path.join(dataset_path, 'keypoints')
        smpl_dir = os.path.join(dataset_path, 'PROXD')

        missing_keypoints = []
        missing_smpl = []

        for fid in tqdm(os.listdir(recordings_dir)):
            img_dir = os.path.join(recordings_dir, fid, 'Color')
            for img_name in os.listdir(img_dir):
                image_path = f'recordings/{fid}/Color/{img_name}'
                image_id = img_name.split('.jpg')[0]
                smpl_file = f'{smpl_dir}/{fid}/results/{image_id}/000.pkl'
                keypoints_file = \
                    f'{keypoints_dir}/{fid}/{image_id}_keypoints.json'
                # check if all scenes only contain single person
                with open(keypoints_file) as f:
                    data = json.load(f)
                    if len(data['people']) < 1:
                        missing_keypoints.append(keypoints_file)
                        continue
                    keypoints2d = np.array(
                        data['people'][0]['pose_keypoints_2d']).reshape(25, 3)
                    keypoints2d[keypoints2d[:, 2] > 0.15,
                                2] = 1  # set based on keypoints confidence

                    vis_keypoints2d = keypoints2d[np.where(
                        keypoints2d[:, 2] > 0)[0]]
                    # bbox
                    bbox_xyxy = [
                        min(vis_keypoints2d[:, 0]),
                        min(vis_keypoints2d[:, 1]),
                        max(vis_keypoints2d[:, 0]),
                        max(vis_keypoints2d[:, 1])
                    ]
                    bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
                    bbox_xywh = self._xyxy2xywh(bbox_xyxy)

                if not os.path.exists(smpl_file):
                    missing_smpl.append(smpl_file)
                else:
                    with open(smpl_file, 'rb') as f:
                        ann = pickle.load(f, encoding='latin1')
                        smplx['body_pose'].append(ann['body_pose'])
                        # smplx['body_pose'].append(ann['body_pose'])
                        smplx['global_orient'].append(ann['global_orient'])
                        smplx['betas'].append(ann['betas'])
                        smplx['transl'].append(ann['transl'])
                        smplx['jaw_pose'].append(ann['jaw_pose'])
                        smplx['leye_pose'].append(ann['leye_pose'])
                        smplx['reye_pose'].append(ann['reye_pose'])
                        smplx['expression'].append(ann['expression'])
                        smplx['pose_embedding'].append(ann['pose_embedding'])

                        R = ann['camera_rotation'].reshape(3, 3)
                        T = ann['camera_translation'].reshape(3)

                        camera_params = CameraParameter(H=h, W=w)
                        camera_params.set_KRT(K, R, T)

                    image_path_.append(image_path)
                    keypoints2d_.append(keypoints2d)
                    cam_param_.append(camera_params)
                    bbox_xywh_.append(bbox_xywh)

        smplx['expression'] = np.array(smplx['expression']).reshape((-1, 10))
        smplx['leye_pose'] = np.array(smplx['leye_pose']).reshape((-1, 3))
        smplx['reye_pose'] = np.array(smplx['reye_pose']).reshape((-1, 3))
        smplx['jaw_pose'] = np.array(smplx['jaw_pose']).reshape((-1, 3))
        smplx['body_pose'] = np.array(smplx['body_pose']).reshape((-1, 21, 3))
        smplx['global_orient'] = np.array(smplx['global_orient']).reshape(
            (-1, 3))
        smplx['betas'] = np.array(smplx['betas']).reshape((-1, 10))
        smplx['transl'] = np.array(smplx['transl']).reshape((-1, 3))
        smplx['pose_embedding'] = np.array(smplx['pose_embedding']).reshape(
            (-1, 32))

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 25, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'prox', 'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['smplx'] = smplx
        human_data['cam_param'] = cam_param_
        human_data['config'] = 'prox'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'prox_train_smplx.npz')
        human_data.dump(out_file)
