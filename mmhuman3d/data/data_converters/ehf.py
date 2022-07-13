import json
import os

import cv2
import numpy as np
from plyfile import PlyData

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter


@DATA_CONVERTERS.register_module()
class EHFConverter(BaseModeConverter):
    """EHF dataset 'Expressive Hands and Face' More details can be found on the
    website:

    https://smpl-x.is.tue.mpg.de
    Args:
        modes (list): 'val' for accepted modes
    """
    NUM_BETAS = 10
    NUM_EXPRESSION = 10
    ACCEPTED_MODES = ['val']

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
                A dict containing keys image_path, bbox_xywh, smplx, meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()
        # structs we use
        image_path_, bbox_xywh_ = [], []
        keypoints2d_ = []
        vertices_ = []
        camera_pose = np.array(
            [-2.9874789618512025, 0.011724572107320893, -0.05704686818955933])
        camera_pose = cv2.Rodrigues(camera_pose)[0]

        datalist = os.listdir(dataset_path)
        for dataname in datalist:
            if os.path.splitext(dataname)[1] == '.jpg':
                image_path_.append(dataname)

        for dataname in image_path_:
            name = dataname.split('_')[0]
            json_path = os.path.join(dataset_path, name + '_2Djnt.json')
            json_data = json.load(open(json_path, 'r'))
            body_kps2d = np.array(
                json_data['people'][0]['pose_keypoints_2d'],
                dtype=np.float32).reshape(-1, 3)
            face_kps2d = np.array(
                json_data['people'][0]['face_keypoints_2d'],
                dtype=np.float32).reshape(-1, 3)
            lhand_kps2d = np.array(
                json_data['people'][0]['hand_left_keypoints_2d'],
                dtype=np.float32).reshape(-1, 3)
            rhand_kps2d = np.array(
                json_data['people'][0]['hand_right_keypoints_2d'],
                dtype=np.float32).reshape(-1, 3)
            keypoints2d = np.concatenate(
                (body_kps2d, lhand_kps2d, rhand_kps2d, face_kps2d), axis=0)
            keypoints2d[:, -1] = 1.0
            conf = keypoints2d[:, -1]
            keypoints2d_.append(keypoints2d)
            bbox = self._keypoints_to_scaled_bbox(
                keypoints2d[:, :2][conf > 0], scale=1.2)
            bbox_xywh = self._xyxy2xywh(bbox)
            bbox_xywh_.append(bbox_xywh)

            align_path = os.path.join(dataset_path, name + '_align.ply')
            aligndata = PlyData.read(align_path)
            vertices = np.concatenate(
                (np.array(aligndata['vertex']['x']).reshape(
                    -1, 1), np.array(aligndata['vertex']['y']).reshape(-1, 1),
                 np.array(aligndata['vertex']['z']).reshape(-1, 1)),
                axis=1)
            vertices = np.matmul(camera_pose, vertices.T).T
            vertices_.append(vertices)

        keypoints2d_ = np.array(keypoints2d_)
        keypoints2d, keypoints2d_mask = convert_kps(
            keypoints2d_, src='openpose_137', dst='human_data')
        vertices_ = np.array(vertices_)

        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask
        human_data['vertices'] = vertices_
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])

        human_data['image_path'] = image_path_

        human_data['bbox_xywh'] = bbox_xywh_
        human_data['config'] = 'ehf'

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'ehf_{}.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
