import os
import pickle
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter


@DATA_CONVERTERS.register_module()
class Pw3dConverter(BaseModeConverter):
    """3D Poses in the Wild dataset `Recovering Accurate 3D Human Pose in The
    Wild Using IMUs and a Moving Camera' ECCV'2018 More details can be found in
    the `paper.

    <https://virtualhumans.mpi-inf.mpg.de/papers/vonmarcardECCV18/
    vonmarcardECCV18.pdf>`__ .

    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """

    ACCEPTED_MODES = ['train', 'test']

    def __init__(self, modes: List = []) -> None:
        super(Pw3dConverter, self).__init__(modes)

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
                A dict containing keys image_path, bbox_xywh, smpl, meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, cam_param_ = [], [], []
        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []
        meta = {}
        meta['gender'] = []

        root_path = dataset_path
        # get a list of .pkl files in the directory
        dataset_path = os.path.join(dataset_path, 'sequenceFiles', mode)
        files = [
            os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
            if f.endswith('.pkl')
        ]

        # go through all the .pkl files
        for filename in tqdm(files):
            with open(filename, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                smpl_pose = data['poses']
                smpl_betas = data['betas']
                poses2d = data['poses2d']
                global_poses = data['cam_poses']
                genders = data['genders']
                valid = np.array(data['campose_valid']).astype(np.bool)
                K = np.array(data['cam_intrinsics'])
                num_people = len(smpl_pose)
                num_frames = len(smpl_pose[0])
                seq_name = str(data['sequence'])
                img_names = np.array([
                    'imageFiles/' + seq_name + f'/image_{str(i).zfill(5)}.jpg'
                    for i in range(num_frames)
                ])
                # get through all the people in the sequence
                for i in range(num_people):
                    valid_pose = smpl_pose[i][valid[i]]
                    valid_betas = np.tile(smpl_betas[i][:10].reshape(1, -1),
                                          (num_frames, 1))
                    valid_betas = valid_betas[valid[i]]
                    valid_keypoints_2d = poses2d[i][valid[i]]
                    valid_img_names = img_names[valid[i]]
                    valid_global_poses = global_poses[valid[i]]
                    gender = genders[i]

                    # consider only valid frames
                    for valid_i in range(valid_pose.shape[0]):
                        keypoints2d = valid_keypoints_2d[valid_i, :, :].T
                        keypoints2d = keypoints2d[keypoints2d[:, 2] > 0, :]
                        bbox_xywh = [
                            min(keypoints2d[:, 0]),
                            min(keypoints2d[:, 1]),
                            max(keypoints2d[:, 0]),
                            max(keypoints2d[:, 1])
                        ]

                        bbox_xywh = self._bbox_expand(
                            bbox_xywh, scale_factor=1.2)

                        image_path = valid_img_names[valid_i]
                        image_abs_path = os.path.join(root_path, image_path)
                        h, w, _ = cv2.imread(image_abs_path).shape

                        # transform global pose
                        pose = valid_pose[valid_i]
                        extrinsic_param = valid_global_poses[valid_i]
                        R = extrinsic_param[:3, :3]
                        T = extrinsic_param[:3, 3]

                        camera = CameraParameter(H=h, W=w)
                        camera.set_KRT(K, R, T)
                        parameter_dict = camera.to_dict()

                        pose[:3] = cv2.Rodrigues(
                            np.dot(R,
                                   cv2.Rodrigues(pose[:3])[0]))[0].T[0]

                        image_path_.append(image_path)
                        bbox_xywh_.append(bbox_xywh)
                        smpl['body_pose'].append(pose[3:].reshape((23, 3)))
                        smpl['global_orient'].append(pose[:3])
                        smpl['betas'].append(valid_betas[valid_i])
                        meta['gender'].append(gender)
                        cam_param_.append(parameter_dict)

        # change list to np array
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))
        meta['gender'] = np.array(meta['gender'])

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smpl'] = smpl
        human_data['meta'] = meta
        human_data['cam_param'] = cam_param_
        human_data['config'] = 'pw3d'

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'pw3d_{}.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
