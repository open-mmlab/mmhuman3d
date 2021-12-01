import json
import os

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_converters.base_converter import BaseModeConverter
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.datasets.pipelines.hybrik_transforms import (
    get_bbox,
    get_intrinsic_matrix,
)


@DATA_CONVERTERS.register_module()
class MpiInf3dhpHybrIKConverter(BaseModeConverter):
    """MPI-INF-3DHP dataset for HybrIK `Monocular 3D Human Pose Estimation In
    The Wild Using Improved CNN Supervision' 3DC`2017 More details can be found
    in the `paper.

    <https://arxiv.org/pdf/1611.09813.pdf>`__.

    Args:
        modes (list): 'test' or 'train' for accepted modes
    """
    ACCEPTED_MODES = ['test', 'train']

    def __init__(self, modes=[]):
        super(MpiInf3dhpHybrIKConverter, self).__init__(modes)

    @staticmethod
    def cam2pixel_matrix(cam_coord: np.ndarray,
                         intrinsic_param: np.ndarray) -> np.ndarray:
        """Convert coordinates from camera to image frame given intrinsic
        matrix
        Args:
            cam_coord (np.ndarray): Coordinates in camera frame
            intrinsic_param (np.ndarray): 3x3 Intrinsic matrix

        Returns:
            img_coord (np.ndarray): Coordinates in image frame
        """
        cam_coord = cam_coord.transpose(1, 0)
        cam_homogeneous_coord = np.concatenate(
            (cam_coord, np.ones((1, cam_coord.shape[1]), dtype=np.float32)),
            axis=0)
        img_coord = np.dot(intrinsic_param, cam_homogeneous_coord) / (
            cam_coord[2, :] + 1e-8)
        img_coord = np.concatenate((img_coord[:2, :], cam_coord[2:3, :]),
                                   axis=0)
        return img_coord.transpose(1, 0)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where hybrik preprocessed
            json files are stored
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, image_height, image_width,
                bbox_xywh, cam_param, root_cam, depth_factor, keypoints3d,
                keypoints3d_mask, keypoints3d_cam, keypoints3d_cam_mask
                stored in HumanData() format
        """
        if mode == 'train':
            ann_file = os.path.join(dataset_path,
                                    'annotation_mpi_inf_3dhp_train_v2.json')
        elif mode == 'test':
            ann_file = os.path.join(dataset_path,
                                    'annotation_mpi_inf_3dhp_test.json')

        with open(ann_file, 'r') as fid:
            database = json.load(fid)

        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, root_cam_, image_width_, image_height_, \
            joint_cam_, joint_img_, depth_factor_ = \
            [], [], [], [], [], [], [], []
        smpl = {}
        smpl['thetas'] = []
        smpl['betas'] = []
        cam_param = {}
        cam_param['f'] = []
        cam_param['c'] = []
        cam_param['intrinsic'] = []

        num_datapoints = len(database['images'])
        for ann_image, ann_annotations in tqdm(
                zip(database['images'], database['annotations']),
                total=num_datapoints):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v

            width, height = ann['width'], ann['height']
            bbox = ann['bbox']

            bbox = get_bbox(np.array(bbox), width, height)

            K = np.array(ann['cam_param']['intrinsic_param'])
            f = np.array([K[0, 0], K[1, 1]])
            c = np.array([K[0, 2], K[1, 2]])
            intrinsic = get_intrinsic_matrix(f, c, inv=True)

            joint_cam = np.array(ann['keypoints_cam'])
            num_joints = joint_cam.shape[0]

            # if train
            if mode == 'train':
                root_idx = 4
                _, sub, seq, vid, im = ann['file_name'].split('/')[-1].split(
                    '_')
                fname = '{}/{}/{}/{}'.format(sub, seq,
                                             vid.replace('V', 'video_'), im)
                # fname = '{}/{}/imageFrames/{}/frame_{}'.format(
                #     sub, seq, vid.replace('V', 'video_'), im)
            elif mode == 'test':
                root_idx = 14
                fname = 'mpi_inf_3dhp_test_set/' + ann['file_name']
                # fname = 'mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/' + ann[
                #     'file_name']

            joint_img = self.cam2pixel_matrix(joint_cam, K)
            joint_img[:, 2] = joint_img[:, 2] - joint_cam[root_idx, 2]
            root_cam = joint_cam[root_idx]
            joint_img = np.hstack([joint_img, np.ones([num_joints, 1])])
            joint_cam = np.hstack([joint_cam, np.ones([num_joints, 1])])

            image_path_.append(fname)
            image_height_.append(height)
            image_width_.append(width)
            bbox_xywh_.append(bbox)
            depth_factor_.append(2000.)
            cam_param['f'].append(f.reshape((-1, 2)))
            cam_param['c'].append(c.reshape((-1, 2)))
            cam_param['intrinsic'].append(intrinsic)
            joint_cam_.append(joint_cam)
            joint_img_.append(joint_img)
            root_cam_.append(root_cam)

        cam_param['f'] = np.array(cam_param['f']).reshape((-1, 2))
        cam_param['c'] = np.array(cam_param['c']).reshape((-1, 2))
        cam_param['intrinsic'] = np.array(cam_param['intrinsic']).reshape(
            (-1, 3, 3))

        if mode == 'train':
            keypoints3d_ = np.array(joint_img_).reshape((-1, 28, 4))
            keypoints3d_cam_ = np.array(joint_cam_).reshape((-1, 28, 4))
            keypoints3d_, keypoints3d_mask = convert_kps(
                keypoints3d_, 'hybrik_hp3d', 'human_data')
            keypoints3d_cam_, keypoints3d_cam_mask = convert_kps(
                keypoints3d_cam_, 'hybrik_hp3d', 'human_data')
        elif mode == 'test':
            keypoints3d_ = np.array(joint_img_).reshape((-1, 17, 4))
            keypoints3d_cam_ = np.array(joint_cam_).reshape((-1, 17, 4))
            keypoints3d_, keypoints3d_mask = convert_kps(
                keypoints3d_, 'mpi_inf_3dhp_test', 'human_data')
            keypoints3d_cam_, _ = convert_kps(keypoints3d_cam_,
                                              'mpi_inf_3dhp_test',
                                              'human_data')

        # convert keypoints
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['depth_factor'] = depth_factor_
        human_data['keypoints3d_mask'] = keypoints3d_mask
        human_data['keypoints3d_cam_mask'] = keypoints3d_mask
        human_data['keypoints3d_cam'] = keypoints3d_cam_
        human_data['keypoints3d'] = keypoints3d_

        human_data['cam_param'] = cam_param
        human_data['root_cam'] = root_cam_
        human_data['image_height'] = image_height_
        human_data['image_width'] = image_width_
        human_data['config'] = 'mpi_inf_3dhp'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'hybrik_mpi_inf_3dhp_{}.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
