import json
import os

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_converters.base_converter import BaseConverter
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.datasets.pipelines.hybrik_transforms import (
    cam2pixel,
    get_bbox,
    get_intrinsic_matrix,
)


@DATA_CONVERTERS.register_module()
class Pw3dHybrIKConverter(BaseConverter):
    """3D Poses in the Wild dataset for HybrIK `Recovering Accurate 3D Human
    Pose in The Wild Using IMUs and a Moving Camera' ECCV'2018 More details can
    be found in the `paper.

    <https://virtualhumans.mpi-inf.mpg.de/papers/vonmarcardECCV18/
    vonmarcardECCV18.pdf>`__ .

    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where hybrik preprocessed
            json files are stored
            out_path (str): Path to directory to save preprocessed npz file

        Returns:
            dict:
                A dict containing keys image_path, image_height, image_width,
                bbox_xywh, smpl, cam_param, root_cam, depth_factor,
                keypoints3d, keypoints3d_mask, keypoints3d17_cam_mask,
                keypoints3d_cam_mask, keypoints3d17_relative_mask,
                keypoints3d_relative_mask, keypoints3d17_cam, keypoints3d17,
                keypoints3d17_relative, keypoints3d_cam
                stored in HumanData() format
        """
        ann_file = os.path.join(dataset_path, '3DPW_test_new.json')

        with open(ann_file, 'r') as fid:
            database = json.load(fid)

        # use HumanData to store all data
        human_data = HumanData()

        image_path_, bbox_xywh_, root_cam_, image_width_, image_height_, \
            joint_cam_, joint_img_, joint_relative_, depth_factor_, \
            joint29_cam_, joint29_img_, joint29_relative_ = \
            [], [], [], [], [], [], [], [], [], [], [], []

        smpl = {}
        smpl['thetas'] = []
        smpl['betas'] = []
        cam_param = {}
        cam_param['f'] = []
        cam_param['c'] = []
        cam_param['intrinsic'] = []

        imgs = {}
        for img in database['images']:
            imgs[img['id']] = img

        for ann_annotations in tqdm(database['annotations']):
            ann = dict()
            for k, v in ann_annotations.items():
                ann[k] = v

            image_id = ann['image_id']
            file_name = str(imgs[image_id]['file_name'])
            height = imgs[image_id]['height']
            width = imgs[image_id]['width']
            sequence = str(imgs[image_id]['sequence'])
            bbox = ann['bbox']
            bbox = get_bbox(np.array(bbox), width, height)
            f, c = np.array(
                imgs[image_id]['cam_param']['focal'],
                dtype=np.float32), np.array(
                    imgs[image_id]['cam_param']['princpt'], dtype=np.float32)
            intrinsic = get_intrinsic_matrix(f, c, inv=True)

            h36m_joints = np.array(ann['h36m_joints']).reshape(17, 3)
            smpl_joints = np.array(ann['smpl_joint_cam']).reshape(24, 3)

            root_cam = np.array(ann['fitted_3d_pose']).reshape(-1, 3)[0]
            path = 'imageFiles/' + sequence + '/' + file_name

            # image_abs_path = os.path.join(image_dir, path)
            # if not os.path.exists(image_abs_path):
            #     print('file does not exist in image dir')

            joint_cam_17 = h36m_joints
            joint_relative_17 = joint_cam_17 - joint_cam_17[0, :]
            joint_img_17 = np.zeros((17, 3))
            joint_img_17 = np.hstack([joint_img_17, np.ones([17, 1])])
            joint_cam_17 = np.hstack([joint_cam_17, np.ones([17, 1])])
            joint_relative_17 = np.hstack(
                [joint_relative_17, np.ones([17, 1])])

            joint_cam = smpl_joints.reshape(24, 3)
            joint_cam_29 = np.zeros((29, 3))
            joint_cam_29[:24, :] = joint_cam.reshape(24, 3)

            joint_img = cam2pixel(joint_cam, f, c)
            joint_img_29 = np.zeros((29, 3))
            joint_img_29[:24, :] = joint_img.reshape(24, 3)
            joint_img_29[:, 2] = joint_img_29[:, 2] - \
                joint_cam_29[0, 2]

            root_cam = joint_cam_29[0].astype(np.float32)
            joint_vis_29 = np.zeros((29, 1))
            joint_vis_29[:24, :] = np.ones((24, 1))
            joint_relative_29 = joint_cam_29 - joint_cam_29[0, :].copy()

            joint_cam_29 = np.hstack([joint_cam_29, joint_vis_29])
            joint_img_29 = np.hstack([joint_img_29, joint_vis_29])
            joint_relative_29 = np.hstack([joint_relative_29, joint_vis_29])

            smpl['betas'].append(
                np.array(ann['smpl_param']['shape']).reshape((10)))
            smpl['thetas'].append(
                np.array(ann['smpl_param']['pose']).reshape((24, 3)))
            image_path_.append(path)
            image_height_.append(height)
            image_width_.append(width)
            bbox_xywh_.append(bbox)
            cam_param['f'].append(f.reshape((-1, 2)))
            cam_param['c'].append(c.reshape((-1, 2)))
            cam_param['intrinsic'].append(intrinsic.reshape(3, 3))
            root_cam_.append(root_cam)
            depth_factor_.append(2.)
            joint_cam_.append(joint_cam_17)
            joint_img_.append(joint_img_17)
            joint_relative_.append(joint_relative_17)
            joint29_cam_.append(joint_cam_29)
            joint29_img_.append(joint_img_29)
            joint29_relative_.append(joint_relative_29)

        smpl['thetas'] = np.array(smpl['thetas']).reshape((-1, 24, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))
        cam_param['f'] = np.array(cam_param['f']).reshape((-1, 2))
        cam_param['c'] = np.array(cam_param['c']).reshape((-1, 2))
        cam_param['intrinsic'] = np.array(cam_param['intrinsic']).reshape(
            (-1, 3, 3))

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])

        joint_img_ = np.array(joint_img_).reshape((-1, 17, 4))
        joint_cam_ = np.array(joint_cam_).reshape((-1, 17, 4))
        joint_relative_ = np.array(joint_relative_).reshape((-1, 17, 4))

        keypoints3d17_, keypoints3d17_mask = convert_kps(
            joint_img_, 'h36m', 'human_data')
        keypoints3d17_cam, _ = convert_kps(joint_cam_, 'h36m', 'human_data')
        keypoints3d17_relative_, _ = convert_kps(joint_relative_, 'h36m',
                                                 'human_data')

        joint29_img_ = np.array(joint29_img_).reshape((-1, 29, 4))
        joint29_cam_ = np.array(joint29_cam_).reshape((-1, 29, 4))
        joint29_relative_ = np.array(joint29_relative_).reshape((-1, 29, 4))
        keypoints3d_, keypoints3d_mask = convert_kps(joint29_img_, 'hybrik_29',
                                                     'human_data')
        keypoints3d_cam_, _ = convert_kps(joint29_cam_, 'hybrik_29',
                                          'human_data')
        keypoints3d_relative_, _ = convert_kps(joint29_relative_, 'hybrik_29',
                                               'human_data')

        human_data['image_path'] = image_path_
        human_data['image_height'] = image_height_
        human_data['image_width'] = image_width_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smpl'] = smpl
        human_data['cam_param'] = cam_param
        human_data['root_cam'] = root_cam_
        human_data['depth_factor'] = depth_factor_
        human_data['keypoints3d17_mask'] = keypoints3d17_mask
        human_data['keypoints3d_mask'] = keypoints3d_mask
        human_data['keypoints3d17_cam_mask'] = keypoints3d17_mask
        human_data['keypoints3d_cam_mask'] = keypoints3d_mask
        human_data['keypoints3d17_relative_mask'] = keypoints3d17_mask
        human_data['keypoints3d_relative_mask'] = keypoints3d_mask
        human_data['keypoints3d17_cam'] = keypoints3d17_cam
        human_data['keypoints3d17'] = keypoints3d17_
        human_data['keypoints3d17_relative'] = keypoints3d17_relative_
        human_data['keypoints3d_relative'] = keypoints3d_relative_
        human_data['keypoints3d_cam'] = keypoints3d_cam_
        human_data['keypoints3d'] = keypoints3d_
        human_data['config'] = 'pw3d'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'hybrik_pw3d_test.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
