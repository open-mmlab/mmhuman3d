import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.datasets.pipelines.hybrik_transforms import (
    cam2pixel,
    get_bbox,
    get_intrinsic_matrix,
)
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

CAMERA_IDX_TO_NAME = {
    1: '54138969',
    2: '55011271',
    3: '58860488',
    4: '60457274',
}


class H36M_Metadata:
    """Extract sequence mappings from Human3.6M Metadata.

    Args:
        metadata_file (str): path to metadata.xml file
    """

    def __init__(self, metadata_file):
        self.subjects = []
        self.sequence_mappings = {}
        self.action_names = {}
        self.camera_ids = []

        tree = ET.parse(metadata_file)
        root = tree.getroot()

        for i, tr in enumerate(root.find('mapping')):
            if i == 0:
                _, _, *self.subjects = [td.text for td in tr]
                self.sequence_mappings = {
                    subject: {}
                    for subject in self.subjects
                }
            elif i < 33:
                action_id, subaction_id, *prefixes = [td.text for td in tr]
                for subject, prefix in zip(self.subjects, prefixes):
                    self.sequence_mappings[subject][(action_id,
                                                     subaction_id)] = prefix

        for i, elem in enumerate(root.find('actionnames')):
            action_id = str(i + 1)
            self.action_names[action_id] = elem.text

        self.camera_ids = [
            elem.text for elem in root.find('dbcameras/index2id')
        ]

    def get_base_filename(self, subject, action, subaction, camera):
        return '{}.{}'.format(
            self.sequence_mappings[subject][(action, subaction)], camera)


@DATA_CONVERTERS.register_module()
class H36mHybrIKConverter(BaseModeConverter):
    """Human3.6M dataset for HybrIK
    `Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments' TPAMI`2014
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    Args:
        modes (list): 'test' or 'train' for accepted modes
    """
    ACCEPTED_MODES = ['test', 'train']

    def __init__(self, modes=[]):
        super(H36mHybrIKConverter, self).__init__(modes)

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
                bbox_xywh, smpl, cam_param, root_cam, depth_factor,
                keypoints3d, keypoints3d_mask, keypoints3d17_cam_mask,
                keypoints3d_cam_mask, keypoints3d17_relative_mask,
                keypoints3d_relative_mask, keypoints3d17_cam, keypoints3d17,
                keypoints3d17_relative, keypoints3d_cam, phi, phi weight
                stored in HumanData() format
        """
        if mode == 'train':
            ann_file = os.path.join(
                dataset_path,
                'Sample_5_train_Human36M_smpl_leaf_twist_protocol_2.json')
        elif mode == 'test':
            ann_file = os.path.join(
                dataset_path, 'Sample_20_test_Human36M_smpl_protocol_2.json')

        with open(ann_file, 'r') as fid:
            database = json.load(fid)

        metadata_file = os.path.join(dataset_path, 'metadata.xml')
        metadata = H36M_Metadata(metadata_file)

        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, root_cam_, image_width_, image_height_, \
            angle_twist_, joint_cam_, joint_img_, joint29_cam_, \
            joint_relative_, joint29_img_, joint29_relative_, depth_factor_, \
            twist_weight_ = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], []

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

            f, c = np.array(
                ann['cam_param']['f'], dtype=np.float32), np.array(
                    ann['cam_param']['c'], dtype=np.float32)
            intrinsic = get_intrinsic_matrix(f, c, inv=True)

            h36m_joints = np.array(ann['h36m_joints']).reshape(17, 3)
            smpl_joints = np.array(ann['smpl_joints'])

            # image_path
            camera = ann['cam_idx']
            subject = ann['subject']
            frame_idx = ann['frame_idx']
            subaction_idx = ann['subaction_idx']
            action_idx = ann['action_idx']
            seq_name = metadata.sequence_mappings['S{}'.format(subject)][(
                '{}'.format(action_idx), '{}'.format(subaction_idx))]
            seq_name = seq_name.replace(' ', '_')

            fname = 'S{}_{}.{}'.format(subject, seq_name,
                                       CAMERA_IDX_TO_NAME[camera])
            image_name = '%s_%06d.jpg' % (fname, frame_idx + 1)
            path = 'S{}/images/{}/{}'.format(subject, fname, image_name)
            # path on sg2
            # path = 'S{}/{}/{}'.format(subject, fname, image_name)

            if mode == 'train':
                angle = np.array(np.array(ann['angle_twist']).item()['angle'])
                sin = np.array(np.array(ann['angle_twist']).item()['sin'])
                cos = np.array(np.array(ann['angle_twist']).item()['cos'])
                assert (np.cos(angle) - cos < 1e-6).all(), np.cos(angle) - cos
                assert (np.sin(angle) - sin < 1e-6).all(), np.sin(angle) - sin
                phi = np.stack((cos, sin), axis=1)
                target_twist = phi.astype(np.float32)  # twist_phi
                phi_weight = (angle >
                              -10) * 1.0  # invalid angles are set to be -999
                phi_weight = np.stack([phi_weight, phi_weight], axis=1)

            # generate joint_cam, joint_img, joint_vis, joint_relative
            joint_cam_17 = h36m_joints
            # joint_vis_17 = np.ones((17, 3))
            joint_relative_17 = joint_cam_17 - joint_cam_17[0, :]
            joint_img_17 = cam2pixel(joint_cam_17, f, c)
            joint_img_17[:, 2] = joint_img_17[:, 2] - \
                joint_cam_17[0, 2]
            joint_img_17 = np.hstack([joint_img_17, np.ones([17, 1])])
            joint_cam_17 = np.hstack([joint_cam_17, np.ones([17, 1])])
            joint_relative_17 = np.hstack(
                [joint_relative_17, np.ones([17, 1])])

            if smpl_joints.size == 24 * 3:
                joint_cam = smpl_joints.reshape(24, 3)
                joint_cam_29 = np.zeros((29, 3))
                joint_cam_29[:24, :] = joint_cam
                root_cam = joint_cam_29[0].astype(np.float32)

                joint_img = cam2pixel(joint_cam, f, c)
                joint_img_29 = np.zeros((29, 3))
                joint_img_29[:24, :] = joint_img.reshape(24, 3)

                joint_vis_29 = np.zeros((29, 1))
                joint_vis_29[:24, :] = np.ones((24, 1))

                joint_relative_29 = joint_cam_29 - joint_cam_29[0, :].copy()

                joint_cam_29 = np.hstack([joint_cam_29, joint_vis_29])
                joint_img_29 = np.hstack([joint_img_29, joint_vis_29])
                joint_relative_29 = np.hstack(
                    [joint_relative_29, joint_vis_29])

            else:
                joint_cam_29 = smpl_joints.reshape(29, 3)
                root_cam = joint_cam_29[0].astype(np.float32)
                joint_relative_29 = joint_cam_29 - joint_cam_29[0, :].copy()
                joint_img_29 = cam2pixel(joint_cam_29, f, c)
                joint_img_29[:, 2] = joint_img_29[:, 2] - \
                    joint_cam_29[0, 2]

                joint_cam_29 = np.hstack([joint_cam_29, np.ones([29, 1])])
                joint_img_29 = np.hstack([joint_img_29, np.ones([29, 1])])
                joint_relative_29 = np.hstack(
                    [joint_relative_29, np.ones([29, 1])])

            if mode == 'train':
                angle_twist_.append(target_twist)
                twist_weight_.append(phi_weight)

            image_path_.append(path)
            image_height_.append(height)
            image_width_.append(width)
            bbox_xywh_.append(bbox)
            depth_factor_.append(2000.)
            cam_param['f'].append(f.reshape((-1, 2)))
            cam_param['c'].append(c.reshape((-1, 2)))
            cam_param['intrinsic'].append(intrinsic.reshape(3, 3))

            smpl['betas'].append(np.array(ann['betas']).reshape((10)))
            smpl['thetas'].append(np.array(ann['thetas']).reshape((24, 3)))

            root_cam_.append(root_cam)
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
        joint29_img_ = np.array(joint29_img_).reshape((-1, 29, 4))
        joint29_cam_ = np.array(joint29_cam_).reshape((-1, 29, 4))
        joint29_relative_ = np.array(joint29_relative_).reshape((-1, 29, 4))

        keypoints3d17_, keypoints3d17_mask = convert_kps(
            joint_img_, 'h36m', 'human_data')
        keypoints3d17_cam_, _ = convert_kps(joint_cam_, 'h36m', 'human_data')
        keypoints3d17_relative_, _ = convert_kps(joint_relative_, 'h36m',
                                                 'human_data')

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
        if mode == 'train':
            human_data['phi'] = angle_twist_
            human_data['phi_weight'] = twist_weight_
        human_data['root_cam'] = root_cam_
        human_data['depth_factor'] = depth_factor_
        human_data['keypoints3d17_mask'] = keypoints3d17_mask
        human_data['keypoints3d_mask'] = keypoints3d_mask
        human_data['keypoints3d17_cam_mask'] = keypoints3d17_mask
        human_data['keypoints3d_cam_mask'] = keypoints3d_mask
        human_data['keypoints3d17_relative_mask'] = keypoints3d17_mask
        human_data['keypoints3d_relative_mask'] = keypoints3d_mask
        human_data['keypoints3d17_cam'] = keypoints3d17_cam_
        human_data['keypoints3d17'] = keypoints3d17_
        human_data['keypoints3d17_relative'] = keypoints3d17_relative_
        human_data['keypoints3d_relative'] = keypoints3d_relative_
        human_data['keypoints3d_cam'] = keypoints3d_cam_
        human_data['keypoints3d'] = keypoints3d_
        human_data['config'] = 'h36m'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        if mode == 'train':
            out_file = os.path.join(out_path, 'hybrik_h36m_train.npz')
        elif mode == 'test':
            out_file = os.path.join(out_path,
                                    'hybrik_h36m_valid_protocol2.npz')
        human_data.dump(out_file)
