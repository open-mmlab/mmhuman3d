import os

import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter


@DATA_CONVERTERS.register_module()
class ExposeSPINSMPLXConverter(BaseModeConverter):
    """SPIN in SMPLX for ExPose 'Monocular Expressive Body Regression through
    Body-Driven Attention' More details can be found on the website:

    https://expose.is.tue.mpg.de/
    Args:
        modes (list): 'train' for accepted modes
    """
    NUM_BETAS = 10
    NUM_EXPRESSION = 10
    ACCEPTED_MODES = ['train']

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
        BODY_THRESH = 0.1
        HAND_THRESH = 0.2
        FACE_THRESH = 0.4
        BODY_IDXS = get_keypoint_idxs_by_part('body', 'human_data')
        LEFT_HAND_IDXS = get_keypoint_idxs_by_part('left_hand', 'human_data')
        RIGHT_HAND_IDXS = get_keypoint_idxs_by_part('right_hand', 'human_data')
        FACE_IDXS = get_keypoint_idxs_by_part('head', 'human_data')
        # use HumanData to store all data
        human_data = HumanData()
        # structs we use
        image_path_, bbox_xywh_ = [], []
        smplx = {}
        smplx['body_pose'] = []
        smplx['jaw_pose'] = []
        smplx['global_orient'] = []
        smplx['betas'] = []
        smplx['left_hand_pose'] = []
        smplx['right_hand_pose'] = []

        dset_names = ['coco', 'lsp', 'lspet', 'mpii']
        spin_data = {}
        for dset_name in dset_names:
            data = np.load(
                os.path.join(dataset_path, '{}.npz'.format(dset_name)))

            has_smpl = np.asarray(data['has_smpl']).astype(np.bool)
            data = {key: data[key][has_smpl] for key in data.keys()}
            data['dset'] = [dset_name] * data['pose'].shape[0]

            if dset_name == 'lsp':
                data['part'][26, [9, 11], :] = data['part'][26, [11, 9], :]
            spin_data[dset_name] = data

        pose = np.concatenate([spin_data[dset]['pose'] for dset in spin_data],
                              axis=0).astype(np.float32)
        keypoints2d = np.concatenate(
            [spin_data[dset]['part'] for dset in spin_data],
            axis=0).astype(np.float32)
        imgnames = np.concatenate(
            [spin_data[dset]['imgname'] for dset in spin_data], axis=0)
        pose = pose.reshape(-1, 55, 3)
        betas = np.concatenate(
            [spin_data[dset]['betas'] for dset in spin_data],
            axis=0).astype(np.float32)
        dsets = np.concatenate([spin_data[dset]['dset'] for dset in spin_data],
                               axis=0)
        centers = np.concatenate(
            [spin_data[dset]['center'] for dset in spin_data],
            axis=0).astype(np.float32)
        scales = np.concatenate(
            [spin_data[dset]['scale'] for dset in spin_data],
            axis=0).astype(np.float32)

        global_pose = pose[:, 0]
        body_pose = pose[:, 1:22]
        jaw_pose = pose[:, 22]
        left_hand_pose = pose[:, 25:25 + 15]
        right_hand_pose = pose[:, -15:]

        keypoints2d, keypoints2d_mask = convert_kps(
            keypoints2d, src='spin_smplx', dst='human_data')

        for index, kps in enumerate(keypoints2d):
            # Remove joints with negative confidence
            kps[kps[:, -1] < 0, -1] = 0
            body_conf = kps[BODY_IDXS, -1]
            left_hand_conf = kps[LEFT_HAND_IDXS, -1]
            right_hand_conf = kps[RIGHT_HAND_IDXS, -1]
            face_conf = kps[FACE_IDXS, -1]

            body_conf = (body_conf >= BODY_THRESH).astype(np.float32)
            left_hand_conf = (left_hand_conf >= HAND_THRESH).astype(np.float32)
            right_hand_conf = (right_hand_conf >= HAND_THRESH).astype(
                np.float32)
            face_conf = (face_conf >= FACE_THRESH).astype(np.float32)

            kps[BODY_IDXS, -1] = body_conf
            kps[LEFT_HAND_IDXS, -1] = left_hand_conf
            kps[RIGHT_HAND_IDXS, -1] = right_hand_conf
            kps[FACE_IDXS, -1] = face_conf
            center = centers[index]
            scale = scales[index]
            bbox = np.concatenate([center - scale * 100, center + scale * 100])
            bbox_xywh = self._xyxy2xywh(bbox)
            bbox_xywh_.append(bbox_xywh)

        for index in range(len(imgnames)):
            if dsets[index] == 'coco':
                image_path_.append(
                    os.path.join('coco/train2014', imgnames[index]))
            elif dsets[index] == 'lspet':
                image_path_.append(
                    os.path.join('lsp/lspet/images', imgnames[index]))
            elif dsets[index] == 'lsp':
                image_path_.append(
                    os.path.join('lsp/lsp_dataset_original/images',
                                 imgnames[index]))
            elif dsets[index] == 'mpii':
                image_path_.append(
                    os.path.join('mpii/images', imgnames[index]))

        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        smplx['body_pose'] = body_pose
        smplx['global_orient'] = global_pose
        smplx['jaw_pose'] = jaw_pose
        smplx['betas'] = betas
        smplx['right_hand_pose'] = right_hand_pose
        smplx['left_hand_pose'] = left_hand_pose

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smplx'] = smplx
        human_data['config'] = 'expose_spin_smplx'

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'spin_smplx_{}.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
