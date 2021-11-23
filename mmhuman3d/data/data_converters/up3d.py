import os
import pickle
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class Up3dConverter(BaseModeConverter):
    """Unite the People dataset `Unite the People – Closing the Loop Between 3D
    and 2D Human Representations' CVPR'2017 More details can be found in the
    `paper.

    <https://arxiv.org/pdf/1701.02468.pdf>`__ .

    Args:
        modes (list): 'test' and/or 'trainval' for accepted modes
    """
    ACCEPTED_MODES = ['test', 'trainval']

    def __init__(self, modes: List = []) -> None:
        super(Up3dConverter, self).__init__(modes)

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
                keypoints2d_mask, smpl stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_ = [], [], []
        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []

        txt_file = os.path.join(dataset_path, '%s.txt' % mode)
        with open(txt_file, 'r') as f:
            img_list = f.read()
        imgs = img_list.split('\n')

        # go over all images
        for img_i in tqdm(imgs):
            # skip empty row in txt
            if len(img_i) == 0:
                continue

            # image name
            img_base = img_i[1:-10]
            img_name = '%s_image.png' % img_base

            # keypoints processing
            keypoints_file = os.path.join(dataset_path,
                                          '%s_joints.npy' % img_base)
            keypoints2d = np.load(keypoints_file)
            keypoints2d = np.transpose(keypoints2d, (1, 0))

            # obtain bbox from mask
            render_name = os.path.join(dataset_path,
                                       '%s_render_light.png' % img_base)
            render_mask = cv2.imread(render_name)
            ys, xs = np.where(np.min(render_mask, axis=2) < 255)
            bbox_xyxy = np.array(
                [np.min(xs),
                 np.min(ys),
                 np.max(xs) + 1,
                 np.max(ys) + 1])
            bbox_xywh = self._bbox_expand(bbox_xyxy, scale_factor=0.9)

            # pose and shape
            pkl_file = os.path.join(dataset_path, '%s_body.pkl' % img_base)
            with open(pkl_file, 'rb') as f:
                pkl = pickle.load(f, encoding='latin1')
            pose = pkl['pose']
            shape = pkl['betas']

            smpl['body_pose'].append(pose[3:].reshape((23, 3)))
            smpl['global_orient'].append(pose[:3])
            smpl['betas'].append(shape)
            image_path_.append(img_name)
            bbox_xywh_.append(bbox_xywh)
            keypoints2d_.append(keypoints2d)

        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 14, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'lsp', 'human_data')

        # change list to np array
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smpl'] = smpl
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['config'] = 'up3d'
        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'up3d_%s.npz' % mode
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
