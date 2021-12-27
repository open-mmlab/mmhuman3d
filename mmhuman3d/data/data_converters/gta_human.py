import glob
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.builder import build_body_model
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS

# TODO:
# 1. camera parameters
# 2. root align using mid-point of hips


@DATA_CONVERTERS.register_module()
class GTAHumanConverter(BaseConverter):
    """GTA-Human dataset `Playing for 3D Human Recovery' arXiv`2021 More
    details can be found in the `paper.

    <https://arxiv.org/pdf/2110.07588.pdf>`__.
    """

    focal_length = 1158.0337
    camera_center = (960, 540)  # xy
    image_size = (1080, 1920)  # (height, width)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    smpl = build_body_model(
        dict(
            type='SMPL',
            keypoint_src='smpl_54',
            keypoint_dst='smpl_49',
            model_path='data/body_models/smpl',
            extra_joints_regressor='data/body_models/J_regressor_extra.npy')
    ).to(device)

    camera = build_cameras(
        dict(
            type='PerspectiveCameras',
            convention='opencv',
            in_ndc=False,
            focal_length=focal_length,
            image_size=image_size,
            principal_point=camera_center))

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file

        Returns:
            dict:
                A dict containing keys video_path, smplh, meta, frame_idx
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        smpl = {}
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []
        smpl['transl'] = []

        # structs we use
        image_path_, bbox_xywh_, keypoints_2d_gta_, keypoints_3d_gta_, \
            keypoints_2d_, keypoints_3d_ = [], [], [], [], [], []

        ann_paths = sorted(
            glob.glob(os.path.join(dataset_path, 'annotations', '*.pkl')))

        for ann_path in tqdm(ann_paths):

            with open(ann_path, 'rb') as f:
                ann = pickle.load(f, encoding='latin1')

            base = os.path.basename(ann_path)  # -> seq_00000001.pkl
            seq_idx, _ = os.path.splitext(base)  # -> seq_00000001
            num_frames = len(ann['body_pose'])

            keypoints_2d_gta, keypoints_2d_gta_mask = convert_kps(
                ann['keypoints_2d'], src='gta', dst='smpl_49')
            keypoints_3d_gta, keypoints_3d_gta_mask = convert_kps(
                ann['keypoints_3d'], src='gta', dst='smpl_49')

            global_orient = ann['global_orient']
            body_pose = ann['body_pose']
            betas = ann['betas']
            transl = ann['transl']

            output = self.smpl(
                global_orient=torch.tensor(global_orient, device=self.device),
                body_pose=torch.tensor(body_pose, device=self.device),
                betas=torch.tensor(betas, device=self.device),
                transl=torch.tensor(transl, device=self.device),
                return_joints=True)

            keypoints_3d = output['joints']
            keypoints_2d_xyd = self.camera.transform_points_screen(
                keypoints_3d)
            keypoints_2d = keypoints_2d_xyd[..., :2]

            keypoints_3d = keypoints_3d.cpu().numpy()
            keypoints_2d = keypoints_2d.cpu().numpy()

            # root align
            root_idx = get_keypoint_idx('pelvis_extra', convention='smpl_49')
            keypoints_3d_gta = \
                keypoints_3d_gta - keypoints_3d_gta[:, [root_idx], :]
            keypoints_3d = keypoints_3d - keypoints_3d[:, [root_idx], :]

            for frame_idx in range(num_frames):

                image_path = os.path.join('images', seq_idx,
                                          '{:08d}.jpeg'.format(frame_idx))
                bbox_xywh = ann['bbox_xywh'][frame_idx]

                # reject examples with bbox center outside the frame
                x, y, w, h = bbox_xywh
                x = max([x, 0.0])
                y = max([y, 0.0])
                w = min([w, 1920 - x])  # x + w <= img_width
                h = min([h, 1080 - y])  # y + h <= img_height
                if not (0 <= x < 1920 and 0 <= y < 1080 and 0 < w < 1920
                        and 0 < h < 1080):
                    continue

                image_path_.append(image_path)
                bbox_xywh_.append([x, y, w, h])

                smpl['global_orient'].append(global_orient[frame_idx])
                smpl['body_pose'].append(body_pose[frame_idx])
                smpl['betas'].append(betas[frame_idx])
                smpl['transl'].append(transl[frame_idx])

                keypoints_2d_gta_.append(keypoints_2d_gta[frame_idx])
                keypoints_3d_gta_.append(keypoints_3d_gta[frame_idx])
                keypoints_2d_.append(keypoints_2d[frame_idx])
                keypoints_3d_.append(keypoints_3d[frame_idx])

        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(-1, 3)
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape(-1, 23, 3)
        smpl['betas'] = np.array(smpl['betas']).reshape(-1, 10)
        smpl['transl'] = np.array(smpl['transl']).reshape(-1, 3)
        human_data['smpl'] = smpl

        keypoints2d = np.array(keypoints_2d_).reshape(-1, 49, 2)
        keypoints2d = np.concatenate(
            [keypoints2d, np.ones([keypoints2d.shape[0], 49, 1])], axis=-1)
        keypoints2d, keypoints2d_mask = \
            convert_kps(keypoints2d, src='smpl_49', dst='human_data')
        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask

        keypoints3d = np.array(keypoints_3d_).reshape(-1, 49, 3)
        keypoints3d = np.concatenate(
            [keypoints3d, np.ones([keypoints3d.shape[0], 49, 1])], axis=-1)
        keypoints3d, keypoints3d_mask = \
            convert_kps(keypoints3d, src='smpl_49', dst='human_data')
        human_data['keypoints3d'] = keypoints3d
        human_data['keypoints3d_mask'] = keypoints3d_mask

        keypoints2d_gta = np.array(keypoints_2d_gta_).reshape(-1, 49, 3)
        keypoints2d_gta, keypoints2d_gta_mask = \
            convert_kps(keypoints2d_gta, src='smpl_49', dst='human_data')
        human_data['keypoints2d_gta'] = keypoints2d_gta
        human_data['keypoints2d_gta_mask'] = keypoints2d_gta_mask

        keypoints3d_gta = np.array(keypoints_3d_gta_).reshape(-1, 49, 4)
        keypoints3d_gta, keypoints3d_gta_mask = \
            convert_kps(keypoints3d_gta, src='smpl_49', dst='human_data')
        human_data['keypoints3d_gta'] = keypoints3d_gta
        human_data['keypoints3d_gta_mask'] = keypoints3d_gta_mask

        human_data['image_path'] = image_path_

        bbox_xywh = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh = np.hstack([bbox_xywh, np.ones([bbox_xywh.shape[0], 1])])
        human_data['bbox_xywh'] = bbox_xywh

        human_data['config'] = 'gta_human'
        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'gta_human.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
