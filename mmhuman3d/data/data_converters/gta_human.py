import glob
import os
import pickle
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

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


def perspective_projection(points: torch.Tensor, focal_length: float,
                           camera_center: Tuple):
    """This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.

    K[:, 0, -1] = camera_center[0]
    K[:, 1, -1] = camera_center[1]

    points = torch.transpose(points, 1, 2)  # (bs, 3, N)
    projected_points = torch.matmul(K, points)  # (bs, 3, N)
    projected_points = torch.transpose(projected_points, 1, 2)  # (bs, N, 3)
    projected_points = projected_points[:, :, :
                                        2] / projected_points[:, :,
                                                              2].unsqueeze(
                                                                  -1
                                                              )  # (bs, N, 2)

    return projected_points


@DATA_CONVERTERS.register_module()
class GTAHumanConverter(BaseConverter):
    """GTA-Human dataset `Playing for 3D Human Recovery' arXiv`2021 More
    details can be found in the `paper.

    <https://arxiv.org/pdf/2110.07588.pdf>`__.
    """

    focal_length = 1158.0337
    camera_center = (960, 540)

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

        body_pose_list = []
        global_orient_list = []
        betas_list = []
        transl_list = []

        image_path_list = []
        bbox_xywh_list = []

        keypoints_2d_gta_list = []
        keypoints_3d_gta_list = []
        keypoints_2d_list = []
        keypoints_3d_list = []

        anno_paths = sorted(glob.glob(
            os.path.join(dataset_path, 'annotations', '*.pkl')))

        for anno_path in tqdm(anno_paths):

            with open(anno_path, 'rb') as f:
                anno = pickle.load(f, encoding='latin1')

            base = os.path.basename(anno_path)  # -> seq_00000001.pkl
            seq_idx, _ = os.path.splitext(base)  # -> seq_00000001
            num_frames = len(anno['body_pose'])

            # keypoints_2d_gta_openpose = map_joints(
            # annotation['keypoints_2d'],
            # joint_mapper_gta_to_openpose,
            # (num_frames, 25, 3))  # 25 joints, (x, y, conf)
            # keypoints_2d_gta_joint24 = map_joints(annotation['keypoints_2d'],
            # joint_mapper_gta_to_joint24,
            # (num_frames, 24, 3))  # 24 joints, (x, y, conf)
            # keypoints_3d_gta_joint24 = map_joints(annotation['keypoints_3d'],
            # joint_mapper_gta_to_joint24,
            # (num_frames, 24, 4))  # 24 joints, (x, y, z, conf)

            keypoints_2d_gta, keypoints_2d_gta_mask = convert_kps(
                anno['keypoints_2d'], src='gta', dst='smpl_49')
            keypoints_3d_gta, keypoints_3d_gta_mask = convert_kps(
                anno['keypoints_3d'], src='gta', dst='smpl_49')

            global_orient = anno['global_orient']
            body_pose = anno['body_pose']
            betas = anno['betas']
            transl = anno['transl']

            # keypoints_2d_smpl_openpose, \
            # keypoints_2d_smpl_joint24, \
            # keypoints_3d_smpl_openpose, \
            # keypoints_3d_smpl_joint24 = \
            #     extract_keypoints_from_smpl(smpl,
            #     global_orient=torch.tensor(global_orient, device=device),
            #     body_pose=torch.tensor(body_pose, device=device),
            #     betas=torch.tensor(betas, device=device),
            #     transl=torch.tensor(transl, device=device)
            #     )

            output = self.smpl(
                global_orient=torch.tensor(global_orient, device=self.device),
                body_pose=torch.tensor(body_pose, device=self.device),
                betas=torch.tensor(betas, device=self.device),
                transl=torch.tensor(transl, device=self.device),
                return_joints=True)

            keypoints_3d = output['joints']
            keypoints_2d = perspective_projection(keypoints_3d,
                                                  self.focal_length,
                                                  self.camera_center)
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
                bbox_xywh = anno['bbox_xywh'][frame_idx]

                # reject examples with bbox center outside the frame
                x, y, w, h = bbox_xywh
                x = max([x, 0.0])
                y = max([y, 0.0])
                w = min([w, 1920 - x])  # x + w <= img_width
                h = min([h, 1080 - y])  # y + h <= img_height
                if not (0 <= x < 1920 and 0 <= y < 1080 and 0 < w < 1920
                        and 0 < h < 1080):
                    continue

                # center = [x + w / 2.0, y + h / 2.0]
                # scale = max(w, h)

                # keypoints_2d_gta_openpose_frame =
                # keypoints_2d_gta_openpose[frame_idx]
                # keypoints_2d_gta_joint24_frame =
                # keypoints_2d_gta_joint24[frame_idx]
                # keypoints_3d_gta_joint24_frame =
                # keypoints_3d_gta_joint24[frame_idx]
                #
                # keypoints_2d_smpl_openpose_frame =
                # keypoints_2d_smpl_openpose[frame_idx]
                # keypoints_2d_smpl_joint24_frame =
                # keypoints_2d_smpl_joint24[frame_idx]
                # keypoints_3d_smpl_joint24_frame =
                # keypoints_3d_smpl_joint24[frame_idx]

                # # re-center 3d keypoints with the pelvis as the origin
                # keypoints_3d_gta_joint24_frame[:, :3] =
                # keypoints_3d_gta_joint24_frame[:, :3] -
                # keypoints_3d_gta_joint24_frame[14, :3]
                # keypoints_3d_smpl_joint24_frame[:, :3] =
                # keypoints_3d_smpl_joint24_frame[:,:3] -
                # keypoints_3d_smpl_joint24_frame[14, :3]

                # # GTA 2D and 3D keypoints
                # keypoints2d_gta.append(keypoints_2d_gta_joint24_frame)
                # keypoints3d_gta.append(keypoints_3d_gta_joint24_frame)
                #
                # # SMPL-derived 2D and 3D keypoints
                # keypoints2d_smpl_.append(keypoints_2d_smpl_joint24_frame)
                # keypoints3d_smpl_.append(keypoints_3d_smpl_joint24_frame)
                # openpose_smpl_.append(keypoints_2d_smpl_openpose_frame)

                image_path_list.append(image_path)
                # center_list.append(center)
                # scale_list.append(scale)
                bbox_xywh_list.append([x, y, w, h])

                global_orient_list.append(global_orient[frame_idx])
                body_pose_list.append(body_pose[frame_idx])
                betas_list.append(betas[frame_idx])
                transl_list.append(transl[frame_idx])

                keypoints_2d_gta_list.append(keypoints_2d_gta[frame_idx])
                keypoints_3d_gta_list.append(keypoints_3d_gta[frame_idx])
                keypoints_2d_list.append(keypoints_2d[frame_idx])
                keypoints_3d_list.append(keypoints_3d[frame_idx])

        human_data['config'] = 'gta_human'

        smpl = {}
        smpl['global_orient'] = np.array(global_orient_list).reshape(-1, 3)
        smpl['body_pose'] = np.array(body_pose_list).reshape(-1, 23, 3)
        smpl['betas'] = np.array(betas_list).reshape(-1, 10)
        smpl['transl'] = np.array(transl_list).reshape(-1, 3)
        human_data['smpl'] = smpl

        keypoints2d = np.array(keypoints_2d_list).reshape(-1, 49, 2)
        keypoints2d = np.concatenate(
            [keypoints2d, np.ones([keypoints2d.shape[0], 49, 1])], axis=-1)
        keypoints2d, keypoints2d_mask = \
            convert_kps(keypoints2d, src='smpl_49', dst='human_data')
        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask

        keypoints3d = np.array(keypoints_3d_list).reshape(-1, 49, 3)
        keypoints3d = np.concatenate(
            [keypoints3d, np.ones([keypoints3d.shape[0], 49, 1])], axis=-1)
        keypoints3d, keypoints3d_mask = \
            convert_kps(keypoints3d, src='smpl_49', dst='human_data')
        human_data['keypoints3d'] = keypoints3d
        human_data['keypoints3d_mask'] = keypoints3d_mask

        keypoints2d_gta = np.array(keypoints_2d_gta_list).reshape(-1, 49, 3)
        # keypoints2d_gta = np.concatenate(
        #     [keypoints2d_gta,
        #      np.ones([keypoints2d_gta.shape[0], 49, 1])],
        #     axis=-1)
        keypoints2d_gta, keypoints2d_gta_mask = \
            convert_kps(keypoints2d_gta, src='smpl_49', dst='human_data')
        human_data['keypoints2d_gta'] = keypoints2d_gta
        human_data['keypoints2d_gta_mask'] = keypoints2d_gta_mask

        keypoints3d_gta = np.array(keypoints_3d_gta_list).reshape(-1, 49, 4)
        # keypoints3d_gta = np.concatenate(
        #     [keypoints3d_gta,
        #      np.ones([keypoints3d_gta.shape[0], 49, 1])],
        #     axis=-1)
        keypoints3d_gta, keypoints3d_gta_mask = \
            convert_kps(keypoints3d_gta, src='smpl_49', dst='human_data')
        human_data['keypoints3d_gta'] = keypoints3d_gta
        human_data['keypoints3d_gta_mask'] = keypoints3d_gta_mask

        human_data['image_path'] = image_path_list

        bbox_xywh = np.array(bbox_xywh_list).reshape((-1, 4))
        bbox_xywh = np.hstack([bbox_xywh, np.ones([bbox_xywh.shape[0], 1])])
        human_data['bbox_xywh'] = bbox_xywh

        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'gta_human.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
