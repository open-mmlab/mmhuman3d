import ast
import glob
import json
import os
import pdb
import random
import time
from typing import List

import cv2
import numpy as np
import pandas as pd
import smplx
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData
# import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import (
    batch_transform_to_camera_frame,
    transform_to_camera_frame,
)
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class BlurhandConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train', 'test']

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.misc_config = dict(
            bbox_source='original',
            smplx_source='original',
            flat_hand_mean=False,
            camera_param_type='perspective',
            kps3d_root_aligned=False,
            image_shape=(512, 334),  # height, width
        )
        self.smplx_shape = {
            'left_hand_pose': (-1, 15, 3),
            'right_hand_pose': (-1, 15, 3),
        }
        self.mano_shape = {
            'pose': (-1, 15, 3),
            'betas': (-1, 10),
            'global_orient': (-1, 3),
            'transl': (-1, 3),
        }

        super(BlurhandConverter, self).__init__(modes)

    # def _generate_patch_image(self, cvimg, bbox, do_flip, scale, rot, out_shape):
    #     img = cvimg.copy()
    #     img_height, img_width, img_channels = img.shape

    #     bb_c_x = float(bbox[0] + 0.5*bbox[2])
    #     bb_c_y = float(bbox[1] + 0.5*bbox[3])
    #     bb_width = float(bbox[2])
    #     bb_height = float(bbox[3])

    #     if do_flip:
    #         img = img[:, ::-1, :]
    #         bb_c_x = img_width - bb_c_x - 1

    #     trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    #     img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    #     img_patch = img_patch.astype(np.float32)
    #     inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    #     return img_patch, trans, inv_trans

    def _get_bbox_from_joints(self, joint_img, joint_valid):

        original_image_shape = self.misc_config['image_shape']

        x_img = joint_img[:, 0][joint_valid == 1]
        y_img = joint_img[:, 1][joint_valid == 1]
        xmin = min(x_img)
        ymin = min(y_img)
        xmax = max(x_img)
        ymax = max(y_img)

        x_center = (xmin + xmax) / 2.
        width = xmax - xmin
        xmin = x_center - 0.5 * width * 1.2
        xmax = x_center + 0.5 * width * 1.2

        y_center = (ymin + ymax) / 2.
        height = ymax - ymin
        ymin = y_center - 0.5 * height * 1.2
        ymax = y_center + 0.5 * height * 1.2

        bbox = np.array([xmin, ymin, xmax - xmin,
                         ymax - ymin]).astype(np.float32)

        # aspect ratio preserving bbox
        w = bbox[2]
        h = bbox[3]
        c_x = bbox[0] + w / 2
        c_y = bbox[1] + h / 2
        aspect_ratio = 1  # (256, 256)
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        bbox[2] = w * 1.25
        bbox[3] = h * 1.25
        bbox[0] = c_x - bbox[2] / 2.
        bbox[1] = c_y - bbox[3] / 2.

        return bbox  # bbox_xyxy

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        print('Converting Blurhand dataset...')

        # annotation dir
        anno_dir = os.path.join(dataset_path, 'annotations', mode)
        anno_mano_path = os.path.join(
            anno_dir, f'BlurHand_{mode}_MANO_NeuralAnnot.json')
        anno_j3d_path = os.path.join(anno_dir,
                                     f'BlurHand_{mode}_joint_3d.json')
        anno_cam_path = os.path.join(anno_dir,
                                     f'BlurHand_{mode}_camera.json')
        anno_data_path = os.path.join(
            anno_dir, f'BlurHand_{mode}_data_reformat.json')

        # load annotations
        with open(anno_mano_path, 'r') as f:
            anno_mano = json.load(f)
        with open(anno_j3d_path, 'r') as f:
            anno_j3d = json.load(f)
        with open(anno_cam_path, 'r') as f:
            anno_cam = json.load(f)
        with open(anno_data_path, 'r') as f:
            anno_data = json.load(f)

        # parse sequences
        seqs = glob.glob(
            os.path.join(dataset_path, f'images', mode, 'Capture*',
                         '*', 'cam*'))

        # use HumanData to store the data
        human_data = HumanData()

        seed = '230828'
        size = 999999

        # initialize
        smplx_ = {}
        for hand_type in ['left', 'right']:
            smplx_[f'{hand_type}_hand_pose'] = []
        bboxs_ = {}
        for hand_type in ['left', 'right']:
            bboxs_[f'{hand_type[0]}hand_bbox_xywh'] = []
        bboxs_['bbox_xywh'] = []
        image_path_, keypoints2d_smplx_ = [], []
        keypoints3d_smplx_ = []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length']:
            meta_[meta_key] = []
        # save mano params for vis purpose
        mano_ = []

        # pdb.set_trace()
        # seqs = seqs[:100]
        # sort by image path
        for seq in tqdm(
                seqs,
                desc=f'Processing {mode}',
                total=len(seqs),
                position=0,
                leave=False):
            seq_name = seq.split(os.path.sep)[-2]

            camera_id = seq.split(os.path.sep)[-1][3:]
            capture_id = seq.split(os.path.sep)[-3][7:]

            img_path_list = glob.glob(os.path.join(seq, '*.png'))

            # assume image shape is same in a sequence
            # img = cv2.imread(img_path_list[0])
            height, width = self.misc_config['image_shape']

            for image_path in tqdm(
                    img_path_list,
                    desc=f'Processing Capture{capture_id} '
                    f'{seq_name} Camera{camera_id}',
                    total=len(img_path_list),
                    position=1,
                    leave=False):

                # get all indexs
                frame_idx = image_path.split(os.path.sep)[-1][5:-4]
                frame_idx = str(int(frame_idx))

                image_path = image_path.replace(dataset_path + os.path.sep, '')

                anno_info = anno_data[image_path.replace(
                    f'images{os.path.sep}{mode}{os.path.sep}', '')]

                hand_param = {}
                for hand_type in ['left', 'right']:
                    # get hand pose
                    try:
                        mano_param = anno_mano[capture_id][frame_idx][
                            hand_type]
                        if mano_param is None:
                            continue
                        hand_param[hand_type] = mano_param
                    except KeyError:
                        continue

                # get joints 3d
                joints3d_world = anno_j3d[capture_id][frame_idx]['world_coord']

                # get bbox
                # pdb.set_trace()
                # bbox_xyxy = anno_info['bbox']
                # bbox_xywh = self._xyxy2xywh(bbox_xyxy)
                bbox_xywh = anno_info['bbox']

                # bug exist in projection

                # get camera params
                t = np.array(
                    anno_cam[capture_id]['campos'][str(camera_id)],
                    dtype=np.float32).reshape(3)
                R = np.array(
                    anno_cam[capture_id]['camrot'][str(camera_id)],
                    dtype=np.float32).reshape(3, 3)
                focal = np.array(
                    anno_cam[capture_id]['focal'][str(camera_id)],
                    dtype=np.float32).reshape(2)
                princpt = np.array(
                    anno_cam[capture_id]['princpt'][str(camera_id)],
                    dtype=np.float32).reshape(2)

                T = -np.dot(R, t.reshape(3, 1)).reshape(3)  # -Rt -> t
                j3d_w = np.array(
                    joints3d_world, dtype=np.float32).reshape(-1, 3)
                j3d_c = np.dot(R, j3d_w.transpose(1, 0)).transpose(
                    1, 0) + T.reshape(1, 3)

                # project 3d joints to 2d
                princpt = (princpt[0], princpt[1])
                focal = (focal[0], focal[1])
                camera = build_cameras(
                    dict(
                        type='PerspectiveCameras',
                        convention='opencv',
                        in_ndc=False,
                        focal_length=focal,
                        principal_point=princpt,
                        image_size=(height, width),
                    )).to(self.device)

                j2d = camera.transform_points_screen(
                    torch.tensor(j3d_c.reshape(1, -1, 3), device=self.device))
                j2d_orig = j2d[0, :, :2].detach().cpu().numpy()
                j2d_conf = np.array(anno_info['joint_valid'])

                # append keypoints
                j2d_orig = np.concatenate(
                    [j2d_orig, j2d_conf.reshape(-1, 1)], axis=-1)
                j3d_orig = np.concatenate(
                    [j3d_c, j2d_conf.reshape(-1, 1)], axis=-1)
                keypoints2d_smplx_.append(j2d_orig)
                keypoints3d_smplx_.append(j3d_orig)

                # append
                image_path_.append(image_path)

                # append bbox
                available_hand_types = list(hand_param.keys())
                for hand_type in ['left', 'right']:
                    if hand_type in available_hand_types:
                        global_orient = np.array(
                            hand_param[hand_type]['pose'][:3])
                        smplx_[f'{hand_type}_hand_pose'].append(
                            hand_param[hand_type]['pose'][3:])
                        bboxs_[f'{hand_type[0]}hand_bbox_xywh'].append(
                            bbox_xywh + [1])

                    else:
                        smplx_[f'{hand_type}_hand_pose'].append(
                            np.zeros((45)).tolist())
                        bboxs_[f'{hand_type[0]}hand_bbox_xywh'].append(
                            bbox_xywh + [0])
                bboxs_['bbox_xywh'].append([0, 0, width, height, 1])

                meta_['focal_length'].append(focal)
                meta_['principal_point'].append(princpt)

                # append mano params
                mano_.append(hand_param)

                # # test overlay j2d
                # img = cv2.imread(f'{dataset_path}/{image_path}')
                # for i in range(len(j2d_orig)):
                #     cv2.circle(img, (int(j2d_orig[i,0]), int(j2d_orig[i,1])), 3, (0,0,255), -1)
                #     pass
                # # write image
                # os.makedirs(f'{out_path}/{mode}', exist_ok=True)
                # cv2.imwrite(f'{out_path}/{mode}/{seq_name}_{frame_idx}_{hand_type}.jpg', img)

        size_i = min(size, len(image_path_))

        # meta
        human_data['meta'] = meta_

        # image path
        human_data['image_path'] = image_path_

        # save bbox
        for bbox_name in bboxs_.keys():
            bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 5)
            human_data[bbox_name] = bbox_

        # save mano
        human_data['mano'] = mano_

        # save smplx
        human_data.skip_keys_check = ['smplx']
        for key in smplx_.keys():
            smplx_[key] = np.concatenate(
                smplx_[key], axis=0).reshape(self.smplx_shape[key])
        human_data['smplx'] = smplx_

        # keypoints2d_smplx
        keypoints2d_smplx = np.concatenate(
            keypoints2d_smplx_, axis=0).reshape(-1, 42, 3)
        keypoints2d_smplx, keypoints2d_smplx_mask = \
                convert_kps(keypoints2d_smplx, src='mano_hands', dst='human_data')
        human_data['keypoints2d_smplx'] = keypoints2d_smplx
        human_data['keypoints2d_smplx_mask'] = keypoints2d_smplx_mask

        # keypoints3d_smplx
        keypoints3d_smplx = np.concatenate(
            keypoints3d_smplx_, axis=0).reshape(-1, 42, 4)
        keypoints3d_smplx, keypoints3d_smplx_mask = \
                convert_kps(keypoints3d_smplx, src='mano_hands', dst='human_data')
        human_data['keypoints3d_smplx'] = keypoints3d_smplx
        human_data['keypoints3d_smplx_mask'] = keypoints3d_smplx_mask

        # misc
        human_data['misc'] = self.misc_config
        human_data['config'] = f'blurhand_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        size_i = min(len(seqs), int(size))
        out_file = os.path.join(out_path,
            f'blurhand_{mode}_{seed}_{"{:06d}".format(size_i)}.npz')
        human_data.dump(out_file)
