import ast
import glob
import itertools
import json
import os
import pdb
import random
import time
from multiprocessing import Pool
from typing import List

import cv2
import numpy as np
import pandas as pd
import smplx
import torch
from pycocotools.coco import COCO
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
from mmhuman3d.models.body_models.utils import transform_to_camera_frame
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class UbodyConverter(BaseModeConverter):

    ACCEPTED_MODES = ['inter', 'intra']

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.misc_config = dict(
            kps3d_root_aligned=False,
            face_bbox='by_dataset',
            hand_bbox='by_dataset',
            bbox='by_dataset',
            flat_hand_mean=False,
        )
        self.smplx_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 21, 3),
            'left_hand_pose': (-1, 15, 3),
            'right_hand_pose': (-1, 15, 3),
            'leye_pose': (-1, 3),
            'reye_pose': (-1, 3),
            'jaw_pose': (-1, 3),
            'expression': (-1, 10)
        }
        self.bbox_mapping = {
            'bbox_xywh': 'bbox',
            'face_bbox_xywh': 'face_box',
            'lhand_bbox_xywh': 'lefthand_box',
            'rhand_bbox_xywh': 'righthand_box'
        }
        self.smplx_mapping = {
            'betas': 'shape',
            'transl': 'trans',
            'global_orient': 'root_pose',
            'body_pose': 'body_pose',
            'left_hand_pose': 'lhand_pose',
            'right_hand_pose': 'rhand_pose',
            'jaw_pose': 'jaw_pose',
            'expression': 'expr'
        }

        super(UbodyConverter, self).__init__(modes)

    def _keypoints_to_scaled_bbox_bfh(self,
                                      keypoints,
                                      occ=None,
                                      body_scale=1.0,
                                      fh_scale=1.0,
                                      convention='smplx'):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale

        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format
        '''
        bboxs = []

        # supported kps.shape: (1, n, k) or (n, k), k = 2 or 3
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        if keypoints.shape[-1] != 2:
            keypoints = keypoints[:, :2]

        for body_part in ['body', 'head', 'left_hand', 'right_hand']:
            if body_part == 'body':
                scale = body_scale
                kps = keypoints
            else:
                scale = fh_scale
                kp_id = get_keypoint_idxs_by_part(
                    body_part, convention=convention)
                kps = keypoints[kp_id]

            if not occ is None:
                occ_p = occ[kp_id]
                if np.sum(occ_p) / len(kp_id) >= 0.1:
                    conf = 0
                    # print(f'{body_part} occluded, occlusion: {np.sum(occ_p) / len(kp_id)}, skip')
                else:
                    # print(f'{body_part} good, {np.sum(self_occ_p + occ_p) / len(kp_id)}')
                    conf = 1
            else:
                conf = 1
            if body_part == 'body':
                conf = 1

            xmin, ymin = np.amin(kps, axis=0)
            xmax, ymax = np.amax(kps, axis=0)

            width = (xmax - xmin) * scale
            height = (ymax - ymin) * scale

            x_center = 0.5 * (xmax + xmin)
            y_center = 0.5 * (ymax + ymin)
            xmin = x_center - 0.5 * width
            xmax = x_center + 0.5 * width
            ymin = y_center - 0.5 * height
            ymax = y_center + 0.5 * height

            bbox = np.stack([xmin, ymin, xmax, ymax, conf],
                            axis=0).astype(np.float32)
            bboxs.append(bbox)

        return bboxs

    def _cal_trunc_mask(self, keypoints2d, meta):

        trunc_mask = np.zeros((keypoints2d.shape[0], keypoints2d.shape[1], 1))

        # check if keypoints2d is inside image
        for idx in range(keypoints2d.shape[0]):
            kps2d = keypoints2d[idx]
            height, width = meta['height'][idx], meta['width'][idx]

            mask_h1 = kps2d[:, 1] > 0
            mask_h2 = kps2d[:, 1] < height
            mask_w1 = kps2d[:, 0] > 0
            mask_w2 = kps2d[:, 0] < width

            mask = mask_h1 * mask_h2 * mask_w1 * mask_w2

            trunc_mask[idx] = mask.reshape(-1, 1)

        return trunc_mask

    def preprocess_ubody(self, vid_p):

        cmd = f'python tools/preprocess/ubody_preprocess.py --vid_p {vid_p}'
        # /home/weichen/zoehuman/mmhuman3d/tools/preprocess/ubody_proprecess.py
        os.system(cmd)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:

        # load scene split and get all video paths
        scene_split = np.load(
            os.path.join(dataset_path, 'splits',
                         f'{mode}_scene_test_list.npy'),
            allow_pickle=True)
        vid_ps_all = glob.glob(
            os.path.join(dataset_path, 'videos', '**', '*.mp4'),
            recursive=True)
        # vid_ps_all_format = glob.glob(os.path.join(dataset_path, 'videos', '**', '*.*'), recursive=True)
        # pdb.set_trace()

        processed_vids = []

        # # build smplx model
        # smplx_model = build_body_model(
        #                 dict(
        #                     type='SMPLX',
        #                     keypoint_src='smplx',
        #                     keypoint_dst='smplx',
        #                     model_path='data/body_models/smplx',
        #                     gender='neutral',
        #                     num_betas=10,
        #                     use_face_contour=True,
        #                     flat_hand_mean=True,
        #                     use_pca=False,
        #                     batch_size=1)).to(self.device)

        # scene_split = scene_split

        test_vids, train_vids = [], []

        # if mode == 'inter':
        #     for scene in scene_split:
        #         if not '_' in scene:
        #             continue
        #         vid_ps = [vid_p for vid_p in vid_ps_all if scene in vid_p]
        #         test_vids += vid_ps
        #     test_vids = list(dict.fromkeys(test_vids))
        #     train_vids = [vid_p for vid_p in vid_ps_all if vid_p not in test_vids]
        # if mode == 'intra':
        #     for scene in scene_split:
        #         if not '_' in scene:
        #             continue
        #         vid_ps = [vid_p for vid_p in vid_ps_all if scene in vid_p]
        #         train_vids += vid_ps[:int((len(vid_ps)+1)*.70)]
        #         test_vids += vid_ps[int((len(vid_ps)+1)*.70):]
        # processed_vids += vid_ps
        # processed_vids = list(dict.fromkeys(processed_vids))

        # for scene in scene_split:
        #     if not '_' in scene:
        #         continue
        #     vid_ps = [vid_p for vid_p in vid_ps_all if scene in vid_p]
        #     test_vids += vid_ps
        # test_vids = list(dict.fromkeys(test_vids))
        # train_vids = [vid_p for vid_p in vid_ps_all if vid_p not in test_vids]

        # vid_ps = vid_ps[:1]

        for batch in ['test', 'train']:

            # num_proc = 4
            # with Pool(num_proc) as p:
            #     r = list(
            #         tqdm(
            #             p.imap(self.preprocess_ubody, vid_ps_batch),
            #             total=len(vid_ps_batch),
            #             desc=f'Scene: {mode} {batch}',
            #             leave=False,
            #             position=1))

            from tools.preprocess.ubody_preprocess import process_vid, process_vid_COCO
            smplx_model = build_body_model(
                dict(
                    type='SMPLX',
                    keypoint_src='smplx',
                    keypoint_dst='smplx',
                    model_path='data/body_models/smplx',
                    gender='neutral',
                    num_betas=10,
                    use_face_contour=True,
                    flat_hand_mean=self.misc_config['flat_hand_mean'],
                    use_pca=False,
                    batch_size=1)).to(self.device)

            vid_batches = [
                'ConductMusic', 'Entertainment', 'Fitness', 'Interview',
                'LiveVlog', 'Magic_show', 'Movie', 'Olympic', 'Online_class',
                'SignLanguage', 'Singing', 'Speech', 'TalkShow', 'TVShow',
                'VideoConference'
            ]
            dst = os.path.join(dataset_path, 'preprocess_db',
                               f'{mode}_{batch}')

            for bid, vscene in enumerate(vid_batches):

                anno_folder = os.path.join(dataset_path, 'annotations', vscene)
                # load seq kp annotation
                # with open(os.path.join(anno_folder, 'keypoint_annotation.json')) as f:
                #     anno_param = json.load(f)
                db = COCO(
                    os.path.join(anno_folder, 'keypoint_annotation.json'))
                # load seq smplx annotation
                with open(os.path.join(anno_folder,
                                       'smplx_annotation.json')) as f:
                    smplx_param = json.load(f)
                # for vid_p in tqdm(vid_ps_scene, desc=f'Preprocessing scene {bid+1}/{len(vid_batches)}',
                #                   position=0, leave=False):
                # process_vid(vid_p, smplx_model, anno_param, smplx_param)

                process_vid_COCO(smplx_model, vscene, scene_split, batch, db,
                                 smplx_param, dst)

                # pdb.set_trace()

            processed_npzs = glob.glob(os.path.join(dst, '*.npz'))
            seed, size = '240409', '99'

            random.seed(int(seed))

            size_i = len(processed_npzs)

            slices = 1
            if batch == 'train':
                slices = 5
            elif batch == 'test':
                slices = 3
            print(f'Seperate in to {slices} files')

            slice_npzs = int(len(processed_npzs) / slices) + 1

            for slice_idx in range(slices):
                # use HumanData to store all data
                human_data = HumanData()

                # initialize output for human_data
                smplx_ = {}
                for keys in self.smplx_shape.keys():
                    smplx_[keys] = []
                keypoints2d_, keypoints3d_, keypoints2d_ubody_ = [], [], []
                bboxs_ = {}
                for bbox_name in [
                        'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                        'rhand_bbox_xywh'
                ]:
                    bboxs_[bbox_name] = []
                meta_ = {}
                for meta_key in [
                        'principal_point',
                        'focal_length',
                        'height',
                        'width',
                        'iscrowd',
                        'num_keypoints',
                        'valid_label',
                        'lefthand_valid',
                        'righthand_valid',
                        'face_valid',
                ]:
                    meta_[meta_key] = []
                image_path_ = []

                # pdb.set_trace()
                for npzp in tqdm(
                        processed_npzs[slice_npzs * slice_idx:slice_npzs *
                                       (slice_idx + 1)]):

                    # load param dict
                    try:
                        param_dict = dict(np.load(npzp, allow_pickle=True))
                    except:
                        print(f'Error in loading {npzp}')
                        continue

                    image_path = param_dict['image_path'].tolist()
                    image_path_ += image_path

                    # append to humandata
                    # bbox
                    for bbox_name in [
                            'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                            'rhand_bbox_xywh'
                    ]:
                        bboxs_[bbox_name] += param_dict[bbox_name].tolist()

                    # keypoints
                    keypoints2d_ += param_dict['keypoints2d'].tolist()
                    keypoints2d_ubody_ += param_dict[
                        'keypoints2d_ubody'].tolist()
                    keypoints3d_ += param_dict['keypoints3d'].tolist()

                    # smplx
                    for smplx_key in self.smplx_mapping.keys():
                        smplx_[smplx_key] += param_dict[smplx_key].tolist()

                    # meta
                    meta_['height'] += param_dict['height'].tolist()
                    meta_['width'] += param_dict['width'].tolist()
                    meta_['focal_length'] += param_dict['focal_length'].tolist(
                    )
                    meta_['principal_point'] += param_dict[
                        'principal_point'].tolist()

                    for key in [
                            'iscrowd', 'num_keypoints', 'valid_label',
                            'lefthand_valid', 'righthand_valid', 'face_valid'
                    ]:
                        meta_[key] += param_dict[key].tolist()

                    size_i += len(image_path)

                    # del param_dict
                    # pdb.set_trace()

                # prepare for output
                # smplx
                for key in smplx_.keys():
                    smplx_[key] = np.array(smplx_[key]).reshape(
                        self.smplx_shape[key])
                human_data['smplx'] = smplx_
                print('Smpl and/or Smplx finished at',
                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

                # bbox
                for key in bboxs_.keys():
                    bbox_ = np.array(bboxs_[key]).reshape((-1, 5))
                    human_data[key] = bbox_
                print('BBox generation finished at',
                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

                # keypoints 2d
                keypoints2d = np.array(keypoints2d_).reshape(-1, 144, 2)
                keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
                keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf],
                                             axis=-1)
                keypoints2d, keypoints2d_mask = \
                        convert_kps(keypoints2d, src='smplx', dst='human_data')
                human_data['keypoints2d_smplx'] = keypoints2d
                human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

                # get keypoints2d trunc mask
                keypoints2d_smplx_trunc_mask = self._cal_trunc_mask(
                    keypoints2d, meta_)
                human_data[
                    'keypoints2d_smplx_trunc_mask'] = keypoints2d_smplx_trunc_mask

                # keypoints 3d
                keypoints3d = np.array(keypoints3d_).reshape(-1, 144, 3)
                keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
                keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf],
                                             axis=-1)
                keypoints3d, keypoints3d_mask = \
                        convert_kps(keypoints3d, src='smplx', dst='human_data')
                human_data['keypoints3d_smplx'] = keypoints3d
                human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

                # keypoints 2d ubody
                keypoints2d_ubody = np.array(keypoints2d_ubody_).reshape(
                    -1, 133, 3)
                keypoints2d_ubody_conf = np.ones(
                    [keypoints2d_ubody.shape[0], 133, 1])
                keypoints2d_ubody = np.concatenate(
                    [keypoints2d_ubody, keypoints2d_ubody_conf], axis=-1)
                keypoints2d_ubody, keypoints2d_ubody_mask = \
                        convert_kps(keypoints2d_ubody, src='coco_wholebody', dst='human_data')
                human_data['keypoints2d_ubody'] = keypoints2d_ubody
                human_data['keypoints2d_ubody_mask'] = keypoints2d_ubody_mask

                # image path
                human_data['image_path'] = image_path_
                print('Image path writing finished at',
                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

                # meta
                human_data['meta'] = meta_
                print('Meta writing finished at',
                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

                # store
                human_data['config'] = f'ubody_{mode}'
                human_data['misc'] = self.misc_config

                human_data.compress_keypoints_by_mask()
                os.makedirs(out_path, exist_ok=True)
                out_file = os.path.join(
                    out_path,
                    f'ubody_{mode}_{batch}_{seed}_{"{:02d}".format(size_i)}_{slice_idx}.npz'
                )
                human_data.dump(out_file)
