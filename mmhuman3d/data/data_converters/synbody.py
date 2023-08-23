import glob
import json
import os
import pdb
import random
import time
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

# import mmcv
# from mmhuman3d.models.body_models.builder import build_body_model
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class SynbodyConverter(BaseModeConverter):
    """Synbody dataset."""
    ACCEPTED_MODES = ['v0_train', 'v0_ehf', 'v0_amass',
                      'v0_agora', 'v0_renew', 'v1_train']

    def __init__(self, modes: List = []) -> None:

        self.smpl_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 23, 3)
        }
        self.smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 21, 3), \
                       'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 'leye_pose': (-1, 3),
                       'reye_pose': (-1, 3), \
                       'jaw_pose': (-1, 3), 'expression': (-1, 10)}
        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            flat_hand_mean=False,
            bbox_source='keypoints2d_original',
            kps3d_root_aligned=False,
            cam_param_type='perspective',
            smplx_source='original',
            focal_length=(640, 640),
            principal_point=(640, 360),
            fps=30)

        super(SynbodyConverter, self).__init__(modes)

    def _keypoints_to_scaled_bbox_fh(self,
                                     keypoints,
                                     occ=None,
                                     scale=1.0,
                                     convention='smplx'):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale

        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format
        '''
        bboxs = []
        for body_part in ['head', 'left_hand', 'right_hand']:
            kp_id = get_keypoint_idxs_by_part(body_part, convention=convention)

            # keypoints_factory=smplx.SMPLX_KEYPOINTS)
            kps = keypoints[kp_id]

            if occ == None:
                conf = 1
            else:
                occ_p = occ[kp_id]

                if np.sum(occ_p) / len(kp_id) >= 0.1:
                    conf = 0
                    # print(f'{body_part} occluded, occlusion: {np.sum(occ_p) / len(kp_id)}, skip')
                else:
                    # print(f'{body_part} good, {np.sum(self_occ_p + occ_p) / len(kp_id)}')
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
        return bboxs[0], bboxs[1], bboxs[2]

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
                keypoints2d_mask, keypoints3d, keypoints3d_mask, cam_param
                stored in HumanData() format
        """
        # get targeted sequence list
        root_dir, prefix = os.path.split(dataset_path)
        preprocessed_dir = os.path.join(dataset_path, 'preprocess', mode)
        npzs = glob.glob(os.path.join(preprocessed_dir, '*', '*', '*.npz'))

        # init random seed
        slice_num = 18
        seed, size = '230804', '94000'
        random.seed(int(seed))
        random.shuffle(npzs)
        # npzs = sorted(npzs)
        npzs = npzs[:int(size)]

        # pdb.set_trace()

        size_n = min(int(size), len(npzs))

        print(f'Seperate in to {slice_num} files')

        slice = int(len(npzs) / slice_num) + 1
        for s in range(slice_num):

            # use HumanData to store all data
            human_data = HumanData()

            # get data shape
            npfile = dict(np.load(npzs[0], allow_pickle=True))
            kp_shape = npfile['keypoints2d'].shape[1]

            # initialize storage
            _keypoints2d = np.array([]).reshape(0, kp_shape, 3)
            _keypoints3d = np.array([]).reshape(0, kp_shape, 4)
            _bboxs = {}
            _meta = {}
            _meta['gender'] = []
            for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
            ]:
                _bboxs[bbox_name] = []
            _image_path = []
            _keypoints2d_list, _keypoints3d_list = [], []

            # initialize smpl and smplx
            _smpl, _smplx = {}, {}

            for key in self.smpl_shape:
                _smpl[key] = np.array([]).reshape(self.smpl_shape[key])
            for key in self.smplx_shape:
                _smplx[key] = np.array([]).reshape(self.smplx_shape[key])
            _smpl_l, _smplx_l = {}, {}
            for key in self.smpl_shape:
                _smpl_l[key] = []
            for key in self.smplx_shape:
                _smplx_l[key] = []

            for npzf in tqdm(npzs[slice * s:slice * (s + 1)], desc='Npzfiles concating'):
                try:
                    npfile = dict(np.load(npzf, allow_pickle=True))

                    # (width, height) = npfile['shape']
                    if 'shape' in npfile.keys():
                        (width, height) = npfile['shape']
                    else:
                        (width, height) = (1280, 720)

                    # seq_folder_id = npzf.split('/').index('preprocessed')
                    # synbody_path = '/mnt/lustre/share_data/meihaiyi/shared_data/'
                    # seq_p = npzf.split('/')[seq_folder_id+1:]
                    # seq_p[-1] = seq_p[-1][:-4]
                    # occ_fp = os.path.join(synbody_path, 'SynBody', '/'.join(seq_p))
                    # image_idx = [int(x[-9: -5]) for x in  npfile['image_path']]
                    # occs_ = []
                    # for idx, i in enumerate(image_idx):
                    #     occ_file = os.path.join(occ_fp, 'occlusion', npfile['npz_name'][idx])
                    #     occ = np.load(occ_file)['occlusion'][i]
                    #     occs_.append(occ)

                    # occs_ = npfile['occlusion']

                    # pdb.set_trace()
                    # os._exit(0)
                    bbox_ = []
                    # bbox_ = npfile['bbox']
                    keypoints2d_ = npfile['keypoints2d'].reshape(
                        len(npfile['image_path']), -1, 2)
                    keypoints3d_ = npfile['keypoints3d'].reshape(
                        len(npfile['image_path']), -1, 3)

                    # root centered
                    valid_id = []
                    conf = npfile['conf']
                    pelvis = keypoints3d_[:, 0, :]

                    for i in range(len(conf)):
                        if conf[i][0] > 0:
                            valid_id.append(i)
                    if len(valid_id) == 0:
                        raise ValueError('No good keypoints found, skip!!')
                    valid_id = np.array(valid_id)
                    # keypoints3d_[:, :, :] -= pelvis[:, None, :]
                    # print('Root centered finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

                except Exception as e:
                    print(f'{npzf} failed because of {e}')
                    continue

                # for kp in keypoints2d_[valid_id]:
                # since the 2d keypoints are not strictly corrcet, a large scale factor is used
                for idx in valid_id:
                    kp = keypoints2d_[idx]
                    # occ = occs_[idx]
                    occ = None
                    bbox_tmp_ = {}
                    bbox_tmp_['bbox_xywh'] = self._keypoints_to_scaled_bbox(
                        kp, 1.2)
                    bbox_tmp_['face_bbox_xywh'], bbox_tmp_[
                        'lhand_bbox_xywh'], bbox_tmp_[
                            'rhand_bbox_xywh'] = self._keypoints_to_scaled_bbox_fh(
                                kp, occ=occ, scale=1.0)
                    for bbox_name in [
                            'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                            'rhand_bbox_xywh'
                    ]:
                        bbox = bbox_tmp_[bbox_name]
                        xmin, ymin, xmax, ymax = bbox[:4]
                        if bbox_name == 'bbox_xywh':
                            bbox_conf = 1
                        else:
                            bbox_conf = bbox[-1]
                            # if bbox_conf == 0:
                            #     print(f'{npzf}, {idx},{bbox_name} invalid')
                        # import pdb; pdb.set_trace()
                        bbox = np.array([
                            max(0, xmin),
                            max(0, ymin),
                            min(width, xmax),
                            min(height, ymax)
                        ])
                        bbox_xywh = self._xyxy2xywh(bbox)
                        bbox_xywh.append(bbox_conf)

                        # pdb.set_trace()

                        # print(f'{bbox_name}: {bbox_xywh}')

                        # bbox_xywh = [0, 0, 0, 0, 0]

                        _bboxs[bbox_name].append(bbox_xywh)

                image_path_ = []
                for imp in npfile['image_path']:
                    imp = imp.split('/')
                    image_path_.append('/'.join(imp[1:]))
                _image_path += np.array(image_path_)[valid_id].tolist()

                # handling keypoints
                keypoints2d_ = np.concatenate((keypoints2d_, conf), axis=2)
                keypoints3d_ = np.concatenate((keypoints3d_, conf), axis=2)

                conf = np.ones_like(keypoints2d_[valid_id][..., 0:1])
                # remove_kp = [39, 35, 38, 23, 36, 37, 41, 44, 42, 43, 22, 20, 18]
                # conf[:, remove_kp, :] = 0

                if 'smpl' in npfile.keys():
                    for k in npfile['smpl'].item().keys():
                        # _smpl[k] = np.concatenate((_smpl[k], npfile['smpl'].item()[k][valid_id]
                        #   .reshape(smpl_shape[k])), axis=0)
                        _smpl_l[k].append(npfile['smpl'].item()[k][valid_id].reshape(
                            self.smpl_shape[k]))
                if 'smplx' in npfile.keys():
                    for k in npfile['smplx'].item().keys():
                        # _smplx[k] = np.concatenate((_smplx[k], npfile['smplx'].item()[k][valid_id]
                        #   .reshape(smplx_shape[k])), axis=0)
                        _smplx_l[k].append(npfile['smplx'].item()[k][valid_id].reshape(
                            self.smplx_shape[k]))
                # pdb.set_trace()
                gender = []
                for idx, meta_tmp in enumerate(npfile['meta'][valid_id]):
                    gender.append(meta_tmp['gender'])

                _meta['gender'] += gender
                # pdb.set_trace()

                # _meta['gender'].append(np.array(gender)[valid_id].tolist())
                # _keypoints2d = np.concatenate((_keypoints2d, keypoints2d_[valid_id]), axis=0)
                # _keypoints3d = np.concatenate((_keypoints3d, keypoints3d_[valid_id]), axis=0)
                _keypoints2d_list.append(keypoints2d_[valid_id])
                _keypoints3d_list.append(keypoints3d_[valid_id])

            print('Starts concatenating...')

            human_data['image_path'] = _image_path
            print('Image path writing finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            for key in _bboxs.keys():
                bbox_ = np.array(_bboxs[key]).reshape((-1, 5))
                # bbox_ = np.hstack([bbox_, np.zeros([bbox_.shape[0], 1])])
                # import pdb; pdb.set_trace()
                human_data[key] = bbox_
            print('BBox generation finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            # for kp_set in _keypoints2d_list:
            #     _keypoints2d = np.concatenate((_keypoints2d, kp_set), axis=0)
            _keypoints2d = np.concatenate(_keypoints2d_list, axis=0)
            _keypoints3d = np.concatenate(_keypoints3d_list, axis=0)
            _keypoints2d, mask = convert_kps(_keypoints2d, 'smplx', 'human_data')
            _keypoints3d, mask = convert_kps(_keypoints3d, 'smplx', 'human_data')

            human_data['keypoints2d_smplx'] = _keypoints2d
            human_data['keypoints3d_smplx'] = _keypoints3d
            human_data['keypoints2d_smplx_mask'] = mask
            human_data['keypoints3d_smplx_mask'] = mask
            print('Keypoint conversion finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            for key in _smpl.keys():
                # for model_param in _smpl_l[key]:
                _smpl[key] = np.concatenate(_smpl_l[key], axis=0)
            for key in _smplx.keys():
                # for model_param in _smplx_l[key]:
                _smplx[key] = np.concatenate(_smplx_l[key], axis=0)

            # human_data['smpl'] = _smpl
            human_data['smplx'] = _smplx
            print('Smpl and/or Smplx finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            human_data['config'] = f'synbody_{mode}'
            human_data['meta'] = _meta
            human_data['misc'] = self.misc_config
            print('MetaData finished at',
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            human_data.compress_keypoints_by_mask()
            # store the data struct
            os.makedirs(out_path, exist_ok=True)

            out_file = os.path.join(out_path,
                                    f'synbody_{mode}_{seed}_{str(size_n)}_{s}.npz')
            human_data.dump(out_file)
