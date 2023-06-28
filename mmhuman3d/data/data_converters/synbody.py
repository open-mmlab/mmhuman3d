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
    ACCEPTED_MODES = ['train']

    def __init__(self, modes: List = []) -> None:

        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            flat_hand_mean=False,
            bbox_source='keypoints2d_original',
            kps3d_root_aligned=False,
            smplx_source='original',
            fps=30)

        super(SynbodyConverter, self).__init__(modes)
        # self.do_npz_merge = do_npz_merge
        # merged_path is the folder (will) contain merged npz
        # self.merged_path = merged_path

    # def _get_imgname(self, v):
    #     rgb_folder = os.path.join(v, 'rgb')
    #     root_folder_id = v.split('/').index('synbody')
    #     imglist = os.path.join('/'.join(v.split('/')[root_folder_id:]), 'rgb')

    #     # image 1 is T-pose, don't use
    #     im = []
    #     images = [img for img in os.listdir(rgb_folder) if img.endswith('.jpeg')]
    #     for i in range(1, len(images)):
    #         imglist_tmp = os.path.join(imglist, f'{i:04d}.jpeg')
    #         im.append(imglist_tmp)

    #     # import pdb; pdb.set_trace()
    #     return im

    # def _get_exrname(self, v):
    #     rgb_folder = os.path.join(v, 'rgb')
    #     root_folder_id = v.split('/').index('synbody')
    #     masklist = os.path.join('/'.join(v.split('/')[root_folder_id:]), 'mask')

    #     # image 1 is T-pose, don't use
    #     images = [img for img in os.listdir(rgb_folder) if img.endswith('.jpeg')]
    #     exr = []
    #     for i in range(1, len(images)):
    #         masklist_tmp = os.path.join(masklist, f'{i:04d}.exr')
    #         exr.append(masklist_tmp)

    #     return exr

    # def _get_npzname(self, p, f_num):

    #     npz = []
    #     npz_tmp = p.split('/')[-1]
    #     for _ in range(f_num):
    #         npz.append(npz_tmp)

    #     return npz

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

    # def _get_mask_conf(self, root, merged):

    #     os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

    #     root_folder_id = root.split('/').index('synbody')

    #     conf = []
    #     for idx, mask_path in enumerate(merged['mask_path']):
    #         exr_path = os.path.join('/'.join(root.split('/')[:root_folder_id]), mask_path)

    #         # import pdb; pdb.set_trace()

    #         image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #         json_path = os.path.join('/'.join(exr_path.split('/')[:-2]), 'seq_data.json')
    #         jsfile_tmp = json.load(open(json_path, 'r'))

    #         p_rgb = [0, 0, 0]
    #         keys = list(jsfile_tmp['Actors']['CharacterActors'].keys())
    #         for key in keys:
    #             if jsfile_tmp['Actors']['CharacterActors'][key]['animation'] == merged['npz_name'][idx][:-4]:
    #                 p_rgb = jsfile_tmp['Actors']['CharacterActors'][key]['mask_rgb_value']
    #                 break
    #         if p_rgb == [0, 0, 0]:
    #             raise ValueError(f'Cannot find info of {merged["npz_name"][idx][:-4]} in {json_path}')

    #         kps2d = merged['keypoints2d'][idx]
    #         v = []

    #         for kp in kps2d:
    #             if (not 0 < kp[1] < 720) or (not 0 < kp[0] < 1280) or \
    #                     sum(image[int(kp[1]), int(kp[0])] * 255 - np.array(p_rgb)) > 3:
    #                 v.append(0)
    #             else:
    #                 v.append(1)
    #         conf.append(v)

    #     return conf

    # def _merge_npz(self, root_path, mode):
    #     # root_path is where the npz files stored. Should ends with 'synbody'
    #     # if not os.path.basename(root_path).endswith('synbody'):
    #     #     root_path = os.path.join(root_path, 'synbody')
    #     batch_paths = [os.path.join(root_path, p) for p in os.listdir(root_path)]
    #     # ple = [p for p in ple if '.' not in p]

    #     # print(ple)
    #     print(f'There are {len(batch_paths)} batches:', batch_paths)
    #     for batch_path in tqdm(batch_paths, desc='batch'):
    #         print(batch_path)

    #         failed, merged = [], {}
    #         for key in ['image_path', 'mask_path', 'npz_name', 'meta', 'keypoints2d', 'keypoints3d']:
    #             merged[key] = []
    #         merged['smpl'] = {}
    #         for key in ['transl', 'global_orient', 'betas', 'body_pose']:
    #             merged['smpl'][key] = []
    #         merged['smplx'] = {}
    #         for key in ['transl', 'global_orient', 'betas', 'body_pose',
    #                     'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
    #             merged['smplx'][key] = []

    #         places = os.listdir(batch_path)
    #         for ple in places:
    #             if '.' in ple:
    #                 continue
    #             seqs = [os.path.join(batch_path, ple, seq_name) for \
    #                 seq_name in os.listdir(os.path.join(batch_path, ple)) if seq_name.startswith('LS')]
    #             for v in tqdm(seqs, desc='Place:' + ple):
    #                 try:
    #                     imgname = self._get_imgname(v)
    #                     exrname = self._get_exrname(v)
    #                     valid_frame_number = len(imgname)
    #                     # for p in tqdm(glob.glob(v + '/smpl_with_joints/*.npz'), desc='person'):
    #                     ps = [os.path.join(v, 'smpl_withJoints_inCamSpace', p) for p in \
    #                         os.listdir(os.path.join(v, 'smpl_withJoints_inCamSpace')) if p.endswith('.npz')]
    #                     for p in sorted(ps):
    #                         npfile_tmp = np.load(p, allow_pickle=True)
    #                         merged['image_path'] += imgname
    #                         merged['mask_path'] += exrname
    #                         merged['npz_name'] += self._get_npzname(p, valid_frame_number)
    #                         # merged['smpl']['transl'].append(npfile_tmp['smpl'].item()['transl'][1:61])
    #                         # merged['smpl']['global_orient'].append(npfile_tmp['smpl'].item()['global_orient'][1:61])
    #                         # betas = npfile_tmp['smpl'].item()['betas']
    #                         # betas = np.repeat(betas, 60, axis=0)
    #                         # merged['smpl']['betas'].append(betas)
    #                         # merged['smpl']['body_pose'].append(npfile_tmp['smpl'].item()['body_pose'][1:61])
    #                         # merged['smpl']['keypoints3d'].append(npfile_tmp['keypoints3d'][1:61])
    #                         # merged['smpl']['keypoints2d'].append(npfile_tmp['keypoints2d'][1:61])

    #                         # import pdb; pdb.set_trace()
    #                         for _ in range(valid_frame_number):
    #                             merged['meta'].append(npfile_tmp['meta'])

    #                         for key in ['betas', 'global_orient', 'transl', 'body_pose']:
    #                             if key == 'betas' and len(npfile_tmp['smpl'].item()['betas']) == 1:
    #                                 betas = np.repeat(npfile_tmp['smpl'].item()[key], valid_frame_number, axis=0)
    #                                 merged['smpl']['betas'].append(betas)
    #                             else:
    #                                 if len(npfile_tmp['smpl'].item()[key]) == valid_frame_number:
    #                                     merged['smpl'][key].append(npfile_tmp['smpl'].item()[key])
    #                                 else:
    #                                     merged['smpl'][key].append(npfile_tmp['smpl'].item()[key][1:valid_frame_number+1])

    #                     ps =  [os.path.join(v, 'smplx_withJoints_inCamSpace', p) for p in \
    #                         os.listdir(os.path.join(v, 'smplx_withJoints_inCamSpace')) if p.endswith('.npz')]
    #                     for p in sorted(ps):
    #                         npfile_tmp = np.load(p, allow_pickle=True)
    #                         merged['keypoints2d'].append(npfile_tmp['keypoints2d'][1:valid_frame_number+1])
    #                         merged['keypoints3d'].append(npfile_tmp['keypoints3d'][1:valid_frame_number+1])
    #                         for key in ['betas', 'global_orient', 'transl', 'body_pose', \
    #                                     'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
    #                             if key == 'betas' and len(npfile_tmp['smplx'].item()['betas']) == 1:
    #                                 betas = np.repeat(npfile_tmp['smplx'].item()[key], valid_frame_number, axis=0)
    #                                 merged['smplx']['betas'].append(betas)
    #                             else:
    #                                 if len(npfile_tmp['smplx'].item()[key]) == valid_frame_number:
    #                                     merged['smplx'][key].append(npfile_tmp['smplx'].item()[key])
    #                                 else:
    #                                     merged['smplx'][key].append(npfile_tmp['smplx'].item()[key][1:valid_frame_number+1])
    #                 except Exception as e:
    #                     failed.append(v)
    #                     with open('log_synbody.json', 'w') as f:
    #                         json.dump(failed, f)
    #                     print(v, 'failed because of', e)

    #         print('total', len(failed), 'failed in batch:', batch_path)

    #         for k in merged['smpl'].keys():
    #             merged['smpl'][k] = np.vstack(merged['smpl'][k])
    #         for k in merged['smplx'].keys():
    #             merged['smplx'][k] = np.vstack(merged['smplx'][k])
    #         for k in ['left_hand_pose', 'right_hand_pose']:
    #             merged['smplx'][k] = merged['smplx'][k].reshape(-1, 15, 3)
    #         merged['smplx']['body_pose'] = merged['smplx']['body_pose'].reshape(-1, 21, 3)

    #         merged['keypoints3d'] = np.vstack(merged['keypoints3d'])
    #         merged['keypoints2d'] = np.vstack(merged['keypoints2d'])

    #         merged['conf'] = np.vstack(self._get_mask_conf(root_path, merged)).reshape(-1, 144, 1)

    #         os.makedirs(self.merged_path, exist_ok=True)
    #         batch_name = os.path.basename(batch_path)
    #         outpath = os.path.join(self.merged_path, f'synbody_{batch_name}_merged.npz')
    #         np.savez(outpath, **merged)

    #     print('Merge npz finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    #     # import pdb; pdb.set_trace()

    #     # os.makedirs(self.merged_path, exist_ok=True)
    #     # outpath = os.path.join(self.merged_path, 'synbody_{mode}_merged.npz')
    #     # np.savez(outpath, **merged)
    #     return merged

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
        # use HumanData to store all data
        human_data = HumanData()

        # get targeted sequence list
        root_dir, prefix = os.path.split(dataset_path)
        print(root_dir)
        preprocessed_dir = os.path.join(root_dir, 'preprocessed_0220_renew')
        npzs = glob.glob(os.path.join(preprocessed_dir, '*', '*', 'LS*.npz'))
        seed, size = '230526_renew', '04000'
        # random.seed(int(seed))
        # random.shuffle(npzs)
        # npzs = npzs[:int(size)]
        # print(npzs[:10])

        # initialize storage
        _bboxs = {}
        _meta = {}
        _meta['gender'] = []
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            _bboxs[bbox_name] = []
        _image_path = []

        # get data shape
        npfile = dict(np.load(npzs[0], allow_pickle=True))
        kp_shape = npfile['keypoints2d'].shape[1]
        _keypoints2d = np.array([]).reshape(0, kp_shape, 3)
        _keypoints3d = np.array([]).reshape(0, kp_shape, 4)
        _keypoints2d_list, _keypoints3d_list = [], []

        # initialize smpl and smplx
        _smpl, _smplx = {}, {}
        smpl_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 23, 3)
        }
        smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 21, 3), \
                       'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 'leye_pose': (-1, 3),
                       'reye_pose': (-1, 3), \
                       'jaw_pose': (-1, 3), 'expression': (-1, 10)}
        for key in smpl_shape:
            _smpl[key] = np.array([]).reshape(smpl_shape[key])
        for key in smplx_shape:
            _smplx[key] = np.array([]).reshape(smplx_shape[key])
        _smpl_l, _smplx_l = {}, {}
        for key in smpl_shape:
            _smpl_l[key] = []
        for key in smplx_shape:
            _smplx_l[key] = []

        # pdb.set_trace()

        size_n = max(int(size), len(npzs))

        for npzf in tqdm(npzs, desc='Npzfiles concating'):
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
            # import IPython; IPython.embed()
            # keypoints2d_, mask = convert_kps(np.concatenate((keypoints2d_merged[valid_id], conf), axis=2), 'smpl_45', 'human_data')
            # keypoints3d_, mask = convert_kps(np.concatenate((keypoints3d_merged[valid_id], conf), axis=2), 'smpl_45', 'human_data')
            # pdb.set_trace()

            for k in npfile['smpl'].item().keys():
                # _smpl[k] = np.concatenate((_smpl[k], npfile['smpl'].item()[k][valid_id].reshape(smpl_shape[k])), axis=0)
                _smpl_l[k].append(npfile['smpl'].item()[k][valid_id].reshape(
                    smpl_shape[k]))
            for k in npfile['smplx'].item().keys():
                # _smplx[k] = np.concatenate((_smplx[k], npfile['smplx'].item()[k][valid_id].reshape(smplx_shape[k])), axis=0)
                _smplx_l[k].append(npfile['smplx'].item()[k][valid_id].reshape(
                    smplx_shape[k]))
            gender = []
            for idx, meta_tmp in enumerate(npfile['meta'][valid_id]):
                gender.append(meta_tmp.item()['gender'])

            _meta['gender'] += gender
            # pdb.set_trace()

            # _meta['gender'].append(np.array(gender)[valid_id].tolist())
            # _keypoints2d = np.concatenate((_keypoints2d, keypoints2d_[valid_id]), axis=0)
            # _keypoints3d = np.concatenate((_keypoints3d, keypoints3d_[valid_id]), axis=0)
            _keypoints2d_list.append(keypoints2d_[valid_id])
            _keypoints3d_list.append(keypoints3d_[valid_id])

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

        human_data['keypoints2d_original'] = _keypoints2d
        human_data['keypoints3d_original'] = _keypoints3d
        human_data['keypoints2d_original_mask'] = mask
        human_data['keypoints3d_original_mask'] = mask
        print('Keypoint conversion finished at',
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        for key in _smpl.keys():
            # for model_param in _smpl_l[key]:
            _smpl[key] = np.concatenate(_smpl_l[key], axis=0)
        for key in _smplx.keys():
            # for model_param in _smplx_l[key]:
            _smplx[key] = np.concatenate(_smplx_l[key], axis=0)

        human_data['smpl'] = _smpl
        human_data['smplx'] = _smplx
        print('Smpl and/or Smplx finished at',
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        human_data['config'] = 'synbody_train'
        human_data['meta'] = _meta
        human_data['misc'] = self.misc_config
        print('MetaData finished at',
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        human_data.compress_keypoints_by_mask()
        # store the data struct
        os.makedirs(out_path, exist_ok=True)

        out_file = os.path.join(out_path,
                                f'synbody_train_{seed}_{str(size_n)}.npz')
        human_data.dump(out_file)
