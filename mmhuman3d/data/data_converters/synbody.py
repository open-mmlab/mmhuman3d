import glob
import os
from typing import List

import cdflib
import h5py
import numpy as np
import json
import cv2
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
import mmcv
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idxs_by_part


@DATA_CONVERTERS.register_module()
class SynbodyConverter(BaseModeConverter):
    """Synbody dataset
    """
    ACCEPTED_MODES = ['train']

    def __init__(self, modes: List = [], merged_path='data/preprocessed_datasets', do_npz_merge=True, do_split=False) -> None:
        super(SynbodyConverter, self).__init__(modes)
        self.do_npz_merge = do_npz_merge
        # merged_path is the folder (will) contain merged npz
        self.merged_path = merged_path

        
    def _get_imgname(self, v):
        rgb_folder = os.path.join(v, 'rgb')
        root_folder_id = v.split('/').index('synbody')
        imglist = os.path.join('/'.join(v.split('/')[root_folder_id:]), 'rgb')

        # image 1 is T-pose, don't use
        im = []
        for i in range(1, len(glob.glob(os.path.join(rgb_folder, '*.jpeg')))):
            imglist_tmp = os.path.join(imglist, f'{i:04d}.jpeg')
            im.append(imglist_tmp)
        
        # import pdb; pdb.set_trace()
        return im

    
    def _get_exrname(self, v):
        rgb_folder = os.path.join(v, 'rgb')
        root_folder_id = v.split('/').index('synbody')
        masklist = os.path.join('/'.join(v.split('/')[root_folder_id:]), 'mask')

        # image 1 is T-pose, don't use
        exr = []
        for i in range(1, len(glob.glob(os.path.join(rgb_folder, '*.jpeg')))):
            masklist_tmp = os.path.join(masklist, f'{i:04d}.exr')
            exr.append(masklist_tmp)

        return exr

    def _get_npzname(self, p, f_num):
        
        npz = []
        npz_tmp = p.split('/')[-1]
        for _ in range(f_num):
            npz.append(npz_tmp)

        return npz

    def _keypoints_to_scaled_bbox_fh(self, keypoints, scale=1.0):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale

        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format
        '''
        bboxs = []
        for body_part in ['head', 'left_hand', 'right_hand']:
            kp_id = get_keypoint_idxs_by_part(body_part, convention='smplx')
                # keypoints_factory=smplx.SMPLX_KEYPOINTS)
            kps = keypoints[kp_id]

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

            bbox = np.stack([xmin, ymin, xmax, ymax], axis=0).astype(np.float32)

            bboxs.append(bbox)
        return bboxs[0], bboxs[1], bboxs[2]


    def _get_mask_conf(self, root, merged):
        
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

        root_folder_id = root.split('/').index('synbody')

        conf = []
        for idx, mask_path in enumerate(merged['mask_path']):
            exr_path = os.path.join('/'.join(root.split('/')[:root_folder_id]), mask_path)

            # import pdb; pdb.set_trace()

            image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            json_path = os.path.join('/'.join(exr_path.split('/')[:-2]), 'seq_data.json')
            jsfile_tmp = json.load(open(json_path, 'r'))

            p_rgb = [0, 0, 0]
            keys = list(jsfile_tmp['Actors']['CharacterActors'].keys())
            for key in keys:
                if jsfile_tmp['Actors']['CharacterActors'][key]['animation'] == merged['npz_name'][idx][:-4]:
                    p_rgb = jsfile_tmp['Actors']['CharacterActors'][key]['mask_rgb_value']
                    break
            if p_rgb == [0, 0, 0]:
                raise ValueError(f'Cannot find info of {merged["npz_name"][idx][:-4]} in {json_path}')

            kps2d = merged['keypoints2d'][idx]
            v = []

            for kp in kps2d:
                if (not 0 < kp[1] < 720) or (not 0 < kp[0] < 1280) or \
                        sum(image[int(kp[1]), int(kp[0])] * 255 - np.array(p_rgb)) > 3:
                    v.append(0)
                else:
                    v.append(1)
            conf.append(v)

        return conf

    
    def _merge_npz(self, root_path, mode):
        # root_path is where the npz files stored. Should ends with 'synbody'
        if not os.path.basename(root_path).endswith('synbody'):
            root_path = os.path.join(root_path, 'synbody')
        batch_paths = glob.glob(os.path.join(root_path, '*'))
        # ple = [p for p in ple if '.' not in p]
                            
        merged = {}
        for key in ['image_path', 'mask_path', 'npz_name', 'meta', 'keypoints2d', 'keypoints3d']:
            merged[key] = []
        merged['smpl'] = {}
        for key in ['transl', 'global_orient', 'betas', 'body_pose']:
            merged['smpl'][key] = []
        merged['smplx'] = {}
        for key in ['transl', 'global_orient', 'betas', 'body_pose', 
                    'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
            merged['smplx'][key] = []

        # print(ple)
        print(f'There are {len(batch_paths)} batches:', batch_paths)
        for batch_path in tqdm(batch_paths, desc='batch'):
            print(batch_path)
            for v in tqdm(glob.glob(os.path.join(batch_path, '*/LS*'))):
                imgname = self._get_imgname(v)
                exrname = self._get_exrname(v)
                valid_frame_number = len(imgname)
                # for p in tqdm(glob.glob(v + '/smpl_with_joints/*.npz'), desc='person'):
                for p in sorted(glob.glob(v + '/smpl_withJoints_inCamSpace/*.npz')):
                    npfile_tmp = np.load(p, allow_pickle=True)
                    merged['image_path'] += imgname
                    merged['mask_path'] += exrname
                    merged['npz_name'] += self._get_npzname(p, valid_frame_number)
                    # merged['smpl']['transl'].append(npfile_tmp['smpl'].item()['transl'][1:61])
                    # merged['smpl']['global_orient'].append(npfile_tmp['smpl'].item()['global_orient'][1:61])
                    # betas = npfile_tmp['smpl'].item()['betas']
                    # betas = np.repeat(betas, 60, axis=0)
                    # merged['smpl']['betas'].append(betas)
                    # merged['smpl']['body_pose'].append(npfile_tmp['smpl'].item()['body_pose'][1:61])
                    # merged['smpl']['keypoints3d'].append(npfile_tmp['keypoints3d'][1:61])
                    # merged['smpl']['keypoints2d'].append(npfile_tmp['keypoints2d'][1:61])

                    # import pdb; pdb.set_trace()
                    for _ in range(valid_frame_number):
                        merged['meta'].append(npfile_tmp['meta'])

                    for key in ['betas', 'global_orient', 'transl', 'body_pose']:
                        if key == 'betas' and len(npfile_tmp['smpl'].item()['betas']) == 1:
                            betas = np.repeat(npfile_tmp['smpl'].item()[key], valid_frame_number, axis=0)
                            merged['smpl']['betas'].append(betas)
                        else:
                            if len(npfile_tmp['smpl'].item()[key]) == valid_frame_number:
                                merged['smpl'][key].append(npfile_tmp['smpl'].item()[key])
                            else:
                                merged['smpl'][key].append(npfile_tmp['smpl'].item()[key][1:valid_frame_number+1])

                for p in sorted(glob.glob(v + '/smplx_withJoints_inCamSpace/*.npz')):
                    npfile_tmp = np.load(p, allow_pickle=True)
                    merged['keypoints2d'].append(npfile_tmp['keypoints2d'][1:valid_frame_number+1])
                    merged['keypoints3d'].append(npfile_tmp['keypoints3d'][1:valid_frame_number+1])
                    for key in ['betas', 'global_orient', 'transl', 'body_pose', \
                                'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                        if key == 'betas' and len(npfile_tmp['smplx'].item()['betas']) == 1:
                            betas = np.repeat(npfile_tmp['smplx'].item()[key], valid_frame_number, axis=0)
                            merged['smplx']['betas'].append(betas)
                        else:
                            if len(npfile_tmp['smplx'].item()[key]) == valid_frame_number:
                                merged['smplx'][key].append(npfile_tmp['smplx'].item()[key])
                            else:
                                merged['smplx'][key].append(npfile_tmp['smplx'].item()[key][1:valid_frame_number+1])
                
                        
                    # betas = npfile_tmp['smpl'].item()['betas']
                    # betas = np.repeat(betas, 60, axis=0)
                    # merged['smplx']['betas'].append(betas)
                    # merged['smplx']['body_pose'].append(npfile_tmp['smpl'].item()['body_pose'][1:61])
                    # merged['keypoints3d'].append(npfile_tmp['keypoints3d'][1:61])
                    # merged['keypoints2d'].append(npfile_tmp['keypoints2d'][1:61])

                    
        for k in merged['smpl'].keys():
            merged['smpl'][k] = np.vstack(merged['smpl'][k])
        for k in merged['smplx'].keys():
            merged['smplx'][k] = np.vstack(merged['smplx'][k])
        for k in ['left_hand_pose', 'right_hand_pose']:
            merged['smplx'][k] = merged['smplx'][k].reshape(-1, 15, 3)
        merged['smplx']['body_pose'] = merged['smplx']['body_pose'].reshape(-1, 21, 3)

        merged['keypoints3d'] = np.vstack(merged['keypoints3d'])
        merged['keypoints2d'] = np.vstack(merged['keypoints2d'])

        import pdb; pdb.set_trace()

        merged['conf'] = np.vstack(self._get_mask_conf(root_path, merged)).reshape(-1, 144, 1)

        # import pdb; pdb.set_trace()

        # os.makedirs(self.merged_path, exist_ok=True)
        # outpath = os.path.join(self.merged_path, 'synbody_{mode}_merged.npz')
        # np.savez(outpath, **merged)
        return merged
    

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

        # structs we use
        if self.do_npz_merge:
            npfile = self._merge_npz(dataset_path, mode=mode)
        else:
            npfile = np.load(os.path.join(self.merged_path, 'synbody_{mode}_merged.npz'), allow_pickle=True)
        
        # filter good instances
        # valid_id = np.load('/mnt/lustre/liushuai/project/debug_synbody/select4.npy')
        # valid_id = np.load('/mnt/lustre/liushuai/project/debug_synbody/synbody_0113_valid_id_new.npy')
        # valid_id = np.load('/mnt/lustre/liushuai/project/debug_synbody/synbody_0113_valid.npy')
        # valid_id = np.array(list(range(len(npfile['image_path']))))w
 
        bbox_ = []
        # bbox_ = npfile['bbox']
        keypoints2d_smpl_merged = npfile['keypoints2d'].reshape(len(npfile['image_path']), -1, 2)
        keypoints3d_smpl_merged = npfile['keypoints3d'].reshape(len(npfile['image_path']), -1, 3)

        # root centered
        valid_id = []
        # conf = keypoints3d_smpl_merged[:, :, -1]
        conf = npfile['conf']
        pelvis = keypoints3d_smpl_merged[:, 0, :]
        for i in range(len(conf)):
            if conf[i][0] > 0:
                valid_id.append(i)
        valid_id = np.array(valid_id)
        keypoints3d_smpl_merged[:, :, :] -= pelvis[:, None, :]

        bboxs_ = {}
        for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
            bboxs_[bbox_name] = []
        for kp in keypoints2d_smpl_merged:
            # since the 2d keypoints are not strictly corrcet, a large scale factor is used
            bbox_tmp_ = {}
            bbox_tmp_['bbox_xywh'] = self._keypoints_to_scaled_bbox(kp, 1.2)
            bbox_tmp_['face_bbox_xywh'], bbox_tmp_['lhand_bbox_xywh'], bbox_tmp_['rhand_bbox_xywh'] = self._keypoints_to_scaled_bbox_fh(kp, 1.0)
            for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
                bbox = bbox_tmp_[bbox_name]
                xmin, ymin, xmax, ymax = bbox
                # import pdb; pdb.set_trace()
                bbox = np.array([max(0, xmin), max(0, ymin), min(1280, xmax), min(720, ymax)])
                bbox_xywh = self._xyxy2xywh(bbox)
                bboxs_[bbox_name].append(bbox_xywh)

        for key in bboxs_.keys():
            bbox_ = np.array(bboxs_[key]).reshape((-1, 4))
            bbox_ = np.hstack([bbox_, np.ones([bbox_.shape[0], 1])])
            # import pdb; pdb.set_trace()
            human_data[key] = bbox_[valid_id]
        


        image_path_ = []
        for imp in npfile['image_path']:
            imp = imp.split('/')
            image_path_.append('/'.join(imp[1:]))
            # import pdb; pdb.set_trace()
        human_data['image_path'] = np.array(image_path_)[valid_id].tolist()
        # human_data['bbox_xywh'] = bbox_b[valid_id]
        # human_data['face_bbox_xywh'] = bbox_f[valid_id]
        # human_data['lhand_bbox_xywh'] = bbox_lh[valid_id]
        # human_data['rhand_bbox_xywh'] = bbox_rh[valid_id]

        keypoints2d_smpl_merged = np.concatenate((keypoints2d_smpl_merged, conf), axis=2)
        keypoints3d_smpl_merged = np.concatenate((keypoints3d_smpl_merged, conf), axis=2)

        conf = np.ones_like(keypoints2d_smpl_merged[valid_id][..., 0:1])
        # remove_kp = [39, 35, 38, 23, 36, 37, 41, 44, 42, 43, 22, 20, 18]
        # conf[:, remove_kp, :] = 0
        # import IPython; IPython.embed()
        # keypoints2d_, mask = convert_kps(np.concatenate((keypoints2d_merged[valid_id], conf), axis=2), 'smpl_45', 'human_data')
        keypoints2d_, mask = convert_kps(keypoints2d_smpl_merged[valid_id], 'smplx', 'human_data')
        # keypoints3d_, mask = convert_kps(np.concatenate((keypoints3d_merged[valid_id], conf), axis=2), 'smpl_45', 'human_data')
        keypoints3d_, mask = convert_kps(keypoints3d_smpl_merged[valid_id], 'smplx', 'human_data')

        # use smpl to generate keypoints3d
        # body_model_config=dict(
        #     type='GenderedSMPL',
        #     keypoint_src='h36m',
        #     keypoint_dst='h36m',
        #     model_path='data/body_models/smpl',
        #     joints_regressor='data/body_models/J_regressor_h36m.npy')
        # body_model = build_body_model(body_model_config)
        # betas = []
        # body_pose = []
        # global_orient = []
        # gender = []
        # smpl_dict = npfile['smpl'].item()
        # for idx in tqdm(range(len(npfile['image_path']))):
        # # for idx in tqdm(range(10)):
        #     betas.append(smpl_dict['betas'][idx])
        #     body_pose.append(smpl_dict['body_pose'][idx])
        #     global_orient.append(smpl_dict['global_orient'][idx])
        #     if npfile['meta'].item()['gender'][idx] == 'm':
        #         gender.append(0)
        #     else:
        #         gender.append(1)
        # import torch
        # betas = torch.FloatTensor(betas)
        # body_pose = torch.FloatTensor(body_pose).view(-1, 69)
        # global_orient = torch.FloatTensor(global_orient)
        # gender = torch.Tensor(gender)
        # gt_output = body_model(
        #     betas=betas,
        #     body_pose=body_pose,
        #     global_orient=global_orient,
        #     gender=gender)
        # gt_keypoints3d = gt_output['joints'].detach().cpu().numpy()
        # # gt_keypoints3d_mask = np.ones((len(npfile), 24))
        # # import IPython; IPython.embed()

        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d'] = keypoints3d_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints3d_mask'] = mask
        
        smpl, smplx = {}, {}
        for k in npfile['smpl'].keys():
            smpl[k] = npfile['smpl'][k][valid_id]
        human_data['smpl'] = smpl
        for k in npfile['smplx'].keys():
            smplx[k] = npfile['smplx'][k][valid_id]
        human_data['smplx'] = smplx
        human_data['config'] = 'synbody_train'
        

        meta = {}
        meta['gender'] = []
        for meta_tmp in npfile['meta']:

            s = meta_tmp.item()['gender']
            if s == 'male':
                meta['gender'].append('m')
            elif s == 'female':
                meta['gender'].append('f')
            else:
                meta['gender'].append('n')
        meta['gender'] = np.array(meta['gender'])[valid_id].tolist()
        human_data['meta'] = meta

        # import pdb; pdb.set_trace()

        human_data.compress_keypoints_by_mask()
        # store the data struct
        os.makedirs(out_path,exist_ok=True)
        out_file = os.path.join(out_path, f'synbody_train.npz')
        human_data.dump(out_file)
