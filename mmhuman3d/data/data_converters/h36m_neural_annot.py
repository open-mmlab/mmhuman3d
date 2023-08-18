import glob
import json
import os
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm
import cv2

from mmhuman3d.core.cameras import build_cameras
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.models.body_models.utils import transform_to_camera_frame
# from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

import pdb


@DATA_CONVERTERS.register_module()
class H36mNeuralConverter(BaseModeConverter):
    """Human3.6M dataset
    `Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments' TPAMI`2014
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    Args:
        modes (list): 'val' or 'train' for accepted modes
        protocol (int): 1 or 2 for available protocols
        extract_img (bool): Store True to extract images into a separate
        folder. Default: False.
        mosh_dir (str, optional): Path to directory containing mosh files.
    """
    ACCEPTED_MODES = ['val', 'train']

    def __init__(self, modes: List = []) -> None:

        self.device = torch.device('cuda:0')
        self.misc = dict(
            bbox_source='by_dataset',
            cam_param_type='prespective',
            cam_param_source='original',
            smplx_source='neural_annot',
        )
        self.smplx_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 21, 3),
            # 'left_hand_pose': (-1, 15, 3),
            # 'right_hand_pose': (-1, 15, 3),
            # 'leye_pose': (-1, 3),
            # 'reye_pose': (-1, 3),
            # 'jaw_pose': (-1, 3),
            # 'expression': (-1, 10)
        }
        super(H36mNeuralConverter, self).__init__(modes)


    def convert_by_mode(self,
                        dataset_path: str,
                        out_path: str,
                        mode: str,
                        enable_multi_human_data: bool = False) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes
            enable_multi_human_data (bool):
                Whether to generate a multi-human data. If set to True,
                stored in MultiHumanData() format.
                Default: False, stored in HumanData() format.

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask, keypoints3d, keypoints3d_mask, cam_param
                stored in HumanData() format
        """
        # get targeted seq list
        targeted_seqs = sorted(
            glob.glob(os.path.join(dataset_path, 'images', 's_*_act_*')))
        
        # get all subject_ids and rearrange the seqs
        subject_ids = [int(os.path.basename(seq)[2:4]) for seq in targeted_seqs]
        subject_ids = list(set(subject_ids))

        subject_seq_dict = {}
        for subject_id in subject_ids:
            subject_seq_dict[subject_id] = []
            for seq in targeted_seqs:
                if int(os.path.basename(seq)[2:4]) == subject_id:
                    subject_seq_dict[subject_id].append(seq)

        # choose sebjetct ids for different mode
        if mode == 'train':
            user_list = [1, 5, 6, 7, 8]
        elif mode == 'val':
            user_list = [11]
        subject_ids = list(set(subject_ids) & set(user_list))

        # calculate size
        seqs_len = 0
        for key in subject_seq_dict:
            seqs_len += len(subject_seq_dict[key])


        # parse seqs
        for s, sid in enumerate(subject_ids):
            
            # use HumanData to store all data
            human_data = HumanData()

            # init seed and size
            seed, size = '230811', '999'
            size_i = min(int(size), seqs_len)
            random.seed(int(seed))
            # targeted_seqs = targeted_seqs[:size_i]
            # random.shuffle(npzs)

            # initialize output for human_data
            smplx_ = {}
            for key in self.smplx_shape.keys():
                smplx_[key] = []
            keypoints2d_smplx_, keypoints3d_smplx_, = [], []
            keypoints2d_orig_, keypoints3d_orig_ = [], []
            bboxs_ = {}
            for bbox_name in ['bbox_xywh']:
                bboxs_[bbox_name] = []
            meta_ = {}
            for key in ['focal_length', 'principal_point', 'height', 'width']:
                meta_[key] = []
            image_path_ = []

            # init smplx model
            smplx_model = build_body_model(
                dict(
                    type='SMPLX',
                    keypoint_src='smplx',
                    keypoint_dst='smplx',
                    model_path='data/body_models/smplx',
                    gender='neutral',
                    num_betas=10,
                    use_face_contour=True,
                    flat_hand_mean=False,
                    use_pca=False,
                    batch_size=1)).to(self.device)

            # load subject annotations
            anno_base_path = os.path.join(dataset_path, 'annotations')
            anno_base_name = f'Human36M_subject{int(sid)}'

            # load camera parameters
            cam_param = f'{anno_base_name}_camera.json'
            with open(os.path.join(anno_base_path, cam_param)) as f:
                cam_params = json.load(f)

            # load data annotations 
            data_an = f'{anno_base_name}_data.json'
            with open(os.path.join(anno_base_path, data_an)) as f:
                data_annos = json.load(f)

            # load joints 3d annotations
            joints3d_an = f'{anno_base_name}_joint_3d.json'
            with open(os.path.join(anno_base_path, joints3d_an)) as f:
                j3d_annos = json.load(f)

            # load smplx annotations (NeuralAnnot)
            smplx_an = f'{anno_base_name}_SMPLX_NeuralAnnot.json'
            with open(os.path.join(anno_base_path, smplx_an)) as f:
                smplx_annos = json.load(f)

            for seq in tqdm(subject_seq_dict[sid], 
                            desc=f'Processing subject {s + 1} / {len(subject_ids)}',
                            position=0, leave=False):
                
                # get ids
                seqn = os.path.basename(seq)
                action_id = str(int(seqn[9:11]))
                subaction_id = str(int(seqn[19:21]))
                camera_id = str(int(seqn[-2:]))

                # get annotation slice
                smplx_anno_seq = smplx_annos[action_id][subaction_id]

                # get frames list
                frames = smplx_anno_seq.keys()

                # get seq annotations
                data_anno_seq = [data_annos['annotations'][idx] | data_annos['images'][idx]
                                 for idx, finfo in enumerate(data_annos['images']) if 
                                 os.path.basename(finfo['file_name'])[:-11] == seqn]
                # pdb.set_trace()
                if len(data_anno_seq) != len(frames):
                    print(f'Warning: {seqn} has different length of frames and annotations')
                    continue

                # get joints 3d annotations
                j3d_anno_seq = j3d_annos[action_id][subaction_id]

                # get camera parameters
                cam_param = cam_params[camera_id]
                R = np.array(cam_param['R']).reshape(3, 3)
                T = np.array(cam_param['t']).reshape(1, 3)
                focal_length = np.array(cam_param['f']).reshape(1, 2)
                principal_point = np.array(cam_param['c']).reshape(1, 2)

                # create extrinsics
                extrinsics = np.eye(4)
                extrinsics[:3, :3] = R
                extrinsics[:3, 3] = T / 1000

                # create intrinsics camera, assume resolution is same in seq
                width, height = data_anno_seq[0]['width'], data_anno_seq[0]['height']
                camera = build_cameras(
                    dict(
                        type='PerspectiveCameras',
                        convention='opencv',
                        in_ndc=False,
                        focal_length=focal_length,
                        image_size=(width, height),
                        principal_point=principal_point)).to(self.device)

                for fid, frame in enumerate(tqdm(frames, desc=f'Processing seq {seqn}',
                                                  position=1, leave=False)):

                    smplx_anno = smplx_anno_seq[frame]

                    # get image and bbox
                    info_anno = data_anno_seq[fid]
                    width, height = info_anno['width'], info_anno['height']
                    bbox_xywh = info_anno['bbox']
                    bbox_xywh.append(1)
                    imgp = os.path.join(dataset_path, 'images', info_anno['file_name'])
                    image_path = imgp.replace(f'{dataset_path}{os.path.sep}', '')

                    # reformat smplx_anno
                    smplx_param = {}
                    smplx_param['global_orient'] = np.array(smplx_anno['root_pose']).reshape(-1, 3)
                    smplx_param['body_pose'] = np.array(smplx_anno['body_pose']).reshape(-1, 21, 3)
                    smplx_param['betas'] = np.array(smplx_anno['shape']).reshape(-1, 10)
                    smplx_param['transl'] = np.array(smplx_anno['trans']).reshape(-1, 3)

                    # get pelvis world
                    intersect_keys = list(
                        set(smplx_param.keys()) & set(self.smplx_shape.keys()))
                    body_model_param_tensor = {
                        key: torch.tensor(
                            np.array(smplx_param[key]).reshape(self.smplx_shape[key]),
                            device=self.device,
                            dtype=torch.float32)
                        for key in intersect_keys
                    }
                    output = smplx_model(**body_model_param_tensor, return_joints=True)

                    keypoints_3d = output['joints']
                    pelvis_world = keypoints_3d.detach().cpu().numpy()[
                        0, get_keypoint_idx('pelvis', 'smplx')]

                    # transform to camera space
                    global_orient, transl = transform_to_camera_frame(
                        global_orient=smplx_param['global_orient'],
                        transl=smplx_param['transl'],
                        pelvis=pelvis_world,
                        extrinsic=extrinsics)

                    # update smplx param
                    smplx_param['global_orient'] = global_orient
                    smplx_param['transl'] = transl

                    # update smplx
                    for update_key in ['global_orient', 'transl']:
                        body_model_param_tensor[update_key] = torch.tensor(
                            np.array(smplx_param[update_key]).reshape(
                                self.smplx_shape[update_key]),
                            device=self.device,
                            dtype=torch.float32)
                    output = smplx_model(**body_model_param_tensor, return_joints=True)
                    keypoints_3d = output['joints']

                    # get kps2d
                    keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
                    keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
                    keypoints_3d = keypoints_3d.detach().cpu().numpy()

                    # get j3d and project to 2d
                    j3d = np.array(j3d_anno_seq[frame]).reshape(-1, 3)
                    j3d_c = np.dot(R, j3d.transpose(1,0)).transpose(1,0) + T.reshape(1,3)
                    j2d = camera.transform_points_screen(torch.tensor(j3d_c, device=self.device, dtype=torch.float32))
                    j2d = j2d.detach().cpu().numpy()[..., :2]

                    # test projection
                    # pdb.set_trace()
                    # kps3d_c = np.dot(R, keypoints_3d[0].transpose(1,0)).transpose(1,0) + T.reshape(1,3)
                    # kps2d = camera.transform_points_screen(torch.tensor(kps3d_c, device=self.device, dtype=torch.float32))
                    # kps2d = kps2d.detach().cpu().numpy()[..., :2]

                    # # test overlay j2d
                    # img = cv2.imread(f'{dataset_path}/{image_path}')
                    # for kp in j2d:
                    #     if  0 < kp[0] < 1920 and 0 < kp[1] < 1080: 
                    #         cv2.circle(img, (int(kp[0]), int(kp[1])), 1, (0,0,255), -1)
                    #     pass
                    # # write image
                    # os.makedirs(f'{out_path}', exist_ok=True)
                    # cv2.imwrite(f'{out_path}/{os.path.basename(seq)}_{fid}.jpg', img)

                    # append image path
                    image_path_.append(image_path)

                    # append keypoints2d and 3d
                    keypoints2d_smplx_.append(keypoints_2d)
                    keypoints3d_smplx_.append(keypoints_3d)
                    keypoints2d_orig_.append(j2d)
                    keypoints3d_orig_.append(j3d_c)

                    # append bbox
                    bboxs_['bbox_xywh'].append(bbox_xywh)

                    # append smpl
                    for key in smplx_param.keys():
                        smplx_[key].append(smplx_param[key])

                    # append meta
                    meta_['principal_point'].append(principal_point)
                    meta_['focal_length'].append(focal_length)
                    meta_['height'].append(height)
                    meta_['width'].append(width)

                    # pdb.set_trace()

            # meta
            human_data['meta'] = meta_

            # image path
            human_data['image_path'] = image_path_

            # save bbox
            for bbox_name in bboxs_.keys():
                bbox_ = np.array(bboxs_[bbox_name]).reshape(-1, 5)
                human_data[bbox_name] = bbox_

            # save smplx
            # human_data.skip_keys_check = ['smplx']
            for key in smplx_.keys():
                smplx_[key] = np.concatenate(
                    smplx_[key], axis=0).reshape(self.smplx_shape[key])
            human_data['smplx'] = smplx_

            # keypoints2d_smplx
            keypoints2d_smplx = np.concatenate(
                keypoints2d_smplx_, axis=0).reshape(-1, 144, 2)
            keypoints2d_smplx_conf = np.ones([keypoints2d_smplx.shape[0], 144, 1])
            keypoints2d_smplx = np.concatenate(
                [keypoints2d_smplx, keypoints2d_smplx_conf], axis=-1)
            keypoints2d_smplx, keypoints2d_smplx_mask = \
                    convert_kps(keypoints2d_smplx, src='smplx', dst='human_data')
            human_data['keypoints2d_smplx'] = keypoints2d_smplx
            human_data['keypoints2d_smplx_mask'] = keypoints2d_smplx_mask

            # keypoints3d_smplx
            keypoints3d_smplx = np.concatenate(
                keypoints3d_smplx_, axis=0).reshape(-1, 144, 3)
            keypoints3d_smplx_conf = np.ones([keypoints3d_smplx.shape[0], 144, 1])
            keypoints3d_smplx = np.concatenate(
                [keypoints3d_smplx, keypoints3d_smplx_conf], axis=-1)
            keypoints3d_smplx, keypoints3d_smplx_mask = \
                    convert_kps(keypoints3d_smplx, src='smplx', dst='human_data')
            human_data['keypoints3d_smplx'] = keypoints3d_smplx
            human_data['keypoints3d_smplx_mask'] = keypoints3d_smplx_mask

            # keypoints2d_orig
            keypoints2d_orig = np.concatenate(
                keypoints2d_orig_, axis=0).reshape(-1, 17, 2)
            keypoints2d_orig_conf = np.ones([keypoints2d_orig.shape[0], 17, 1])
            keypoints2d_orig = np.concatenate(
                [keypoints2d_orig, keypoints2d_orig_conf], axis=-1)
            keypoints2d_orig, keypoints2d_orig_mask = \
                    convert_kps(keypoints2d_orig, src='h36m', dst='human_data')
            human_data['keypoints2d_original'] = keypoints2d_orig
            human_data['keypoints2d_original_mask'] = keypoints2d_orig_mask

            # keypoints3d_orig
            keypoints3d_orig = np.concatenate(
                keypoints3d_orig_, axis=0).reshape(-1, 17, 3)
            keypoints3d_orig_conf = np.ones([keypoints3d_orig.shape[0], 17, 1])
            keypoints3d_orig = np.concatenate(
                [keypoints3d_orig, keypoints3d_orig_conf], axis=-1)
            keypoints3d_orig, keypoints3d_orig_mask = \
                    convert_kps(keypoints3d_orig, src='h36m', dst='human_data')
            human_data['keypoints3d_original'] = keypoints3d_orig
            human_data['keypoints3d_original_mask'] = keypoints3d_orig_mask

            # misc
            human_data['misc'] = self.misc
            human_data['config'] = f'h36m_neural_annot_{mode}'

            # save
            human_data.compress_keypoints_by_mask()
            os.makedirs(out_path, exist_ok=True)
            out_file = os.path.join(
                out_path,
                f'h36m_neural_{mode}_{seed}_{"{:04d}".format(size_i)}_subject{sid}.npz')
            human_data.dump(out_file)