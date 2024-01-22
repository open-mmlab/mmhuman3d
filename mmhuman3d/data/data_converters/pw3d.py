import glob
import json
import os
import random
import pickle
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
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame
# from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

import pdb


@DATA_CONVERTERS.register_module()
class Pw3dConverter(BaseModeConverter):
    """3D Poses in the Wild dataset `Recovering Accurate 3D Human Pose in The
    Wild Using IMUs and a Moving Camera' ECCV'2018 More details can be found in
    the `paper.

    <https://virtualhumans.mpi-inf.mpg.de/papers/vonmarcardECCV18/
    vonmarcardECCV18.pdf>`__ .

    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """

    ACCEPTED_MODES = ['train', 'test', 'val']

    def __init__(self, modes: List = []):

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.misc_config = dict(
            bbox_source='keypoints2d_smpl',
            smpl_source='original',
            cam_param_type='prespective',
            bbox_scale=1.2,
            kps3d_root_aligned=False,
            has_gender=True,
        )
        
        self.smpl_shape = {
            'body_pose': (-1, 69),
            'betas': (-1, 10),
            'global_orient': (-1, 3),
            'transl': (-1, 3),} 
        
        super(Pw3dConverter, self).__init__(modes)

    def convert_by_mode(self,
                        dataset_path: str,
                        out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, smpl, meta
                stored in HumanData() format
        """

        # use HumanData to store all data
        human_data = HumanData()

        # find sequences
        seq_ps = sorted(glob.glob(os.path.join(dataset_path, 'sequenceFiles', mode, '*.pkl')))

        # build smpl model
        smpl_gendered = {}
        for gender in ['male', 'female', 'neutral']:
            smpl_gendered[gender] = build_body_model(
                dict(
                    type='SMPL',
                    keypoint_src='smpl_45',
                    keypoint_dst='smpl_45',
                    model_path='data/body_models/smpl',
                    gender=gender,
                    num_betas=10,
                    use_pca=False,
                    batch_size=1)).to(self.device)

        # initialize
        smpl_ = {}
        for key in self.smpl_shape.keys():
            smpl_[key] = []
        bboxs_ = {}
        for key in ['bbox_xywh']:  
            bboxs_[key] = []
        image_path_, keypoints2d_original_ = [], []
        keypoints2d_smpl_, keypoints3d_smpl_ = [], []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length', 'height', 'width', 
                         'gender', 'track_id', 'sequence_name', 'RT']:
            meta_[meta_key] = []

        seed = '240116'
        size = 999

        # add track id
        random_ids = np.random.RandomState(seed=int(seed)).permutation(999999)
        used_id_num = 0


        for seq_p in tqdm(seq_ps):

            # load sequence annotation
            with open(seq_p, 'rb') as f:
                data = pickle.load(f, encoding='latin1')

            seq = os.path.basename(seq_p).split('.')[0]

            image_ps = sorted(glob.glob(os.path.join(dataset_path, 'imageFiles', seq, '*.jpg')))
            frame_len = len(image_ps)
            img_sample = cv2.imread(image_ps[0])
            height, width = img_sample.shape[:2]

            # load smpl params
            smpl_param = {}
            smpl_param['global_orient'] = np.array(data['poses'])[:, :, :3]
            smpl_param['body_pose'] = np.array(data['poses'])[:, :, 3:]
            try:
                smpl_param['betas'] = np.array([betas[:10] for betas in data['betas']]).reshape(-1, 1, 10).repeat(frame_len, axis=1).reshape(-1, frame_len, 10)
            except:
                pdb.set_trace()
            smpl_param['transl'] = np.array(data['trans'])

            # load gender
            genders = [] 
            for gender in data['genders']:
                if gender == 'm':
                    genders.append('male')
                if gender == 'f':
                    genders.append('female')
                if gender == 'n':
                    genders.append('neutral')    
            
            # load camera and build camera
            intrinsics = np.array(data['cam_intrinsics'])
            extrinsics = np.array(data['cam_poses'])
            focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
            principal_point = [intrinsics[0, 2], intrinsics[1, 2]]

            # build camera
            camera = build_cameras(
                dict(
                    type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=focal_length,
                    image_size=(width, height),
                    principal_point=principal_point)).to(self.device)
            
            for gid in range(len(genders)):
                
                track_id = random_ids[used_id_num]
                used_id_num += 1

                body_model_param_tensor = {key: torch.tensor(
                        np.array(smpl_param[key][gid:gid+1, ...].reshape(self.smpl_shape[key])),
                                device=self.device, dtype=torch.float32)
                                for key in smpl_param.keys()}
                output = smpl_gendered[genders[gid]](**body_model_param_tensor, return_verts=False)
                kps3d = output['joints'].detach().cpu().numpy()
                
                # get pelvis world and transl
                pelvis_world = kps3d[:, get_keypoint_idx('pelvis', 'smpl'), :] 
                transl = smpl_param['transl'][gid, ...]
                global_orient = smpl_param['global_orient'][gid, ...]
                body_pose = smpl_param['body_pose'][gid, ...]
                betas = smpl_param['betas'][gid, ...]
                
                # batch transform smpl to camera frame
                global_orient, transl = batch_transform_to_camera_frame(
                global_orient, transl, pelvis_world, extrinsics)

                output = smpl_gendered[genders[gid]](
                    global_orient=torch.Tensor(global_orient).to(self.device),
                    body_pose=torch.Tensor(body_pose).to(self.device),
                    betas=torch.Tensor(betas).to(self.device),
                    transl=torch.Tensor(transl).to(self.device),
                    return_verts=False, )
                smpl_joints = output['joints']
                kps3d_c = smpl_joints.detach().cpu().numpy()
                kps2d = camera.transform_points_screen(smpl_joints)[..., :2].detach().cpu().numpy()


                # test 2d overlay
                # for kp in kps2d[0]:
                #     if  0 < kp[0] < width and 0 < kp[1] < height: 
                #         cv2.circle(img_sample, (int(kp[0]), int(kp[1])), 3, (0,0,255), 1)
                #     pass
                # # write image
                # os.makedirs(f'{out_path}', exist_ok=True)
                # cv2.imwrite(f'{out_path}/{os.path.basename(seq)}.jpg', img_sample)

                # append bbox
                for kp2d in kps2d:
                    # get bbox
                    bbox_xyxy = self._keypoints_to_scaled_bbox(kp2d, scale=self.misc_config['bbox_scale'])
                    bbox_xywh = self._xyxy2xywh(bbox_xyxy)
                    bboxs_['bbox_xywh'].append(bbox_xywh)

                # append image path
                image_paths = [imgp.replace(f'{dataset_path}/', '') for imgp in image_ps]
                image_path_ += image_paths

                # append keypoints
                keypoints2d_smpl_.append(kps2d)
                keypoints3d_smpl_.append(kps3d_c)

                # append smpl
                smpl_['global_orient'].append(global_orient)
                smpl_['body_pose'].append(body_pose)
                smpl_['betas'].append(betas)
                smpl_['transl'].append(transl)

                # append meta
                meta_['principal_point'] += [principal_point for pp in range(frame_len)]
                meta_['focal_length'] += [focal_length for fl in range(frame_len)]
                meta_['height'] += [height for h in range(frame_len)]
                meta_['width'] += [width for w in range(frame_len)]
                meta_['RT'] += [extrinsics[rt] for rt in range(frame_len)]
                meta_['track_id'] += [track_id for tid in range(frame_len)]
                meta_['gender'] += [genders[gid] for g in range(frame_len)]
                meta_['sequence_name'] += [f'{seq}_{track_id}' for sn in range(frame_len)]

        size_i = min(size, len(seq_ps))

        # append smpl
        for key in smpl_.keys():
            smpl_[key] = np.concatenate(
                smpl_[key], axis=0).reshape(self.smpl_shape[key])
        human_data['smpl'] = smpl_

        # append bbox
        for key in bboxs_.keys():
            bbox_ = np.array(bboxs_[key]).reshape((-1, 4))
            # add confidence
            conf_ = np.ones(bbox_.shape[0])
            bbox_ = np.concatenate([bbox_, conf_[..., None]], axis=-1)
            human_data[key] = bbox_

        # append keypoints 2d
        keypoints2d = np.concatenate(
            keypoints2d_smpl_, axis=0).reshape(-1, 45, 2)
        keypoints2d_conf = np.ones([keypoints2d.shape[0], 45, 1])
        keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf],
                                        axis=-1)
        keypoints2d, keypoints2d_mask = \
                convert_kps(keypoints2d, src='smpl_45', dst='human_data')
        human_data['keypoints2d_smpl'] = keypoints2d
        human_data['keypoints2d_smpl_mask'] = keypoints2d_mask

        # append keypoints 3d
        keypoints3d = np.concatenate(
            keypoints3d_smpl_, axis=0).reshape(-1, 45, 3)
        keypoints3d_conf = np.ones([keypoints3d.shape[0], 45, 1])
        keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf],
                                        axis=-1)
        keypoints3d, keypoints3d_mask = \
                convert_kps(keypoints3d, src='smpl_45', dst='human_data')
        human_data['keypoints3d_smpl'] = keypoints3d
        human_data['keypoints3d_smpl_mask'] = keypoints3d_mask

        # append image path
        human_data['image_path'] = image_path_

        # append meta
        human_data['meta'] = meta_

        # append misc
        human_data['misc'] = self.misc_config
        human_data['config'] = f'pw3d_{mode}'

        # save
        os.makedirs(f'{out_path}', exist_ok=True)
        out_file = f'{out_path}/pw3d_{mode}_{seed}_{"{:03d}".format(size_i)}.npz'
        human_data.dump(out_file)






        # pdb.set_trace()


