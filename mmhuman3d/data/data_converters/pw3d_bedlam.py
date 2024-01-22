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
from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

import pdb

@DATA_CONVERTERS.register_module()
class Pw3dBedlamConverter(BaseModeConverter):
    """3D Poses in the Wild dataset `Recovering Accurate 3D Human Pose in The
    Wild Using IMUs and a Moving Camera' ECCV'2018 More details can be found in
    the `paper.

    <https://virtualhumans.mpi-inf.mpg.de/papers/vonmarcardECCV18/
    vonmarcardECCV18.pdf>`__ .

    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """

    ACCEPTED_MODES = ['train']

    def __init__(self, modes: List = []):

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.misc_config = dict(
            bbox_source='keypoints2d_smplx',
            smpl_source='original',
            cam_param_type='prespective',
            bbox_scale=1.2,
            kps3d_root_aligned=False,
            flat_hand_mean=False,
            has_gender=True,
        )
        
        self.smplx_shape = {
            'betas': (-1, 10),
            'transl': (-1, 3),
            'global_orient': (-1, 3),
            'body_pose': (-1, 21, 3),
            'left_hand_pose': (-1, 15, 3),
            'right_hand_pose': (-1, 15, 3),
            # 'leye_pose': (-1, 3),
            # 'reye_pose': (-1, 3),
            # 'jaw_pose': (-1, 3),
        }
        self.anno_key_map = {
            'shape': 'betas',
            'trans': 'transl',
            'root_pose': 'global_orient',
            'body_pose': 'body_pose', 
        }
        super(Pw3dBedlamConverter, self).__init__(modes)


    def _fullpose_to_params(self, fullpose):
        
        fullpose = fullpose.reshape(-1, 55, 3)
        params = {}
        params['global_orient'] = fullpose[:, 0].reshape(-1, 3)
        params['body_pose'] = fullpose[:, 1:22].reshape(-1, 63)
        params['jaw_pose'] = fullpose[:, 22].reshape(-1, 3)
        params['leye_pose'] = fullpose[:, 23].reshape(-1, 3)
        params['reye_pose'] = fullpose[:, 24].reshape(-1, 3)
        params['left_hand_pose'] = fullpose[:, 25:40].reshape(-1, 45)
        params['right_hand_pose'] = fullpose[:, 40:55].reshape(-1, 45)

        return params
    
    def get_transform(self, center, scale, res, rot=0):
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1] / 2
            t_mat[1, 2] = -res[0] / 2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
        return t

    def convert_by_mode(self,
                        dataset_path: str,
                        out_path: str,
                        mode: str,
                        enable_multi_human_data: bool = False) -> dict:
        """
        Use bedlam pose to replace neural annot
        """
        # use HumanData to store all data
        human_data = HumanData()

        # initialize output for human_data
        smplx_, smplx_extra_ = {}, {}
        for key in self.smplx_shape.keys():
            smplx_[key] = []
        keypoints2d_smplx_, keypoints3d_smplx_, = [], []
        keypoints2d_orig_ = []
        bboxs_ = {}
        for bbox_name in ['bbox_xywh']:
            bboxs_[bbox_name] = []
        meta_ = {}
        for key in ['focal_length', 'principal_point', 'height', 'width']:
            meta_[key] = []
        image_path_ = []

        # load train val test split
        anno_p = os.path.join(dataset_path, f'3DPW_{mode}_reformat.json')
        with open(anno_p, 'r') as f:
            info_annos = json.load(f)

        # load smplx annotaion
        smplx_p = os.path.join(dataset_path, f'3DPW_{mode}_SMPLX_NeuralAnnot.json')
        with open(smplx_p, 'r') as f:
            smplx_annos = json.load(f)

        # load annot - download from bedlam website (3dpw_train_smplx.npz) and rename
        annot_param = dict(np.load(f'{dataset_path}/3dpw_bedlam_{mode}_smplx.npz', allow_pickle=True))
        # dict_keys(['imgname', 'center', 'scale', 'pose_cam', 'shape', 'gender', 'gtkps', 'smplx_pose', 'smplx_shape', 'smplx_trans', 'cam_int'])


        # verify valid image and smplx
        print('Selecting valid image and smplx instances...')
        smplx_instances = list(smplx_annos.keys())
        for sid in smplx_instances:
            if sid not in info_annos.keys():
                smplx_annos.pop(sid)
        targeted_frame_ids = list(smplx_annos.keys())
        
        # init seed and size
        seed, size = '230821', '99999'
        size_i = min(int(size), len(targeted_frame_ids))
        random.seed(int(seed))
        targeted_frame_ids = targeted_frame_ids[:size_i]

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

        print('Converting...')
        for sid in tqdm(targeted_frame_ids):

            smplx_anno = smplx_annos[sid]
            camera_param = info_annos[sid]['cam_param']
            info_anno = info_annos[sid]

            # get bbox
            width, height = info_anno['width'], info_anno['height']
            bbox_xywh = info_anno['bbox']
            if bbox_xywh[2] * bbox_xywh[3] > 0:
                bbox_xywh.append(1)
            else:
                bbox_xywh.append(0)

            # get image path
            imgp = os.path.join(dataset_path, 'imageFiles', str(info_anno['image_name']))
            if not os.path.exists(imgp):
                pdb.set_trace()
                print('missing image: ', imgp)
                continue
            image_path = imgp.replace(f'{dataset_path}{os.path.sep}', '')

            # look for the same path in annot_param
            neural_pose = np.array(smplx_anno['body_pose']).reshape(-1, 3)
            losses, aids = [], []
            for aid, imgname in enumerate(annot_param['imgname']):
                if imgname == image_path:
                    
                    bedlam_pose = annot_param['smplx_pose'][aid].reshape(-1, 3)[1:22, :]

                    # cal error
                    loss = torch.abs(torch.tensor(aa_to_rotmat(bedlam_pose)) - 
                                     torch.tensor(aa_to_rotmat(neural_pose)))
                    loss = loss.detach().cpu().numpy().mean()
                    
                    losses.append(loss)
                    aids.append(aid)

            if len(aids) == 0:
                print('missing bedlam annotation: ', image_path)
                continue

            # select one with lowest loss
            aid = aids[np.argmin(losses)]
            bedlam_betas = annot_param['smplx_shape'][aid][:10].reshape(1, 10)
            bedlam_global_orient = annot_param['smplx_pose'][aid].reshape(-1, 3)[0:1, :]
            bedlam_pose = annot_param['smplx_pose'][aid].reshape(-1, 3)[1:22, :]
            bedlam_lhand_pose = annot_param['smplx_pose'][aid].reshape(-1, 3)[25:40, :]
            bedlam_rhand_pose = annot_param['smplx_pose'][aid].reshape(-1, 3)[40:55, :]

            # get camera parameters and create camera
            focal_length = camera_param['focal']
            principal_point = camera_param['princpt']
            camera = build_cameras(
                dict(
                    type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=focal_length,
                    image_size=(width, height),
                    principal_point=principal_point)).to(self.device)

            # reformat smplx_anno
            smplx_param = {}
            for key in self.anno_key_map.keys():
                smplx_key = self.anno_key_map[key]
                smplx_shape = self.smplx_shape[smplx_key]
                smplx_param[smplx_key] = np.array(smplx_anno[key]).reshape(smplx_shape)

            # change pose to bedlam pose
            smplx_param['betas'] = bedlam_betas.reshape(-1, 10)
            smplx_param['global_orient'] = bedlam_global_orient.reshape(-1, 3)
            smplx_param['body_pose'] = bedlam_pose.reshape(-1, 21, 3)
            smplx_param['left_hand_pose'] = bedlam_lhand_pose.reshape(-1, 15, 3)
            smplx_param['right_hand_pose'] = bedlam_rhand_pose.reshape(-1, 15, 3)
            
            # build smplx model and get output
            intersect_keys = list(
                set(smplx_param.keys()) & set(self.smplx_shape.keys()))
            body_model_param_tensor = {
                key: torch.tensor(
                    np.array(smplx_param[key]).reshape(self.smplx_shape[key]),
                    device=self.device, dtype=torch.float32)
                for key in intersect_keys}
            output = smplx_model(**body_model_param_tensor, return_joints=True)

            # get kps2d and 3d
            keypoints_3d = output['joints']
            keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
            keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
            keypoints_3d = keypoints_3d.detach().cpu().numpy()

            # get kps2d original
            j2d_body = np.array(info_anno['openpose_result'])
            # j2d_ft = info_anno['foot_kpts']
            # j2d_lh = info_anno['lefthand_kpts']
            # j2d_rh = info_anno['righthand_kpts']
            # j2d_face = info_anno['face_kpts']

            # j2d = np.concatenate([j2d_body, j2d_ft, j2d_lh, j2d_rh, j2d_face], axis=0)
            j2d = j2d_body.reshape(-1, 3)

            # change conf 0, 1, 2 to 0, 1
            j2d_conf = j2d[:, -1]
            j2d_conf = (j2d_conf != 0).astype(int)
            j2d[:, -1] = j2d_conf

            # print('j2d_body', len(j2d_body))
            # print('j2d_lh', len(j2d_lh))
            # print('j2d_rh', len(j2d_rh))
            # print('j2d_face', len(j2d_face))

            # append image path
            image_path_.append(image_path)

            # append keypoints2d and 3d
            keypoints2d_smplx_.append(keypoints_2d)
            keypoints3d_smplx_.append(keypoints_3d)
            keypoints2d_orig_.append(j2d)

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
        for key in smplx_extra_.keys():
            smplx_[key] = np.array(smplx_extra_[key])
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
            keypoints2d_orig_, axis=0).reshape(-1, 18, 3)
        # keypoints2d_orig_conf = np.ones([keypoints2d_orig.shape[0], 18, 1])
        # keypoints2d_orig = np.concatenate( 
        #     [keypoints2d_orig[:, :, :2], keypoints2d_orig_conf], axis=-1)
        keypoints2d_orig, keypoints2d_orig_mask = \
                convert_kps(keypoints2d_orig, src='pw3d', dst='human_data')
        human_data['keypoints2d_original'] = keypoints2d_orig
        human_data['keypoints2d_original_mask'] = keypoints2d_orig_mask

        # misc
        human_data['misc'] = self.misc_config
        human_data['config'] = f'pw3d_neural_annot_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path,
            f'pw3d_bedlam_{mode}_{seed}_{"{:05d}".format(size_i)}.npz')
        human_data.dump(out_file)

    # def convert_by_mode(self,
    #                     dataset_path: str,
    #                     out_path: str,
    #                     mode: str) -> dict:
    #     """
    #     Args:
    #         dataset_path (str): Path to directory where raw images and
    #         annotations are stored.
    #         out_path (str): Path to directory to save preprocessed npz file
    #         mode (str): Mode in accepted modes

    #     Returns:
    #         dict:
    #             A dict containing keys image_path, bbox_xywh, smpl, meta
    #             stored in HumanData() format
    #     """

    #     # use HumanData to store all data
    #     human_data = HumanData()

    #     # build smpl model
    #     smplx_gendered = {}
    #     for gender in ['male', 'female', 'neutral']:
    #         smplx_gendered[gender] = build_body_model(
    #             dict(
    #                 type='SMPLX',
    #                 keypoint_src='smplx',
    #                 keypoint_dst='smplx',
    #                 model_path='data/body_models/smplx',
    #                 gender=gender,
    #                 num_betas=10,
    #                 flat_hand_mean=self.misc_config['flat_hand_mean'],     
    #                 use_face_contour=True,
    #                 use_pca=False,
    #                 batch_size=1)).to(self.device)

    #     # initialize
    #     smplx_ = {}
    #     for key in self.smplx_shape.keys():
    #         smplx_[key] = []
    #     bboxs_ = {}
    #     for key in ['bbox_xywh']:  
    #         bboxs_[key] = []
    #     image_path_, keypoints2d_original_ = [], []
    #     keypoints2d_smpl_, keypoints3d_smpl_ = [], []
    #     meta_ = {}
    #     for meta_key in ['principal_point', 'focal_length', 'height', 'width', 
    #                      'gender', 'track_id', 'sequence_name', 'RT']:
    #         meta_[meta_key] = []

    #     seed = '240116'
    #     size = 99999

    #     # add track id
    #     random_ids = np.random.RandomState(seed=int(seed)).permutation(999999)
    #     used_id_num = 0

    #     # load annot - download from bedlam website (3dpw_train_smplx.npz) and rename
    #     annot_param = dict(np.load(f'{dataset_path}/3dpw_bedlam_{mode}_smplx.npz', allow_pickle=True))
    #     # dict_keys(['imgname', 'center', 'scale', 'pose_cam', 'shape', 'gender', 'gtkps', 'smplx_pose', 'smplx_shape', 'smplx_trans', 'cam_int'])

    #     seq_previous, height, width = '', 0, 0
    #     for aid, image_path in enumerate(tqdm(annot_param['imgname'])):

    #         seq = image_path.split('/')[-2]

    #         # get width and height
    #         if seq != seq_previous:
    #             img_sample = cv2.imread(f'{dataset_path}/{image_path}')
    #             height, width = img_sample.shape[:2]
    #             seq_previous = seq

    #         # load smplx params
    #         smplx_param = self._fullpose_to_params(annot_param['smplx_pose'][aid])
    #         smplx_param['betas'] = annot_param['smplx_shape'][aid][:10].reshape(1, 10)
    #         smplx_param['transl'] = annot_param['smplx_trans'][aid].reshape(1, 3)
            
    #         # load gender
    #         gender_a = annot_param['gender'][aid]
    #         if gender_a == 'm':
    #             gender = 'male'
    #         if gender_a == 'f':
    #             gender = 'female'
    #         if gender_a == 'n':
    #             gender = 'neutral'   

    #         # load camera and build camera
    #         intrinsics = annot_param['cam_int'][aid]
    #         # extrinsics = annot_param['pose_cam'][aid]            
    #         scale = annot_param['scale'][aid]
    #         center = annot_param['center'][aid]

    #         # bbox2image = self.get_transform(center, scale, (224, 224))

    #         # K1 = (intrinsics.T @ bbox2image.T).T
    #         K1 = intrinsics

    #         focal_length = [K1[0, 0] , K1[1, 1]]
    #         principal_point = [K1[0, 2] , K1[1, 2]]

    #         # focal_length = [intrinsics[0, 0] / scale, intrinsics[1, 1] / scale]
    #         # principal_point = [intrinsics[0, 2] - center[0], intrinsics[1, 2] - center[1]]

    #         # build camera
    #         camera = build_cameras(
    #             dict(
    #                 type='PerspectiveCameras',
    #                 convention='opencv',
    #                 in_ndc=False,
    #                 focal_length=focal_length,
    #                 image_size=(width, height),
    #                 principal_point=principal_point)).to(self.device)

    #         body_model_param_tensor = {key: torch.tensor(
    #                 np.array(smplx_param[key]),
    #                         device=self.device, dtype=torch.float32)
    #                         for key in smplx_param.keys()}
    #         output = smplx_gendered[gender](**body_model_param_tensor, return_verts=True)
    #         kps3d = output['joints'].detach().cpu().numpy()



    #         # get pelvis world and transl
    #         # pelvis_world = kps3d[:, get_keypoint_idx('pelvis', 'smpl'), :] 
    #         # transl = smpl_param['transl'][gid, ...]
    #         # global_orient = smpl_param['global_orient'][gid, ...]
    #         # body_pose = smpl_param['body_pose'][gid, ...]
    #         # betas = smpl_param['betas'][gid, ...]
            
    #         # # batch transform smpl to camera frame
    #         # global_orient, transl = batch_transform_to_camera_frame(
    #         # global_orient, transl, pelvis_world, extrinsics)

    #         # output = smpl_gendered[genders[gid]](
    #         #     global_orient=torch.Tensor(global_orient).to(self.device),
    #         #     body_pose=torch.Tensor(body_pose).to(self.device),
    #         #     betas=torch.Tensor(betas).to(self.device),
    #         #     transl=torch.Tensor(transl).to(self.device),
    #         #     return_verts=False, )
    #         smpl_joints = output['joints']
    #         kps2d = camera.transform_points_screen(smpl_joints)[..., :2].detach().cpu().numpy()

    #         verts = camera.transform_points_screen(output['vertices'])[..., :2].detach().cpu().numpy()


    #         # test 2d overlay
    #         img_sample = cv2.imread(f'{dataset_path}/{image_path}')
    #         # img_sample = cv2.rotate(img_sample, cv2.ROTATE_90_CLOCKWISE)
    #         for kp in verts[0]:
    #             kp = kp / 1000
    #             if  0 < kp[0] < width and 0 < kp[1] < height: 
    #                 cv2.circle(img_sample, (int(kp[0]), int(kp[1])), 3, (0,0,255), 1)
    #             pass
    #         # write image
    #         os.makedirs(f'{out_path}', exist_ok=True)
    #         cv2.imwrite(f'{out_path}/{os.path.basename(seq)}.jpg', img_sample)

    #         # append bbox
    #         # for kp2d in kps2d:
    #         #     # get bbox
    #         #     bbox_xyxy = self._keypoints_to_scaled_bbox(kp2d, scale=self.misc_config['bbox_scale'])
    #         #     bbox_xywh = self._xyxy2xywh(bbox_xyxy)
    #         #     bboxs_['bbox_xywh'].append(bbox_xywh)

    #         pdb.set_trace()

    #         # append image path
    #         image_path_ += [image_path]

    #         # append keypoints
    #         keypoints2d_smpl_.append(kps2d)
    #         keypoints3d_smpl_.append(kps3d_c)

    #         # append smplx
    #         for key in smplx_.keys():
    #             smplx_[key].append(smplx_param[key])

    #         # append meta
    #         meta_['principal_point'] += [principal_point]
    #         meta_['focal_length'] += [focal_length]
    #         meta_['height'] += [height]
    #         meta_['width'] += [width]
    #         meta_['sequence_name'] += [seq]

    #         # meta_['principal_point'] += [principal_point for pp in range(frame_len)]
    #         # meta_['focal_length'] += [focal_length for fl in range(frame_len)]
    #         # meta_['height'] += [height for h in range(frame_len)]
    #         # meta_['width'] += [width for w in range(frame_len)]
    #         # meta_['RT'] += [extrinsics[rt] for rt in range(frame_len)]
    #         # meta_['track_id'] += [track_id for tid in range(frame_len)]
    #         # meta_['sequence_name'] += [f'{seq}_{track_id}' for sn in range(frame_len)]

    #     size_i = min(size, len(seq_ps))

    #     # append smpl
    #     for key in smpl_.keys():
    #         smpl_[key] = np.concatenate(
    #             smpl_[key], axis=0).reshape(self.smpl_shape[key])
    #     human_data['smpl'] = smpl_

    #     # append bbox
    #     for key in bboxs_.keys():
    #         bbox_ = np.array(bboxs_[key]).reshape((-1, 4))
    #         # add confidence
    #         conf_ = np.ones(bbox_.shape[0])
    #         bbox_ = np.concatenate([bbox_, conf_[..., None]], axis=-1)
    #         human_data[key] = bbox_

    #     # append keypoints 2d
    #     keypoints2d = np.concatenate(
    #         keypoints2d_smpl_, axis=0).reshape(-1, 45, 2)
    #     keypoints2d_conf = np.ones([keypoints2d.shape[0], 45, 1])
    #     keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf],
    #                                     axis=-1)
    #     keypoints2d, keypoints2d_mask = \
    #             convert_kps(keypoints2d, src='smpl_45', dst='human_data')
    #     human_data['keypoints2d_smpl'] = keypoints2d
    #     human_data['keypoints2d_smpl_mask'] = keypoints2d_mask

    #     # append keypoints 3d
    #     keypoints3d = np.concatenate(
    #         keypoints3d_smpl_, axis=0).reshape(-1, 45, 3)
    #     keypoints3d_conf = np.ones([keypoints3d.shape[0], 45, 1])
    #     keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf],
    #                                     axis=-1)
    #     keypoints3d, keypoints3d_mask = \
    #             convert_kps(keypoints3d, src='smpl_45', dst='human_data')
    #     human_data['keypoints3d_smpl'] = keypoints3d
    #     human_data['keypoints3d_smpl_mask'] = keypoints3d_mask

    #     # append image path
    #     human_data['image_path'] = image_path_

    #     # append meta
    #     human_data['meta'] = meta_

    #     # append misc
    #     human_data['misc'] = self.misc_config
    #     human_data['config'] = f'pw3d_{mode}'

    #     # save
    #     os.makedirs(f'{out_path}', exist_ok=True)
    #     out_file = f'{out_path}/pw3d_{mode}_{seed}_{"{:03d}".format(size_i)}.npz'
    #     human_data.dump(out_file)
