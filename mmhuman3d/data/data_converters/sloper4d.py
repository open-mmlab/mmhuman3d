import glob
import os
import pdb
import random
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

from scipy.spatial.transform import Rotation

@DATA_CONVERTERS.register_module()
class Sloper4dConverter(BaseModeConverter):

    ACCEPTED_MODES = ['train']

    def __init__(self, modes=[], *args, **kwargs):

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.misc_config = dict(
            bbox_source='by_dataset',
            smpl_source='original',
            cam_param_type='prespective',
            kps3d_root_aligned=False,
        )
        INTRINSICS = [599.628, 599.466, 971.613, 540.258]
        DIST       = [0.003, -0.003, -0.001, 0.004, 0.0]
        LIDAR2CAM  = [[[-0.0355545576, -0.999323133, -0.0094419378, -0.00330376451], 
                    [0.00117895777, 0.00940596282, -0.999955068, -0.0498469479], 
                    [0.999367041, -0.0355640917, 0.00084373493, -0.0994979365], 
                    [0.0, 0.0, 0.0, 1.0]]]
        self.default_camera = {'fps':20, 'width': 1920, 'height':1080, 
                'intrinsics':INTRINSICS, 'lidar2cam':LIDAR2CAM, 'dist':DIST}
        self.smpl_shape = {
            'body_pose': (-1, 69),
            'betas': (-1, 10),
            'global_orient': (-1, 3),
            'transl': (-1, 3),}
        
        super(Sloper4dConverter, self).__init__(modes)

    
    def _world_to_camera(self, X, extrinsic_matrix):
        n = X.shape[0]
        X = np.concatenate((X, np.ones((n, 1))), axis=-1).T
        X = np.dot(extrinsic_matrix, X).T
        return X[..., :3]

    
    def _camera_to_pixel(self, X, intrinsics, distortion_coefficients=[0, 0, 0, 0, 0]):
        # focal length
        f = intrinsics[:2]
        # center principal point
        c = intrinsics[2:]
        k = np.array([distortion_coefficients[0],
                    distortion_coefficients[1], distortion_coefficients[4]])
        p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
        XX = X[..., :2] / (X[..., 2:])
        # XX = pd.to_numeric(XX, errors='coere')
        r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

        radial = 1 + np.sum(k * np.concatenate((r2, r2**2, r2**3),
                            axis=-1), axis=-1, keepdims=True)

        tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
        XXX = XX * (radial + tan) + r2 * p[..., ::-1]
        return f * XXX + c


    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        print('Converting Sloper4d dataset...')
        
        # parse sequences
        seqs = glob.glob(os.path.join(dataset_path, 'seq*'))

        # build smpl model
        gendered_model = {}
        for gender in ['male', 'female', 'neutral']:
            gendered_model[gender] = build_body_model(
                dict(
                    type='SMPL',
                    keypoint_src='smpl_45',
                    keypoint_dst='smpl_45',
                    model_path='data/body_models/smpl',
                    gender=gender,
                    num_betas=10,
                    use_pca=False,
                    batch_size=1)).to(self.device)
            
        # use HumanData to store the data
        human_data = HumanData()

        # initialize
        smpl_ = {}
        for key in self.smpl_shape.keys():
            smpl_[key] = []
        bboxs_ = {}
        for key in ['bbox_xywh']:  
            bboxs_[key] = []
        image_path_, keypoints2d_smpl_ = [], []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length', 'height', 'width']:
            meta_[meta_key] = []

        seed = '230714'
        size = 99

        # sort by seq
        for sid, seq in enumerate(seqs):

            seq_name = os.path.basename(seq)

            # load meta params
            meta_path = os.path.join(seq, 'dataset_params.json')
            meta_info = json.load(open(meta_path, 'r'))
            
            # load pickle annotations   
            pickle_path = os.path.join(seq, f'{seq_name}_labels.pkl')
            pickle_info = dict(np.load(pickle_path, allow_pickle=True))

            framerate = pickle_info['framerate']

            # get frame name for sequence
            frame_names = pickle_info['RGB_frames']['file_basename']

            # get lidar transform (world2lidar)
            lidar_traj = pickle_info['first_person']['lidar_traj']
            world2lidar   = np.array([np.eye(4)] * len(frame_names))
            world2lidar[:, :3, :3] = Rotation.from_quat(lidar_traj[:len(frame_names), 4: 8]).inv().as_matrix()
            world2lidar[:, :3, 3:] = -world2lidar[:, :3, :3] @ lidar_traj[:len(frame_names), 1:4].reshape(-1, 3, 1)

            # get bbox (in xyxy format)
            bbox = pickle_info['RGB_frames']['bbox'] 
            bbox_xywh = [self._xyxy2xywh(b) + [1] if len(b) > 0
                         else [0, 0, 0, 0, 0] for b in bbox]

            # get skel_2d
            skel_2d = pickle_info['RGB_frames']['skel_2d']

            # get smpl params
            seq_length = len(frame_names)

            # prepare smpl params
            sp = pickle_info['second_person']
            global_orient = np.array(sp['opt_pose'][:seq_length][:, :3]).reshape(-1, 3)
            body_pose = np.array(sp['opt_pose'][:seq_length][:, 3:]).reshape(-1, 23 * 3)
            betas = np.array(sp['beta']).reshape(-1, 10)
            if len(betas) == 1:
                betas = betas.repeat(seq_length, axis=0)
            transl = np.array(sp['trans'][:seq_length]).reshape(-1, 3)
            gender = sp['gender']

            # print(global_orient.shape, body_pose.shape, betas.shape, transl.shape)
            # pdb.set_trace()

            output = gendered_model[gender](
                global_orient=torch.Tensor(global_orient).to(self.device),
                body_pose=torch.Tensor(body_pose).to(self.device),
                betas=torch.Tensor(betas).to(self.device),
                transl=torch.Tensor(transl).to(self.device),
                return_verts=True, )
            kps3d = output['joints'].detach().cpu().numpy()
            pelvis_world = kps3d[:, get_keypoint_idx('pelvis', 'smpl'), :] 

            # load camera params / or use default params
            if 'RGB_info' in pickle_info.keys():
                cam_param = pickle_info['RGB_info']
            else:
                cam_param = self.default_camera
            # prepare camera params
            extrinsics = pickle_info['RGB_frames']['cam_pose']
            height = cam_param['height']
            width = cam_param['width']
            
            # build camera
            intrinsics = np.eye(3)
            intrinsics[0, 0] = cam_param['intrinsics'][0]
            intrinsics[1, 1] = cam_param['intrinsics'][1]
            intrinsics[0, 2] = cam_param['intrinsics'][2]
            intrinsics[1, 2] = cam_param['intrinsics'][3]

            camera = build_cameras(
                    dict(
                        type='PerspectiveCameras',
                        convention='opencv',
                        in_ndc=False,
                        focal_length=(intrinsics[0, 0], intrinsics[1, 1]),
                        principal_point=(intrinsics[0, 2], intrinsics[1, 2]),
                        image_size=(width, height),
                        )).to(self.device)
            
            # intrinsics_undistorted, _ = cv2.getOptimalNewCameraMatrix(
            #     intrinsics,
            #     np.array(cam_param['dist']),
            #     (width, height),
            #     alpha=0)
            
            # camera_undist = build_cameras(
            #         dict(
            #             type='PerspectiveCameras',
            #             convention='opencv',
            #             in_ndc=False,
            #             focal_length=(intrinsics_undistorted[0, 0], intrinsics_undistorted[1, 1]),
            #             principal_point=(intrinsics_undistorted[0, 2], intrinsics_undistorted[1, 2]),
            #             image_size=(width, height),
            #             )).to(self.device)
            
            # prepare meta
            focal_length = (cam_param['intrinsics'][0], cam_param['intrinsics'][1])
            principal_point = (cam_param['intrinsics'][2], cam_param['intrinsics'][3])

            # batch transform smpl to camera frame
            global_orient, transl = batch_transform_to_camera_frame(
                global_orient, transl, pelvis_world, extrinsics)

            # sort by frames
            for fid, fname in tqdm(enumerate(frame_names), total=seq_length, 
                                   desc=f'Converting {seq_name}, Seq {sid+1} / total {len(seqs)}', leave=False):
                
                imgp = os.path.join(seq, 'rgb_data', f'{seq_name}_imgs', fname)
                image_path = imgp.replace(dataset_path + os.path.sep, '')

                # pickle info keys 'file_basename', 'bbox', 'skel_2d', 
                # 'human_points', 'extrinsic', 'cam_pose', 'tstamp', 'beta', 
                # 'lidar_tstamps', 'lidar_fname', 'smpl_pose', 'global_trans' 

                extrinsic = extrinsics[fid]
                # transform to cam space
                j3d_w = kps3d[fid]
                j3d_c = self._world_to_camera(j3d_w, extrinsic)

                # intrinsics
                j2d = camera.transform_points_screen(torch.tensor(
                        j3d_c.reshape(1, -1, 3), device=self.device, dtype=torch.float32))
                j2d = j2d[0, :, :2].detach().cpu().numpy()
                j2d_conf = np.ones((j2d.shape[0], 1))
                j2d_orig = np.concatenate((j2d, j2d_conf), axis=-1)

                # append kps2d
                keypoints2d_smpl_.append(j2d_orig)

                # append image path
                image_path_.append(image_path)


                # # test overlay j2d
                # img = cv2.imread(f'{dataset_path}/{image_path}')
                # for kp in j2d_orig:
                #     if  0 < kp[0] < 1920 and 0 < kp[1] < 1080: 
                #         cv2.circle(img, (int(kp[0]), int(kp[1])), 1, (0,0,255), -1)
                #     pass
                # # write image
                # os.makedirs(f'{out_path}', exist_ok=True)
                # cv2.imwrite(f'{out_path}/{seq_name}_{fname}.jpg', img)

            # append bbox xywh
            bboxs_['bbox_xywh'] += bbox_xywh

            # append meta
            meta_['principal_point'] += [principal_point] * seq_length
            meta_['focal_length'] += [focal_length] * seq_length
            meta_['height'] += [height] * seq_length
            meta_['width'] += [width] * seq_length

            # append smpl params
            smpl_['body_pose'] += body_pose.tolist()
            smpl_['betas'] += betas.tolist()
            smpl_['global_orient'] += global_orient.tolist()
            smpl_['transl'] += transl.tolist()

        size_i = min(size, len(seqs))
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
        for key in smpl_.keys():
            smpl_[key] = np.concatenate(
                smpl_[key], axis=0).reshape(self.smpl_shape[key])
        human_data['smpl'] = smpl_

        # keypoints2d_smplx
        keypoints2d_smpl = np.concatenate(
            keypoints2d_smpl_, axis=0).reshape(-1, 45, 3)
        keypoints2d_smpl, keypoints2d_smpl_mask = \
                convert_kps(keypoints2d_smpl, src='smpl_45', dst='human_data')
        human_data['keypoints2d_smpl'] = keypoints2d_smpl
        human_data['keypoints2d_smpl_mask'] = keypoints2d_smpl_mask

        # misc
        human_data['misc'] = self.misc_config
        human_data['config'] = f'sloper4d_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path,
            f'sloper4d_{mode}_{seed}_{"{:02d}".format(size_i)}.npz')
        human_data.dump(out_file)