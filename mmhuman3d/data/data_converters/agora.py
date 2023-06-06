import os
from typing import List, Tuple

import cv2
import numpy as np
# import pickle5 as pickle
import pickle
from tqdm import tqdm
import torch
import json
import pdb
import time

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class AgoraConverter(BaseModeConverter):
    """AGORA dataset
    `AGORA: Avatars in Geography Optimized for Regression Analysis' CVPR`2021
    More details can be found in the `paper
    <https://arxiv.org/pdf/2104.14643.pdf>`__.

    Args:
        modes (list): 'validation' or 'train' for accepted modes
        fit (str): 'smpl' or 'smplx for available body model fits
        res (tuple): (1280, 720) or (3840, 2160) for available image resolution
    """
    ACCEPTED_MODES = ['validation_3840', 'train_3840', 'train_1280', 'validation_1280']

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.misc_config = dict(bbox_body_scale=1.2, bbox_facehand_scale=1.0, bbox_source='keypoints2d_smplx',
                                cam_param_source='original', smplx_source='original')

        self.smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3),
                            'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3),
                            'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}

        super(AgoraConverter, self).__init__(modes)

    def _focalLength_mm2px(self, focalLength, principal):

        dslr_sens_width = 36
        dslr_sens_height = 20.25

        focal_pixel_x = (focalLength / dslr_sens_width) * principal[0] * 2
        focal_pixel_y = (focalLength / dslr_sens_height) * principal[1] * 2

        return np.array([focal_pixel_x, focal_pixel_y]).reshape(-1, 2)

    def _get_focal_length(self, imgPath):
        if 'hdri' in imgPath:
            focalLength = 50
        elif 'cam00' in imgPath:
            focalLength = 18
        elif 'cam01' in imgPath:
            focalLength = 18
        elif 'cam02' in imgPath:
            focalLength = 18
        elif 'cam03' in imgPath:
            focalLength = 18
        elif 'ag2' in imgPath:
            focalLength = 28
        else:
            focalLength = 28

        return focalLength

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:

        # check pytorch device
        device = self.device

        # confirm batch
        batch_info = mode.split('_')[0]
        res_info = mode.split('_')[1]
        if res_info == '1280':
            self.misc_config['image_size'] = (1280, 720)
        elif res_info == '3840':
            self.misc_config['image_size'] = (3840, 2160)

        # use HumanData to store all data
        human_data = HumanData()

        # initialize output for human_data
        smplx_ = {}
        for keys in self.smplx_shape.keys():
            smplx_[keys] = []
        keypoints2d_, keypoints3d_, = [], []
        bboxs_ = {}
        for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
            bboxs_[bbox_name] = []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length', 'gender']:
            meta_[meta_key] = []
        image_path_ = []

        seed, size = '230606', '999999'

        # build smplx model
        smplx_model = {}
        for gender in ['male', 'female', 'neutral']:
            smplx_model[gender] = build_body_model(dict(
                type='SMPLX',
                keypoint_src='smplx',
                keypoint_dst='smplx',
                model_path='data/body_models/smplx',
                gender=gender,
                num_betas=10,
                use_face_contour=True,
                flat_hand_mean=False,
                use_pca=False,
                batch_size=1
            )).to(device)

        # data info path
        data_info_path = os.path.join(dataset_path, f'AGORA_{batch_info}_fix_betas.json')

        # read data info
        with open(data_info_path, 'r') as f:
            param = json.load(f)
        image_param = param['images']
        anno_param = param['annotations']

        for anno_info in tqdm(anno_param, desc=f'Processing Agora {mode}'):

            image_id = anno_info['image_id']

            # get image details
            image_info = image_param[image_id]
            image_path = image_info[f"file_name_{self.misc_config['image_size'][0]}x"
                                    f"{self.misc_config['image_size'][1]}"]

            # collect bbox
            for key in ['bbox', 'face_bbox', 'lhand_bbox', 'rhand_bbox']:
                bboxs_[f'{key}_xywh'].append(np.array(anno_info[key] + [1]))

            # collect smplx_params
            smplx_path = os.path.join(dataset_path, anno_info['smplx_param_path'])
            smplx_param = pickle.load(open(smplx_path, 'rb'))

            for key in self.smplx_shape.keys():
                smplx_[key].append(smplx_param[key])

            # collect keypoints
            smplx_joints_2d_path = os.path.join(dataset_path, anno_info['smplx_joints_2d_path'])
            smplx_joints_2d = pickle.load(open(smplx_joints_2d_path, 'rb'))
            keypoints2d_.append(smplx_joints_2d)

            smplx_joints_3d_path = os.path.join(dataset_path, anno_info['smplx_joints_3d_path'])
            smplx_joints_3d = pickle.load(open(smplx_joints_3d_path, 'rb'))
            keypoints3d_.append(smplx_joints_3d)

            # get camera parameters
            principal_point = np.array([self.misc_config['image_size'][0] / 2, self.misc_config['image_size'][1] / 2])
            focal_length = self._focalLength_mm2px(self._get_focal_length(image_path), principal_point)

            # collect meta
            meta_['gender'].append(anno_info['gender'])
            meta_['principal_point'].append(principal_point)
            meta_['focal_length'].append(focal_length)

            # pdb.set_trace()
            # build camera
            # camera = build_cameras(
            # dict(
            #     type='PerspectiveCameras',
            #     convention='opencv',
            #     in_ndc=False,
            #     focal_length=focal_length,
            #     image_size=(self.misc_config['image_size'][0], self.misc_config['image_size'][1]),
            #     principal_point=np.array(principal_point).reshape(-1, 2))).to(device)

            # # test smplx
            # intersect_key = list(set(smplx_param.keys()) & set(self.smplx_shape.keys()))
            # body_model_param_tensor = {key: torch.tensor(
            #         np.array(smplx_param[key]).reshape(self.smplx_shape[key]),
            #                 device=device, dtype=torch.float32)
            #                 for key in intersect_key
            #                 if len(smplx_param[key]) > 0}
            # output = smplx_model[gender](**body_model_param_tensor, return_verts=True)
            # smplx_joints = output['joints']
            # kps2d = camera.transform_points_screen(smplx_joints)[..., :2].detach().cpu().numpy()

            # pdb.set_trace()

        # prepare for output
        # smplx
        for key in smplx_.keys():
            smplx_[key] = np.array(smplx_[key]).reshape(self.smplx_shape[key])
        human_data['smplx'] = smplx_
        print('Smpl and/or Smplx finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # bbox
        for key in bboxs_.keys():
            bbox_ = np.array(bboxs_[key]).reshape((-1, 5))
            human_data[key] = bbox_
        print('BBox generation finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # keypoints 2d
        keypoints2d = np.array(keypoints2d_).reshape(-1, 144, 2)
        keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
        keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
        keypoints2d, keypoints2d_mask = \
            convert_kps(keypoints2d, src='smplx', dst='human_data')
        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask

        # keypoints 3d
        keypoints3d = np.array(keypoints3d_).reshape(-1, 144, 3)
        keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
        keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
        keypoints3d, keypoints3d_mask = \
            convert_kps(keypoints3d, src='smplx', dst='human_data')
        human_data['keypoints3d'] = keypoints3d
        human_data['keypoints3d_mask'] = keypoints3d_mask

        # image path
        human_data['image_path'] = image_path_
        print('Image path writting finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # meta
        human_data['meta'] = meta_
        print('Meta writting finished at', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # store
        human_data['config'] = f'agora_{mode}'
        human_data['misc'] = self.misc_config

        size_i = str(max(int(size), len(anno_param)))

        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, f'agora_{mode}_{seed}_{"{:06d}".format(int(size_i))}_.npz')
        human_data.dump(out_file)


'''
    def __init__(self, modes: List = [], fit: str = 'smpl',
                 res: Tuple[int, int] = (1280, 720)) -> None:  # yapf: disable
        super(AgoraConverter, self).__init__(modes)
        accepted_fits = ['smpl', 'smplx']
        if fit not in accepted_fits:
            raise ValueError('Input fit not in accepted fits. \
                Use either smpl or smplx')
        self.fit = fit

        accepted_res = [(1280, 720), (3840, 2160)]
        if res not in accepted_res:
            raise ValueError('Input resolution not in accepted resolution. \
                Use either (1280, 720) or (3840, 2160)')
        self.res = res

    def get_global_orient(self,
                          imgPath,
                          df,
                          i,
                          pNum,
                          globalOrient=None,
                          meanPose=False):
        """Modified from https://github.com/pixelite1201/agora_evaluation/blob/
        master/agora_evaluation/projection.py specific to AGORA.

        Args:
            imgPath: image path
            df: annotation dataframe
            i: frame index
            pNum: person index
            globalOrient: original global orientation
            meanPose: Store True for mean pose from vposer

        Returns:
            globalOrient: rotated global orientation
        """
        if 'hdri' in imgPath:
            camYaw = 0
            camPitch = 0

        elif 'cam00' in imgPath:
            camYaw = 135
            camPitch = 30
        elif 'cam01' in imgPath:
            camYaw = -135
            camPitch = 30
        elif 'cam02' in imgPath:
            camYaw = -45
            camPitch = 30
        elif 'cam03' in imgPath:
            camYaw = 45
            camPitch = 30
        elif 'ag2' in imgPath:
            camYaw = 0
            camPitch = 15
        else:
            camYaw = df.iloc[i]['camYaw']
            camPitch = 0

        if meanPose:
            yawSMPL = 0
        else:
            yawSMPL = df.iloc[i]['Yaw'][pNum]

        # scans have a 90deg rotation, but for mean pose from vposer there is
        # no such rotation
        if meanPose:
            rotMat, _ = cv2.Rodrigues(
                np.array([[0, (yawSMPL) / 180 * np.pi, 0]], dtype=float))
        else:
            rotMat, _ = cv2.Rodrigues(
                np.array([[0, ((yawSMPL - 90) / 180) * np.pi, 0]],
                         dtype=float))

        camera_rotationMatrix, _ = cv2.Rodrigues(
            np.array([0, ((-camYaw) / 180) * np.pi, 0]).reshape(3, 1))
        camera_rotationMatrix2, _ = cv2.Rodrigues(
            np.array([camPitch / 180 * np.pi, 0, 0]).reshape(3, 1))

        # flip pose
        R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
        R_root = cv2.Rodrigues(globalOrient.reshape(-1))[0]
        new_root = R_root.dot(R_mod)
        globalOrient = cv2.Rodrigues(new_root)[0].reshape(3)

        # apply camera matrices
        globalOrient = self.rotate_global_orient(rotMat, globalOrient)
        globalOrient = self.rotate_global_orient(camera_rotationMatrix,
                                                 globalOrient)
        globalOrient = self.rotate_global_orient(camera_rotationMatrix2,
                                                 globalOrient)

        return globalOrient

    @staticmethod
    def rotate_global_orient(rotMat, global_orient):
        """Transform global orientation given rotation matrix.

        Args:
            rotMat: rotation matrix
            global_orient: original global orientation

        Returns:
            new_global_orient: transformed global orientation
        """
        new_global_orient = cv2.Rodrigues(
            np.dot(rotMat,
                   cv2.Rodrigues(global_orient.reshape(-1))[0]))[0].T[0]
        return new_global_orient

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
                keypoints3d, keypoints2d_mask, keypoints3d_mask, meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # structs we use
        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []

        # get a list of .pkl files in the directory
        img_path = os.path.join(dataset_path, 'images', mode)

        body_model = {}
        body_model['body_pose'] = []
        body_model['global_orient'] = []
        body_model['betas'] = []
        body_model['transl'] = []

        if self.fit == 'smplx':
            annot_path = os.path.join(dataset_path, 'camera_dataframe')
            body_model['left_hand_pose'] = []
            body_model['right_hand_pose'] = []
            body_model['expression'] = []
            body_model['leye_pose'] = []
            body_model['reye_pose'] = []
            body_model['jaw_pose'] = []
            num_keypoints = 127
            keypoints_convention = 'agora'
            num_body_pose = 21
        else:
            annot_path = os.path.join(dataset_path, 'camera_dataframe_smpl')
            num_keypoints = 45
            keypoints_convention = 'smpl_45'
            num_body_pose = 23

        meta = {}
        meta['gender'] = []
        meta['age'] = []
        meta['ethnicity'] = []
        meta['kid'] = []
        meta['occlusion'] = []

        # go through all the .pkl files
        annot_dataframes = [
            os.path.join(annot_path, f) for f in os.listdir(annot_path)
            if f.endswith('.pkl') and '{}'.format(mode) in f
        ]

        for filename in tqdm(sorted(annot_dataframes)):
            df = pickle.load(open(filename, 'rb'))
            for idx in tqdm(range(len(df))):
                imgname = df.iloc[idx]['imgPath']
                if self.res == (1280, 720):
                    imgname = imgname.replace('.png', '_1280x720.png')
                img_path = os.path.join('images', mode, imgname)
                valid_pers_idx = np.where(df.iloc[idx].at['isValid'])[0]
                for pidx in valid_pers_idx:
                    # obtain meta data
                    gender = df.iloc[idx]['gender'][pidx]
                    age = df.iloc[idx]['age'][pidx]
                    kid = df.iloc[idx]['kid'][pidx]
                    occlusion = df.iloc[idx]['occlusion'][pidx]
                    ethnicity = df.iloc[idx]['ethnicity'][pidx]

                    # obtain keypoints
                    keypoints2d = df.iloc[idx]['gt_joints_2d'][pidx]
                    if self.res == (1280, 720):
                        keypoints2d *= (720 / 2160)
                    keypoints3d = df.iloc[idx]['gt_joints_3d'][pidx]

                    gt_bodymodel_path = os.path.join(
                        dataset_path,
                        df.iloc[idx][f'gt_path_{self.fit}'][pidx])
                    gt_bodymodel_path = gt_bodymodel_path.replace(
                        '.obj', '.pkl')
                    ann = pickle.load(open(gt_bodymodel_path, 'rb'))

                    if self.fit == 'smplx':
                        # obtain smplx data
                        body_model['body_pose'].append(ann['body_pose'])
                        global_orient = ann['global_orient']
                        body_model['betas'].append(
                            ann['betas'].reshape(-1)[:10])
                        body_model['transl'].append(ann['transl'])
                        body_model['left_hand_pose'].append(
                            ann['left_hand_pose'])
                        body_model['right_hand_pose'].append(
                            ann['right_hand_pose'])
                        body_model['jaw_pose'].append(ann['jaw_pose'])
                        body_model['leye_pose'].append(ann['leye_pose'])
                        body_model['reye_pose'].append(ann['reye_pose'])
                        body_model['expression'].append(ann['expression'])

                    else:
                        # obtain smpl data
                        body_model['body_pose'].append(
                            ann['body_pose'].cpu().detach().numpy())
                        global_orient = ann['root_pose'].cpu().detach().numpy()
                        body_model['betas'].append(
                            ann['betas'].cpu().detach().numpy().reshape(
                                -1)[:10])
                        body_model['transl'].append(
                            ann['translation'].cpu().detach().numpy())

                    global_orient = self.get_global_orient(
                        img_path, df, idx, pidx, global_orient.reshape(-1))

                    # add confidence column
                    keypoints2d = np.hstack(
                        [keypoints2d, np.ones((num_keypoints, 1))])
                    keypoints3d = np.hstack(
                        [keypoints3d, np.ones((num_keypoints, 1))])

                    bbox_xyxy = [
                        min(keypoints2d[:, 0]),
                        min(keypoints2d[:, 1]),
                        max(keypoints2d[:, 0]),
                        max(keypoints2d[:, 1])
                    ]
                    bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
                    bbox_xywh = self._xyxy2xywh(bbox_xyxy)

                    keypoints2d_.append(keypoints2d)
                    keypoints3d_.append(keypoints3d)
                    bbox_xywh_.append(bbox_xywh)
                    image_path_.append(img_path)
                    body_model['global_orient'].append(global_orient)
                    meta['gender'].append(gender)
                    meta['age'].append(age)
                    meta['kid'].append(kid)
                    meta['occlusion'].append(occlusion)
                    meta['ethnicity'].append(ethnicity)

        # change list to np array
        if self.fit == 'smplx':
            body_model['left_hand_pose'] = np.array(
                body_model['left_hand_pose']).reshape((-1, 15, 3))
            body_model['right_hand_pose'] = np.array(
                body_model['right_hand_pose']).reshape((-1, 15, 3))
            body_model['expression'] = np.array(
                body_model['expression']).reshape((-1, 10))
            body_model['leye_pose'] = np.array(
                body_model['leye_pose']).reshape((-1, 3))
            body_model['reye_pose'] = np.array(
                body_model['reye_pose']).reshape((-1, 3))
            body_model['jaw_pose'] = np.array(body_model['jaw_pose']).reshape(
                (-1, 3))

        body_model['body_pose'] = np.array(body_model['body_pose']).reshape(
            (-1, num_body_pose, 3))
        body_model['global_orient'] = np.array(
            body_model['global_orient']).reshape((-1, 3))
        body_model['betas'] = np.array(body_model['betas']).reshape((-1, 10))
        body_model['transl'] = np.array(body_model['transl']).reshape((-1, 3))

        meta['gender'] = np.array(meta['gender'])
        meta['age'] = np.array(meta['age'])
        meta['kid'] = np.array(meta['kid'])
        meta['occlusion'] = np.array(meta['occlusion'])
        meta['ethnicity'] = np.array(meta['ethnicity'])

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])

        # change list to np array
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, num_keypoints, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, keypoints_convention,
                                         'human_data')
        keypoints3d_ = np.array(keypoints3d_).reshape((-1, num_keypoints, 4))
        keypoints3d_, _ = convert_kps(keypoints3d_, keypoints_convention,
                                      'human_data')

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints3d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['keypoints3d'] = keypoints3d_
        human_data['meta'] = meta
        human_data['config'] = 'agora'
        if self.fit == 'smplx':
            human_data['smplx'] = body_model
        else:
            human_data['smpl'] = body_model
        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = f'agora_{mode}_{self.fit}.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
'''