import json
import os
import pdb
# import pickle5 as pickle
import pickle
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from pycocotools.coco import COCO
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
    ACCEPTED_MODES = [
        'validation_3840', 'train_3840', 'train_1280', 'validation_1280'
    ]

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            bbox_source='keypoints2d_smplx',
            cam_param_source='original',
            smplx_source='original',
            has_betas_extra=True,
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
            'expression': (-1, 10),
            'betas_extra': (-1, 1),  # how close to kid template
            # 'betas_fixed': (-1, 10),  # fixed gendered betas
            'betas_neutral': (-1, 10),  # neutral betas
        }

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
        for bbox_name in [
            'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
            'rhand_bbox_xywh'
        ]:
            bboxs_[bbox_name] = []
        meta_ = {}
        for meta_key in ['principal_point', 'focal_length', 'gender', 'annot_valid']:
            meta_[meta_key] = []
        image_path_ = []

        seed, size = '231031', '999999'

        # build smplx model
        smplx_model = {}
        for gender in ['male', 'female', 'neutral']:
            smplx_model[gender] = build_body_model(
                dict(
                    type='SMPLX',
                    keypoint_src='smplx',
                    keypoint_dst='smplx',
                    model_path='data/body_models/smplx',
                    gender=gender,
                    num_betas=10,
                    use_face_contour=True,
                    flat_hand_mean=False,
                    use_pca=False,
                    batch_size=1)).to(device)

        # data info path and bf (betas fixed)
        data_info_path = os.path.join(dataset_path,
                                      f'AGORA_{batch_info}.json')
        data_info_path_bf = os.path.join(dataset_path,
                                         f'AGORA_{batch_info}_fix_betas.json')

        # read data info
        with open(data_info_path, 'r') as f:
            param = json.load(f)
        with open(data_info_path_bf, 'r') as f:
            param_bf = json.load(f)
        # image_param = param['images']
        # anno_param = param['annotations']
        # anno_param_bf = param_bf['annotations']

        db = COCO(data_info_path)
        db_bf = COCO(data_info_path_bf)

        print('Converting...')
        for aid in tqdm(db.anns.keys(), desc=f'Processing Agora {mode}'):
            anno_info = db.anns[aid]
            anno_info_bf = db_bf.anns[aid]

        # for aid, anno_info in enumerate(tqdm(anno_param, desc=f'Processing Agora {mode}')):
            image_id = anno_info['image_id']

            # anno_info_bf = anno_param_bf[aid]

            # print(anno_info['gender'])

            # get image details
            image_info = db.loadImgs(image_id)[0]
            # image_info = image_param[image_id]
            image_path = image_info[
                f"file_name_{self.misc_config['image_size'][0]}x"
                f"{self.misc_config['image_size'][1]}"]

            try:
                smplx_param_bf = pickle.load(open(
                    os.path.join(dataset_path, anno_info_bf['smplx_param_path']), 'rb'))
            except:
                print(f'{anno_info_bf["smplx_param_path"]}, not found')
                continue

            # collect bbox and resize bbox for 1280
            if res_info == '1280':
                scale = 3840 / int(res_info)
                for key in ['bbox', 'face_bbox', 'lhand_bbox', 'rhand_bbox']:
                    bboxs_[f'{key}_xywh'].append(
                        np.array(anno_info[key] + [scale]) / scale)
            else:
                for key in ['bbox', 'face_bbox', 'lhand_bbox', 'rhand_bbox']:
                    bboxs_[f'{key}_xywh'].append(np.array(anno_info[key] + [1]))

            # collect smplx_params
            smplx_path = os.path.join(dataset_path,
                                      anno_info['smplx_param_path'])
            
            # smplx_param = pickle.load(open(smplx_path, 'rb'))

            smplx_param = smplx_param_bf
            smplx_param['betas_fixed'] = smplx_param_bf['betas'][:, :10]

            if smplx_param['betas'].shape[1] != 10:
                smplx_param['betas_extra'] = smplx_param['betas'][:, 10:]
                smplx_param['betas'] = smplx_param['betas_neutral'][:, :10]
            else:
                smplx_param['betas_extra'] = np.zeros([1, 1])
            for key in self.smplx_shape.keys():
                smplx_[key].append(smplx_param[key])

            # # collect keypoints
            # smplx_joints_2d_path = os.path.join(dataset_path, anno_info['smplx_joints_2d_path'])
            # smplx_joints_2d = json.load(open(smplx_joints_2d_path, 'rb'), encoding='latin1')
            # keypoints2d_.append(smplx_joints_2d)
            #
            # smplx_joints_3d_path = os.path.join(dataset_path, anno_info['smplx_joints_3d_path'])
            # smplx_joints_3d = json.load(open(smplx_joints_3d_path, 'rb'), encoding='latin1')
            # keypoints3d_.append(smplx_joints_3d)

            # get camera parameters
            principal_point = np.array([
                self.misc_config['image_size'][0] / 2,
                self.misc_config['image_size'][1] / 2
            ])
            focal_length = self._focalLength_mm2px(
                self._get_focal_length(image_path), principal_point)

            # collect meta
            meta_['gender'].append(anno_info['gender'])
            # meta_['gender'].append('neutral')
            meta_['principal_point'].append(principal_point)
            meta_['focal_length'].append(focal_length)
            meta_['annot_valid'].append(anno_info['is_valid'])

            # print(smplx_param['betas'].shape)

            # pdb.set_trace()
            # build camera
            camera = build_cameras(
                dict(
                    type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=focal_length,
                    image_size=(self.misc_config['image_size'][0], self.misc_config['image_size'][1]),
                    principal_point=np.array(principal_point).reshape(-1, 2))).to(device)

            # # test smplx
            intersect_key = list(set(smplx_param.keys()) & set(self.smplx_shape.keys()))
            body_model_param_tensor = {key: torch.tensor(
                np.array(smplx_param[key]).reshape(self.smplx_shape[key]),
                device=device, dtype=torch.float32)
                for key in intersect_key
                if len(smplx_param[key]) > 0}
            output = smplx_model[gender](**body_model_param_tensor, return_verts=True)
            smplx_joints = output['joints']
            kps3d = smplx_joints.detach().cpu().numpy()
            kps2d = camera.transform_points_screen(smplx_joints)[..., :2].detach().cpu().numpy()

            keypoints2d_.append(kps2d)
            keypoints3d_.append(kps3d)
            image_path_.append(image_path)
            # pdb.set_trace()
            pass

        # prepare for output
        # smplx
        # pdb.set_trace()
        for key in smplx_.keys():
            # print('doing', key)
            smplx_[key] = np.concatenate(smplx_[key], axis=0).reshape(self.smplx_shape[key])
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
        keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
        keypoints2d, keypoints2d_mask = \
            convert_kps(keypoints2d, src='smplx', dst='human_data')
        human_data['keypoints2d_smplx'] = keypoints2d
        human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

        # keypoints 3d
        keypoints3d = np.array(keypoints3d_).reshape(-1, 144, 3)
        keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
        keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
        keypoints3d, keypoints3d_mask = \
            convert_kps(keypoints3d, src='smplx', dst='human_data')
        human_data['keypoints3d_smplx'] = keypoints3d
        human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

        # image path
        human_data['image_path'] = image_path_
        print('Image path writing finished at',
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # meta
        human_data['meta'] = meta_
        print('Meta writing finished at',
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # store
        human_data['config'] = f'agora_{mode}'
        human_data['misc'] = self.misc_config

        size_i = str(min(int(size), len(db.anns.keys())))

        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path,
            f'agora_{mode}_{seed}_{"{:06d}".format(int(size_i))}.npz')
        # pdb.set_trace()
        human_data.dump(out_file)
