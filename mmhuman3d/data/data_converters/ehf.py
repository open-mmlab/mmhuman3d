import glob
import json
import os
import random
from typing import List

import numpy as np
import torch
from tqdm import tqdm

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


@DATA_CONVERTERS.register_module()
class EhfConverter(BaseModeConverter):
    """EHF dataset 'Expressive Hands and Face' More details can be found on the
    website:

    https://smpl-x.is.tue.mpg.de
    Args:
        modes (list): 'val' for accepted modes
    """

    ACCEPTED_MODES = ['val']

    def __init__(self, modes: List = []) -> None:
        # check pytorch device
        self.device = torch.device('cuda:0')
        self.misc = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            bbox_source='keypoints2d_smplx',
            cam_param_type='prespective',
            cam_param_source='original',
            smplx_source='original',
            focal_length=(1498.224, 1498.224),
            principal_point=(790.263706, 578.90334),
            image_size=(1600, 1200),
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
            'expression': (-1, 10)
        }
        self.T = np.array([-0.03609917, 0.43416458, 2.37101226])
        self.R = np.array([[0.9992447, -0.00488005, 0.03855169],
                           [-0.01071995, -0.98820424, 0.15276562],
                           [0.03735144, -0.15306349, -0.9875102]])

        super(EhfConverter, self).__init__(modes)

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

            if occ is not None:
                occ_p = occ[kp_id]
                if np.sum(occ_p) / len(kp_id) >= 0.1:
                    conf = 0
                else:
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
                A dict containing keys image_path, bbox_xywh, smplx, meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # get targeted frame list
        image_paths = sorted(
            glob.glob(os.path.join(dataset_path, '*_img.png')))

        # init seed and size
        seed, size = '230807', '999'
        size_i = min(int(size), len(image_paths))
        random.seed(int(seed))
        # random.shuffle(npzs)

        # initialize output for human_data
        smplx_ = {}
        for key in self.smplx_shape.keys():
            smplx_[key] = []
        keypoints2d_, keypoints3d_, kps2d_orig_ = [], [], []
        bboxs_ = {}
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            bboxs_[bbox_name] = []
        meta_ = {}
        image_path_ = []

        # get camera parameters - extrinsic
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = self.R
        camera_pose[:3, 3] = self.T

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

        # build camera
        camera = build_cameras(
            dict(
                type='PerspectiveCameras',
                convention='opencv',
                in_ndc=False,
                focal_length=self.misc['focal_length'],
                image_size=self.misc['image_size'],
                principal_point=self.misc['principal_point'])).to(self.device)

        # parse images
        for imgp in tqdm(image_paths):

            # get image path
            image_path = imgp.replace(f'{dataset_path}{os.path.sep}', '')

            # get frame name
            name = os.path.basename(imgp).split('_')[0]

            # load keypoints 2d original
            json_path = os.path.join(dataset_path, name + '_2Djnt.json')
            json_data = json.load(open(json_path, 'r'))
            body_kps2d = np.array(
                json_data['people'][0]['pose_keypoints_2d'],
                dtype=np.float32).reshape(-1, 3)
            face_kps2d = np.array(
                json_data['people'][0]['face_keypoints_2d'],
                dtype=np.float32).reshape(-1, 3)
            lhand_kps2d = np.array(
                json_data['people'][0]['hand_left_keypoints_2d'],
                dtype=np.float32).reshape(-1, 3)
            rhand_kps2d = np.array(
                json_data['people'][0]['hand_right_keypoints_2d'],
                dtype=np.float32).reshape(-1, 3)
            kps2d = np.concatenate(
                (body_kps2d, lhand_kps2d, rhand_kps2d, face_kps2d), axis=0)
            kps2d[:, -1] = 1.0
            conf = kps2d[:, -1]

            # load smplx
            smplx_path = os.path.join(dataset_path, 'fitted_params',
                                      f'{name}_align.npz')
            smplx_param = dict(np.load(smplx_path))

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
                extrinsic=camera_pose)

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

            # test image overlay
            # img = cv2.imread(imgp)
            # for kp in keypoints_2d[0]:
            #     cv2.circle(img, (int(kp[0]), int(kp[1])), 5, (0, 0, 255), -1)
            # cv2.imwrite(f'{dataset_path}/output/{name}_overlay.png', img)

            # get bbox from 2d keypoints
            bboxs = self._keypoints_to_scaled_bbox_bfh(
                keypoints_2d,
                body_scale=self.misc['bbox_body_scale'],
                fh_scale=self.misc['bbox_facehand_scale'])
            for i, bbox_name in enumerate([
                    'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                    'rhand_bbox_xywh'
            ]):
                xmin, ymin, xmax, ymax, conf = bboxs[i]
                bbox = np.array([
                    max(0, xmin),
                    max(0, ymin),
                    min(self.misc['image_size'][0], xmax),
                    min(self.misc['image_size'][1], ymax)
                ])
                bbox_xywh = self._xyxy2xywh(bbox)  # list of len 4
                bbox_xywh.append(conf)  # (5,)
                bboxs_[bbox_name].append(bbox_xywh)

            # append image path
            image_path_.append(image_path)

            # append smplx
            for key in intersect_keys:
                smplx_[key].append(smplx_param[key])

            # append keypoints
            kps2d_orig_.append(kps2d)
            keypoints2d_.append(keypoints_2d)
            keypoints3d_.append(keypoints_3d)

        # save keypoints 2d original
        keypoints2d_original = np.array(kps2d_orig_)
        keypoints2d_original, keypoints2d_original_mask = convert_kps(
            keypoints2d_original, src='openpose_137', dst='human_data')
        human_data['keypoints2d_original'] = keypoints2d_original
        human_data['keypoints2d_original_mask'] = keypoints2d_original_mask

        # save keypoints 2d smplx
        keypoints2d = np.concatenate(keypoints2d_, axis=0)
        keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
        keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
        keypoints2d, keypoints2d_mask = convert_kps(
            keypoints2d, src='smplx', dst='human_data')
        human_data['keypoints2d_smplx'] = keypoints2d
        human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

        # save keypoints 3d smplx
        keypoints3d = np.concatenate(keypoints3d_, axis=0)
        keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
        keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
        keypoints3d, keypoints3d_mask = convert_kps(
            keypoints3d, src='smplx', dst='human_data')
        human_data['keypoints3d_smplx'] = keypoints3d
        human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

        # save bbox
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            bbox_xywh_ = np.array(bboxs_[bbox_name]).reshape((-1, 5))
            human_data[bbox_name] = bbox_xywh_

        # save smplx
        for key in smplx_.keys():
            smplx_[key] = np.concatenate(
                smplx_[key], axis=0).reshape(self.smplx_shape[key])
        human_data['smplx'] = smplx_

        # save image path
        human_data['image_path'] = image_path_

        # save meta and misc
        human_data['config'] = 'ehf'
        human_data['misc'] = self.misc
        human_data['meta'] = meta_

        size_i = min(int(size), len(image_paths))

        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path, f'ehf_{mode}_{seed}_{"{:03d}".format(size_i)}.npz')
        human_data.dump(out_file)
