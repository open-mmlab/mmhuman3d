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

from pycocotools.coco import COCO

import pdb


@DATA_CONVERTERS.register_module()
class MpiiNeuralConverter(BaseModeConverter):
    """MPII Dataset `2D Human Pose Estimation: New Benchmark and State of the
    Art Analysis' CVPR'2014. More details can be found in the `paper.

    <http://human-pose.mpi-inf.mpg.de/contents/andriluka14cvpr.pdf>`__ .
    """

    ACCEPTED_MODES = ['test', 'train']

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
        super(MpiiNeuralConverter, self).__init__(modes)


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

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, keypoints2d,
                keypoints2d_mask stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()

        # initialize output for human_data
        smplx_ = {}
        for key in self.smplx_shape.keys():
            smplx_[key] = []
        keypoints2d_smplx_, keypoints3d_smplx_, = [], []
        keypoints2d_orig_ = [ ]
        bboxs_ = {}
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            bboxs_[bbox_name] = []
        meta_ = {}
        for key in ['focal_length', 'principal_point', 'height', 'width']:
            meta_[key] = []
        image_path_ = []

        # load data seperate
        split_path = os.path.join(dataset_path, 'annotations', f'{mode}.json')
        db = COCO(split_path)

        # with open(split_path, 'r') as f:
        #     image_data = json.load(f)

        # load smplx annot
        smplx_path = os.path.join(dataset_path, 'annotations', f'MPII_train_SMPLX_NeuralAnnot.json')
        with open(smplx_path, 'r') as f:
            smplx_data = json.load(f)

        # get targeted frame list
        image_list = list(db.anns.keys())

        # init seed and size
        seed, size = '231016', '90999'
        size_i = min(int(size), len(image_list))
        random.seed(int(seed))
        image_list = image_list[:size_i]
        # random.shuffle(npzs)

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

        for aid in tqdm(db.anns.keys(), desc=f'Converting MPII {mode} data'):
        # for aid in db.anns.keys():
            
            # get info slice
            ann = db.anns[aid]
            image_info = ann
            img = db.loadImgs(ann['image_id'])[0]
            # pdb.set_trace()
            
            # prepare image path
            image_path = img['file_name']
            imgp = os.path.join(dataset_path, image_path)
            if not os.path.exists(imgp):
                continue

            # access image info
            annot_id = image_info['id']
            width = img['width']
            height = img['height']

            # read keypoints2d and bbox
            j2d = np.array(image_info['keypoints']).reshape(-1, 3)

            bbox_xywh = image_info['bbox']
            bbox_xywh.append(1)
            
            # read smplx annot info
            annot_info = smplx_data[str(annot_id)]
            smplx_info = annot_info['smplx_param']
            cam_info = annot_info['cam_param']

            # reformat smplx anno
            smplx_param = {}
            smplx_param['global_orient'] = np.array(smplx_info['root_pose']).reshape(-1, 3)
            smplx_param['body_pose'] = np.array(smplx_info['body_pose']).reshape(-1, 21, 3)
            smplx_param['betas'] = np.array(smplx_info['shape']).reshape(-1, 10)
            smplx_param['transl'] = np.array(smplx_info['trans']).reshape(-1, 3)

            # get camera param and build camera
            focal_length = cam_info['focal']
            principal_point = cam_info['princpt']

            camera = build_cameras(
                dict(
                    type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=focal_length,
                    image_size=(width, height),
                    principal_point=principal_point)).to(self.device)

            # get smplx output
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
        
            # get kps2d from projection
            keypoints_3d = output['joints']
            keypoints_2d_xyd = camera.transform_points_screen(keypoints_3d)
            keypoints_2d = keypoints_2d_xyd[..., :2].detach().cpu().numpy()
            keypoints_3d = keypoints_3d.detach().cpu().numpy()


            # # test overlay j2d
            # img = cv2.imread(f'{dataset_path}/{image_path}')
            # for kp in keypoints_2d[0]:
            #     if  0 < kp[0] < 1920 and 0 < kp[1] < 1080: 
            #         cv2.circle(img, (int(kp[0]), int(kp[1])), 1, (0,0,255), -1)
            #     pass
            # # write image
            # os.makedirs(f'{out_path}', exist_ok=True)
            # cv2.imwrite(f'{out_path}/{fname}', img)

            # append image path
            image_path_.append(image_path)

            # append keypoints2d and 3d
            keypoints2d_smplx_.append(keypoints_2d)
            keypoints3d_smplx_.append(keypoints_3d)
            keypoints2d_orig_.append(j2d)

            # append bbox
            bboxs = self._keypoints_to_scaled_bbox_bfh(
                keypoints_2d,
                body_scale=1.2,
                fh_scale=1.0)
            for i, bbox_name in enumerate([
                    'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                    'rhand_bbox_xywh'
            ]):
                xmin, ymin, xmax, ymax, conf = bboxs[i]
                bbox = np.array([
                    max(0, xmin),
                    max(0, ymin),
                    min(width, xmax),
                    min(height, ymax)
                ])
                bbox_xywh = self._xyxy2xywh(bbox)  # list of len 4
                bbox_xywh.append(conf)  # (5,)
                bboxs_[bbox_name].append(bbox_xywh)

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

        # keypoints2d_orig
        keypoints2d_orig = np.concatenate(
            keypoints2d_orig_, axis=0).reshape(-1, 16, 3)
        keypoints2d_orig, keypoints2d_orig_mask = \
                convert_kps(keypoints2d_orig, src='mpii', dst='human_data')
        human_data['keypoints2d_original'] = keypoints2d_orig
        human_data['keypoints2d_original_mask'] = keypoints2d_orig_mask

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

        # misc
        human_data['misc'] = self.misc
        human_data['config'] = f'mpii_neural_annot_{mode}'

        # save
        human_data.compress_keypoints_by_mask()
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            out_path,
            f'mpii_neural_{mode}_{seed}_{"{:05d}".format(size_i)}.npz')
        human_data.dump(out_file)