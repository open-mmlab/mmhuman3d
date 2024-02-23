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
import torch

# import mmcv
# from mmhuman3d.models.body_models.builder import build_body_model
# from mmhuman3d.core.conventions.keypoints_mapping import smplx
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from mmhuman3d.models.body_models.utils import batch_transform_to_camera_frame
from mmhuman3d.models.body_models.utils import transform_to_camera_frame
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.core.cameras import build_cameras

@DATA_CONVERTERS.register_module()
class SynbodyWhacConverter(BaseModeConverter):
    """Synbody dataset."""
    ACCEPTED_MODES = ['AMASS_tracking', 'DuetDance']

    def __init__(self, modes: List = []) -> None:

        self.device = torch.device('cuda:0')
        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            bbox_source='keypoints2d_original',
            flat_hand_mean=True,
            cam_param_type='prespective',
            cam_param_source='original',
            smplx_source='original',
            contact_label=['part_segmentation'],
            part_segmentation=['left_foot', 'right_foot'],
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
            'expression': (-1, 10)
        }

        super(SynbodyWhacConverter, self).__init__(modes)

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
       
        # get target sequences
        seqs_targeted = glob.glob(os.path.join(dataset_path, 'Synbody_whac', 
                                               f'{mode}*', '*', 
                                               '*_*', 'smplx_adjusted', '*.npz'))

        # bulid smplx model
        gendered_smplx = {}
        for gender in ['male', 'female', 'neutral']:
            gendered_smplx[gender] = build_body_model(
                dict(
                    type='SMPLX',
                    keypoint_src='smplx',
                    keypoint_dst='smplx',
                    model_path='data/body_models/smplx',
                    gender=gender,
                    num_betas=10,
                    use_face_contour=True,
                    flat_hand_mean=self.misc_config['flat_hand_mean'],
                    use_pca=False,
                    batch_size=1)).to(self.device)
        
        focal_length = [960.0, 960.0]
        principal_point = [960.0, 540.0]
        width, height = 1920, 1080
        # build cameras
        camera = build_cameras(
                dict(
                    type='PerspectiveCameras',
                    convention='opencv',
                    in_ndc=False,
                    focal_length=focal_length,
                    image_size=(width, height),
                    principal_point=principal_point)).to(self.device)
        
        # use HumanData to store all data
        human_data = HumanData()

        # init seed and size
        seed, size = '240222', '999'
        size_i = min(int(size), len(seqs_targeted))
        random.seed(int(seed))
        np.set_printoptions(suppress=True)
        random_ids = np.random.RandomState(seed=int(seed)).permutation(999999)
        used_id_num = 0
        
        seqs_targeted = seqs_targeted[:size_i]

        # initialize output for human_data
        smplx_ = {}
        for key in self.smplx_shape.keys():
            smplx_[key] = []
        keypoints2d_, keypoints3d_ = [], []
        bboxs_ = {}
        for bbox_name in [
                'bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh',
                'rhand_bbox_xywh'
        ]:
            bboxs_[bbox_name] = []
        meta_ = {}
        for meta_name in ['principal_point', 'focal_length', 'height', 'width', 'RT',
                          'sequence_name', 'track_id', 'gender']:
            meta_[meta_name] = []
        image_path_ = []
        contact_ = {}
        for contact_key in self.misc_config['contact_label']:
            contact_[contact_key] = []

        # load contact region
        from tools.utils.convert_contact_label import vertex2part_smplx_dict
        left_foot_idxs = np.array(vertex2part_smplx_dict['left_foot'])
        right_foot_idxs = np.array(vertex2part_smplx_dict['right_foot'])

        for seq in tqdm(seqs_targeted, desc=f'Processing {mode}', leave=False, position=0):
            
            # preprocess sequence

            # get track id
            track_id = random_ids[used_id_num]
            used_id_num += 1

            # load smplx
            param = dict(np.load(seq, allow_pickle=True))
            smplx_param = param['smplx'].item()
            datalen = param['__data_len__']
            gender = param['meta'].item()['gender']

            # expend betas
            smplx_param['betas'] = np.tile(smplx_param['betas'], (datalen, 1))

            # parse camera params and image
            seq_base = os.path.dirname(os.path.dirname(seq))

            # prepare smplx tensor
            smplx_param_tensor = {}
            for key in self.smplx_shape.keys():
                smplx_param_tensor[key] = torch.tensor(smplx_param[key]
                                                        .reshape(self.smplx_shape[key])).to(self.device)
                
            # get output
            output = gendered_smplx[gender](**smplx_param_tensor, return_verts=True)
                
            kps3d = output['joints'].detach().cpu().numpy()
            pelvis_world = kps3d[:, get_keypoint_idx('pelvis', 'smplx'), :]

            # get vertices and contact
            vertices = output['vertices'].detach().cpu().numpy()

            # height is -y, get lowest from frame 0
            left_foot_y_lowest = np.sort(vertices[1, left_foot_idxs, 1])[-1]
            right_foot_y_lowest = np.sort(vertices[1, right_foot_idxs, 1])[-1]

            left_foot_contact = np.zeros([datalen])
            right_foot_contact = np.zeros([datalen])

            threshold = 0.01
            for i in range(datalen):
                left_foot_contact[i] = 1 if (vertices[i, left_foot_idxs, 1].max() > (left_foot_y_lowest - threshold)) else 0
                right_foot_contact[i] = 1 if (vertices[i, right_foot_idxs, 1].max() > (right_foot_y_lowest - threshold)) else 0

            # cids
            cids = sorted(os.listdir(os.path.join(seq_base, 'img')))

            for cid in cids:

                # prepare sequence name
                sequence_name = f'{os.path.basename(seq_base)}_{cid}'

                # retrieve images
                img_f = os.path.join(seq_base, 'img', cid)
                img_ps = [os.path.join(img_f, ip) for ip in
                            os.listdir(img_f) if ip.endswith('.jpeg')]

                # retrieve smplx
                cam_f = os.path.join(seq_base, 'camera_params', cid)
                cam_ps = [os.path.join(cam_f, cp) for cp in 
                            os.listdir(cam_f) if cp.endswith('.json')]

                # get valid index array & remove frame 0
                valid_idxs_img = np.array([int(os.path.basename(img_p).split('.')[0]) for img_p in img_ps])
                valid_idex_cam = np.array([int(os.path.basename(cam_p).split('.')[0]) for cam_p in cam_ps])
                valid_idxs = np.intersect1d(valid_idxs_img, valid_idex_cam)
                # valid_idxs = valid_idxs[valid_idxs > 0]

                for vid in tqdm(valid_idxs, desc=f'Processing {sequence_name}, {cid} / {cids[-1]}', 
                                leave=False, position=1):
                    if vid == 0:
                        continue
                    # get image path
                    img_p = os.path.join(img_f, f'{vid:04d}.jpeg')
                    image_path = img_p.replace(dataset_path + '/', '')

                    # get camera path
                    cam_p = os.path.join(cam_f, f'{vid:04d}.json')
                    with open(cam_p, 'r') as f:
                        cam_param = json.load(f)

                    Rt = np.eye(4)
                    Rt[:3, :3] = np.array(cam_param['extrinsic_r'])
                    Rt[:3, 3] = np.array(cam_param['extrinsic_t'])

                    ue2opencv = np.array([[-1.0, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
                    
                    # Rt = (Rt.T @ ue2opencv).T

                    # transform to cam space
                    global_orient, transl = transform_to_camera_frame(
                        global_orient=smplx_param['global_orient'][vid].reshape(-1, 3),
                        transl=smplx_param['transl'][vid].reshape(-1, 3),
                        pelvis=pelvis_world[vid].reshape(-1, 3),
                        extrinsic=Rt)
                    
                    smplx_param_frame = {k: v[vid] for k, v in smplx_param.items()}
                    smplx_param_frame['global_orient'] = global_orient
                    smplx_param_frame['transl'] = transl

                    # tensor 
                    smplx_param_frame_tensor = {
                        k: torch.tensor(v.reshape(self.smplx_shape[k])).to(self.device)
                          for k, v in smplx_param_frame.items()}
                    
                    # get output
                    output = gendered_smplx[gender](**smplx_param_frame_tensor)

                    # # build intrinsics camera
                    # intrinsics = np.array(cam_param['intrinsic'])
                    # height, width = cam_param['height'], cam_param['width']

                    # focal_length = [intrinsics[0, 0], intrinsics[1, 1]]
                    # principal_point = [intrinsics[0, 2], intrinsics[1, 2]]

                    # project 3d to 2d
                    kps3d_c = output['joints']
                    kps2d = camera.transform_points(kps3d_c).detach().cpu().numpy().squeeze()[:, :2]
                    kps3d_c = kps3d_c.detach().cpu().numpy().squeeze()

                    kps2d = [1920, 1080] - kps2d

                    # get bbox from 2d keypoints
                    bboxs = self._keypoints_to_scaled_bbox_bfh(
                        kps2d,
                        body_scale=self.misc_config['bbox_body_scale'],
                        fh_scale=self.misc_config['bbox_facehand_scale'])
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

                    # append contact
                    contact_['part_segmentation'].append([left_foot_contact[vid], 
                                                          right_foot_contact[vid]])
                    # if 0 in [left_foot_contact[vid], right_foot_contact[vid]]:
                    #     # test overlay
                    #     img = cv2.imread(img_p)
                    #     for kp in kps2d:
                    #         cv2.circle(img, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                    #     if left_foot_contact[vid] == 0:
                    #         cv2.putText(img, 'left foot', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    #     if right_foot_contact[vid] == 0:
                    #         cv2.putText(img, 'right foot', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    #     cv2.imwrite(f'{out_path}/{os.path.basename(seq_base)}_{cid}_{vid}.jpg', img)

                    # append image path
                    image_path_.append(image_path)

                    # append keypoints
                    keypoints2d_.append(kps2d)
                    keypoints3d_.append(kps3d_c)

                    # append smplx
                    for key in self.smplx_shape.keys():
                        smplx_[key].append(smplx_param_frame[key])

                    # append meta
                    meta_['principal_point'].append(principal_point)
                    meta_['focal_length'].append(focal_length)
                    meta_['height'].append(height)
                    meta_['width'].append(width)
                    meta_['RT'].append(Rt)
                    meta_['sequence_name'].append(sequence_name)
                    meta_['track_id'].append(track_id)
                    meta_['gender'].append(gender)

        # get size
        size_i = min(int(size), len(seqs_targeted))

        # save keypoints 2d smplx
        keypoints2d = np.concatenate(keypoints2d_, axis=0).reshape(-1, 144, 2)
        keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
        keypoints2d = np.concatenate([keypoints2d, keypoints2d_conf], axis=-1)
        keypoints2d, keypoints2d_mask = convert_kps(
            keypoints2d, src='smplx', dst='human_data')
        human_data['keypoints2d_smplx'] = keypoints2d
        human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

        # save keypoints 3d smplx
        keypoints3d = np.concatenate(keypoints3d_, axis=0).reshape(-1, 144, 3)
        keypoints3d_conf = np.ones([keypoints3d.shape[0], 144, 1])
        keypoints3d = np.concatenate([keypoints3d, keypoints3d_conf], axis=-1)
        keypoints3d, keypoints3d_mask = convert_kps(
            keypoints3d, src='smplx', dst='human_data')
        human_data['keypoints3d_smplx'] = keypoints3d
        human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

        # pdb.set_trace()
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

        # save contact
        human_data['contact'] = contact_

        # save meta and misc
        human_data['config'] = 'synbody_whac'
        human_data['misc'] = self.misc_config
        human_data['meta'] = meta_

        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(
            # out_path, f'moyo_{self.misc_config["flat_hand_mean"]}.npz')
            out_path, f'synbody_whac_{mode}_{seed}_{"{:03d}".format(size_i)}.npz')
        human_data.dump(out_file)