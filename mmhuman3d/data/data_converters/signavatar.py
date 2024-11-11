import glob
import json
import os
import pdb
import random
import pickle
import time
from typing import List

import cv2
import numpy as np
from tqdm import tqdm
import torch
# from scipy.spatial.distance import cdist

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
class SignAvatarConverter(BaseModeConverter):
    """Synbody dataset."""
    ACCEPTED_MODES = ['lan2m', 'ham2m', 'word2m']

    def __init__(self, modes: List = []) -> None:

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.misc_config = dict(
            bbox_body_scale=1.2,
            bbox_facehand_scale=1.0,
            bbox_source='keypoints2d_original',
            flat_hand_mean=False,
            cam_param_type='prespective',
            cam_param_source='original',
            smplx_source='original',
            # contact_label=['part_segmentation', 'contact_region'],
            # part_segmentation=['left_foot', 'right_foot'],
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

        super(SignAvatarConverter, self).__init__(modes)
        
    def split_video(self, video_path: str, annot_len: int = 0) -> bool:
        # split video into 06d frames using ffmpeg

        # create folder to store frames
        frame_folder = video_path.replace('.mp4', '').replace('videos', 'images')
        os.makedirs(frame_folder, exist_ok=True)
        
        existing_files = glob.glob(os.path.join(frame_folder, '*.jpg'))
        if annot_len != 0:
            if len(existing_files) == annot_len:
                return True
        
        # split video into frames
        os.system(f'ffmpeg -loglevel quiet -i {video_path} {frame_folder}/%06d.jpg')
        
        # check for number of frames
        frame_files = glob.glob(os.path.join(frame_folder, '*.jpg'))
        if annot_len != 0:
            if len(frame_files) == annot_len:
                return True
            else:
                return False
        
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
        
        assert mode == 'lan2m'
        
        base_folder_dict = {'lan2m': 'language2motion'}
        base_folder = base_folder_dict[mode]
        
        # load annotations
        annot_base_folder = os.path.join(dataset_path, base_folder, 'annotations')
        annot_files = glob.glob(os.path.join(annot_base_folder, '*.pkl'))
        
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
            
        # init seed and size
        seed, size = '241106', '99999'
        size_i = min(int(size), len(annot_files))
        random.seed(int(seed))
        np.set_printoptions(suppress=True)
        random_ids = np.random.RandomState(seed=int(seed)).permutation(999999)
        used_id_num = 0
        
        print('Total sequences:', len(annot_files))
        
        slice_num = 8
        slice_len = len(annot_files) // slice_num
        
        for sid in range(slice_num):
            print(f'Slice {sid+1}/{slice_num}')
            # use HumanData to store all data
            human_data = HumanData()
        
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
            for meta_name in ['principal_point', 'focal_length', 'height', 'width', 'gender',
                            'sequence_name', 'left_hand_valid', 'right_hand_valid']:
                meta_[meta_name] = []
            image_path_ = []
                
            # annot_files = annot_files[:size_i]
            annot_files2process = annot_files[sid*slice_len:(sid+1)*slice_len]

            
            # for annot_path in tqdm(annot_files, desc=f'Splitting {mode}', 
            #                        leave=False, position=0):
            #     vid_path = annot_path.replace('annotations', 'videos').replace('.pkl', '.mp4')
            #     self.split_video(vid_path)
                
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm

            # 使用线程池并行处理视频分割
            # with ThreadPoolExecutor(max_workers=16) as executor:
            #     futures = [
            #         executor.submit(self.split_video, annot_path.replace('annotations', 'videos').replace('.pkl', '.mp4'))
            #         for annot_path in annot_files
            #     ]

            #     # 使用 tqdm 追踪任务进度
            #     for future in tqdm(as_completed(futures), desc=f'Splitting {mode}', leave=False, position=0, total=len(annot_files)):
            #         # try:
            #         future.result()  # 捕获异常并确保进度条准确
                    # except Exception as e:
                    #     print(f"Error processing file: {e}")
            
            # test_seqs = ['_20g7MG8K1U_3-8-rgb_front', '_Dh512GX6d8_14-8-rgb_front', 
            #              '00kppw3aqus_11-3-rgb_front']
            # annot_files = [f'{annot_base_folder}/{seq}.pkl' for seq in test_seqs]
                
            for annot_path in tqdm(annot_files2process, desc=f'Converting {mode}', 
                                leave=False, position=0):
                
                # load annot pickle
                annot_seq = np.load(annot_path, allow_pickle=True)
                # for key in annot_seq.keys():
                #     print(key, annot_seq[key].shape)
                vid_path = annot_path.replace('annotations', 'videos').replace('.pkl', '.mp4')
                frame_folder = vid_path.replace('.mp4', '').replace('videos', 'images')

                annot_len = annot_seq['smplx'].shape[0]
                split_success = self.split_video(vid_path, annot_len)
                if not split_success:
                    # pdb.set_trace()
                    continue
                
                smplx_seq = annot_seq['smplx'].copy()
                gender = 'neutral'
                smplx_param = {
                    'global_orient': smplx_seq[:, :3],
                    'body_pose': smplx_seq[:, 3:66],
                    'left_hand_pose': smplx_seq[:, 66:111],
                    'right_hand_pose': smplx_seq[:, 111:156],
                    'jaw_pose': smplx_seq[:, 156:159],
                    'betas': smplx_seq[:, 159:169],
                    'expression': smplx_seq[:, 169:179],
                    'transl': smplx_seq[:, 179:182]
                }
                for key in self.smplx_shape.keys():
                    if key in smplx_param.keys():
                        smplx_param[key] = smplx_param[key].reshape(self.smplx_shape[key])
                    else:
                        pad_shape = np.array(self.smplx_shape[key])
                        pad_shape[0] = annot_len
                        pad_shape = tuple(pad_shape)
                        smplx_param[key] = np.zeros(pad_shape)

                # prepare smplx tensor
                smplx_param_tensor = {}
                for key in self.smplx_shape.keys():
                        smplx_param_tensor[key] = torch.tensor(smplx_param[key].reshape(self.smplx_shape[key]),
                                                            dtype=torch.float).to(self.device)
                
                # ue2opencv = np.array([[-1.0, 0, 0, 0],
                #         [0, -1, 0, 0],
                #         [0, 0, 1, 0],
                #         [0, 0, 0, 1]])
                
                # get output
                output = gendered_smplx[gender](**smplx_param_tensor)
                kps3d_c = output['joints']
                # kps3d_c = output['joints'].detach().cpu().numpy()
                # pelvis_world = kps3d_c[:, get_keypoint_idx('pelvis', 'smplx'), :]
                
                # # transform to cam space
                # global_orient, transl = batch_transform_to_camera_frame(
                #     global_orient=smplx_param['global_orient'].reshape(-1, 3),
                #     transl=smplx_param['transl'].reshape(-1, 3),
                #     pelvis=pelvis_world.reshape(-1, 3),
                #     extrinsic=ue2opencv)
                
                # smplx_param['global_orient'] = global_orient
                # smplx_param['transl'] = transl

                # # prepare smplx tensor
                # smplx_param_tensor = {}
                # for key in self.smplx_shape.keys():
                #         smplx_param_tensor[key] = torch.tensor(smplx_param[key].reshape(self.smplx_shape[key]),
                #                                                dtype=torch.float).to(self.device)

                # get image size
                img_path = os.path.join(frame_folder, '000001.jpg')
                img = cv2.imread(img_path)
                height, width, _ = img.shape 
                
                for fid in tqdm(annot_seq['total_valid_index'], position=1, leave=False):
                    # get image path
                    img_p = os.path.join(frame_folder, f'{fid+1:06d}.jpg')
                    image_path = img_p.replace(dataset_path + '/', '')
                    if not os.path.exists(img_path):
                        pdb.set_trace()

                    left_valid = annot_seq['left_valid'][fid].cpu().item()
                    right_valid = annot_seq['right_valid'][fid].cpu().item()
                    # smplx_valid = True if str(fid) in annot_seq['total_valid_index'] else False
                    
                    focal_length = list(annot_seq['focal'][fid])
                    principal_point = list(annot_seq['princpt'][fid])
                    
                    camera = build_cameras(
                        dict(
                            type='PerspectiveCameras',
                            convention='opencv',
                            in_ndc=False,
                            focal_length=focal_length,
                            image_size=(width, height),
                            principal_point=principal_point)).to(self.device)
                    
                    # 3d -> 2d
                    kps2d = camera.transform_points_screen(kps3d_c[fid]).detach().cpu().numpy().squeeze()[:, :2]
                    kps3d = kps3d_c[fid].detach().cpu().numpy().squeeze()

                    # test overlay
                    # img = cv2.imread(img_p)
                    # for kp in kps2d:
                    #     cv2.circle(img, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
                    # cv2.imwrite(f'{out_path}/{os.path.basename(frame_folder)}_{fid}.jpg', img)
                    
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
                        
                    # append image path
                    image_path_.append(image_path)

                    # append keypoints
                    keypoints2d_.append(kps2d)
                    keypoints3d_.append(kps3d)

                    # append smplx
                    for key in self.smplx_shape.keys():
                        # try:
                        smplx_[key].append(smplx_param[key][fid])
                        # except:
                        #     pdb.set_trace()

                    # append meta
                    meta_['principal_point'].append(principal_point)
                    meta_['focal_length'].append(focal_length)
                    meta_['height'].append(height)
                    meta_['width'].append(width)
                    meta_['gender'].append(gender)
                    meta_['sequence_name'].append(os.path.basename(frame_folder))
                    meta_['left_hand_valid'].append(left_valid)
                    meta_['right_hand_valid'].append(right_valid)
                    
            # get size
            size_i = len(annot_files)
            
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
            # human_data['contact'] = contact_

            # save meta and misc
            human_data['config'] = f'signavatar_{mode}'
            human_data['misc'] = self.misc_config
            human_data['meta'] = meta_

            os.makedirs(out_path, exist_ok=True)
            out_file = os.path.join(
                # out_path, f'moyo_{self.misc_config["flat_hand_mean"]}.npz')
                out_path, f'signavatar_{mode}_{seed}_{"{:05d}".format(size_i)}_{sid}.npz')
            human_data.dump(out_file) 