import glob
import os
import random

import numpy as np
import torch
from tqdm import tqdm
import pdb

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS
from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idxs_by_part

smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}
smplx_shape_except_expression = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3)}
smplx_shape = smplx_shape_except_expression

@DATA_CONVERTERS.register_module()
class GTAHuman2Converter(BaseModeConverter):
    """GTA-Human++ dataset

    Args:
        modes (list): 'single', 'multiple' for accepted modes
    """

    ACCEPTED_MODES = ['single', 'multiple']

    def __init__(self, modes=[], *args, **kwargs):
        super(GTAHuman2Converter, self).__init__(modes, *args, **kwargs)

        focal_length = 1158.0337  # default setting
        camera_center = (960, 540)  # xy
        image_size = (1080, 1920)  # (height, width)

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        # self.smplx = build_body_model(
        #     dict(
        #         type='SMPLX',
        #         keypoint_src='smplx',
        #         keypoint_dst='smplx',
        #         model_path='data/body_models/smplx',
        #         num_betas=10,
        #         use_face_contour=True,
        #         flat_hand_mean=True,
        #         use_pca=True,
        #         num_pca_comps=24,
        #         batch_size=55,
        #     )).to(self.device)

        self.camera = build_cameras(
            dict(
                type='PerspectiveCameras',
                convention='opencv',
                in_ndc=False,
                focal_length=focal_length,
                image_size=image_size,
                principal_point=camera_center)).to(self.device)
        
        self.misc = dict(
            focal_length=np.array([focal_length, focal_length]),
            camera_center=np.array(camera_center),
            image_shape=np.array(image_size),
            bbox_source='keypoints2d_smplx', smplx_source='smplifyx',
            cam_param_type='prespective', flat_hand_mean=True,)


    
    def _keypoints_to_scaled_bbox_fh(self, keypoints, occ, self_occ, scale=1.0, convention='gta'):
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
            occ_p = occ[kp_id]
            self_occ_p = self_occ[kp_id]

            if np.sum(self_occ_p) / len(kp_id) >= 0.5 or np.sum(occ_p) / len(kp_id) >= 0.5:
                conf = 0
                # print(f'{body_part} occluded, occlusion: {np.sum(self_occ_p + occ_p) / len(kp_id)}, skip')
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

            bbox = np.stack([xmin, ymin, xmax, ymax, conf], axis=0).astype(np.float32)

            bboxs.append(bbox)
        return bboxs[0], bboxs[1], bboxs[2]


    def _revert_smplx_hands_pca(self, param_dict, num_pca_comps):
        # gta-human++ 24
        hl_pca = param_dict['left_hand_pose']
        hr_pca = param_dict['right_hand_pose']

        smplx_model = dict(np.load('data/body_models/smplx/SMPLX_NEUTRAL.npz', allow_pickle=True))

        hl = smplx_model['hands_componentsl'] # 45, 45
        hr = smplx_model['hands_componentsr'] # 45, 45

        hl_pca = np.concatenate((hl_pca, np.zeros((len(hl_pca), 45 - num_pca_comps))), axis=1)
        hr_pca = np.concatenate((hr_pca, np.zeros((len(hr_pca), 45 - num_pca_comps))), axis=1)

        hl_reverted = np.einsum('ij, jk -> ik', hl_pca, hl).astype(np.float32)
        hr_reverted = np.einsum('ij, jk -> ik', hr_pca, hr).astype(np.float32)

        param_dict['left_hand_pose'] = hl_reverted
        param_dict['right_hand_pose'] = hr_reverted

        return param_dict


    def convert_by_mode(self, dataset_path: str, out_path: str, mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Accetped mode 'single' and 'multi', pointing to different dataset
        Returns:
            dict:
                A dict containing keys video_path, smplh, meta, frame_idx
                stored in HumanData() format
        """

        if mode == 'single':
            ann_paths_total = sorted(
                glob.glob(os.path.join(dataset_path, 'annotations_single_smplx', '*.npz')))
        elif mode == 'multiple':
            ann_paths_total = sorted(
                glob.glob(os.path.join(dataset_path, 'annotations_multiple_smplx', '*.npz')))

        print(f'{len(ann_paths_total)} sequences are avliable in this mode: {mode}')

        seed, size = '230619', '04000'
        size = str(max([int(size), len(ann_paths_total)]))
        random.seed(int(seed))
        # random.shuffle(ann_paths_total)
        # ann_paths_total = ann_paths_total[:int(size)]

        s_num = 9
        print(f'Seperate in to {s_num} files')

        slice = int(int(size)/s_num) + 1

        for i in range(s_num):

            body_model = build_body_model(
                        dict(
                            type='SMPLX',
                            keypoint_src='smplx',
                            keypoint_dst='smplx',
                            model_path='data/body_models/smplx',
                            num_betas=10,
                            use_face_contour=True,
                            flat_hand_mean=True,
                            use_pca=False,
                            batch_size=1,
                        )).to(self.device)

            ann_paths = ann_paths_total[slice*i:slice*(i+1)]

            # use HumanData to store all data
            human_data = HumanData()
        
            smplx = {}
            smplx['body_pose'], smplx['transl'], smplx['global_orient'], smplx['betas'] = [], [], [], []
            smplx['left_hand_pose'], smplx['right_hand_pose']  = [], []
            smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 21, 3), \
                    'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 'leye_pose': (-1, 3),
                    'reye_pose': (-1, 3), \
                    'jaw_pose': (-1, 3), 'expression': (-1, 10)}
            bboxs = {}
            for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
                bboxs[bbox_name] = []

            # structs we use
            image_path_, bbox_xywh_, keypoints_2d_gta_, keypoints_3d_gta_, \
                keypoints_2d_, keypoints_3d_= [], [], [], [], [], []


            for ann_path in tqdm(ann_paths, desc=f'Processing {i}/{s_num} slices'):

                # with open(ann_path, 'rb') as f:
                #     ann = pickle.load(f, encoding='latin1')

                ann = dict(np.load(ann_path, allow_pickle=True))

                base = os.path.basename(ann_path)  # -> seq_00090376_154131.npz -> seq_00090376_154131
                seq_idx, ped_idx = base[4:12], base[13:19]  # -> 00090376, 154131
                num_frames = len(ann['body_pose'])

                
                # convention
                # try:
                #     aaa =  ann['keypoints_2d']
                # except:
                #     print(ann.keys())


                keypoints_2d_gta, keypoints_2d_gta_mask = convert_kps(
                    ann['keypoints_2d'], src='gta', dst='smplx')
                keypoints_3d_gta, keypoints_3d_gta_mask = convert_kps(
                    ann['keypoints_3d'], src='gta', dst='smplx')
                
                global_orient = np.array(ann['global_orient'])
                body_pose = ann['body_pose']
                betas = ann['betas']
                transl = ann['transl']
                left_hand_pose = ann['left_hand_pose']
                right_hand_pose = ann['right_hand_pose']

                # normally gta-human++ hands is presented in pca=24
                hand_pca_comps = left_hand_pose.shape[1]
                if hand_pca_comps < 45:
                    ann = self._revert_smplx_hands_pca(param_dict=ann, num_pca_comps=hand_pca_comps)
                    left_hand_pose = ann['left_hand_pose']
                    right_hand_pose = ann['right_hand_pose']


                intersect_key = list(set(ann.keys()) & set(smplx_shape.keys()))
                
                # prepare tensor
                body_model_param_tensor = {}
                batch_frames = global_orient.shape[0]
                for key in smplx_shape.keys():
                    if key in intersect_key:
                        body_model_param_tensor[key] = torch.tensor(np.array(ann[key]).reshape(smplx_shape[key]),
                                device=self.device, dtype=torch.float32)
                    else:
                        shape = np.array(smplx_shape[key])
                        shape[0] = batch_frames
                        zero_tensor = np.zeros(shape)
                        body_model_param_tensor[key] = torch.tensor(zero_tensor, device=self.device, dtype=torch.float32)


                # for keys in body_model_param_tensor.keys():
                #     print(keys, body_model_param_tensor[keys].shape)
                output = body_model(return_verts=True, **body_model_param_tensor)

                # output = body_model(
                #     global_orient=torch.tensor(global_orient, device=self.device),
                #     body_pose=torch.tensor(body_pose, device=self.device),
                #     betas=torch.tensor(betas, device=self.device),
                #     transl=torch.tensor(transl, device=self.device),
                #     left_hand_pose=torch.tensor(left_hand_pose, device=self.device),
                #     right_hand_pose=torch.tensor(right_hand_pose, device=self.device),
                #     return_joints=True)


                keypoints_3d = output['joints']
                keypoints_2d_xyd = self.camera.transform_points_screen(keypoints_3d)
                keypoints_2d = keypoints_2d_xyd[..., :2]

                keypoints_3d = keypoints_3d.detach().cpu().numpy()
                keypoints_2d = keypoints_2d.detach().cpu().numpy()

                if np.sum(np.isnan(keypoints_2d)) + np.sum(np.isnan(keypoints_3d)) > 0:
                    print(f'{base} skip due to nan in data')
                    continue
                    # raise ValueError(f'{base} skip due to nan in data')

                # root align
                # root_idx = get_keypoint_idx('pelvis', convention='smplx')
                # keypoints_3d_gta_ra = \
                #     keypoints_3d_gta - keypoints_3d_gta[:, [root_idx], :]
                # keypoints_3d_ra = keypoints_3d - keypoints_3d[:, [root_idx], :]



                for frame_idx in range(num_frames):
                    
                    image_path = os.path.join('images_' + mode, 'seq_' + seq_idx, '{:08d}.jpeg'.format(frame_idx))
                    image_path_real = os.path.join(f'/mnt/e/gta_human2/images_{mode}', 'seq_' + seq_idx, '{:08d}.jpeg'.format(frame_idx))
                    if not os.path.exists(image_path_real):
                        continue
                        # raise FileNotFoundError(image_path_real)
                    # import pdb; pdb.set_trace()
                    # print(image_path)
                    # reject examples with bbox center outside the frame
                    # x, y, w, h = bbox_xywh
                    # x = max([x, 0.0])
                    # y = max([y, 0.0])
                    # w = min([w, 1920 - x])  # x + w <= img_width
                    # h = min([h, 1080 - y])  # y + h <= img_height
                    # if not (0 <= x < 1920 and 0 <= y < 1080 and 0 < w < 1920
                    #         and 0 < h < 1080):
                    #     continue

                    image_path_.append(image_path)
                    # bbox_xywh_.append([x, y, w, h])

                    kp = ann['keypoints_2d'][frame_idx]
                    kp = kp[:, :2]
                    occ = ann['occ'][frame_idx]
                    self_occ = ann['self_occ'][frame_idx]

                    # import pdb; pdb.set_trace()
                    # since the 2d keypoints are nearly correct, scale a little bit
                    bbox_tmp_ = {}

                    # bbox = np.array([max(0, xmin), max(0, ymin), min(1920, xmax), min(1080, ymax)])
                    # import pdb; pdb.set_trace()
                    if np.sum(occ)/len(kp) >= 0.5:
                        body_conf = 0
                    else:
                        body_conf = 1

                    bbox_tmp_['bbox_xywh'] = ann['bbox_xywh'][frame_idx]
                    bbox_tmp_['face_bbox_xywh'], bbox_tmp_['lhand_bbox_xywh'], bbox_tmp_[
                        'rhand_bbox_xywh'] = self._keypoints_to_scaled_bbox_fh(kp, occ, self_occ, 1.0)
                    for bbox_name in ['bbox_xywh', 'face_bbox_xywh', 'lhand_bbox_xywh', 'rhand_bbox_xywh']:
                        # import pdb; pdb.set_trace()
                        if bbox_name != 'bbox_xywh':
                            bbox = bbox_tmp_[bbox_name]
                            xmin, ymin, xmax, ymax, conf = bbox
                            bbox = np.array([max(0, xmin), max(0, ymin), min(1920, xmax), min(1080, ymax)])
                            bbox_xywh = self._xyxy2xywh(bbox)
                        else:
                            bbox_xywh = bbox_tmp_[bbox_name].tolist()
                            xmin, ymin, w, h = bbox_xywh
                            xmax = w + xmin
                            ymax = h + ymin
                            bbox = np.array([max(0, xmin), max(0, ymin), min(1920, xmax), min(1080, ymax)])
                            conf = body_conf

                        if bool(set(bbox).intersection([0, 1280, 1920])):
                            bbox_xywh.append(0)
                        else: 
                            bbox_xywh.append(conf)
                        bboxs[bbox_name].append(bbox_xywh)
                        # print(bbox_xywh)

                    smplx['global_orient'].append(global_orient[frame_idx])
                    smplx['body_pose'].append(body_pose[frame_idx])
                    smplx['betas'].append(betas[frame_idx])
                    smplx['transl'].append(transl[frame_idx])
                    smplx['left_hand_pose'].append(left_hand_pose[frame_idx])
                    smplx['right_hand_pose'].append(right_hand_pose[frame_idx])

                    keypoints_2d_gta_.append(keypoints_2d_gta[frame_idx])
                    keypoints_3d_gta_.append(keypoints_3d_gta[frame_idx])
                    keypoints_2d_.append(keypoints_2d[frame_idx])
                    keypoints_3d_.append(keypoints_3d[frame_idx])

            smplx['global_orient'] = np.array(smplx['global_orient']).reshape(-1, 3)
            smplx['body_pose'] = np.array(smplx['body_pose']).reshape(-1, 21, 3)
            smplx['betas'] = np.array(smplx['betas']).reshape(-1, 10)
            smplx['transl'] = np.array(smplx['transl']).reshape(-1, 3)
            smplx['left_hand_pose'] = np.array(smplx['left_hand_pose']).reshape(-1, 15, 3)
            smplx['right_hand_pose'] = np.array(smplx['right_hand_pose']).reshape(-1, 15, 3)

            human_data['smplx'] = smplx
            # import pdb; pdb.set_trace()
            for key in bboxs.keys():
                bbox_ = np.array(bboxs[key]).reshape((-1, 5))
                # print(bbox_[:, -1])
                # bbox_ = np.hstack([bbox_, np.ones([bbox_.shape[0], 1])])
                human_data[key] = bbox_
            
            keypoints2d = np.array(keypoints_2d_).reshape(-1, 144, 2)
            keypoints2d_conf = np.ones([keypoints2d.shape[0], 144, 1])
            keypoints2d_conf[:][-1] = 0
            keypoints2d = np.concatenate(
                [keypoints2d, keypoints2d_conf], axis=-1)
            keypoints2d, keypoints2d_mask = \
                convert_kps(keypoints2d, src='smplx', dst='human_data')
            # import pdb; pdb.set_trace()
            human_data['keypoints2d_smplx'] = keypoints2d
            human_data['keypoints2d_smplx_mask'] = keypoints2d_mask

            keypoints3d = np.array(keypoints_3d_).reshape(-1, 144, 3)
            keypoints2d_conf = np.ones([keypoints3d.shape[0], 144, 1])
            keypoints2d_conf[-1][-1] = 0
            keypoints3d = np.concatenate(
                [keypoints3d, np.ones([keypoints3d.shape[0], 144, 1])], axis=-1)
            keypoints3d, keypoints3d_mask = \
                convert_kps(keypoints3d, src='smplx', dst='human_data')
            human_data['keypoints3d_smplx'] = keypoints3d
            human_data['keypoints3d_smplx_mask'] = keypoints3d_mask

            keypoints2d_gta = np.array(keypoints_2d_gta_).reshape(-1, 144, 3)
            keypoints2d_gta, keypoints2d_gta_mask = \
                convert_kps(keypoints2d_gta, src='smplx', dst='human_data')
            human_data['keypoints2d_original'] = keypoints2d_gta
            human_data['keypoints2d_original_mask'] = keypoints2d_gta_mask

            keypoints3d_gta = np.array(keypoints_3d_gta_).reshape(-1, 144, 4)
            keypoints3d_gta, keypoints3d_gta_mask = \
                convert_kps(keypoints3d_gta, src='smplx', dst='human_data')
            human_data['keypoints3d_original'] = keypoints3d_gta
            human_data['keypoints3d_original_mask'] = keypoints3d_gta_mask

            human_data['image_path'] = image_path_

            # bbox_xywh = np.array(bbox_xywh_).reshape((-1, 4))
            # bbox_xywh = np.hstack([bbox_xywh, np.ones([bbox_xywh.shape[0], 1])])
            # human_data['bbox_xywh'] = bbox_xywh+

            human_data['config'] = 'gta_human2' + mode

            # import pdb; pdb.set_trace()
            human_data.compress_keypoints_by_mask()

            # store data
            if not os.path.isdir(out_path):
                os.makedirs(out_path)

            file_name = f'gta_human2{mode}_{str(seed)}_{str(size)}_{str(i)}.npz'
            out_file = os.path.join(out_path, file_name)
            human_data.dump(out_file)

            # import pdb; pdb.set_trace()