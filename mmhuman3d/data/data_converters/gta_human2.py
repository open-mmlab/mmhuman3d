import glob
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS

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
        # use HumanData to store all data
        human_data = HumanData()

        smplx = {}
        smplx['body_pose'], smplx['transl'], smplx['global_orient'], smplx['betas'] = [], [], [], []
        smplx['left_hand_pose'], smplx['right_hand_pose']  = [], []

        # structs we use
        image_path_, bbox_xywh_, keypoints_2d_gta_, keypoints_3d_gta_, \
            keypoints_2d_, keypoints_3d_ = [], [], [], [], [], []

        if mode == 'single':
            ann_paths = sorted(
                glob.glob(os.path.join(dataset_path, 'annotations_single_person', '*.npz')))
        elif mode == 'multiple':
            ann_paths = sorted(
                glob.glob(os.path.join(dataset_path, 'annotations_multiple_person', '*.npz')))

        for ann_path in tqdm(ann_paths):

            # with open(ann_path, 'rb') as f:
            #     ann = pickle.load(f, encoding='latin1')

            ann = dict(np.load(ann_path, allow_pickle=True))

            base = os.path.basename(ann_path)  # -> seq_00090376_154131.npz -> seq_00090376_154131
            seq_idx, ped_idx = base[4:12], base[13:19]  # -> 00090376, 154131
            num_frames = len(ann['body_pose'])

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
                    batch_size=len(betas),
                )).to(self.device)
            output = body_model(
                global_orient=torch.tensor(global_orient, device=self.device),
                body_pose=torch.tensor(body_pose, device=self.device),
                betas=torch.tensor(betas, device=self.device),
                transl=torch.tensor(transl, device=self.device),
                left_hand_pose=torch.tensor(left_hand_pose, device=self.device),
                right_hand_pose=torch.tensor(right_hand_pose, device=self.device),
                return_joints=True)


            keypoints_3d = output['joints']
            keypoints_2d_xyd = self.camera.transform_points_screen(keypoints_3d)
            keypoints_2d = keypoints_2d_xyd[..., :2]

            keypoints_3d = keypoints_3d.detach().cpu().numpy()
            keypoints_2d = keypoints_2d.detach().cpu().numpy()

            # root align
            root_idx = get_keypoint_idx('pelvis_extra', convention='smplx')
            keypoints_3d_gta = \
                keypoints_3d_gta - keypoints_3d_gta[:, [root_idx], :]
            keypoints_3d = keypoints_3d - keypoints_3d[:, [root_idx], :]

            for frame_idx in range(num_frames):
                
                image_path = os.path.join('images_' + mode, 'seq_' + seq_idx, '{:08d}.jpeg'.format(frame_idx))
                print(image_path)
                bbox_xywh = ann['bbox_xywh'][frame_idx]

                # reject examples with bbox center outside the frame
                x, y, w, h = bbox_xywh
                x = max([x, 0.0])
                y = max([y, 0.0])
                w = min([w, 1920 - x])  # x + w <= img_width
                h = min([h, 1080 - y])  # y + h <= img_height
                if not (0 <= x < 1920 and 0 <= y < 1080 and 0 < w < 1920
                        and 0 < h < 1080):
                    continue

                image_path_.append(image_path)
                bbox_xywh_.append([x, y, w, h])

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
        smplx['left_hand_pose'] = np.array(smplx['left_hand_pose']).reshape(-1, 45)
        smplx['right_hand_pose'] = np.array(smplx['right_hand_pose']).reshape(-1, 45)

        human_data['smplx'] = smplx

        import pdb; pdb.set_trace()

        keypoints2d = np.array(keypoints_2d_).reshape(-1, 144, 2)
        keypoints2d = np.concatenate(
            [keypoints2d, np.ones([keypoints2d.shape[0], 144, 1])], axis=-1)
        keypoints2d, keypoints2d_mask = \
            convert_kps(keypoints2d, src='smplx', dst='human_data')
        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask

        keypoints3d = np.array(keypoints_3d_).reshape(-1, 144, 3)
        keypoints3d = np.concatenate(
            [keypoints3d, np.ones([keypoints3d.shape[0], 144, 1])], axis=-1)
        keypoints3d, keypoints3d_mask = \
            convert_kps(keypoints3d, src='smplx', dst='human_data')
        human_data['keypoints3d'] = keypoints3d
        human_data['keypoints3d_mask'] = keypoints3d_mask

        keypoints2d_gta = np.array(keypoints_2d_gta_).reshape(-1, 144, 3)
        keypoints2d_gta, keypoints2d_gta_mask = \
            convert_kps(keypoints2d_gta, src='smplx', dst='human_data')
        human_data['keypoints2d_gta'] = keypoints2d_gta
        human_data['keypoints2d_gta_mask'] = keypoints2d_gta_mask

        keypoints3d_gta = np.array(keypoints_3d_gta_).reshape(-1, 144, 4)
        keypoints3d_gta, keypoints3d_gta_mask = \
            convert_kps(keypoints3d_gta, src='smplx', dst='human_data')
        human_data['keypoints3d_gta'] = keypoints3d_gta
        human_data['keypoints3d_gta_mask'] = keypoints3d_gta_mask

        human_data['image_path'] = image_path_

        bbox_xywh = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh = np.hstack([bbox_xywh, np.ones([bbox_xywh.shape[0], 1])])
        human_data['bbox_xywh'] = bbox_xywh

        human_data['config'] = 'gta_human2' + mode
        human_data.compress_keypoints_by_mask()

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'gta_human2' + mode + '.npz'
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)