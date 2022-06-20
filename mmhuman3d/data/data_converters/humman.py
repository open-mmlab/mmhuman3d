import glob
import os

import numpy as np
import torch
from tqdm import tqdm

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_idx,
    get_keypoint_num,
)
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.data.data_structures.smc_reader import SMCReader
from mmhuman3d.models.body_models.builder import build_body_model
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class HuMManConverter(BaseModeConverter):
    """A mysterious dataset that will be announced soon."""

    ACCEPTED_MODES = ['test', 'train']

    def __init__(self, *args, **kwargs):
        super(HuMManConverter, self).__init__(*args, **kwargs)

        self.skip_no_iphone = True
        self.skip_no_keypoints3d = True
        self.downsample_ratio = 10  # uniformly sampling

        self.keypoints2d_humman_mask = None
        self.keypoints3d_humman_mask = None

        self.keypoint_convention_humman = 'coco_wholebody'
        self.num_keypoints_humman = \
            get_keypoint_num(self.keypoint_convention_humman)

        self.keypoint_convention_smpl = 'smpl_54'
        self.num_keypoints_smpl = \
            get_keypoint_num(self.keypoint_convention_smpl)

        self.left_hip_idx_humman = get_keypoint_idx(
            'left_hip', convention=self.keypoint_convention_humman)
        self.right_hip_idx_humman = get_keypoint_idx(
            'right_hip', convention=self.keypoint_convention_humman)
        self.root_idx_smpl = get_keypoint_idx(
            'pelvis_extra', convention=self.keypoint_convention_smpl)

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Body model used for keypoint computation
        self.smpl = build_body_model(
            dict(
                type='SMPL',
                keypoint_src='smpl_54',
                keypoint_dst=self.keypoint_convention_smpl,
                model_path='data/body_models/smpl',
                extra_joints_regressor='data/body_models/J_regressor_extra.npy'
            )).to(self.device)

        # Body model used for pelvis computation in SMCReader
        self.smpl_smc = build_body_model(
            dict(
                type='SMPL',
                gender='neutral',
                num_betas=10,
                keypoint_src='smpl_45',
                keypoint_dst='smpl_45',
                model_path='data/body_models/smpl',
                batch_size=1,
            )).to(self.device)

    def _derive_keypoints(self, global_orient, body_pose, betas, transl,
                          focal_length, image_size, camera_center):
        """Get SMPL-derived keypoints."""
        camera = build_cameras(
            dict(
                type='PerspectiveCameras',
                convention='opencv',
                in_ndc=False,
                focal_length=torch.tensor(focal_length,
                                          dtype=torch.float).reshape(1, 2),
                image_size=torch.tensor(image_size,
                                        dtype=torch.float).reshape(1, 2),
                principal_point=torch.tensor(camera_center,
                                             dtype=torch.float).reshape(
                                                 1, 2))).to(self.device)

        output = self.smpl(
            global_orient=torch.tensor(global_orient, device=self.device),
            body_pose=torch.tensor(body_pose, device=self.device),
            betas=torch.tensor(betas, device=self.device),
            transl=torch.tensor(transl, device=self.device),
            return_joints=True)

        keypoints3d = output['joints']
        keypoints2d_xyd = camera.transform_points_screen(keypoints3d)
        keypoints2d = keypoints2d_xyd[..., :2]

        keypoints3d = keypoints3d.cpu().numpy()
        keypoints2d = keypoints2d.cpu().numpy()

        # root align
        keypoints3d = keypoints3d - keypoints3d[:, [self.root_idx_smpl], :]

        return {'keypoints2d': keypoints2d, 'keypoints3d': keypoints3d}

    def _make_human_data(
        self,
        smpl,
        image_path,
        image_id,
        bbox_xywh,
        keypoints2d_smpl,
        keypoints3d_smpl,
        keypoints2d_humman,
        keypoints3d_humman,
    ):
        # use HumanData to store all data
        human_data = HumanData()

        # frames
        num_frames = len(image_path)

        # downsample idx
        selected_inds = np.arange(len(image_path))
        selected_inds = selected_inds[::self.downsample_ratio]

        smpl['global_orient'] = np.concatenate(
            smpl['global_orient'], axis=0).reshape(-1, 3)[selected_inds]
        smpl['body_pose'] = np.concatenate(
            smpl['body_pose'], axis=0).reshape(-1, 23, 3)[selected_inds]
        smpl['betas'] = np.concatenate(
            smpl['betas'], axis=0).reshape(-1, 10)[selected_inds]
        smpl['transl'] = np.concatenate(
            smpl['transl'], axis=0).reshape(-1, 3)[selected_inds]

        human_data['smpl'] = smpl

        # Save derived keypoints as ground truth
        # 2D SMPL keypoints
        keypoints2d_smpl = np.concatenate(
            keypoints2d_smpl, axis=0).reshape(num_frames,
                                              self.num_keypoints_smpl, 2)
        keypoints2d_smpl, keypoints2d_smpl_mask = convert_kps(
            keypoints2d_smpl,
            src=self.keypoint_convention_smpl,
            dst='human_data')
        keypoints2d_smpl = np.concatenate(
            [keypoints2d_smpl,
             np.ones([*keypoints2d_smpl.shape[:2], 1])],
            axis=-1)
        human_data['keypoints2d'] = keypoints2d_smpl[selected_inds]
        human_data['keypoints2d_mask'] = keypoints2d_smpl_mask

        # 3D SMPL keypoints
        keypoints3d_smpl = np.concatenate(
            keypoints3d_smpl, axis=0).reshape(num_frames,
                                              self.num_keypoints_smpl, 3)
        keypoints3d_smpl, keypoints3d_smpl_mask = convert_kps(
            keypoints3d_smpl,
            src=self.keypoint_convention_smpl,
            dst='human_data')
        keypoints3d_smpl = np.concatenate(
            [keypoints3d_smpl,
             np.ones([*keypoints3d_smpl.shape[:2], 1])],
            axis=-1)
        human_data['keypoints3d'] = keypoints3d_smpl[selected_inds]
        human_data['keypoints3d_mask'] = keypoints3d_smpl_mask

        # Save HuMMan keypoints
        # 2D HuMMan Keypoints
        keypoints2d_humman = np.concatenate(
            keypoints2d_humman, axis=0).reshape(num_frames,
                                                self.num_keypoints_humman, 3)
        keypoints2d_humman, keypoints2d_humman_mask = convert_kps(
            keypoints2d_humman,
            mask=self.keypoints2d_humman_mask,
            src=self.keypoint_convention_humman,
            dst='human_data')

        human_data['keypoints2d_humman'] = keypoints2d_humman[selected_inds]
        human_data['keypoints2d_humman_mask'] = keypoints2d_humman_mask

        # 3D HuMMan Keypoints
        keypoints3d_humman = np.concatenate(
            keypoints3d_humman, axis=0).reshape(num_frames,
                                                self.num_keypoints_humman, 4)
        keypoints3d_humman, keypoints3d_humman_mask = convert_kps(
            keypoints3d_humman,
            mask=self.keypoints3d_humman_mask,
            src=self.keypoint_convention_humman,
            dst='human_data')

        human_data['keypoints3d_humman'] = keypoints3d_humman[selected_inds]
        human_data['keypoints3d_humman_mask'] = keypoints3d_humman_mask

        # Save bboxes
        bbox_xywh = np.array(bbox_xywh).reshape((num_frames, 4))
        bbox_xywh = np.concatenate(
            [bbox_xywh, np.ones([num_frames, 1])], axis=-1)
        assert bbox_xywh.shape == (num_frames, 5)
        human_data['bbox_xywh'] = bbox_xywh[selected_inds]

        # Save other attributes
        human_data['image_path'] = [image_path[i] for i in selected_inds]
        human_data['image_id'] = [image_id[i] for i in selected_inds]
        human_data['config'] = 'humman'

        human_data.compress_keypoints_by_mask()

        return human_data

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
                keypoints2d_mask stored in HumanData() format
        """

        kinect_smpl = {}
        kinect_smpl['body_pose'] = []
        kinect_smpl['global_orient'] = []
        kinect_smpl['betas'] = []
        kinect_smpl['transl'] = []

        iphone_smpl = {}
        iphone_smpl['body_pose'] = []
        iphone_smpl['global_orient'] = []
        iphone_smpl['betas'] = []
        iphone_smpl['transl'] = []

        # structs we use
        kinect_image_path_, kinect_image_id_, kinect_bbox_xywh_, \
            kinect_keypoints2d_smpl_, kinect_keypoints3d_smpl_, \
            kinect_keypoints2d_humman_, kinect_keypoints3d_humman_ = \
            [], [], [], [], [], [], []
        iphone_image_path_, iphone_image_id_, iphone_bbox_xywh_, \
            iphone_keypoints2d_smpl_, iphone_keypoints3d_smpl_, \
            iphone_keypoints2d_humman_, iphone_keypoints3d_humman_ = \
            [], [], [], [], [], [], []

        ann_paths = sorted(glob.glob(os.path.join(dataset_path, '*.smc')))

        with open(os.path.join(dataset_path, f'{mode}.txt'), 'r') as f:
            split = set(f.read().splitlines())

        for ann_path in tqdm(ann_paths):

            if os.path.basename(ann_path) not in split:
                continue

            try:
                smc_reader = SMCReader(ann_path, body_model=self.smpl_smc)
            except OSError:
                print(f'Unable to load {ann_path}.')
                continue

            if self.skip_no_keypoints3d and not smc_reader.keypoint_exists:
                continue
            if self.skip_no_iphone and not smc_reader.iphone_exists:
                continue

            num_kinect = smc_reader.get_num_kinect()
            num_iphone = smc_reader.get_num_iphone()
            num_frames = smc_reader.get_kinect_num_frames()

            device_list = [('Kinect', i) for i in range(num_kinect)] + \
                [('iPhone', i) for i in range(num_iphone)]
            assert len(device_list) == num_kinect + num_iphone

            for device, device_id in device_list:
                assert device in {
                    'Kinect', 'iPhone'
                }, f'Undefined device: {device}, ' \
                   f'should be "Kinect" or "iPhone"'

                if device == 'Kinect':
                    image_id_ = kinect_image_id_
                    image_path_ = kinect_image_path_
                    bbox_xywh_ = kinect_bbox_xywh_
                    keypoints2d_humman_ = kinect_keypoints2d_humman_
                    keypoints3d_humman_ = kinect_keypoints3d_humman_
                    keypoints2d_smpl_ = kinect_keypoints2d_smpl_
                    keypoints3d_smpl_ = kinect_keypoints3d_smpl_
                    smpl_ = kinect_smpl
                    width, height = \
                        smc_reader.get_kinect_color_resolution(device_id)
                    intrinsics = smc_reader.get_kinect_color_intrinsics(
                        device_id)
                    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                    focal_length = (fx, fy)
                    camera_center = (cx, cy)  # xy
                    image_size = (height, width)  # (height, width)

                else:
                    image_id_ = iphone_image_id_
                    image_path_ = iphone_image_path_
                    bbox_xywh_ = iphone_bbox_xywh_
                    keypoints2d_humman_ = iphone_keypoints2d_humman_
                    keypoints3d_humman_ = iphone_keypoints3d_humman_
                    keypoints2d_smpl_ = iphone_keypoints2d_smpl_
                    keypoints3d_smpl_ = iphone_keypoints3d_smpl_
                    smpl_ = iphone_smpl
                    width, height = smc_reader.get_iphone_color_resolution()
                    intrinsics = smc_reader.get_iphone_intrinsics()
                    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                    focal_length = (fx, fy)
                    camera_center = (cx, cy)  # xy
                    image_size = (height, width)  # (height, width)

                assert device_id >= 0, f'Negative device id: {device_id}'

                keypoint_convention_humman = \
                    smc_reader.get_keypoints_convention()
                assert self.keypoint_convention_humman == \
                       keypoint_convention_humman

                # get keypoints2d (all frames)
                keypoints2d_humman, keypoints2d_humman_mask = \
                    smc_reader.get_keypoints2d(device, device_id)

                if self.keypoints2d_humman_mask is None:
                    self.keypoints2d_humman_mask = keypoints2d_humman_mask
                assert np.allclose(self.keypoints2d_humman_mask,
                                   keypoints2d_humman_mask)

                keypoints2d_humman_.append(keypoints2d_humman)

                # get keypoints3d (all frames)
                keypoints3d_humman, keypoints3d_humman_mask = \
                    smc_reader.get_keypoints3d(device, device_id)

                if self.keypoints3d_humman_mask is None:
                    self.keypoints3d_humman_mask = keypoints3d_humman_mask
                assert np.allclose(self.keypoints3d_humman_mask,
                                   keypoints3d_humman_mask)

                # root-align keypoints3d
                left_hip_keypoints = \
                    keypoints3d_humman[:, [self.left_hip_idx_humman], :3]
                right_hip_keypoints = \
                    keypoints3d_humman[:, [self.right_hip_idx_humman], :3]
                root_keypoints = \
                    (left_hip_keypoints + right_hip_keypoints) / 2.0
                keypoints3d_humman[..., :3] = \
                    keypoints3d_humman[..., :3] - root_keypoints
                keypoints3d_humman_.append(keypoints3d_humman)

                # get smpl (all frames)
                smpl_dict = smc_reader.get_smpl(device, device_id)
                smpl_['body_pose'].append(smpl_dict['body_pose'])
                smpl_['global_orient'].append(smpl_dict['global_orient'])
                smpl_['transl'].append(smpl_dict['transl'])

                # expand betas
                betas_expanded = np.tile(smpl_dict['betas'],
                                         num_frames).reshape(-1, 10)
                smpl_['betas'].append(betas_expanded)

                # get keypoints derived from SMPL and use them as supervision
                smpl_keypoints = self._derive_keypoints(
                    **smpl_dict,
                    focal_length=focal_length,
                    image_size=image_size,
                    camera_center=camera_center)
                keypoints2d_smpl = smpl_keypoints['keypoints2d']
                keypoints3d_smpl = smpl_keypoints['keypoints3d']
                keypoints2d_smpl_.append(keypoints2d_smpl)
                keypoints3d_smpl_.append(keypoints3d_smpl)

                # compute bbox from keypoints2d
                for kp2d in keypoints2d_smpl:
                    assert kp2d.shape == (self.num_keypoints_smpl, 2)
                    xs, ys = kp2d[:, 0], kp2d[:, 1]
                    xmin = max(np.min(xs), 0)
                    xmax = min(np.max(xs), width - 1)
                    ymin = max(np.min(ys), 0)
                    ymax = min(np.max(ys), height - 1)
                    bbox_xyxy = [xmin, ymin, xmax, ymax]
                    bbox_xyxy = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
                    bbox_xywh = self._xyxy2xywh(bbox_xyxy)
                    bbox_xywh_.append(bbox_xywh)

                # get image paths (smc paths)
                image_path = os.path.basename(ann_path)
                for frame_id in range(num_frames):
                    image_id = (device, device_id, frame_id)
                    image_id_.append(image_id)
                    image_path_.append(image_path)

        os.makedirs(out_path, exist_ok=True)

        # make kinect human data
        kinect_human_data = self._make_human_data(
            kinect_smpl, kinect_image_path_, kinect_image_id_,
            kinect_bbox_xywh_, kinect_keypoints2d_smpl_,
            kinect_keypoints3d_smpl_, kinect_keypoints2d_humman_,
            kinect_keypoints3d_humman_)

        file_name = f'humman_{mode}_kinect_ds{self.downsample_ratio}_smpl.npz'
        out_file = os.path.join(out_path, file_name)
        kinect_human_data.dump(out_file)

        # make iphone human data
        iphone_human_data = self._make_human_data(
            iphone_smpl, iphone_image_path_, iphone_image_id_,
            iphone_bbox_xywh_, iphone_keypoints2d_smpl_,
            iphone_keypoints3d_smpl_, iphone_keypoints2d_humman_,
            iphone_keypoints3d_humman_)

        file_name = f'humman_{mode}_iphone_ds{self.downsample_ratio}_smpl.npz'
        out_file = os.path.join(out_path, file_name)
        iphone_human_data.dump(out_file)
