import glob
import os

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.data.data_structures import SMCReader
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class HuMManConverter(BaseConverter):
    """A mysterious dataset that will be announced soon."""

    skip_no_iphone = True
    skip_no_keypoints3d = True

    def _make_human_data(
        self,
        smpl,
        keypoints_convention,
        image_path,
        image_id,
        bbox_xywh,
        keypoints_2d,
        keypoints2d_mask,
        keypoints_3d,
        keypoints3d_mask,
    ):
        # use HumanData to store all data
        human_data = HumanData()

        smpl['global_orient'] = np.concatenate(
            smpl['global_orient'], axis=0).reshape(-1, 3)
        smpl['body_pose'] = np.concatenate(
            smpl['body_pose'], axis=0).reshape(-1, 23, 3)
        smpl['betas'] = np.concatenate(smpl['betas'], axis=0).reshape(-1, 10)
        smpl['transl'] = np.concatenate(smpl['transl'], axis=0).reshape(-1, 3)

        human_data['smpl'] = smpl

        num_keypoints = get_keypoint_num(keypoints_convention)

        keypoints2d = np.concatenate(
            keypoints_2d, axis=0).reshape(-1, num_keypoints, 3)
        keypoints2d, keypoints2d_mask = convert_kps(
            keypoints2d,
            mask=keypoints2d_mask,
            src=keypoints_convention,
            dst='human_data')
        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask

        keypoints3d = np.concatenate(
            keypoints_3d, axis=0).reshape(-1, num_keypoints, 4)
        keypoints3d, keypoints3d_mask = convert_kps(
            keypoints3d,
            mask=keypoints3d_mask,
            src=keypoints_convention,
            dst='human_data')
        human_data['keypoints3d'] = keypoints3d
        human_data['keypoints3d_mask'] = keypoints3d_mask

        human_data['image_path'] = image_path
        human_data['image_id'] = image_id

        bbox_xywh = np.array(bbox_xywh).reshape((-1, 4))
        bbox_xywh = np.concatenate(
            [bbox_xywh, np.ones([bbox_xywh.shape[0], 1])], axis=-1)
        human_data['bbox_xywh'] = bbox_xywh

        human_data['config'] = 'humman'

        human_data.compress_keypoints_by_mask()

        return human_data

    def convert(self, dataset_path: str, out_path: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file

        Returns:
            dict:
                A dict containing keys video_path, smplh, meta, frame_idx
                stored in HumanData() format
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
            kinect_keypoints_2d_, kinect_keypoints_3d_ = [], [], [], [], []
        iphone_image_path_, iphone_image_id_, iphone_bbox_xywh_, \
            iphone_keypoints_2d_, iphone_keypoints_3d_ = [], [], [], [], []
        keypoints_convention_, keypoints2d_mask_, keypoints3d_mask_ = \
            None, None, None

        ann_paths = sorted(glob.glob(os.path.join(dataset_path, '*.smc')))

        for ann_path in tqdm(ann_paths):

            try:
                smc_reader = SMCReader(ann_path)
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
                    keypoints_2d_ = kinect_keypoints_2d_
                    keypoints_3d_ = kinect_keypoints_3d_
                    smpl_ = kinect_smpl
                else:
                    image_id_ = iphone_image_id_
                    image_path_ = iphone_image_path_
                    bbox_xywh_ = iphone_bbox_xywh_
                    keypoints_2d_ = iphone_keypoints_2d_
                    keypoints_3d_ = iphone_keypoints_3d_
                    smpl_ = iphone_smpl

                assert device_id >= 0, f'Negative device id: {device_id}'

                keypoints_convention = smc_reader.get_keypoints_convention()
                if keypoints_convention_ is None:
                    keypoints_convention_ = keypoints_convention
                assert keypoints_convention_ == keypoints_convention

                # get keypoints2d (all frames)
                keypoints2d, keypoints2d_mask = smc_reader.get_keypoints2d(
                    device, device_id)
                if keypoints2d_mask_ is None:
                    keypoints2d_mask_ = keypoints2d_mask
                assert (keypoints2d_mask_ == keypoints2d_mask).all()
                keypoints_2d_.append(keypoints2d)

                # compute bbox from keypoints2d
                xs, ys = keypoints2d[:, :, 0], keypoints2d[:, :, 1]
                xmins, xmaxs = np.min(xs, axis=1), np.max(xs, axis=1)
                ymins, ymaxs = np.min(ys, axis=1), np.max(ys, axis=1)

                for xmin, xmax, ymin, ymax in zip(xmins, xmaxs, ymins, ymaxs):
                    bbox_xyxy = [xmin, ymin, xmax, ymax]
                    bbox_xywh = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
                    bbox_xywh_.append(bbox_xywh)

                # get keypoints3d (all frames)
                keypoints3d, keypoints3d_mask = smc_reader.get_keypoints3d(
                    device, device_id)
                if keypoints3d_mask_ is None:
                    keypoints3d_mask_ = keypoints3d_mask
                assert (keypoints3d_mask_ == keypoints3d_mask).all()
                keypoints_3d_.append(keypoints3d)

                # get smpl (all frames)
                smpl_dict = smc_reader.get_smpl()
                smpl_['body_pose'].append(smpl_dict['body_pose'])
                smpl_['global_orient'].append(
                    smpl_dict['global_orient'])
                smpl_['transl'].append(smpl_dict['transl'])

                # expand betas
                betas_expanded = np.tile(
                    smpl_dict['betas'], num_frames).reshape(-1, 10)
                smpl_['betas'].append(betas_expanded)

                # get image paths (smc paths)
                image_path = os.path.basename(ann_path)
                for frame_id in range(num_frames):
                    image_id = (device, device_id, frame_id)
                    image_id_.append(image_id)
                    image_path_.append(image_path)

                assert len(keypoints2d) == num_frames
                assert len(keypoints3d) == num_frames

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        # make kinect human data
        kinect_human_data = self._make_human_data(
            kinect_smpl, keypoints_convention_, kinect_image_path_,
            kinect_image_id_, kinect_bbox_xywh_, kinect_keypoints_2d_,
            keypoints2d_mask_, kinect_keypoints_3d_, keypoints3d_mask_)

        # store kinect human data
        file_name = 'humman_kinect.npz'
        out_file = os.path.join(out_path, file_name)
        kinect_human_data.dump(out_file)

        # make iphone human data
        iphone_human_data = self._make_human_data(
            iphone_smpl, keypoints_convention_, iphone_image_path_,
            iphone_image_id_, iphone_bbox_xywh_, iphone_keypoints_2d_,
            keypoints2d_mask_, iphone_keypoints_3d_, keypoints3d_mask_)

        # store iphone human data
        file_name = 'humman_iphone.npz'
        out_file = os.path.join(out_path, file_name)
        iphone_human_data.dump(out_file)
