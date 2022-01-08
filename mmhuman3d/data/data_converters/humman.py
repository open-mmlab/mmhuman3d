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
from mmhuman3d.models.builder import build_body_model
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures import SMCReader


@DATA_CONVERTERS.register_module()
class HuMManConverter(BaseConverter):
    """ A mysterious dataset that will be announced soon """

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

        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(-1, 3)
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape(-1, 23, 3)
        smpl['betas'] = np.array(smpl['betas']).reshape(-1, 10)
        smpl['transl'] = np.array(smpl['transl']).reshape(-1, 3)

        human_data['smpl'] = smpl

        keypoints2d = np.array(keypoints_2d).reshape(-1, 49, 2)
        keypoints2d = np.concatenate(
            [keypoints2d, np.ones([keypoints2d.shape[0], 49, 1])], axis=-1)
        keypoints2d, keypoints2d_mask = \
            convert_kps(keypoints2d, mask=keypoints2d_mask, src=keypoints_convention, dst='human_data')
        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask

        keypoints3d = np.array(keypoints_3d).reshape(-1, 49, 3)
        keypoints3d = np.concatenate(
            [keypoints3d, np.ones([keypoints3d.shape[0], 49, 1])], axis=-1)
        keypoints3d, keypoints3d_mask = \
            convert_kps(keypoints3d, mask=keypoints3d_mask, src=keypoints_convention, dst='human_data')
        human_data['keypoints3d'] = keypoints3d
        human_data['keypoints3d_mask'] = keypoints3d_mask

        human_data['image_path'] = image_path
        human_data['image_id'] = image_id

        bbox_xywh = np.array(bbox_xywh).reshape((-1, 4))
        bbox_xywh = np.hstack([bbox_xywh, np.ones([bbox_xywh.shape[0], 1])])
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
        kinect_image_path_, kinect_image_id_, kinect_bbox_xywh_, kinect_keypoints_2d_, kinect_keypoints_3d_ = \
            [], [], [], [], []
        iphone_image_path_, iphone_image_id_, iphone_bbox_xywh_, iphone_keypoints_2d_, iphone_keypoints_3d_ = \
            [], [], [], [], []
        keypoints_convention_, keypoints2d_mask_, keypoints3d_mask_ = None, None, None

        ann_paths = sorted(
            glob.glob(os.path.join(dataset_path, '*.smc')))

        for ann_path in tqdm(ann_paths):

            smc_reader = SMCReader(ann_path)

            num_kinect = smc_reader.get_num_kinect()
            num_iphone = smc_reader.get_num_iphone()

            device_list = [('Kinect', i) for i in range(num_kinect)] + \
                [('iPhone', i) for i in range(num_iphone)]
            assert len(device_list) == num_kinect + num_iphone

            for device, device_id in device_list:
                assert device in {
                    'Kinect', 'iPhone'
                }, f'Undefined device: {device}, should be "Kinect" or "iPhone"'
                assert device_id >= 0, f'Negative device id: {device_id}'

                keypoints_convention = smc_reader.get_keypoints_convention()
                if keypoints_convention_ is None:
                    keypoint_convention_ = keypoints_convention
                assert keypoint_convention_ == keypoints_convention

                # get keypoints2d (all frames)
                keypoints2d, keypoints2d_mask = self._get_keypoints2d(
                    device, device_id)
                if keypoints2d_mask_ is None:
                    keypoints2d_mask_ = keypoints2d_mask
                assert keypoints2d_mask_ == keypoints2d_mask

                # compute bbox from keypoints2d
                xs, ys = keypoints2d[:, :, 0], keypoints2d[:, :, 1]
                xmins, xmaxs = np.min(xs, axis=1), np.max(xs, axis=1)
                ymins, ymaxs = np.min(ys, axis=1), np.max(ys, axis=1)

                bbox_xywhs = []
                for xmin, xmax, ymin, ymax in zip(xmins, xmaxs, ymins, ymaxs):
                    bbox_xyxy = [xmin, ymin, xmax, ymax]
                    bbox_xywh = self._bbox_expand(
                        bbox_xyxy, scale_factor=1.2)
                    bbox_xywhs.append(bbox_xywh)

                # get keypoints3d (all frames)
                keypoints3d, keypoints3d_mask = smc_reader.get_keypoints3d(
                    device, device_id)
                if keypoints3d_mask_ is None:
                    keypoints3d_mask_ = keypoints3d_mask
                assert keypoints3d_mask_ == keypoints3d_mask

                # get smpl (all frames)
                smpl_dict = smc_reader.get_smpl()

                # get image paths (smc paths)
                image_path = os.path.basename(ann_path)

                num_frames = smc_reader.get_kinect_num_frames()
                for ann in [keypoints2d, keypoints2d_mask, keypoints3d, keypoints3d_mask]:
                    assert len(ann) == num_frames

                # build image idx
                image_ids = []
                for frame_id in range(num_frames):
                    image_id = (device, device_id, frame_id)
                    image_ids.append(image_id)

                # save data in structs
                if device == 'Kinect':
                    kinect_image_path_.append(image_path)
                    kinect_image_id_.append(image_id)
                    kinect_bbox_xywh_.extend(bbox_xywhs)
                    kinect_keypoints_2d_.append(keypoints2d)
                    kinect_keypoints_3d_.append(keypoints3d)
                    kinect_smpl['body_pose'].append(smpl_dict['body_pose'])
                    kinect_smpl['global_orient'].append(smpl_dict['global_orient'])
                    kinect_smpl['betas'].append(smpl_dict['betas'])
                    kinect_smpl['transl'].append(smpl_dict['transl'])

                else:
                    iphone_image_path_.append(image_path)
                    iphone_image_id_.append(image_id)
                    iphone_bbox_xywh_.extend(bbox_xywhs)
                    iphone_keypoints_2d_.append(keypoints2d)
                    iphone_keypoints_3d_.append(keypoints3d)
                    iphone_smpl['body_pose'].append(smpl_dict['body_pose'])
                    iphone_smpl['global_orient'].append(smpl_dict['global_orient'])
                    iphone_smpl['betas'].append(smpl_dict['betas'])
                    iphone_smpl['transl'].append(smpl_dict['transl'])

            if not os.path.isdir(out_path):
                os.makedirs(out_path)

            # make kinect human data
            kinect_human_data = self._make_human_data(
                kinect_smpl,
                keypoints_convention_,
                kinect_image_path_,
                kinect_image_id_,
                kinect_bbox_xywh_,
                kinect_keypoints_2d_,
                keypoints2d_mask_,
                kinect_keypoints_3d_,
                keypoints3d_mask_
            )

            # store kinect human data
            file_name = 'humman_kinect.npz'
            out_file = os.path.join(out_path, file_name)
            kinect_human_data.dump(out_file)

            # make iphone human data
            iphone_human_data = self._make_human_data(
                iphone_smpl,
                keypoints_convention_,
                iphone_image_path_,
                iphone_image_id_,
                iphone_bbox_xywh_,
                iphone_keypoints_2d_,
                keypoints2d_mask_,
                iphone_keypoints_3d_,
                keypoints3d_mask_
            )

            # store iphone human data
            file_name = 'humman_iphone.npz'
            out_file = os.path.join(out_path, file_name)
            iphone_human_data.dump(out_file)
