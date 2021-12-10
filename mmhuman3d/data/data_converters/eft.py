import json
import os
from typing import List

import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from mmhuman3d.utils.transforms import rotmat_to_aa
from .base_converter import BaseModeConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class EftConverter(BaseModeConverter):
    """Eft dataset `Exemplar Fine-Tuning for 3D Human Pose Fitting Towards In-
    the-Wild 3D Human Pose Estimation' 3DV'2021 More details can be found in
    the `paper.

    <https://arxiv.org/pdf/2004.03686.pdf>`__ .

    Args:
        modes (list): 'coco_all', 'coco_part', 'mpii' and/or 'lspet' for
        accepted modes
    """
    ACCEPTED_MODES = ['coco_all', 'coco_part', 'mpii', 'lspet']

    def __init__(self, modes: List = []) -> None:
        super(EftConverter, self).__init__(modes)
        self.json_mapping_dict = {
            'coco_all':
            ['coco_2014_train_fit/COCO2014-All-ver01.json', 'train2014/'],
            'coco_part':
            ['coco_2014_train_fit/COCO2014-Part-ver01.json', 'train2014/'],
            'lspet': ['LSPet_fit/LSPet_ver01.json', ''],
            'mpii': ['MPII_fit/MPII_ver01.json', 'images/']
        }

    @staticmethod
    def center_scale_to_bbox(center: List[float], scale: float) -> List[float]:
        """obtain bbox from center and scale with pixel_std=200."""
        w, h = scale * 200, scale * 200
        x, y = center[0] - w / 2, center[1] - h / 2
        return [x, y, w, h]

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
                keypoints2d_mask, smpl stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()
        image_path_, bbox_xywh_, keypoints2d_ = [], [], []
        smpl = {}
        smpl['betas'] = []
        smpl['body_pose'] = []
        smpl['global_orient'] = []

        if mode in self.json_mapping_dict.keys():
            annot_file = os.path.join(dataset_path,
                                      self.json_mapping_dict[mode][0])
            image_prefix = self.json_mapping_dict[mode][1]
        else:
            raise ValueError('provided dataset is not in eft fittings')

        with open(annot_file, 'r') as f:
            eft_data = json.load(f)
        eft_data_all = eft_data['data']

        for data in tqdm(eft_data_all):
            beta = np.array(data['parm_shape']).reshape(10)
            # 3D rotation matrix for 24 joints (24, 3, 3)
            pose_rotmat = np.array(data['parm_pose'])
            pose_rotmat = rotmat_to_aa(pose_rotmat)

            bbox_scale = data['bbox_scale']
            bbox_center = data['bbox_center']
            bbox_xywh = self.center_scale_to_bbox(bbox_center, bbox_scale)

            # (49, 3) keypoints2d in image space according to SPIN format
            gt_keypoint_2d = data['gt_keypoint_2d']

            image_name = data['imageName']

            smpl['body_pose'].append(pose_rotmat[1:].reshape((23, 3)))
            smpl['global_orient'].append(pose_rotmat[0].reshape(-1, 3))
            smpl['betas'].append(beta)

            # store data
            image_path_.append(image_prefix + image_name)
            bbox_xywh_.append(bbox_xywh)
            keypoints2d_.append(gt_keypoint_2d)

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        smpl['body_pose'] = np.array(smpl['body_pose']).reshape((-1, 23, 3))
        smpl['global_orient'] = np.array(smpl['global_orient']).reshape(
            (-1, 3))
        smpl['betas'] = np.array(smpl['betas']).reshape((-1, 10))
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 49, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'smpl_49', 'human_data')
        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['smpl'] = smpl
        human_data['config'] = 'eft'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        out_file = os.path.join(out_path, 'eft_{}.npz'.format(mode))
        human_data.dump(out_file)
