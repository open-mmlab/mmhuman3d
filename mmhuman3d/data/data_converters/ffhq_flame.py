import os

import numpy as np
import torch

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter


@DATA_CONVERTERS.register_module()
class FFHQFlameConverter(BaseModeConverter):
    """Flickr-Faces-High-Quality More details can be found in the repository.

    https://github.com/NVlabs/ffhq-dataset

    The regressed flame parameters are produced by running
    RingNet on FFHQ and then fitting to FAN 2D landmarks by flame-fitting.
    More details of RingNet can be found on the website.
    https://ringnet.is.tue.mpg.de/

    More details of flame-fitting can be found in the repository.
    https://github.com/HavenFeng/photometric_optimization

    Args:
        modes (list): 'val' and/or 'train' for accepted modes
    """
    NUM_BETAS = 100
    NUM_EXPRESSION = 50
    ACCEPTED_MODES = ['val', 'train']

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
        # structs we use
        image_path_, bbox_xywh_ = [], []
        smplx = {}
        smplx['jaw_pose'] = []
        smplx['global_orient'] = []
        smplx['betas'] = []
        smplx['expression'] = []

        data = np.load(os.path.join(dataset_path, 'ffhq_annotations.npz'))
        global_orient = data['global_pose']
        jaw_pose = data['jaw_pose']
        betas = data['betas']
        expression = data['expression']
        smplx['betas'] = betas
        smplx['expression'] = expression
        smplx['jaw_pose'] = jaw_pose
        smplx['global_orient'] = global_orient

        keypoints2d_flame = data['keypoints2d']
        keypoints2d_flame = np.concatenate(
            [keypoints2d_flame,
             np.ones([keypoints2d_flame.shape[0], 73, 1])],
            axis=-1)
        keypoints2d, keypoints2d_mask = \
            convert_kps(keypoints2d_flame, src='flame', dst='human_data')
        human_data['keypoints2d'] = keypoints2d
        human_data['keypoints2d_mask'] = keypoints2d_mask

        img_fnames = data['img_fnames']
        image_path_ = np.array([
            os.path.join('ffhq_global_images_1024', img_fname)
            for img_fname in img_fnames
        ])
        bbox_xywh_ = torch.from_numpy(np.array([0, 0, 1024, 1024])).expand(
            len(image_path_), 4)
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        human_data['image_path'] = image_path_.tolist()
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smplx'] = smplx
        human_data['config'] = 'ffhq'

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        if mode == 'train':
            human_data = human_data.get_slice(0, int(0.8 * len(image_path_)))
        else:
            human_data = human_data.get_slice(
                int(0.8 * len(image_path_)), len(image_path_))

        file_name = 'ffhq_flame_{}.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
