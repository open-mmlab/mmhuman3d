import glob
import os

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class PennActionConverter(BaseConverter):
    """Penn Action dataset `From Actemes to Action: A Strongly-supervised
    Representation for Detailed Action Understanding' ICCV'2012 More details
    can be found in the `paper.

    <https://openaccess.thecvf.com/content_iccv_2013/papers/
        Zhang_From_Actemes_to_2013_ICCV_paper.pdf>`__ .
    """

    @staticmethod
    def load_mat(path: str) -> dict:
        """Filter keys from mat file."""
        mat = loadmat(path)
        del mat['pose'], mat['__header__'], mat['__globals__'], \
            mat['__version__'], mat['train'], mat['action']
        mat['nframes'] = mat['nframes'][0][0]

        return mat

    def convert(self, dataset_path: str, out_path: str) -> dict:
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

        image_path_, bbox_xywh_, keypoints2d_ = [], [], []

        annot_files = sorted(glob.glob(dataset_path + '/labels/' + '*.mat'))

        for fname in tqdm(annot_files):
            vid_dict = self.load_mat(fname)
            imgs = sorted(
                glob.glob(dataset_path + '/frames/' +
                          fname.strip().split('/')[-1].split('.')[0] +
                          '/*.jpg'))
            kp_2d = np.zeros((vid_dict['nframes'], 13, 3))

            kp_2d[:, :, 0] = vid_dict['x']
            kp_2d[:, :, 1] = vid_dict['y']
            kp_2d[:, :, 2] = 1

            counter = 0
            for kp, img_path in zip(kp_2d, imgs):
                counter += 1
                if counter % 10 != 1:
                    continue

                # calculate bbox from keypoints bounds for visible joints
                bbox_xyxy = [
                    min(kp[:, 0]),
                    min(kp[:, 1]),
                    max(kp[:, 0]),
                    max(kp[:, 1])
                ]
                bbox_xywh = self._bbox_expand(bbox_xyxy, scale_factor=1.2)
                # store relative instead of absolute image path
                image_path_.append(img_path.replace(dataset_path + '/', ''))
                bbox_xywh_.append(bbox_xywh)
                keypoints2d_.append(kp)

        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 13, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'penn_action',
                                         'human_data')
        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints2d_
        human_data['config'] = 'penn_action'
        human_data.compress_keypoints_by_mask()

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'penn_action_train.npz')
        human_data.dump(out_file)
