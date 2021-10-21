import glob
import os

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from .base_converter import BaseConverter
from .builder import DATA_CONVERTERS


@DATA_CONVERTERS.register_module()
class PennActionConverter(BaseConverter):

    @staticmethod
    def load_mat(path):
        mat = loadmat(path)
        del mat['pose'], mat['__header__'], mat['__globals__'], \
            mat['__version__'], mat['train'], mat['action']
        mat['nframes'] = mat['nframes'][0][0]

        return mat

    def convert(self, dataset_path, out_path):

        total_dict = {}

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

                image_path_.append(img_path)
                bbox_xywh_.append(bbox_xywh)
                keypoints2d_.append(kp)

        keypoints2d_ = np.array(keypoints2d_).reshape((-1, 13, 3))
        keypoints2d_, mask = convert_kps(keypoints2d_, 'penn_action',
                                         'human_data_1.0')
        total_dict['image_path'] = image_path_
        total_dict['bbox_xywh'] = bbox_xywh_
        total_dict['keypoints2d'] = keypoints2d_
        total_dict['mask'] = mask
        total_dict['config'] = 'penn_action'

        # store the data struct
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, 'penn_action_train.npz')
        np.savez_compressed(out_file, **total_dict)
