import csv
import os

import mmcv
import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter


@DATA_CONVERTERS.register_module()
class StirlingConverter(BaseModeConverter):
    """Stirling/ESRC 3D More details can be found on the website.

    http://pics.psych.stir.ac.uk/ESRC/index.htm
    """
    ACCEPTED_MODES = ['test']

    def convert_by_mode(self, dataset_path: str, out_path: str, mode: str,
                        img_quality: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes
            img_quality (str): HQ/LQ

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh,  meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()
        # structs we use
        image_path_, bbox_xywh_ = [], []
        keypoints3d_ = []
        vertices_ = []

        raw_img_path = os.path.join(dataset_path, 'Subset_2D_FG2018',
                                    img_quality)

        for fname in sorted(os.listdir(raw_img_path)):

            gender = fname[0]
            obj_folder = os.path.join(dataset_path, f'{gender}_3D_N')
            obj_file = os.path.join(obj_folder,
                                    fname.split('_')[0].lower() + '_N.obj')
            if not os.path.exists(obj_file):
                obj_file = os.path.join(obj_folder,
                                        fname.split('_')[0].upper() + '_N.obj')
            annot_folder = os.path.join(dataset_path, 'annotations',
                                        f'{gender}_3D_N')
            annot_file = os.path.join(annot_folder,
                                      fname.split('_')[0].lower() + '_N.lnd')
            if not os.path.exists(annot_file):
                annot_file = os.path.join(
                    annot_folder,
                    fname.split('_')[0].upper() + '_N.lnd')

            if not os.path.exists(annot_file) or not os.path.exists(
                    obj_file):  # file lost
                continue

            # store data
            image_path = os.path.join('Subset_2D_FG2018', img_quality, fname)
            image_path_.append(image_path)
            with open(obj_file) as file:
                vertices = []
                while 1:
                    line = file.readline()
                    if not line:
                        break
                    strs = line.split(' ')
                    if strs[0] == 'v':
                        vertices.append(
                            (float(strs[1]), float(strs[2]), float(strs[3])))
                    if strs[0] == 'vt':
                        break

            vertices = np.array(vertices)
            vertices_.append(vertices)

            img = mmcv.imread(os.path.join(dataset_path, image_path))
            H, W, _ = img.shape
            bbox_xywh_.append(np.array([0, 0, W - 1, H - 1], dtype=np.float32))
            keypoints3d = []
            with open(annot_file, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                for row in reader:
                    if row:
                        keypoints3d.append(
                            np.array([row[1], row[2], row[3]],
                                     dtype=np.float32))
            keypoints3d_.append(keypoints3d)

        keypoints3d_ = np.array(keypoints3d_)
        keypoints3d_ = np.concatenate(
            [keypoints3d_,
             np.ones([keypoints3d_.shape[0], 7, 1])], axis=2)
        vertices_ = np.array(vertices_)

        keypoints3d, keypoints3d_mask = \
            convert_kps(keypoints3d_, src='face3d', dst='human_data')

        human_data['keypoints3d'] = keypoints3d
        human_data['keypoints3d_mask'] = keypoints3d_mask
        human_data['vertices'] = vertices_

        bbox_xywh_ = np.array(bbox_xywh_)
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['config'] = 'stirling'

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'stirling_ESRC3D_{}.npz'.format(img_quality)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
