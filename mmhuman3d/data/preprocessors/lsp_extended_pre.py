import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps


def _bbox_expand(bbox_xyxy, scale_factor):
    center = [(bbox_xyxy[0] + bbox_xyxy[2]) / 2,
              (bbox_xyxy[1] + bbox_xyxy[3]) / 2]
    x1 = scale_factor * (bbox_xyxy[0] - center[0]) + center[0]
    y1 = scale_factor * (bbox_xyxy[1] - center[1]) + center[1]
    x2 = scale_factor * (bbox_xyxy[2] - center[0]) + center[0]
    y2 = scale_factor * (bbox_xyxy[3] - center[1]) + center[1]
    return [x1, y1, x2 - x1, y2 - y1]


def lsp_extended_extract(dataset_path, out_path):

    # total dictionary to store all data
    total_dict = {}

    # training mode
    png_path = os.path.join(dataset_path, '*.png')
    imgs = glob.glob(png_path)
    imgs.sort()

    # structs we use
    image_path_, bbox_xywh_, keypoints2d_ = [], [], []

    # annotation files
    annot_file = os.path.join(dataset_path, 'joints.mat')
    keypoints2d = sio.loadmat(annot_file)['joints']

    for i, imgname in enumerate(tqdm(imgs)):
        # image name
        imgname = imgname.split('/')[-1]
        image_path = os.path.join(dataset_path, imgname)
        im = cv2.imread(image_path)
        h, w, _ = im.shape

        # keypoints
        keypoints2d14 = keypoints2d[:, :2, i]
        keypoints2d14 = np.hstack([keypoints2d14, np.ones([14, 1])])

        # bbox
        bbox_xywh = [
            min(keypoints2d14[:, 0]),
            min(keypoints2d14[:, 1]),
            max(keypoints2d14[:, 0]),
            max(keypoints2d14[:, 1])
        ]

        if 0 <= bbox_xywh[0] <= w and 0 <= bbox_xywh[2] <= w and \
                0 <= bbox_xywh[1] <= h and 0 <= bbox_xywh[3] <= h:
            bbox_xywh = _bbox_expand(bbox_xywh, scale_factor=1.2)
        else:
            print(
                'Bbox out of image bounds. Skipping image {}'.format(imgname))
            continue

        # store data
        image_path_.append(image_path)
        bbox_xywh_.append(bbox_xywh)
        keypoints2d_.append(keypoints2d14)

    keypoints2d_ = np.array(keypoints2d_).reshape((-1, 14, 3))
    keypoints2d_, mask = convert_kps(keypoints2d_, 'lsp', 'smplx')
    total_dict['image_path'] = image_path_
    total_dict['bbox_xywh'] = bbox_xywh_
    total_dict['keypoints2d'] = keypoints2d_
    total_dict['mask'] = mask
    total_dict['config'] = 'hr-lspet'

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(out_path, 'hr-lspet_train.npz')
    np.savez_compressed(out_file, **total_dict)
