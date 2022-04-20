"""This script is modified from https://github.com/
isarandi/synthetic-occlusion. 
Original license please see docs/additional_licenses.md.
"""

import sys
import cv2
import random
import joblib
import os.path
import functools
import PIL.Image
import numpy as np
import skimage.data
from loguru import logger
import xml.etree.ElementTree
import matplotlib.pyplot as plt
from ..builder import PIPELINES

from mmhuman3d.core.conventions.keypoints_mapping.smpl import SMPL_49_KEYPOINTS

def main(type='coco'):
    """Demo of how to use the code"""

    # path = 'something/something/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    path = sys.argv[1]

    print('Loading occluders from Pascal VOC dataset...')
    occluders = load_pascal_occluders(path)
    print('Found {} suitable objects'.format(len(occluders)))

    original_im = cv2.resize(skimage.data.astronaut(), (256, 256))
    fig, axarr = plt.subplots(3, 3, figsize=(7, 7))
    for ax in axarr.ravel():
        occluded_im = occlude_with_pascal_objects(original_im, occluders)
        ax.imshow(occluded_im, interpolation="none")
        ax.axis('off')

    fig.tight_layout(h_pad=0)
    plt.savefig('examples.jpg', dpi=150, bbox_inches='tight')


def load_pascal_occluders(occluders_file,pascal_voc_root_path):
    if os.path.isfile(occluders_file):
        return joblib.load(occluders_file)
    else:
        occluders = []
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

        annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
        for annotation_path in annotation_paths:
            xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
            is_segmented = (xml_root.find('segmented').text != '0')

            if not is_segmented:
                continue

            boxes = []
            for i_obj, obj in enumerate(xml_root.findall('object')):
                is_person = (obj.find('name').text == 'person')
                is_difficult = (obj.find('difficult').text != '0')
                is_truncated = (obj.find('truncated').text != '0')
                if not is_person and not is_difficult and not is_truncated:
                    bndbox = obj.find('bndbox')
                    box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                    boxes.append((i_obj, box))

            if not boxes:
                continue

            im_filename = xml_root.find('filename').text
            seg_filename = im_filename.replace('jpg', 'png')

            im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
            seg_path = os.path.join(pascal_voc_root_path, 'SegmentationObject', seg_filename)

            im = np.asarray(PIL.Image.open(im_path))
            labels = np.asarray(PIL.Image.open(seg_path))

            for i_obj, (xmin, ymin, xmax, ymax) in boxes:
                object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8) * 255
                object_image = im[ymin:ymax, xmin:xmax]
                if cv2.countNonZero(object_mask) < 500:
                    # Ignore small objects
                    continue

                # Reduce the opacity of the mask along the border for smoother blending
                eroded = cv2.erode(object_mask, structuring_element)
                object_mask[eroded < object_mask] = 192
                object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)

                # Downscale for efficiency
                object_with_mask = resize_by_factor(object_with_mask, 0.5)
                occluders.append(object_with_mask)

        print('Saving pascal occluders')
        joblib.dump(occluders, './data/occlusion_augmentation/pascal_occluders.pkl')
        return occluders


def load_coco_occluders(path=None):
    occluders = joblib.load(COCO_OCCLUDERS_FILE)
    joint_occ_freq = np.array([len(v) for k, v in occluders['stats'].items()])
    joint_occ_freq = joint_occ_freq / joint_occ_freq.sum()
    occluders['joint_occ_freq'] = joint_occ_freq
    return occluders


def occlude_with_pascal_objects(im, occluders):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""

    result = im.copy()
    width_height = np.asarray([im.shape[1], im.shape[0]])
    im_scale_factor = min(width_height) / 256
    count = np.random.randint(1, 8)

    # logger.debug(f'Number of augmentation objects: {count}')

    for _ in range(count):
        occluder = random.choice(occluders)

        center = np.random.uniform([0, 0], width_height)
        random_scale_factor = np.random.uniform(0.2, 1.0)
        scale_factor = random_scale_factor * im_scale_factor

        # logger.debug(f'occluder size: {occluder.shape}, scale_f: {scale_factor}, img_scale: {im_scale_factor}')

        occluder = resize_by_factor(occluder, scale_factor)

        paste_over(im_src=occluder, im_dst=result, center=center)

    return result


def occlude_with_coco_objects(im, kp2d, occluders, img_size=224, max_n_objects=4):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""
    # kp2d is in normalized coordinates

    result = im.copy()
    # width_height = np.asarray([im.shape[1], im.shape[0]])
    # im_scale_factor = min(width_height) / 256

    # get GT joints
    kp2d = kp2d[25:]
    # denormalize the keypoints
    kp2d[:, :-1] = 0.5 * img_size * (kp2d[:, :-1] + 1)
    joint_names = SMPL_49_KEYPOINTS[25:]

    count = np.random.randint(0, max_n_objects)

    count = min((kp2d[:, 2] > 0.3).shape[0], count)

    # logger.debug(f'Number of augmentation objects: {count}')

    for _ in range(count):
        # select a random joint based on probability being occluded
        j_occ_prob = occluders['joint_occ_freq'].copy()

        # make nonvisible joint probability zero
        # j_occ_prob[kp2d[:, 2] < 0.5] = 0.0
        # print(j_occ_prob)

        nonvis = True
        while nonvis:
            jid = np.random.multinomial(1, j_occ_prob, size=1).argmax()
            nonvis = kp2d[jid, 2] < 0.5

      
        occluder_info = random.choice(occluders['stats'][joint_names[jid]])
        try:
            occluder_obj_id = random.choice(np.argwhere(occluders['obj_class'] == occluder_info[0]))[0]
        except:
            continue

        occluder_obj_mask = occluders['object_with_mask'][occluder_obj_id]

      
        occluder_obj_height = occluder_obj_mask.shape[0]
        scale_factor = 1. / (occluder_obj_mask.shape[0]/img_size) * np.random.uniform(0.05, 0.7)


        occluder_obj_mask = resize_by_factor(occluder_obj_mask, scale_factor)

        paste_over(im_src=occluder_obj_mask, im_dst=result, center=center)

    return result

def paste_over(im_src, im_dst, center):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.

    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).

    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """

    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32) / 255

    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))




@PIPELINES.register_module()
class SyntheticOcclusion:
    """Data augmentation with synthetic occlusion.

    Required keys: 'img'
    Modifies key: 'img'
    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        flip_pairs (list[int]): list of left-right keypoint pairs for flipping
        occ_aug_dataset (str): name of occlusion dataset. Default: pascal
        pascal_voc_root_path (str): the path to pascal voc dataset, 
        which can generate occluders file. 
        occluders_file (str): occluders file. 
    """

    def __init__(self, occ_aug_dataset = 'pascal', pascal_voc_root_path = 'data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012', occluders_file = ''):
        self.occluders = None
        self.occ_aug_dataset = occ_aug_dataset
        if self.occ_aug_dataset == 'pascal':
            self.occluders = load_pascal_occluders(occluders_file = occluders_file,pascal_voc_root_path=pascal_voc_root_path)
        else:
            raise NotImplementedError()
        
    def __call__(self, results):
        """Perform data augmentation with random channel noise."""
        img = results['img']
        # Each channel is multiplied with a number
        # in the area [1-self.noise_factor, 1+self.noise_factor]
        
        if self.occ_aug_dataset == 'pascal':
            img = occlude_with_pascal_objects(img, self.occluders)
        else:
            raise NotImplementedError()
        results['img'] = img
        return results
if __name__ == '__main__':
    main()

