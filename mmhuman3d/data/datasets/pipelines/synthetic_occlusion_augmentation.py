"""This script is modified from https://github.com/ isarandi/synthetic-
occlusion.

Original license please see docs/additional_licenses.md.
"""
import os.path
import random

import cv2
import numpy as np

from ..builder import PIPELINES


def load_pascal_occluders(occluders_file):
    """load pascal occluders from the occluder file."""

    if os.path.isfile(occluders_file):
        return np.load(occluders_file, allow_pickle=True)
    else:
        raise NotImplementedError()


def occlude_with_pascal_objects(im, occluders):
    """Returns an augmented version of `im`, containing some occluders from the
    Pascal VOC dataset."""

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

        # logger.debug(f'occluder size: {occluder.shape},
        # scale_f: {scale_factor}, img_scale: {im_scale_factor}')
        occluder = resize_by_factor(occluder, scale_factor)

        paste_over(im_src=occluder, im_dst=result, center=center)

    return result


def paste_over(im_src, im_dst, center):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha
    blending, in place.

    Locations outside the bounds of `im_dst`
    are handled as expected (only a part or none of `im_src` becomes visible).

    Args:
        im_src: The RGBA image to be pasted onto `im_dst`.
                Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src`
                controlling the alpha blending at each pixel.
                Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where
                the center of `im_src` should be placed.
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
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up
    and area interp for downscaling."""
    new_size = tuple(
        np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    """list the file paths."""
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

    def __init__(self, occluders_file='', occluders=None):
        self.occluders = None
        if occluders is not None:
            self.occluders = occluders

        else:
            self.occluders = load_pascal_occluders(
                occluders_file=occluders_file, )

    def __call__(self, results):
        """Perform data augmentation with random channel noise."""
        img = results['img']

        img = occlude_with_pascal_objects(img, self.occluders)

        results['img'] = img
        return results
