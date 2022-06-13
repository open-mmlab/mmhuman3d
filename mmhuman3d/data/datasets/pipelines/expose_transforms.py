import math
import random

from typing import Tuple
from unittest import result
import cv2
import numpy as np
import torch
from ..builder import PIPELINES


@PIPELINES.register_module()
class BBoxCenterJitter(object):
    def __init__(self, factor=0.0, dist='normal'):
        super(BBoxCenterJitter, self).__init__()
        self.factor = factor
        self.dist = dist
        assert self.dist in ['normal', 'uniform'], (
            f'Distribution must be normal or uniform, not {self.dist}')

    def __str__(self):
        return f'BBoxCenterJitter({self.factor:0.2f})'

    def __call__(self, results):
        # body model: no process
        if self.factor <= 1e-3:
            return results

        bbox_size = results['scale'][0]

        jitter = bbox_size * self.factor

        if self.dist == 'normal':
            center_jitter = np.random.randn(2) * jitter
        elif self.dist == 'uniform':
            center_jitter = np.random.rand(2) * 2 * jitter - jitter

        center = results['center']
        H, W = results['img_shape']
        new_center = center + center_jitter
        new_center[0] = np.clip(new_center[0], 0, W)
        new_center[1] = np.clip(new_center[1], 0, H)

        results['center'] = new_center
        return results

@PIPELINES.register_module()
class SimulateLowRes(object):
    def __init__(
        self,
        dist: str = 'categorical',
        factor: float = 1.0,
        cat_factors: Tuple[float] = (1.0,),
        factor_min: float = 1.0,
        factor_max: float = 1.0
    ) -> None:
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.dist = dist
        self.cat_factors = cat_factors
        assert dist in ['uniform', 'categorical']

    def __str__(self) -> str:
        if self.dist == 'uniform':
            dist_str = (
                f'{self.dist.title()}: [{self.factor_min}, {self.factor_max}]')
        else:
            dist_str = (
                f'{self.dist.title()}: [{self.cat_factors}]')
        return f'SimulateLowResolution({dist_str})'

    def _sample_low_res(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        '''
        '''
        if self.dist == 'uniform':
            downsample = self.factor_min != self.factor_max
            if not downsample:
                return image
            factor = np.random.rand() * (
                self.factor_max - self.factor_min) + self.factor_min
        elif self.dist == 'categorical':
            if len(self.cat_factors) < 2:
                return image
            idx = np.random.randint(0, len(self.cat_factors))
            factor = self.cat_factors[idx]

        H, W, _ = image.shape
        downsampled_image = cv2.resize(
            image, (int(W // factor), int(H // factor)), cv2.INTER_NEAREST
        )
        resized_image = cv2.resize(
            downsampled_image, (W, H), cv2.INTER_LINEAR_EXACT)
        return resized_image

    def __call__(self, results):
        '''
        '''
        img = results['img']
        img = self._sample_low_res(img)
        results['img'] = img

        return results


def bbox_area(bbox):
    if torch.is_tensor(bbox):
        if bbox is None:
            return 0.0
        xmin, ymin, xmax, ymax = torch.split(bbox.reshape(-1, 4), 1, dim=1)
        return torch.abs((xmax - xmin) * (ymax - ymin)).squeeze(dim=-1)
    else:
        if bbox is None:
            return 0.0
        xmin, ymin, xmax, ymax = np.split(bbox.reshape(-1, 4), 4, axis=1)
        return np.abs((xmax - xmin) * (ymax - ymin))

def keyps_to_bbox(keypoints, conf, min_valid_keypoints=6, scale=1.0):
    valid_keypoints = keypoints[conf > 0]
    if len(valid_keypoints) < min_valid_keypoints:
        return None
    xmin, ymin = np.amin(valid_keypoints, axis=0)
    xmax, ymax = np.amax(valid_keypoints, axis=0)

    width = (xmax - xmin) * scale
    height = (ymax - ymin) * scale

    x_center = 0.5 * (xmax + xmin)
    y_center = 0.5 * (ymax + ymin)
    xmin = x_center - 0.5 * width
    xmax = x_center + 0.5 * width
    ymin = y_center - 0.5 * height
    ymax = y_center + 0.5 * height

    bbox = np.stack([xmin, ymin, xmax, ymax], axis=0).astype(np.float32)
    if bbox_area(bbox) > 0:
        return bbox
    else:
        return None