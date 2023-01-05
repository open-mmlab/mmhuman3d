from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class BaseConverter(metaclass=ABCMeta):
    """Base dataset.

    Args:
        prefix (str): the prefix of data path
        modes (list): the modes of data for converter
    """

    ACCEPTED_MODES = None

    def __init__(self, modes=[]):
        self.modes = modes

        for mode in self.modes:
            if mode not in self.ACCEPTED_MODES:
                raise ValueError(f'Input mode not in {self.ACCEPTED_MODES}')

    @abstractmethod
    def convert(self):
        pass

    @staticmethod
    def _bbox_expand(bbox_xyxy: List[float],
                     scale_factor: float) -> List[float]:
        """Expand bbox in xyxy format by scale factor
        Args:
            bbox_xyxy (List[float]): Bounding box in xyxy format
            scale_factor (float): Scale factor to expand bbox

        Returns:
            bbox_xyxy (List[float]): Expanded bounding box in xyxy format
        """
        center = [(bbox_xyxy[0] + bbox_xyxy[2]) / 2,
                  (bbox_xyxy[1] + bbox_xyxy[3]) / 2]
        x1 = scale_factor * (bbox_xyxy[0] - center[0]) + center[0]
        y1 = scale_factor * (bbox_xyxy[1] - center[1]) + center[1]
        x2 = scale_factor * (bbox_xyxy[2] - center[0]) + center[0]
        y2 = scale_factor * (bbox_xyxy[3] - center[1]) + center[1]
        return [x1, y1, x2, y2]

    @staticmethod
    def _xyxy2xywh(bbox_xyxy: List[float]) -> List[float]:
        """Obtain bbox in xywh format given bbox in xyxy format
        Args:
            bbox_xyxy (List[float]): Bounding box in xyxy format

        Returns:
            bbox_xywh (List[float]): Bounding box in xywh format
        """
        x1, y1, x2, y2 = bbox_xyxy
        return [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def _keypoints_to_scaled_bbox(keypoints, scale=1.0):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale

        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format

        '''
        xmin, ymin = np.amin(keypoints, axis=0)
        xmax, ymax = np.amax(keypoints, axis=0)

        width = (xmax - xmin) * scale
        height = (ymax - ymin) * scale

        x_center = 0.5 * (xmax + xmin)
        y_center = 0.5 * (ymax + ymin)
        xmin = x_center - 0.5 * width
        xmax = x_center + 0.5 * width
        ymin = y_center - 0.5 * height
        ymax = y_center + 0.5 * height

        bbox = np.stack([xmin, ymin, xmax, ymax], axis=0).astype(np.float32)
        return bbox


class BaseModeConverter(BaseConverter):
    """Convert datasets by mode.

    Args:
        prefix (str): the prefix of data path
        modes (list): the modes of data for converter
    """

    def convert(self, dataset_path: str, out_path: str, *args, **kwargs):
        for mode in self.modes:
            self.convert_by_mode(dataset_path, out_path, mode, *args, **kwargs)

    @abstractmethod
    def convert_by_mode(self):
        pass
