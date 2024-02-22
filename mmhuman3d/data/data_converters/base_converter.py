from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import get_keypoint_idxs_by_part

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
        kp_x = np.array(keypoints[:, 0])
        kp_y = np.array(keypoints[:, 1])

        kp_x = kp_x[kp_x > 0]
        kp_y = kp_y[kp_y > 0]

        if len(kp_x) * len(kp_y) == 0:
            return np.array([0, 0, 0, 0])

        xmin = np.min(kp_x)
        ymin = np.min(kp_y)
        xmax = np.max(kp_x)
        ymax = np.max(kp_y)

        # xmin, ymin, = np.amin(keypoints, axis=0)
        # xmax, ymax = np.amax(keypoints, axis=0)

        # import pdb; pdb.set_trace()

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


    def _keypoints_to_scaled_bbox_bfh(self,
                                    keypoints,
                                    occ=None,
                                    body_scale=1.0,
                                    fh_scale=1.0,
                                    convention='smplx'):
        '''Obtain scaled bbox in xyxy format given keypoints
        Args:
            keypoints (np.ndarray): Keypoints
            scale (float): Bounding Box scale
        Returns:
            bbox_xyxy (np.ndarray): Bounding box in xyxy format
        '''
        bboxs = []

        # supported kps.shape: (1, n, k) or (n, k), k = 2 or 3
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        if keypoints.shape[-1] != 2:
            keypoints = keypoints[:, :2]

        for body_part in ['body', 'head', 'left_hand', 'right_hand']:
            if body_part == 'body':
                scale = body_scale
                kps = keypoints
            else:
                scale = fh_scale
                kp_id = get_keypoint_idxs_by_part(
                    body_part, convention=convention)
                kps = keypoints[kp_id]

            if not occ is None:
                occ_p = occ[kp_id]
                if np.sum(occ_p) / len(kp_id) >= 0.1:
                    conf = 0
                    # print(f'{body_part} occluded, occlusion: {np.sum(occ_p) / len(kp_id)}, skip')
                else:
                    # print(f'{body_part} good, {np.sum(self_occ_p + occ_p) / len(kp_id)}')
                    conf = 1
            else:
                conf = 1
            if body_part == 'body':
                conf = 1

            xmin, ymin = np.amin(kps, axis=0)
            xmax, ymax = np.amax(kps, axis=0)

            width = (xmax - xmin) * scale
            height = (ymax - ymin) * scale

            x_center = 0.5 * (xmax + xmin)
            y_center = 0.5 * (ymax + ymin)
            xmin = x_center - 0.5 * width
            xmax = x_center + 0.5 * width
            ymin = y_center - 0.5 * height
            ymax = y_center + 0.5 * height

            bbox = np.stack([xmin, ymin, xmax, ymax, conf],
                            axis=0).astype(np.float32)
            bboxs.append(bbox)

        return bboxs


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
