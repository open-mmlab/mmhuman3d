from abc import ABCMeta, abstractmethod
from typing import List


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
        """Obtain bbox in xywh format given bbox in xyxy format
        Args:
            bbox_xyxy (List[float]): Bounding box in xyxy format
            scale_factor (float): Scale factor to expand bbox

        Returns:
            bbox_xywh (List[float]): Bounding box in xywh format
        """
        center = [(bbox_xyxy[0] + bbox_xyxy[2]) / 2,
                  (bbox_xyxy[1] + bbox_xyxy[3]) / 2]
        x1 = scale_factor * (bbox_xyxy[0] - center[0]) + center[0]
        y1 = scale_factor * (bbox_xyxy[1] - center[1]) + center[1]
        x2 = scale_factor * (bbox_xyxy[2] - center[0]) + center[0]
        y2 = scale_factor * (bbox_xyxy[3] - center[1]) + center[1]
        return [x1, y1, x2 - x1, y2 - y1]


class BaseModeConverter(BaseConverter):
    """Convert datasets by mode.

    Args:
        prefix (str): the prefix of data path
        modes (list): the modes of data for converter
    """

    def convert(self, dataset_path: str, out_path: str):
        for mode in self.modes:
            self.convert_by_mode(dataset_path, out_path, mode)

    @abstractmethod
    def convert_by_mode(self):
        pass
