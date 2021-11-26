import numpy as np

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _xyxy2xywh(bbox_xyxy):
    """T ransform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[..., 2] = bbox_xywh[..., 2] - bbox_xywh[..., 0] + 1
    bbox_xywh[..., 3] = bbox_xywh[..., 3] - bbox_xywh[..., 1] + 1

    return bbox_xywh


def _xywh2xyxy(bbox_xywh):
    """T ransform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, width, height, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[..., 2] = bbox_xyxy[..., 2] + bbox_xyxy[..., 0] - 1
    bbox_xyxy[..., 3] = bbox_xyxy[..., 3] + bbox_xyxy[..., 1] - 1

    return bbox_xyxy


def convert_crop_cam_to_orig_img(cam: np.ndarray, bbox: np.ndarray,
                                 img_width: int, img_height: int):
    """Copied from VIBE.

    Args:
        cam (np.ndarray): cam (ndarray, shape=(frame, 3)):
        weak perspective camera in cropped img coordinates
        bbox (np.ndarray): bbox coordinates (c_x, c_y, h)
        img_width (int): original image width
        img_height (int): original image height

    Returns:
        orig_cam: shape = (frame, 4)
    """
    cx, cy, h = bbox[..., 0], bbox[..., 1], bbox[..., 2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[..., 0] * (1. / (img_width / h))
    sy = cam[..., 0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[..., 1]
    ty = ((cy - hh) / hh / sy) + cam[..., 2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def convert_bbox_to_intrinsic(bboxes: np.ndarray,
                              img_width: int = 224,
                              img_height: int = 224,
                              bbox_scale_factor: float = 1.25,
                              bbox_format: Literal['xyxy', 'xywh'] = 'xyxy'):
    """Convert bbox to intrinsic parameters.

    Args:
        bbox (np.ndarray): (frame, num_person, 4)
        img_width (int): image width of training data.
        img_height (int): image height of training data.
        bbox_scale_factor (float): scale factor for expanding the bbox.
        bbox_format (Literal['xyxy', 'xywh'] ): 'xyxy' means the left-up point
            and right-bottomn point of the bbox.
            'xywh' means the left-up point and the width and height of the
            bbox.
    Returns:
        np.ndarray: (frame, num_person, 3, 3)
    """
    assert bbox_format in ['xyxy', 'xywh']

    if bbox_format == 'xyxy':
        bboxes = _xyxy2xywh(bboxes)

    center_x = bboxes[..., 0] + bboxes[..., 2] / 2.0
    center_y = bboxes[..., 1] + bboxes[..., 3] / 2.0

    W = np.max(bboxes[..., 2:], axis=-1) * bbox_scale_factor

    frame_num = bboxes.shape[0]
    person_num = bboxes.shape[1]

    Ks = np.zeros((frame_num, person_num, 3, 3))

    Ks[:, :, 0, 0] = W / img_width
    Ks[:, :, 1, 1] = W / img_height
    Ks[:, :, 0, 2] = center_x - W / 2.0
    Ks[:, :, 1, 2] = center_y - W / 2.0
    Ks[:, :, 2, 2] = 1
    return Ks


def get_default_hmr_intrinsic(num_frame=1,
                              focal_length=1000,
                              det_width=224,
                              det_height=224) -> np.ndarray:
    """Get default hmr instrinsic, defined by how you trained.

    Args:
        num_frame (int, optional): num of frames. Defaults to 1.
        focal_length (int, optional): defined same as your training.
            Defaults to 1000.
        det_width (int, optional): the size you used to detect.
            Defaults to 224.
        det_height (int, optional): the size you used to detect.
            Defaults to 224.

    Returns:
        np.ndarray: shape of (N, 3, 3)
    """
    K = np.zeros((num_frame, 3, 3))
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 0, 2] = det_width / 2
    K[:, 1, 2] = det_height / 2
    K[:, 2, 2] = 1
    return K


def convert_kp2d_to_bbox(
        kp2d: np.ndarray,
        bbox_format: Literal['xyxy', 'xywh'] = 'xyxy') -> np.ndarray:
    """Convert kp2d to bbox.

    Args:
        kp2d (np.ndarray):  shape should be (num_frame, num_points, 2/3)
            or (num_frame, num_person, num_points, 2/3).
        bbox_format (Literal['xyxy', 'xywh'], optional): Defaults to 'xyxy'.

    Returns:
        np.ndarray: shape will be (num_frame, num_person, 4)
    """
    assert bbox_format in ['xyxy', 'xywh']
    if kp2d.ndim == 2:
        kp2d = kp2d[None, None]
    elif kp2d.ndim == 3:
        kp2d = kp2d[:, None]
    num_frame, num_person, _, _ = kp2d.shape
    x1 = np.max(kp2d[..., 0], axis=-2)
    y1 = np.max(kp2d[..., 1], axis=-2)
    x2 = np.max(kp2d[..., 2], axis=-2)
    y2 = np.max(kp2d[..., 3], axis=-2)
    bbox = np.concatenate([x1, y1, x2, y2], axis=-1)
    assert bbox.shape == (num_frame, num_person, 4)
    if bbox_format == 'xywh':
        bbox = _xyxy2xywh(bbox)
    return bbox
