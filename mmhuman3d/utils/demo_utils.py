import os
import warnings
from pathlib import Path

import mmcv
import numpy as np

from mmhuman3d.core.filter import build_filter
from mmhuman3d.utils.path_utils import check_input_path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

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


def xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

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


def box2cs(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.

    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch

    Args:
    xyxy (list, tuple or numpy.ndarray): bbox in format (xmin, ymin,
     xmax, ymax). If numpy.ndarray is provided, we expect multiple bounding
     boxes with shape `(N, 4)`.
    width (int or float): Boundary width.
    height (int or float): Boundary height.

    Returns:
    xyxy (list, tuple or numpy.ndarray): clipped bbox in format (xmin, ymin,
     xmax, ymax) and input type
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std],
                     dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def convert_crop_cam_to_orig_img(cam: np.ndarray, bbox: np.ndarray,
                                 img_width: int, img_height: int):
    """This function is modified from [VIBE](https://github.com/
    mkocabas/VIBE/blob/master/lib/utils/demo_utils.py#L242-L259). Original
    license please see docs/additional_licenses.md.

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
        bbox (np.ndarray): (frame, num_person, 4) or (frame, 4)
        img_width (int): image width of training data.
        img_height (int): image height of training data.
        bbox_scale_factor (float): scale factor for expanding the bbox.
        bbox_format (Literal['xyxy', 'xywh'] ): 'xyxy' means the left-up point
            and right-bottomn point of the bbox.
            'xywh' means the left-up point and the width and height of the
            bbox.
    Returns:
        np.ndarray: (frame, num_person, 3, 3) or  (frame, 3, 3)
    """
    assert bbox_format in ['xyxy', 'xywh']

    if bbox_format == 'xyxy':
        bboxes = xyxy2xywh(bboxes)

    center_x = bboxes[..., 0] + bboxes[..., 2] / 2.0
    center_y = bboxes[..., 1] + bboxes[..., 3] / 2.0

    W = np.max(bboxes[..., 2:], axis=-1) * bbox_scale_factor

    num_frame = bboxes.shape[0]
    if bboxes.ndim == 3:
        num_person = bboxes.shape[1]
        Ks = np.zeros((num_frame, num_person, 3, 3))
    elif bboxes.ndim == 2:
        Ks = np.zeros((num_frame, 3, 3))

    Ks[..., 0, 0] = W / img_width
    Ks[..., 1, 1] = W / img_height
    Ks[..., 0, 2] = center_x - W / 2.0
    Ks[..., 1, 2] = center_y - W / 2.0
    Ks[..., 2, 2] = 1
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
        bbox = xyxy2xywh(bbox)
    return bbox


def conver_verts_to_cam_coord(verts, pred_cams, bboxes_xy, focal_length=5000.):
    """Convert vertices from the world coordinate to camera coordinate.

    Args:
        verts ([np.ndarray]): The vertices in the world coordinate.
            The shape is (frame,num_person,6890,3) or (frame,6890,3).
        pred_cams ([np.ndarray]): Camera parameters estimated by HMR or SPIN.
            The shape is (frame,num_person,3) or (frame,6890,3).
        bboxes_xy ([np.ndarray]): (frame, num_person, 4|5) or (frame, 4|5)
        focal_length ([float],optional): Defined same as your training.

    Returns:
        np.ndarray: The vertices in the camera coordinate.
            The shape is (frame,num_person,6890,3) or (frame,6890,3).
        np.ndarray: The instrinsic parameters of the pred_cam.
            The shape is (num_frame, 3, 3).
    """
    K0 = get_default_hmr_intrinsic(
        focal_length=focal_length, det_height=224, det_width=224)
    K1 = convert_bbox_to_intrinsic(bboxes_xy, bbox_format='xyxy')
    # K1K0(RX+T)-> K0(K0_inv K1K0)
    Ks = np.linalg.inv(K0) @ K1 @ K0
    # convert vertices from world to camera
    cam_trans = np.concatenate([
        pred_cams[..., [1]], pred_cams[..., [2]], 2 * focal_length /
        (224 * pred_cams[..., [0]] + 1e-9)
    ], -1)
    verts = verts + cam_trans[..., None, :]
    if verts.ndim == 4:
        verts = np.einsum('fnij,fnkj->fnki', Ks, verts)
    elif verts.ndim == 3:
        verts = np.einsum('fij,fkj->fki', Ks, verts)
    return verts, K0


def smooth_process(x, smooth_type='savgol'):
    """Smooth the array with the specified smoothing type.

    Args:
        x (np.ndarray): Shape should be (frame,num_person,K,C)
            or (frame,K,C).
        smooth_type (str, optional): Smooth type.
            choose in ['oneeuro', 'gaus1d', 'savgol'].
            Defaults to 'savgol'.
    Raises:
        ValueError: check the input smoothing type.

    Returns:
        np.ndarray: Smoothed data. The shape should be
            (frame,num_person,K,C) or (frame,K,C).
    """
    x = x.copy()

    assert x.ndim == 3 or x.ndim == 4

    smooth_func = build_filter(dict(type=smooth_type))

    if x.ndim == 4:
        for i in range(x.shape[1]):
            x[:, i] = smooth_func(x[:, i])
    elif x.ndim == 3:
        x = smooth_func(x)

    return x


def process_mmtracking_results(mmtracking_results):
    """Process mmtracking results.

    Args:
        mmtracking_results ([list]): mmtracking_results.

    Returns:
        list: a list of tracked bounding boxes
    """
    person_results = []
    # 'track_results' is changed to 'track_bboxes'
    # in https://github.com/open-mmlab/mmtracking/pull/300
    if 'track_bboxes' in mmtracking_results:
        tracking_results = mmtracking_results['track_bboxes'][0]
    elif 'track_results' in mmtracking_results:
        tracking_results = mmtracking_results['track_results'][0]

    for track in tracking_results:
        person = {}
        person['track_id'] = int(track[0])
        person['bbox'] = track[1:]
        person_results.append(person)
    person_results = sorted(person_results, key=lambda x: x.get('track_id', 0))
    return person_results


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    Args:
        mmdet_results (list|tuple): mmdet results.
        cat_id (int): category id (default: 1 for human)

    Returns:
        person_results (list): a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def prepare_frames(image_path=None, video_path=None):
    """Prepare frames from input image path or video path.

    Args:
        image_path (str, optional): Defaults to None.
        video_path (str, optional): Defaults to None.

    Raises:
        ValueError: check the input path.

    Returns:
        (Union[np.ndarray, object]): prepared frames
    """
    if (image_path is not None) and (video_path is not None):
        warnings.warn('Redundant input, will ignore video')
    # prepare input
    if image_path is not None:
        file_list = []
        if Path(image_path).is_file():
            check_input_path(
                input_path=image_path,
                path_type='file',
                allowed_suffix=['.png', '.jpg'])
            file_list = [image_path]
        elif Path(image_path).is_dir():
            file_list = [
                os.path.join(image_path, fn) for fn in os.listdir(image_path)
                if fn.lower().endswith(('.png', '.jpg'))
            ]
        else:
            raise ValueError('Image path should be an image or image folder.'
                             f' Got invalid image path: {image_path}')
        file_list.sort()
        img_list = [mmcv.imread(img_path) for img_path in file_list]
        assert len(img_list), f'Failed to load image from {image_path}'
    elif video_path is not None:
        check_input_path(
            input_path=video_path,
            path_type='file',
            allowed_suffix=['.mp4', '.flv'])
        video = mmcv.VideoReader(video_path)
        assert video.opened, f'Failed to load video file {video_path}'
    else:
        raise ValueError('No image path or video path provided.')

    frames_iter = img_list if image_path is not None else video

    return frames_iter
