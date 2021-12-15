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
    bbox_xywh[..., 2] = bbox_xywh[..., 2] - bbox_xywh[..., 0]
    bbox_xywh[..., 3] = bbox_xywh[..., 3] - bbox_xywh[..., 1]

    return bbox_xywh


def xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (np.ndarray): Bounding boxes (with scores), shaped
        (frame, 4) or (n, 5). (left, top, width, height, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[..., 2] = bbox_xyxy[..., 2] + bbox_xyxy[..., 0] - 1
    bbox_xyxy[..., 3] = bbox_xyxy[..., 3] + bbox_xyxy[..., 1] - 1

    return bbox_xyxy


def box2cs(x, y, w, h, aspect_ratio=1.0, bbox_scale_factor=1.25):
    """Convert xywh coordinates to center and scale.

    Args:
    x (Union[numpy.ndarray,float]): the x coordinate of the bbox_xywh.
        When the type is `numpy.ndarray`, the shape can be
        (frame, num_person) or (frame,)
    y (Union[numpy.ndarray,float]): the y coordinate of the bbox_xywh
    w (Union[numpy.ndarray,float]): the width of the bbox_xywh
    h (Union[numpy.ndarray,float]): the height of the bbox_xywh
    aspect_ratio (int, optional): Defaults to 1.0
    bbox_scale_factor (float, optional): Defaults to 1.25
    Returns:
        numpy.ndarray: center of the bbox
        numpy.ndarray: the scale of the bbox w & h
    """
    pixel_std = 1
    center = np.stack([x + w * 0.5, y + h * 0.5], -1)

    mask_h = w > aspect_ratio * h
    mask_w = ~mask_h
    if isinstance(x, np.ndarray):
        h[mask_h] = w[mask_h] / aspect_ratio
        w[mask_w] = h[mask_w] * aspect_ratio
    else:
        if mask_h:
            h = w / aspect_ratio
        if mask_w:
            w = h * aspect_ratio

    scale = np.stack([w * 1.0 / pixel_std, h * 1.0 / pixel_std], -1)
    scale = scale * bbox_scale_factor

    return center, scale


def xywh2cs(bbox_xywh, aspect_ratio=1, bbox_scale_factor=1.25):
    """Convert bbox_xywh coordinates to center and scale.

    Args:
        bbox_xywh (numpy.ndarry): Bounding boxes, shaped (n, 4)
        (left, top, width, height, [score])
        aspect_ratio (int, optional): Defaults to 1.0
        bbox_scale_factor (float, optional): Defaults to 1.25

    Returns:
        numpy.ndarray: Bounding boxes, shaped (n, 4)
        (center_x, center_y, scale_x, scale_y)
    """
    bbox_xywh = bbox_xywh[..., :4].copy()
    x, y, w, h = [x[..., -1] for x in np.split(bbox_xywh, 4, -1)]
    center, scale = box2cs(
        x,
        y,
        w,
        h,
        aspect_ratio=aspect_ratio,
        bbox_scale_factor=bbox_scale_factor)
    bbox_cs = np.concatenate([center, scale], axis=-1)
    return bbox_cs


def convert_crop_cam_to_orig_img(cam: np.ndarray,
                                 bbox: np.ndarray,
                                 img_width: int,
                                 img_height: int,
                                 aspect_ratio: float = 1.0,
                                 bbox_scale_factor: float = 1.25,
                                 bbox_format: Literal['xyxy', 'xywh',
                                                      'cs'] = 'xyxy'):
    """This function is modified from [VIBE](https://github.com/
    mkocabas/VIBE/blob/master/lib/utils/demo_utils.py#L242-L259). Original
    license please see docs/additional_licenses.md.

    Args:
        cam (np.ndarray): cam (ndarray, shape=(frame, 3) or
        (frame,num_person, 3)):
        weak perspective camera in cropped img coordinates
        bbox (np.ndarray): bbox coordinates
        img_width (int): original image width
        img_height (int): original image height
        aspect_ratio (float, optional):  Defaults to 1.0.
        bbox_scale_factor (float, optional):  Defaults to 1.25.
        bbox_format (Literal['xyxy', 'xywh', 'cs']): Defaults to 'xyxy'.
            'xyxy' means the left-up point and right-bottomn point of the
            bbox.
            'xywh' means the left-up point and the width and height of the
            bbox.
            'cs' means the center of the bbox (x,y) and the scale of the
            bbox w & h.
    Returns:
        orig_cam: shape = (frame, 4) or (frame, num_person, 4)
    """
    bbox = bbox.copy()
    if bbox_format == 'xyxy':
        bbox_xywh = xyxy2xywh(bbox)
        bbox_cs = xywh2cs(bbox_xywh, aspect_ratio, bbox_scale_factor)
    elif bbox_format == 'xywh':
        bbox_cs = xywh2cs(bbox, aspect_ratio, bbox_scale_factor)
    elif bbox_format == 'cs':
        bbox_cs = bbox
    else:
        raise ValueError('Only supports the format of `xyxy`, `cs` and `xywh`')

    cx, cy, h = bbox_cs[..., 0], bbox_cs[..., 1], bbox_cs[..., 2] + 1e-6
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[..., 0] * (1. / (img_width / h))
    sy = cam[..., 0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / (sx + 1e-6)) + cam[..., 1]
    ty = ((cy - hh) / hh / (sy + 1e-6)) + cam[..., 2]

    orig_cam = np.stack([sx, sy, tx, ty], axis=-1)
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


def conver_verts_to_cam_coord(verts,
                              pred_cams,
                              bboxes_xy,
                              focal_length=5000.,
                              bbox_scale_factor=1.1,
                              bbox_format='xyxy'):
    """Convert vertices from the world coordinate to camera coordinate.

    Args:
        verts ([np.ndarray]): The vertices in the world coordinate.
            The shape is (frame,num_person,6890,3) or (frame,6890,3).
        pred_cams ([np.ndarray]): Camera parameters estimated by HMR or SPIN.
            The shape is (frame,num_person,3) or (frame,6890,3).
        bboxes_xy ([np.ndarray]): (frame, num_person, 4|5) or (frame, 4|5)
        focal_length ([float],optional): Defined same as your training.
        bbox_scale_factor (float): scale factor for expanding the bbox.
        bbox_format (Literal['xyxy', 'xywh'] ): 'xyxy' means the left-up point
            and right-bottomn point of the bbox.
            'xywh' means the left-up point and the width and height of the
            bbox.
    Returns:
        np.ndarray: The vertices in the camera coordinate.
            The shape is (frame,num_person,6890,3) or (frame,6890,3).
        np.ndarray: The instrinsic parameters of the pred_cam.
            The shape is (num_frame, 3, 3).
    """
    K0 = get_default_hmr_intrinsic(
        focal_length=focal_length, det_height=224, det_width=224)
    K1 = convert_bbox_to_intrinsic(
        bboxes_xy,
        bbox_scale_factor=bbox_scale_factor,
        bbox_format=bbox_format)
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


def process_mmtracking_results(mmtracking_results, max_track_id):
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
        if max_track_id < int(track[0]):
            max_track_id = int(track[0])
        person['bbox'] = track[1:]
        person_results.append(person)
    person_results = sorted(person_results, key=lambda x: x.get('track_id', 0))
    instance_num = len(tracking_results)
    return person_results, max_track_id, instance_num


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
        List[np.ndarray]: prepared frames
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

    frames_iter = img_list if image_path is not None else list(video)

    return frames_iter


def extract_feature_sequence(extracted_results,
                             frame_idx,
                             causal,
                             seq_len,
                             step=1):
    """Extract the target frame from person results, and pad the sequence to a
    fixed length.

    Args:
        extracted_results (List[List[Dict]]): Multi-frame feature extraction
            results stored in a nested list. Each element of the outer list
            is the feature extraction results of a single frame, and each
            element of the inner list is the feature information of one person,
            which contains:
                features (ndarray): extracted features
                track_id (int): unique id of each person, required when
                    ``with_track_id==True```
                bbox ((4, ) or (5, )): left, right, top, bottom, [score]
        frame_idx (int): The index of the frame in the original video.
        causal (bool): If True, the target frame is the first frame in
            a sequence. Otherwise, the target frame is in the middle of a
            sequence.
        seq_len (int): The number of frames in the input sequence.
        step (int): Step size to extract frames from the video.

    Returns:
        List[List[Dict]]: Multi-frame feature extraction results stored in a
            nested list with a length of seq_len.
        int: The target frame index in the padded sequence.
    """

    if causal:
        frames_left = 0
        frames_right = seq_len - 1
    else:
        frames_left = (seq_len - 1) // 2
        frames_right = frames_left
    num_frames = len(extracted_results)

    # get the padded sequence
    pad_left = max(0, frames_left - frame_idx // step)
    pad_right = max(0, frames_right - (num_frames - 1 - frame_idx) // step)
    start = max(frame_idx % step, frame_idx - frames_left * step)
    end = min(num_frames - (num_frames - 1 - frame_idx) % step,
              frame_idx + frames_right * step + 1)
    extracted_results_seq = [extracted_results[0]] * pad_left + \
        extracted_results[start:end:step] + [extracted_results[-1]] * pad_right
    return extracted_results_seq
