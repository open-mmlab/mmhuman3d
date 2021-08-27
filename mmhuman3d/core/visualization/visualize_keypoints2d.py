import os
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Tuple, Union

import cv2
import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping.smplx import (
    SMPLX_LIMBS_INDEX,
    SMPLX_PALETTE,
)
from mmhuman3d.utils.ffmpeg_utils import images_to_video
from .keypoint_utils import get_different_colors, search_limbs


def plot_kp2d_frame(kp2d_person: np.ndarray,
                    canvas: np.ndarray,
                    limbs: dict = SMPLX_LIMBS_INDEX,
                    palette: Optional[Union[dict, np.ndarray]] = None,
                    draw_bbox: bool = False,
                    with_number: bool = False,
                    font_size: Union[float, int] = 0.5) -> np.ndarray:
    """Plot a single frame with keypoints, limbs, bbox, index.

    Args:
        kp2d_person : array shape of (J * 2).
        canvas : cv2 image, (H * W * 3).
        limbs (optional): dict of limb connections. Defaults to
                SMPLX_LIMBS_INDEX.
        palette (optional): The whole limbs and keypoints will be in same
                color if is an array, prepared for multi-person
                visualization. Defaults to SMPLX_PALETTE.
        draw_bbox (optional): whether need to draw bounding boxes.
                Defaults to False.
        with_number (optional): whether need to draw index numbers.
                Defaults to False.
        font_size (optional): the font size of the index.
                Defaults to 0.5.

    Returns:
        np.ndarray: opencv image of shape (H * W * 3).
    """
    if kp2d_person.shape[-1] == 3:
        kp2d_person = kp2d_person[..., :-1]
    if kp2d_person.ndim == 3 and kp2d_person.shape[0] == 1:
        kp2d_person = kp2d_person[0]
    assert kp2d_person.ndim == 2

    if draw_bbox:
        bbox = _get_bbox(kp2d_person, canvas, expand=True)
    else:
        bbox = None
    if palette is None:
        palette = SMPLX_PALETTE
    for part_name, part_limbs in limbs.items():
        scatter_points_index = list(
            set(np.array([part_limbs]).reshape(-1).tolist()))
        if isinstance(palette, dict) and part_name == 'body':
            thickness = 2
            radius = 3
            color = get_different_colors(len(scatter_points_index))
        else:
            thickness = 2
            radius = 2
            if isinstance(palette, np.ndarray):
                color = palette.astype(np.int32)
            elif isinstance(palette, dict):
                color = np.array(palette[part_name]).astype(np.int32)

        for limb_index, limb in enumerate(part_limbs):
            limb_index = min(limb_index, len(color) - 1)
            cv2.line(
                canvas,
                tuple(kp2d_person[limb[0]].astype(np.int32)),
                tuple(kp2d_person[limb[1]].astype(np.int32)),
                color=tuple(color[limb_index].tolist()),
                thickness=thickness)

        for index in scatter_points_index:
            x, y = kp2d_person[index, :2]
            if np.isnan(x) or np.isnan(y):
                continue
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:

                cv2.circle(
                    canvas, (int(x), int(y)),
                    radius,
                    color[min(color.shape[0] - 1,
                              len(scatter_points_index) - 1)].tolist(),
                    thickness=-1)

                if with_number:
                    cv2.putText(
                        canvas, str(index), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size,
                        np.array([255, 255, 255]).astype(np.int32).tolist(), 2)
    if bbox is not None:
        bbox = bbox.astype(np.int32)
        cv2.rectangle(canvas, (bbox[0], bbox[2]), (bbox[1], bbox[3]),
                      (0, 255, 255), 1)
    return canvas


def _get_bbox(keypoint_np: np.ndarray,
              img_mat: Optional[np.ndarray] = None,
              expand: bool = False):
    x_max = np.max(keypoint_np[:, 0])
    x_min = np.min(keypoint_np[:, 0])
    y_max = np.max(keypoint_np[:, 1])
    y_min = np.min(keypoint_np[:, 1])
    if expand and img_mat is not None:
        x_expand = (x_max - x_min) * 0.1
        y_expand = (y_max - y_min) * 0.1
        x_min = max(0, x_min - x_expand)
        x_max = min(img_mat.shape[1], x_max + x_expand)
        y_min = max(0, y_min - y_expand)
        y_max = min(img_mat.shape[0], y_max + y_expand)
    return np.asarray([x_min, x_max, y_min, y_max])


def visualize_kp2d(
    kp2d: np.ndarray,
    output_path: str,
    frame_list: Optional[List[str]] = None,
    limbs: Optional[Union[np.ndarray, List[int]]] = None,
    palette: Optional[Iterable[int]] = None,
    data_source: str = 'mmpose',
    mask: Optional[Union[list, np.ndarray]] = None,
    start: int = 0,
    end: int = -1,
    force: bool = False,
    with_file_name: bool = False,
    resolution: Optional[Union[Tuple[int, int], list]] = None,
    fps: Union[float, int] = 30,
    draw_bbox: bool = False,
    with_number: bool = False,
    pop_parts: Iterable[str] = [],
) -> NoReturn:
    """Visualize 2d keypoints to a video or into a folder of frames.

    Args:
        kp2d (np.ndarray): should be array of shape (f * J * 2)
                                or (f * n * J * 2)]
        output_path (str): output video path or image folder.
        frame_list (List[str]): list of frame paths, if None, would be
                initialized as white background.
        limbs (Optional[Union[np.ndarray, List[int]]], optional):
                if not specified, the limbs will be searched by search_limbs,
                this option is for free skeletons like BVH file.
                Defaults to None.
        palette (Iterable, optional): specified palette, three int represents
                (B, G, R). Should be tuple or list.
                Defaults to None.
        data_source (str, optional): data source type. Defaults to 'mmpose'.
        mask (Optional[Union[list, np.ndarray]], optional):
                mask to mask out the incorrect points. Defaults to None.
        start (int, optional): start frame index. Defaults to 0.
        end (int, optional): end frame index. Defaults to -1.
        force (bool, optional): whether replace the origin frames.
                Defaults to False.
        with_file_name (bool, optional): whether write origin frame name on
                the images. Defaults to False.
        resolution (Optional[Union[Tuple[int, int], list]], optional):
                (width, height) of the output video
                will be the same size as the original images if not specified.
                Defaults to None.
        fps (Union[float, int], optional): fps. Defaults to 30.
        draw_bbox (bool, optional): whether need to draw bounding boxes.
                Defaults to False.
        with_number (bool, optional): whether draw index number.
                Defaults to False.
        pop_parts (Iterable[str], optional): The body part names you do not
                want to visualize. Defaults to [].

    Raises:
        FileNotFoundError: check output video path.
        FileNotFoundError: check input frame paths.

    Returns:
        NoReturn.
    """
    output_pathinfo = Path(output_path)
    if not output_pathinfo.parent.is_dir():
        raise FileNotFoundError(f'The output root folder does not exist:\
                    {output_pathinfo.parent}')
    if output_pathinfo.suffix == '':
        if not output_pathinfo.is_dir():
            os.mkdir(output_path)
    elif not output_pathinfo.suffix.lower() in ['.mp4']:
        raise FileNotFoundError(
            f'The output file should be .mp4: {output_path}')
    if kp2d.ndim == 3:
        kp2d = kp2d[:, np.newaxis]
    assert kp2d.ndim == 4
    num_frame, num_person = kp2d.shape[0], kp2d.shape[1]
    if frame_list is not None:
        for frame_path in frame_list:
            if not (Path(frame_path).is_file()
                    and Path(frame_path).suffix.lower() in ['.png', '.jpg']):
                raise FileNotFoundError(
                    f'The frame should be .png or .jpg: {frame_path}')
    if output_pathinfo.suffix.lower() in ['.mp4']:
        temp_folder = os.path.join(
            '.',
            os.path.basename(output_path) + '_temp_images')
    else:
        temp_folder = output_path
    os.makedirs(temp_folder, exist_ok=True)
    if limbs is not None:
        limbs_target, limbs_palette = {
            'body': limbs.tolist() if isinstance(limbs, np.ndarray) else limbs
        }, get_different_colors(len(limbs))
    else:
        limbs_target, limbs_palette = search_limbs(
            data_source=data_source, mask=mask)
    if palette:
        limbs_palette = np.array(palette, dtype=np.uint8)[None]
    num_frame = min(len(frame_list), num_frame)
    end = (min(num_frame, end) + num_frame) % num_frame
    kp2d = kp2d[start:end + 1]
    for part_name in pop_parts:
        limbs_target.pop(part_name)
    for frame_index in range(kp2d.shape[0]):
        kp2d_frame = kp2d[frame_index]
        if kp2d_frame.ndim == 2:
            kp2d_frame = kp2d_frame[np.newaxis]
        if frame_list is not None:
            image_array = cv2.imread(frame_list[frame_index])
        else:
            image_array = np.ones(
                (resolution[1], resolution[0], 3), dtype=np.uint8) * 255
        if resolution is not None:
            w_scale = resolution[0] / image_array.shape[1]
            h_scale = resolution[1] / image_array.shape[0]
            image_array = cv2.resize(image_array, resolution, cv2.INTER_CUBIC)
            kp2d_frame *= np.array([[w_scale, h_scale]])
        for person_index in range(num_person):
            if num_person >= 2:
                limbs_palette = get_different_colors(1)
            image_array = plot_kp2d_frame(
                kp2d_frame[person_index],
                image_array,
                limbs_target,
                limbs_palette,
                draw_bbox=draw_bbox,
                with_number=with_number,
                font_size=0.5)
        if output_pathinfo.suffix.lower() in ['.mp4']:
            if with_file_name:
                h, w, _ = image_array.shape
                cv2.putText(
                    image_array, str(Path(frame_list[frame_index]).name),
                    (w // 2, h // 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * h / 500,
                    np.array([255, 255, 255]).astype(np.int32).tolist(), 2)
            cv2.imwrite(
                os.path.join(temp_folder, '%06d.png' % frame_index),
                image_array)
        else:
            if Path(temp_folder).absolute() == Path(
                    frame_list[frame_index]).parent.absolute() and (not force):
                raise FileExistsError(
                    f'{temp_folder} exists (set --force to overwrite).')
            else:
                if with_file_name:
                    h, w, _ = image_array.shape
                    cv2.putText(
                        image_array, str(Path(frame_list[frame_index]).name),
                        (w // 2, h // 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * h / 500,
                        np.array([255, 255, 255]).astype(np.int32).tolist(), 2)
                cv2.imwrite(
                    os.path.join(temp_folder,
                                 Path(frame_list[frame_index]).name),
                    image_array)
    if output_pathinfo.suffix.lower() in ['.mp4']:
        images_to_video(
            input_folder=temp_folder,
            output_path=output_path,
            remove_raw_file=True,
            fps=fps)
