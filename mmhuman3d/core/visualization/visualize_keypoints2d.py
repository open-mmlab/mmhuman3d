import glob
import os
import os.path as osp
import shutil
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY
from mmhuman3d.core.conventions.keypoints_mapping.smplx import (
    SMPLX_LIMBS_INDEX,
    SMPLX_PALETTE,
)
from mmhuman3d.utils.ffmpeg_utils import images_to_video, video_to_images
from mmhuman3d.utils.keypoint_utils import get_different_colors, search_limbs
from mmhuman3d.utils.path_utils import (
    Existence,
    check_input_path,
    check_path_existence,
    check_path_suffix,
    prepare_output_path,
)


def plot_kp2d_frame(kp2d_person: np.ndarray,
                    canvas: np.ndarray,
                    limbs: Union[list, dict, np.ndarray] = SMPLX_LIMBS_INDEX,
                    palette: Optional[Union[dict, np.ndarray]] = None,
                    draw_bbox: bool = False,
                    with_number: bool = False,
                    font_size: Union[float, int] = 0.5) -> np.ndarray:
    """Plot a single frame(array) with keypoints, limbs, bbox, index.

    Args:
        kp2d_person (np.ndarray): `np.ndarray` shape of (J * 2).
        canvas (np.ndarray): cv2 image, (H * W * 3) array.
        limbs (Union[list, dict, np.ndarray], optional): limbs in form of
            `dict` or 2-dimensional `list` or `np.ndarray` of shape
            (num_limb, 2).
            `dict` is used mainly for function `visualize_kp2d`, you can also
            get the limbs by function `search_limbs`.
            Defaults to `SMPLX_LIMBS_INDEX`.
        palette (Optional[Union[dict, np.ndarray, list]], optional):
            Pass an (1, 3) `np.ndarray` or `list` [B, G, R] if want the whole
            limbs and keypoints will be in same color.
            Pass `None` to use our colorful palette.
            Pass an (num_limb, 3) `np.ndarray` to get each limb your specific
            color.
            `dict` is used mainly for function `visualize_kp2d`, you can also
            get the palette by function `search_limbs`.
            Defaults to `SMPLX_PALETTE`.
        draw_bbox (bool, optional): whether need to draw bounding boxes.
            Defaults to False.
        with_number (bool, optional): whether need to draw index numbers.
            Defaults to False.
        font_size (Union[float, int], optional): the font size of the index.
            Defaults to 0.5.

    Returns:
        np.ndarray: opencv image of shape (H * W * 3).
    """
    # slice the kp2d array
    kp2d_person = kp2d_person.copy()
    if kp2d_person.shape[-1] >= 3:
        kp2d_person = kp2d_person[..., :-1]
        warnings.warn(
            'The input array has more than 2-Dimensional coordinates, will'
            'keep only the first 2-Dimensions of the last axis. The new'
            f'array shape: {kp2d_person.shape}')
    if kp2d_person.ndim == 3 and kp2d_person.shape[0] == 1:
        kp2d_person = kp2d_person[0]
    assert kp2d_person.ndim == 2 and kp2d_person.shape[
        -1] == 2, f'Wrong input array shape {kp2d_person.shape}, \
            should be (num_kp, 2)'

    if draw_bbox:
        bbox = _get_bbox(kp2d_person, canvas, expand=True)
    else:
        bbox = None

    # determine the limb connections and palette
    if isinstance(limbs, list):
        limbs = {'body': limbs}
    elif isinstance(limbs, np.ndarray):
        limbs = {'body': limbs.reshape(-1, 2).astype(np.int32).tolist()}
    else:
        assert set(limbs.keys()).issubset(SMPLX_LIMBS_INDEX)

    if palette is None:
        palette = {'body': None}
    elif isinstance(palette, dict):
        assert set(palette.keys()) == set(limbs.keys())

    # draw by part to specify the thickness and color
    for part_name, part_limbs in limbs.items():
        # scatter_points_index means the limb end points
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
            elif isinstance(palette, list):
                color = np.array(palette).reshape(-1, 3).astype(np.int32)
        for limb_index, limb in enumerate(part_limbs):
            limb_index = min(limb_index, len(color) - 1)
            cv2.line(
                canvas,
                tuple(kp2d_person[limb[0]].astype(np.int32)),
                tuple(kp2d_person[limb[1]].astype(np.int32)),
                color=tuple(color[limb_index].tolist()),
                thickness=thickness)
        # draw the points inside the image region
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
    # draw the bboxes
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


def _prepare_limb_palette(limbs,
                          palette,
                          pop_parts,
                          data_source,
                          mask,
                          search_limbs_func=search_limbs):
    """Prepare limbs and their palette for plotting.

    Args:
        limbs (Union[np.ndarray, List[int]]):
            The preset limbs. This option is for free skeletons like BVH file.
            In most cases, it's set to None,
            this function will search a result for limbs automatically.
        palette (Iterable):
            The preset palette for limbs. Specified palette,
            three int represents (B, G, R). Should be tuple or list.
            In most cases, it's set to None,
            a palette will be generated with the result of search_limbs.
        pop_parts (Iterable[str]):
            The body part names you do not
            want to visualize.
            When it's none, nothing will be removed.
        data_source (str):
            Data source type.
        mask (Union[list, np.ndarray):
            A mask to mask out the incorrect points.

    Returns:
        Tuple[dict, dict]: (limbs_target, limbs_palette).
    """
    if limbs is not None:
        limbs_target, limbs_palette = {
            'body': limbs.tolist() if isinstance(limbs, np.ndarray) else limbs
        }, get_different_colors(len(limbs))
    else:
        limbs_target, limbs_palette = search_limbs_func(
            data_source=data_source, mask=mask)

    if palette:
        limbs_palette = np.array(palette, dtype=np.uint8)[None]

    # check and pop the pop_parts
    assert set(pop_parts).issubset(
        SMPLX_PALETTE), f'wrong part_names in pop_parts, supported parts are\
            {set(SMPLX_PALETTE.keys())}'

    for part_name in pop_parts:
        if part_name in limbs_target:
            limbs_target.pop(part_name)
            limbs_palette.pop(part_name)
    return limbs_target, limbs_palette


def _prepare_output_path(output_path, overwrite):
    prepare_output_path(
        output_path,
        allowed_suffix=['.mp4', ''],
        tag='output video',
        path_type='auto',
        overwrite=overwrite)
    # output_path is a directory
    if check_path_suffix(output_path, ['']):
        temp_folder = output_path
        os.makedirs(temp_folder, exist_ok=True)
    else:
        temp_folder = output_path + '_temp_images'
        if check_path_existence(temp_folder, 'directory') in [
                Existence.FolderExistNotEmpty, Existence.FolderExistEmpty
        ]:
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder, exist_ok=True)
    return temp_folder


def _check_frame_path(frame_list):
    for frame_path in frame_list:
        if check_path_existence(frame_path, 'file') != Existence.FileExist or \
                 not check_path_suffix(frame_path, ['.png', '.jpg', '.jpeg']):
            raise FileNotFoundError(
                f'The frame should be .png or .jp(e)g: {frame_path}')


def _check_temp_path(temp_folder, frame_list, overwrite):
    if not overwrite and frame_list is not None and len(frame_list) > 0:
        if Path(temp_folder).absolute() == \
                Path(frame_list[0]).parent.absolute():
            raise FileExistsError(
                f'{temp_folder} exists (set --overwrite to overwrite).')


class _CavasProducer:

    def __init__(self, frame_list, resolution, kp2d, default_scale=1.5):
        # check the origin background frames
        if frame_list is not None:
            _check_frame_path(frame_list)
            self.frame_list = frame_list
        else:
            self.frame_list = []
        self.frame_list = frame_list
        self.resolution = resolution
        self.kp2d = kp2d
        self.resolution = resolution
        if len(self.frame_list) > 1 and \
                check_path_existence(
                    self.frame_list[0], 'file') == Existence.FileExist:
            tmp_image_array = cv2.imread(self.frame_list[0])
            self.auto_resolution = \
                [tmp_image_array.shape[1], tmp_image_array.shape[0]]
        else:
            self.auto_resolution = \
                [np.max(kp2d) * default_scale, np.max(kp2d) * default_scale]

    def get_data(self, frame_index):
        # frame file exists, resolution not set
        if frame_index < len(self.frame_list) and self.resolution is None:
            image_array = cv2.imread(self.frame_list[frame_index])
            kp2d_frame = self.kp2d[frame_index]
        # no frame file, resolution has been set
        elif frame_index >= len(self.frame_list) and \
                self.resolution is not None:
            image_array = np.ones((self.resolution[1], self.resolution[0], 3),
                                  dtype=np.uint8) * 255
            kp2d_frame = self.kp2d[frame_index]
        # frame file exists, resolution has been set
        elif frame_index < len(self.frame_list) and \
                self.resolution is not None:
            image_array = cv2.imread(self.frame_list[frame_index])
            w_scale = self.resolution[0] / image_array.shape[1]
            h_scale = self.resolution[1] / image_array.shape[0]
            image_array = \
                cv2.resize(image_array, self.resolution, cv2.INTER_CUBIC)
            kp2d_frame = \
                np.array([[w_scale, h_scale]]) * self.kp2d[frame_index]
        # no frame file, no resolution
        else:
            image_array = \
                np.ones((self.auto_resolution[1], self.auto_resolution[0], 3),
                        dtype=np.uint8) * 255
            kp2d_frame = self.kp2d[frame_index]
        return image_array, kp2d_frame


def update_frame_list(frame_list, origin_frames, start, end):
    input_temp_folder = None
    # choose in frame_list or origin_frames
    if frame_list is None and origin_frames is None:
        print('No background provided, will use pure white background.')
    elif frame_list is not None and origin_frames is not None:
        warnings.warn('Redundant input, will only use frame_list.')
        origin_frames = None

    if origin_frames is not None:
        check_input_path(
            input_path=origin_frames,
            allowed_suffix=['.mp4', '.gif', ''],
            tag='origin frames',
            path_type='auto')
        if Path(origin_frames).is_file():
            input_temp_folder = origin_frames + '_temp_images/'
            video_to_images(
                origin_frames, input_temp_folder, start=start, end=end)
            frame_list = glob.glob(osp.join(input_temp_folder, '*.png'))
        else:
            frame_list = []
            for im_name in os.listdir(origin_frames):
                if Path(im_name).suffix.lower() in ['.png', '.jpg', 'jpeg']:
                    frame_list.append(osp.join(origin_frames, im_name))
    return frame_list, input_temp_folder


def visualize_kp2d(kp2d: np.ndarray,
                   output_path: str,
                   frame_list: Optional[List[str]] = None,
                   origin_frames: Optional[str] = None,
                   limbs: Optional[Union[np.ndarray, List[int]]] = None,
                   palette: Optional[Iterable[int]] = None,
                   data_source: str = 'mmpose',
                   mask: Optional[Union[list, np.ndarray]] = None,
                   start: int = 0,
                   end: int = -1,
                   overwrite: bool = False,
                   with_file_name: bool = True,
                   resolution: Optional[Union[Tuple[int, int], list]] = None,
                   fps: Union[float, int] = 30,
                   draw_bbox: bool = False,
                   with_number: bool = False,
                   pop_parts: Iterable[str] = [],
                   disable_tqdm: bool = False,
                   keypoints_factory: dict = KEYPOINTS_FACTORY) -> None:
    """Visualize 2d keypoints to a video or into a folder of frames.

    Args:
        kp2d (np.ndarray): should be array of shape (f * J * 2)
                                or (f * n * J * 2)]
        output_path (str): output video path or image folder.
        frame_list (Optional[List[str]], optional): list of origin brackground
            frame paths, element in list each should be a image path like
            `*.jpg` or `*.png`. Higher priority than `origin_frames`.
            Use this when your file names is hard to sort or you only want to
            render a small number frames.
            Defaults to None.
        origin_frames (Optional[str], optional): origin brackground frame path,
            could be `.mp4`, `.gif`(will be sliced into a folder) or an image
            folder. Lower priority than `frame_list`.
            Defaults to None.
        limbs (Optional[Union[np.ndarray, List[int]]], optional):
                if not specified, the limbs will be searched by search_limbs,
                this option is for free skeletons like BVH file.
                Defaults to None.
        palette (Iterable, optional): specified palette, three int represents
                (B, G, R). Should be tuple or list.
                Defaults to None.
        data_source (str, optional): data source type. Defaults to 'mmpose'.
        mask (Optional[Union[list, np.ndarray]], optional):
                mask to mask out the incorrect point.
                Pass a `np.ndarray` of shape (J,) or `list` of length J.
                Defaults to None.
        start (int, optional): start frame index. Defaults to 0.
        end (int, optional): end frame index. Defaults to -1.
        overwrite (bool, optional): whether replace the origin frames.
                Defaults to False.
        with_file_name (bool, optional): whether write origin frame name on
                the images. Defaults to True.
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
                want to visualize. Supported parts are ['left_eye','right_eye'
                ,'nose', 'mouth', 'face', 'left_hand', 'right_hand'].
                Defaults to [].
        disable_tqdm (bool, optional):
            Whether to disable the entire progressbar wrapper.
            Defaults to False.
        keypoints_factory (dict, optional): Dict of all the conventions.
            Defaults to KEYPOINTS_FACTORY.

    Raises:
        FileNotFoundError: check output video path.
        FileNotFoundError: check input frame paths.

    Returns:
        None.
    """
    # check output path
    temp_folder = _prepare_output_path(output_path, overwrite)

    frame_list, input_temp_folder = update_frame_list(frame_list,
                                                      origin_frames, start,
                                                      end)
    # check whether temp_folder will overwrite frame_list by accident
    _check_temp_path(temp_folder, frame_list, overwrite)

    # check the input array shape, reshape to (num_frame, num_person, J, 2)
    kp2d = kp2d[..., :2].copy()
    if kp2d.ndim == 3:
        kp2d = kp2d[:, np.newaxis]
    assert kp2d.ndim == 4
    num_frame, num_person = kp2d.shape[0], kp2d.shape[1]

    # check data_source & mask
    if data_source not in keypoints_factory:
        raise ValueError('Wrong data_source. Should choose in'
                         f'{list(keypoints_factory.keys())}')
    if mask is not None:
        if isinstance(mask, list):
            mask = np.array(mask).reshape(-1)
        assert mask.shape == (
            len(keypoints_factory[data_source]),
        ), f'mask length should fit with keypoints number \
            {len(keypoints_factory[data_source])}'

    # search the limb connections and palettes from superset smplx
    # check and pop the pop_parts
    limbs_target, limbs_palette = _prepare_limb_palette(
        limbs, palette, pop_parts, data_source, mask)

    # slice the input array temporally
    num_frame = min(len(frame_list),
                    num_frame) if frame_list is not None else num_frame
    end = (min(num_frame - 1, end) + num_frame) % num_frame
    kp2d = kp2d[start:end + 1]

    canvas_producer = _CavasProducer(frame_list, resolution, kp2d)

    # start plotting by frame
    for frame_index in tqdm(range(kp2d.shape[0]), disable=disable_tqdm):
        image_array, kp2d_frame = canvas_producer.get_data(frame_index)
        # start plotting by person
        for person_index in range(num_person):
            if num_person >= 2:
                limbs_palette = get_different_colors(
                    num_person)[person_index].reshape(1, 3)
            image_array = plot_kp2d_frame(
                kp2d_person=kp2d_frame[person_index],
                canvas=image_array,
                limbs=limbs_target,
                palette=limbs_palette,
                draw_bbox=draw_bbox,
                with_number=with_number,
                font_size=0.5)
        if with_file_name and frame_list is not None:
            h, w, _ = image_array.shape
            cv2.putText(image_array, str(Path(frame_list[frame_index]).name),
                        (w // 2, h // 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * h / 500,
                        np.array([255, 255, 255]).astype(np.int32).tolist(), 2)
        # write the frame with opencv
        if frame_list is not None and check_path_suffix(output_path, []):
            frame_path = \
                os.path.join(temp_folder, Path(frame_list[frame_index]).name)
        else:
            frame_path = \
                os.path.join(temp_folder, f'{frame_index:06d}.png')
        cv2.imwrite(frame_path, image_array)

    if input_temp_folder is not None:
        shutil.rmtree(input_temp_folder)
    # convert frames to video
    if check_path_suffix(output_path, ['.mp4']):
        images_to_video(
            input_folder=temp_folder,
            output_path=output_path,
            remove_raw_file=True,
            fps=fps)
