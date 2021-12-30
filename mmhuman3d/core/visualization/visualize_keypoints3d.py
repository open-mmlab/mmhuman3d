import warnings
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

import mmhuman3d.core.conventions.keypoints_mapping as keypoints_mapping
from mmhuman3d.core.visualization.renderer import Axes3dJointsRenderer
from mmhuman3d.utils import (
    get_different_colors,
    prepare_output_path,
    search_limbs,
)


def _norm_pose(pose_numpy: np.ndarray, min_value: Union[float, int],
               max_value: Union[float, int], mask: Union[np.ndarray, list]):
    """Normalize the poses and make the center close to axis center."""
    assert max_value > min_value
    pose_np_normed = pose_numpy.copy()
    if not mask:
        mask = list(range(pose_numpy.shape[-2]))
    axis_num = 3
    axis_stat = np.zeros(shape=[axis_num, 4])
    for axis_index in range(axis_num):
        axis_data = pose_np_normed[..., mask, axis_index]
        axis_min = np.min(axis_data)
        axis_max = np.max(axis_data)
        axis_mid = (axis_min + axis_max) / 2.0
        axis_span = axis_max - axis_min
        axis_stat[axis_index] = np.asarray(
            (axis_min, axis_max, axis_mid, axis_span))
    target_mid = (max_value + min_value) / 2.0
    max_span = np.max(axis_stat[:, 3])
    target_span = max_value - min_value
    for axis_index in range(axis_num):
        pose_np_normed[..., axis_index] = \
            pose_np_normed[..., axis_index] - \
            axis_stat[axis_index, 2]
    pose_np_normed = pose_np_normed / max_span * target_span
    pose_np_normed = pose_np_normed + target_mid
    return pose_np_normed


def visualize_kp3d(
    kp3d: np.ndarray,
    output_path: Optional[str] = None,
    limbs: Optional[Union[np.ndarray, List[int]]] = None,
    palette: Optional[Iterable[int]] = None,
    data_source: str = 'coco',
    mask: Optional[Union[list, tuple, np.ndarray]] = None,
    start: int = 0,
    end: Optional[int] = None,
    resolution: Union[list, Tuple[int, int]] = (1024, 1024),
    fps: Union[float, int] = 30,
    frame_names: Optional[Union[List[str], str]] = None,
    orbit_speed: Union[float, int] = 0.5,
    value_range: Union[Tuple[int, int], list] = (-100, 100),
    pop_parts: Iterable[str] = (),
    disable_limbs: bool = False,
    return_array: Optional[bool] = None,
    convention: str = 'opencv',
    keypoints_factory: dict = keypoints_mapping.KEYPOINTS_FACTORY,
) -> Union[None, np.ndarray]:
    """Visualize 3d keypoints to a video with matplotlib. Support multi person
    and specified limb connections.

    Args:
        kp3d (np.ndarray): shape could be (f * J * 4/3/2) or
            (f * num_person * J * 4/3/2)
        output_path (str): output video path image folder.
        limbs (Optional[Union[np.ndarray, List[int]]], optional):
            if not specified, the limbs will be searched by search_limbs,
            this option is for free skeletons like BVH file.
            Defaults to None.
        palette (Iterable, optional): specified palette, three int represents
            (B, G, R). Should be tuple or list.
            Defaults to None.
        data_source (str, optional): data source type. Defaults to 'coco'.
            choose in ['coco', 'smplx', 'smpl', 'coco_wholebody',
            'mpi_inf_3dhp', 'mpi_inf_3dhp_test', 'h36m', 'pw3d', 'mpii']
        mask (Optional[Union[list, tuple, np.ndarray]], optional):
            mask to mask out the incorrect points. Defaults to None.
        start (int, optional): start frame index. Defaults to 0.
        end (int, optional): end frame index.
            Could be positive int or negative int or None.
            None represents include all the frames.
            Defaults to None.
        resolution (Union[list, Tuple[int, int]], optional):
            (width, height) of the output video
            will be the same size as the original images if not specified.
            Defaults to None.
        fps (Union[float, int], optional): fps. Defaults to 30.
        frame_names (Optional[Union[List[str], str]], optional): List(should be
            the same as frame numbers) or single string or string format
            (like 'frame%06d')for frame title, no title if None.
            Defaults to None.
        orbit_speed (Union[float, int], optional): orbit speed of camera.
            Defaults to 0.5.
        value_range (Union[Tuple[int, int], list], optional):
            range of axis value. Defaults to (-100, 100).
        pop_parts (Iterable[str], optional): The body part names you do not
            want to visualize. Choose in ['left_eye','right_eye', 'nose',
            'mouth', 'face', 'left_hand', 'right_hand']Defaults to [].
        disable_limbs (bool, optional): whether need to disable drawing limbs.
            Defaults to False.
        return_array (bool, optional): Whether to return images as opencv array
            .If None, an array will be returned when frame number is below 100.
            Defaults to None.
        keypoints_factory (dict, optional): Dict of all the conventions.
            Defaults to KEYPOINTS_FACTORY.
    Raises:
        TypeError: check the type of input keypoints.
        FileNotFoundError: check the output video path.

    Returns:
        Union[None, np.ndarray].
    """
    # check input shape
    if not isinstance(kp3d, np.ndarray):
        raise TypeError(
            f'Input type is {type(kp3d)}, which should be numpy.ndarray.')
    kp3d = kp3d.copy()
    if kp3d.shape[-1] == 2:
        kp3d = np.concatenate([kp3d, np.zeros_like(kp3d)[..., 0:1]], axis=-1)
        warnings.warn(
            'The input array is 2-Dimensional coordinates, will concatenate ' +
            f'zeros to the last axis. The new array shape: {kp3d.shape}')
    elif kp3d.shape[-1] >= 4:
        kp3d = kp3d[..., :3]
        warnings.warn(
            'The input array has more than 3-Dimensional coordinates, will ' +
            'keep only the first 3-Dimensions of the last axis. The new ' +
            f'array shape: {kp3d.shape}')
    if kp3d.ndim == 3:
        kp3d = np.expand_dims(kp3d, 1)
    num_frames = kp3d.shape[0]
    assert kp3d.ndim == 4
    assert kp3d.shape[-1] == 3

    if return_array is None:
        if num_frames > 100:
            return_array = False
        else:
            return_array = True

    # check data_source & mask
    if data_source not in keypoints_factory:
        raise ValueError('Wrong data_source. Should choose in' +
                         f'{list(keypoints_factory.keys())}')
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask).reshape(-1)
        assert mask.shape == (
            len(keypoints_factory[data_source]),
        ), f'mask length should fit with keypoints number \
            {len(keypoints_factory[data_source])}'

    # check the output path
    if output_path is not None:
        prepare_output_path(
            output_path,
            path_type='auto',
            tag='output video',
            allowed_suffix=['.mp4', '.gif', ''])

    # slice the frames
    end = num_frames if end is None else end
    kp3d = kp3d[start:end]
    # norm the coordinates
    if value_range is not None:
        # norm pose location to value_range (70% value range)
        mask_index = np.where(np.array(mask) > 0) if mask is not None else None
        margin_width = abs(value_range[1] - value_range[1]) * 0.15
        pose_np_normed = _norm_pose(kp3d, value_range[0] + margin_width,
                                    value_range[1] - margin_width, mask_index)
        input_pose_np = pose_np_normed
    else:
        input_pose_np = kp3d

    # determine the limb connections and palettes
    if limbs is not None:
        limbs_target, limbs_palette = {
            'body': limbs.tolist() if isinstance(limbs, np.ndarray) else limbs
        }, get_different_colors(len(limbs))
    else:
        limbs_target, limbs_palette = search_limbs(
            data_source=data_source, mask=mask)
    if palette is not None:
        limbs_palette = np.array(palette, dtype=np.uint8)[None]

    # check and pop the pop_parts
    assert set(pop_parts).issubset(
        keypoints_mapping.human_data.HUMAN_DATA_PALETTE.keys(
        )), f'wrong part_names in pop_parts, could only \
        choose in{set(keypoints_mapping.human_data.HUMAN_DATA_PALETTE.keys())}'

    for part_name in pop_parts:
        if part_name in limbs_target:
            limbs_target.pop(part_name)

    # initialize renderer and start render
    renderer = Axes3dJointsRenderer()
    renderer.init_camera(cam_hori_speed=orbit_speed, cam_elev_speed=0.2)
    renderer.set_connections(limbs_target, limbs_palette)
    if isinstance(frame_names, str):
        if '%' in frame_names:
            frame_names = [
                frame_names % index for index in range(input_pose_np.shape[0])
            ]
        else:
            frame_names = [frame_names] * input_pose_np.shape[0]
    image_array = renderer.render_kp3d_to_video(
        input_pose_np,
        output_path,
        convention,
        fps=fps,
        resolution=resolution,
        visual_range=value_range,
        frame_names=frame_names,
        disable_limbs=disable_limbs,
        return_array=return_array)
    return image_array
