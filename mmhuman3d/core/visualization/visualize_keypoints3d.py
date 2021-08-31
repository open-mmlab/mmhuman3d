import warnings
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Tuple, Union

import numpy as np

from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY
from mmhuman3d.core.visualization.renderer import matplotlib3d_renderer
from mmhuman3d.utils.keypoint_utils import get_different_colors, search_limbs


def _norm_pose(pose_numpy: np.ndarray, min_value: Union[float, int],
               max_value: Union[float, int], mask: Union[np.ndarray, list]):
    assert max_value > min_value
    pose_np_positive = pose_numpy
    if not mask:
        mask = list(range(pose_numpy.shape[-2]))
    for dim_index in range(3):
        pose_np_positive[..., dim_index] = \
            pose_np_positive[..., dim_index] - \
            np.min(pose_np_positive[:, :, mask, dim_index])
    global_max_value = np.max(pose_np_positive[:, :, mask])
    pose_np_positive = \
        pose_np_positive / global_max_value * (max_value-min_value)
    pose_np_normalized = pose_np_positive + min_value
    return pose_np_normalized


def visualize_kp3d(
    kp3d: np.ndarray,
    output_path: str,
    limbs: Optional[Union[np.ndarray, List[int]]] = None,
    palette: Optional[Iterable[int]] = None,
    data_source: str = 'mmpose',
    mask: Optional[Union[list, tuple, np.ndarray]] = None,
    start: int = 0,
    end: int = -1,
    resolution: Union[list, Tuple[int, int]] = (1280, 1280),
    fps: Union[float, int] = 30,
    frame_names: Optional[Union[List[str], str]] = None,
    orbit_speed: Union[float, int] = 0.5,
    value_range: Union[Tuple[int, int], list] = (-100, 100),
    pop_parts: Iterable[str] = ()
) -> NoReturn:
    """Visualize 3d keypoints to a video with matplotlib. Support multi person
    and specified limb connections.

    Args:
        kp3d (np.ndarray): shape could be (f * J * 4/3/2) or
                (f * num_person * J * 4/3/2)
        output_path (str): output video path.
        limbs (Optional[Union[np.ndarray, List[int]]], optional):
                if not specified, the limbs will be searched by search_limbs,
                this option is for free skeletons like BVH file.
                Defaults to None.
        palette (Iterable, optional): specified palette, three int represents
                (B, G, R). Should be tuple or list.
                Defaults to None.
        data_source (str, optional): data source type. Defaults to 'mmpose'.
                choose in ['coco', 'smplx', 'smpl', 'mmpose', 'mpi_inf_3dhp',
                'mpi_inf_3dhp_test', 'h36m', 'pw3d', 'mpii']
        mask (Optional[Union[list, tuple, np.ndarray]], optional):
                mask to mask out the incorrect points. Defaults to None.
        start (int, optional): start frame index. Defaults to 0.
        end (int, optional): end frame index. Defaults to -1.
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
                want to visualize. Defaults to ().
    Raises:
        TypeError: check the type of input keypoints.
        FileNotFoundError: check the output video path.

    Returns:
        NoReturn.
    """
    # check input shape
    if not isinstance(kp3d, np.ndarray):
        raise TypeError(
            f'Input type is {type(kp3d)}, which should be numpy.ndarray.')
    kp3d = kp3d.copy()
    if kp3d.shape[-1] == 2:
        kp3d = np.concatenate([kp3d, np.zeros_like(kp3d)[..., 0:1]], axis=-1)
        warnings.warn(
            f'The input array is 2-Dimensional coordinates, will concatenate\
                 zeros to the last axis. The new array shape: {kp3d.shape}')
    elif kp3d.shape[-1] >= 4:
        kp3d = kp3d[..., :3]
        warnings.warn(
            f'The input array has more than 3-Dimensional coordinates, will\
                keep only the first 3-Dimensions of the last axis. The new\
                    array shape: {kp3d.shape}')
    if kp3d.ndim == 3:
        kp3d = np.expand_dims(kp3d, 1)
    num_frames = kp3d.shape[0]
    assert kp3d.ndim == 4
    assert kp3d.shape[-1] == 3

    # check data_source & mask
    if data_source not in KEYPOINTS_FACTORY:
        raise ValueError(f'Wrong data_source. Should choose in \
                {list(KEYPOINTS_FACTORY.keys())}')
    if mask is not None:
        assert mask.shape == (
            len(KEYPOINTS_FACTORY[data_source]),
        ), f'mask length should fit with keypoints number \
            {len(KEYPOINTS_FACTORY[data_source])}'

    # check the output path
    if not Path(output_path).parent.is_dir():
        raise FileNotFoundError(
            f'The output folder does not exist: {Path(output_path).parent}')
    if not Path(output_path).suffix.lower() in ['.mp4']:
        raise FileNotFoundError(
            f'The output file should be .mp4: {output_path}')

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

    # slice the frames
    end = (min(end, num_frames) + num_frames) % num_frames
    input_pose_np = input_pose_np[start:end + 1]

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
    for part_name in pop_parts:
        limbs_target.pop(part_name)

    # initialize renderer and start render
    renderer = matplotlib3d_renderer.Axes3dJointsRenderer()
    renderer.init_camera(cam_hori_speed=orbit_speed, cam_elev_speed=0.2)
    renderer.set_connections(limbs_target, limbs_palette)
    if isinstance(frame_names, str):
        if '%' in frame_names:
            frame_names = [
                frame_names % index
                for index in range(len(input_pose_np.shape[0]))
            ]
        else:
            frame_names = [frame_names] * input_pose_np.shape[0]
    renderer.render_kp3d_to_video(
        input_pose_np,
        output_path,
        sign=(-1, -1, -1),
        axis='xzy',
        fps=fps,
        resolution=resolution,
        visual_range=value_range,
        frame_names=frame_names)
