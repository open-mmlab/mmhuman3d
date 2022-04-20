import io
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Union

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

from mmhuman3d.core.conventions.cameras import enc_camera_convention
from mmhuman3d.utils import get_different_colors
from mmhuman3d.utils.ffmpeg_utils import images_to_video
from mmhuman3d.utils.path_utils import check_path_suffix


class Axes3dBaseRenderer(object):
    """Base renderer."""

    def init_camera(self,
                    cam_elev_angle=10,
                    cam_elev_speed=0.0,
                    cam_hori_angle=45,
                    cam_hori_speed=0.5):
        """Initiate the route of camera with arguments.

        Args:
            cam_elev_angle (int, optional):
                The pitch angle where camera starts.
                Defaults to 10.
            cam_elev_speed (float, optional):
                The pitch angle camera steps in one frame.
                It will go back and forth between -30 and 30 degree.
                Defaults to 0.0.
            cam_hori_angle (int, optional):
                The yaw angle where camera starts. Defaults to 45.
            cam_hori_speed (float, optional):
                The yaw angle camera steps in one frame.
                It will go back and forth between 0 and 90 degree.
                Defaults to 0.5.
        """
        self.cam_elevation_args = [cam_elev_angle, cam_elev_speed]
        self.cam_horizon_args = [cam_hori_angle, cam_hori_speed]
        self.if_camera_init = True

    def _get_camera_vector_list(self, frame_number):
        """Generate self.cam_vector_list according to hori and elev arguments.

        Args:
            frame_number (int):
                Number of frames.

        Returns:
            List[List[float, float]]:
                A list of float vectors.
        """
        self.cam_vector_list = [
            [self.cam_elevation_args[0], self.cam_horizon_args[0]],
        ]
        ele_sign = 1
        hor_sign = 1
        for _ in range(frame_number - 1):
            new_ele_angle = ele_sign * self.cam_elevation_args[
                1] + self.cam_vector_list[-1][0]
            #  if elevation angle out of range, go backwards
            if new_ele_angle <= self.cam_elevation_args[
                    1] or new_ele_angle >= 30:
                ele_sign = (-1) * ele_sign
                new_ele_angle = (
                    ele_sign * self.cam_elevation_args[1] +
                    self.cam_vector_list[-1][0])
            new_hor_angle = (
                hor_sign * self.cam_horizon_args[1] +
                self.cam_vector_list[-1][1])
            #  if horizon angle out of range, go backwards
            if new_hor_angle >= 90 - 2 * self.cam_horizon_args[
                    1] or new_hor_angle <= 2 * self.cam_horizon_args[1]:
                hor_sign = (-1) * hor_sign
                new_hor_angle = (
                    hor_sign * self.cam_horizon_args[1] +
                    self.cam_vector_list[-1][1])
            self.cam_vector_list.append([new_ele_angle, new_hor_angle])
        return self.cam_vector_list

    @staticmethod
    def _get_visual_range(points: np.ndarray) -> np.ndarray:
        """Calculate the visual range according to the input points. It make
        sure that no point is absent.

        Args:
            points (np.ndarray):
                An array of 3D points.
                Axis at the last dim.

        Returns:
            np.ndarray:
                An array in shape [3, 2].
                It marks the lower bound and the upper bound
                along each axis.
        """
        axis_num = points.shape[-1]
        axis_stat = np.zeros(shape=[axis_num, 4])
        for axis_index in range(axis_num):
            axis_data = points[..., axis_index]
            axis_min = np.min(axis_data)
            axis_max = np.max(axis_data)
            axis_mid = (axis_min + axis_max) / 2.0
            axis_span = axis_max - axis_min
            axis_stat[axis_index] = np.asarray(
                (axis_min, axis_max, axis_mid, axis_span))
        max_span = np.max(axis_stat[:, 3])
        visual_range = np.zeros(shape=[axis_num, 2])
        for axis_index in range(axis_num):
            visual_range[axis_index, 0] =\
                axis_stat[axis_index, 2] - max_span/2.0
            visual_range[axis_index, 1] =\
                axis_stat[axis_index, 2] + max_span/2.0
        return visual_range

    def _draw_scene(self,
                    visual_range,
                    axis_len=1.0,
                    cam_elev_angle=10,
                    cam_hori_angle=45):
        """Draw an empty scene according to visual range and camera vector.

        Args:
            visual_range (np.ndarray):
                Return value of _get_visual_range().
            axis_len (float, optional):
                The length of every axis.
                Defaults to 1.0.
            cam_elev_angle (int, optional):
                Pitch angle of the camera.
                Defaults to 10.
            cam_hori_angle (int, optional):
                Yaw angle of the camera.
                Defaults to 45.

        Returns:
            list: Figure and Axes3D
        """
        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        ax.set_xlim(*visual_range[0])
        ax.set_ylim(*visual_range[1])
        ax.set_zlim(*visual_range[2])
        ax.view_init(cam_elev_angle, cam_hori_angle)
        mid_point = [
            np.average(visual_range[0]),
            np.average(visual_range[1]),
            np.average(visual_range[2]),
        ]
        # draw axis
        zero_point = np.array([0, 0, 0])
        x_axis = np.array([(visual_range[0][1] - mid_point[0]) * axis_len, 0,
                           0])
        y_axis = np.array(
            [0, (visual_range[1][1] - mid_point[1]) * axis_len, 0])
        z_axis = np.array(
            [0, 0, (visual_range[2][1] - mid_point[2]) * axis_len])
        ax = _plot_line_on_fig(ax, zero_point, x_axis, 'r')
        ax = _plot_line_on_fig(ax, zero_point, y_axis, 'g')
        ax = _plot_line_on_fig(ax, zero_point, z_axis, 'b')
        return fig, ax


class Axes3dJointsRenderer(Axes3dBaseRenderer):
    """Render of joints."""

    def __init__(self):
        self.if_camera_init = False
        self.cam_vector_list = None
        self.if_connection_setup = False
        self.if_frame_updated = False
        self.temp_path = ''

    def set_connections(self, limbs_connection, limbs_palette):
        """set body limbs."""
        self.limbs_connection = limbs_connection
        self.limbs_palette = limbs_palette
        self.if_connection_setup = True

    def render_kp3d_to_video(
        self,
        keypoints_np: np.ndarray,
        output_path: Optional[str] = None,
        convention='opencv',
        fps: Union[float, int] = 30,
        resolution: Iterable[int] = (720, 720),
        visual_range: Iterable[int] = (-100, 100),
        frame_names: Optional[List[str]] = None,
        disable_limbs: bool = False,
        return_array: bool = False,
    ) -> None:
        """Render 3d keypoints to a video.

        Args:
            keypoints_np (np.ndarray): shape of input array should be
                    (f * n * J * 3).
            output_path (str): output video path or frame folder.
            sign (Iterable[int], optional): direction of the axis.
                    Defaults to (1, 1, 1).
            axis (str, optional): axis convention.
                    Defaults to 'xzy'.
            fps (Union[float, int], optional): fps.
                    Defaults to 30.
            resolution (Iterable[int], optional): (width, height) of
                    output video.
                    Defaults to (720, 720).
            visual_range (Iterable[int], optional): range of axis value.
                    Defaults to (-100, 100).
            frame_names (Optional[List[str]], optional):  List of string
                    for frame title, no title if None. Defaults to None.
            disable_limbs (bool, optional): whether need to disable drawing
                limbs.
                Defaults to False.
        Returns:
            None.
        """
        assert self.if_camera_init is True
        assert self.if_connection_setup is True
        sign, axis = enc_camera_convention(convention)
        if output_path is not None:
            if check_path_suffix(output_path, ['.mp4', '.gif']):
                self.temp_path = os.path.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_output_temp')
                mmcv.mkdir_or_exist(self.temp_path)
                print('make dir', self.temp_path)
                self.remove_temp = True
            else:
                self.temp_path = output_path
                self.remove_temp = False
        else:
            self.temp_path = None
        keypoints_np = _set_new_pose(keypoints_np, sign, axis)
        if not self.if_frame_updated:
            if self.cam_vector_list is None:
                self._get_camera_vector_list(
                    frame_number=keypoints_np.shape[0])
            assert len(self.cam_vector_list) == keypoints_np.shape[0]
            if visual_range is None:
                visual_range = self._get_visual_range(keypoints_np)
            else:
                visual_range = np.asarray(visual_range)
                if len(visual_range.shape) == 1:
                    one_dim_visual_range = np.expand_dims(visual_range, 0)
                    visual_range = one_dim_visual_range.repeat(3, axis=0)
            image_array = self._export_frames(keypoints_np, resolution,
                                              visual_range, frame_names,
                                              disable_limbs, return_array)
            self.if_frame_updated = True

        if output_path is not None:
            if check_path_suffix(output_path, '.mp4'):
                images_to_video(
                    self.temp_path,
                    output_path,
                    img_format='frame_%06d.png',
                    fps=fps)
        return image_array

    def _export_frames(self, keypoints_np, resolution, visual_range,
                       frame_names, disable_limbs, return_array):
        """Write output/temp images."""
        image_array = []
        for frame_index in range(keypoints_np.shape[0]):
            keypoints_frame = keypoints_np[frame_index]
            cam_ele, cam_hor = self.cam_vector_list[frame_index]
            fig, ax = \
                self._draw_scene(visual_range=visual_range, axis_len=0.5,
                                 cam_elev_angle=cam_ele,
                                 cam_hori_angle=cam_hor)
            #  draw limbs
            num_person = keypoints_frame.shape[0]
            for person_index, keypoints_person in enumerate(keypoints_frame):
                if num_person >= 2:
                    self.limbs_palette = get_different_colors(
                        num_person)[person_index].reshape(-1, 3)
                if not disable_limbs:
                    for part_name, limbs in self.limbs_connection.items():
                        if part_name == 'body':
                            linewidth = 2
                        else:
                            linewidth = 1
                        if isinstance(self.limbs_palette, np.ndarray):
                            color = self.limbs_palette.astype(
                                np.int32).reshape(-1, 3)
                        elif isinstance(self.limbs_palette, dict):
                            color = np.array(
                                self.limbs_palette[part_name]).astype(np.int32)
                        for limb_index, limb in enumerate(limbs):
                            limb_index = min(limb_index, len(color) - 1)

                            ax = _plot_line_on_fig(
                                ax,
                                keypoints_person[limb[0]],
                                keypoints_person[limb[1]],
                                color=np.array(color[limb_index]) / 255.0,
                                linewidth=linewidth)
                scatter_points_index = list(
                    set(
                        np.array(self.limbs_connection['body']).reshape(
                            -1).tolist()))
                ax.scatter(
                    keypoints_person[scatter_points_index, 0],
                    keypoints_person[scatter_points_index, 1],
                    keypoints_person[scatter_points_index, 2],
                    c=np.array([0, 0, 0]).reshape(1, -1),
                    s=10,
                    marker='o')
            if num_person >= 2:
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])
                ax.zaxis.set_ticklabels([])
                labels = []
                custom_lines = []
                for person_index in range(num_person):
                    color = get_different_colors(
                        num_person)[person_index].reshape(1, 3) / 255.0
                    custom_lines.append(
                        Line2D([0], [0],
                               linestyle='-',
                               color=color[0],
                               lw=2,
                               marker='',
                               markeredgecolor='k',
                               markeredgewidth=.1,
                               markersize=20))
                    labels.append(f'person_{person_index + 1}')
                ax.legend(
                    handles=custom_lines,
                    labels=labels,
                    loc='upper left',
                )
            plt.close('all')
            rgb_mat = _get_cv2mat_from_buf(fig)
            resized_mat = cv2.resize(rgb_mat, resolution)
            if frame_names is not None:
                cv2.putText(
                    resized_mat, str(frame_names[frame_index]),
                    (resolution[0] // 10, resolution[1] // 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * resolution[0] / 500,
                    np.array([255, 255, 255]).astype(np.int32).tolist(), 2)
            if self.temp_path is not None:
                frame_path = os.path.join(self.temp_path,
                                          'frame_%06d.png' % frame_index)
                cv2.imwrite(frame_path, resized_mat)
            if return_array:
                image_array.append(resized_mat[None])
        if return_array:
            image_array = np.concatenate(image_array)
            return image_array
        else:
            return None

    def __del__(self):
        """remove temp images."""
        self.remove_temp_frames()

    def remove_temp_frames(self):
        """remove temp images."""
        if self.temp_path is not None:
            if Path(self.temp_path).is_dir() and self.remove_temp:
                shutil.rmtree(self.temp_path)


def _set_new_pose(pose_np, sign, axis):
    """set new pose with axis convention."""
    target_sign = [-1, 1, -1]
    target_axis = ['x', 'z', 'y']

    pose_rearrange_axis_result = pose_np.copy()
    for axis_index, axis_name in enumerate(target_axis):
        src_axis_index = axis.index(axis_name)
        pose_rearrange_axis_result[..., axis_index] = \
            pose_np[..., src_axis_index]

    for dim_index in range(pose_rearrange_axis_result.shape[-1]):
        pose_rearrange_axis_result[
            ..., dim_index] = sign[dim_index] / target_sign[
                dim_index] * pose_rearrange_axis_result[..., dim_index]
    return pose_rearrange_axis_result


def _plot_line_on_fig(ax,
                      point1_location,
                      point2_location,
                      color,
                      linewidth=1):
    """Draw line on fig with matplotlib."""
    ax.plot([point1_location[0], point2_location[0]],
            [point1_location[1], point2_location[1]],
            [point1_location[2], point2_location[2]],
            color=color,
            linewidth=linewidth)
    return ax


def _get_cv2mat_from_buf(fig, dpi=180):
    """Get numpy image from IO."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
