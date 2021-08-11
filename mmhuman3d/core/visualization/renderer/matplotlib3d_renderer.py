import io
import os
import shutil
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mmhuman3d.core.visualization.ffmpeg_utils import images_to_video


class Axes3dBaseRenderer(object):

    def init_camera(self,
                    cam_elev_angle=10,
                    cam_elev_speed=0.0,
                    cam_hori_angle=45,
                    cam_hori_speed=0.5,
                    cam_vector_list=None):
        if cam_vector_list is not None:
            self.cam_vector_list = cam_vector_list
        else:
            self.cam_elevation_args = [cam_elev_angle, cam_elev_speed]
            self.cam_horizon_args = [cam_hori_angle, cam_hori_speed]
        self.if_camera_init = True

    def _get_camera_vector_list(self, frame_number):
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
    def _get_visual_range(self, points: np.ndarray) -> np.ndarray:
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
        fig = plt.figure()
        ax = Axes3D(fig)
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

    def __init__(self):
        self.if_camera_init = False
        self.cam_vector_list = None
        self.if_connection_setup = False
        self.if_frame_updated = False
        self.frames_dir_path = ""

    def set_connections(self, limbs_connection, limbs_palette):
        self.limbs_connection = limbs_connection
        self.limbs_palette = limbs_palette
        self.if_connection_setup = True

    def render_kp3d_to_video(
        self,
        keypoints_np: np.ndarray,
        export_path: str,
        sign: Iterable[int] = (1, 1, 1),
        axis: str = 'xzy',
        fps: Union[float, int] = 30,
        resolution: Iterable[int] = (720, 720),
        visual_range: Iterable[int] = (-100, 100),
        frame_names: Optional[List[str]] = None,
    ) -> NoReturn:
        """Render 3d keypoints to a video.

        Args:
            keypoints_np (np.ndarray): shape of input array should be
                    (f * n * J * 3).
            export_path (str): output video path.
            sign (Iterable[int], optional): direction of the aixs.
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
        Returns:
            NoReturn.
        """
        assert self.if_camera_init is True
        assert self.if_connection_setup is True
        if not Path(export_path).parent.is_dir():
            raise FileNotFoundError('Wrong output video path.')
        temp_dir = os.path.join(
            Path(export_path).parent,
            Path(export_path).name + "_temp")
        self.frames_dir_path = temp_dir
        keypoints_np = _set_new_pose(keypoints_np, sign, axis)
        if not self.if_frame_updated:
            if not os.path.exists(temp_dir) or\
                    not os.path.isdir(temp_dir):
                os.makedirs(temp_dir)
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
            self._export_frames(keypoints_np, temp_dir, resolution,
                                visual_range, self.cam_vector_list,
                                frame_names)
            self.if_frame_updated = True
        images_to_video(
            temp_dir, export_path, img_format="frame_%06d.png", fps=fps)

    def _export_frames(self, keypoints_np, temp_dir, resolution, visual_range,
                       cam_vector_list, frame_names):
        for frame_index in range(keypoints_np.shape[0]):
            keypoints_frame = keypoints_np[frame_index]
            cam_ele, cam_hor = cam_vector_list[frame_index]
            fig, ax = \
                self._draw_scene(visual_range=visual_range, axis_len=0.5,
                                 cam_elev_angle=cam_ele,
                                 cam_hori_angle=cam_hor)
            #  draw limbs

            num_person = keypoints_frame.shape[0]
            for keypoints_person in keypoints_frame:
                if num_person >= 2:
                    self.limbs_palette = np.random.randint(
                        0, high=255, size=(1, 3), dtype=np.uint8)
                for part_name, limbs in self.limbs_connection.items():
                    if part_name == 'body':
                        linewidth = 2
                    else:
                        linewidth = 1
                    if isinstance(self.limbs_palette, np.ndarray):
                        color = self.limbs_palette.astype(np.int32)
                    elif isinstance(self.limbs_palette, dict):
                        color = np.array(self.limbs_palette[part_name]).astype(
                            np.int32)
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
            plt.close("all")
            rgb_mat = _get_cv2mat_from_buf(fig)
            resized_mat = cv2.resize(rgb_mat, resolution)
            if frame_names is not None:
                cv2.putText(
                    resized_mat, str(frame_names[frame_index]),
                    (resolution[0] // 10, resolution[1] // 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * resolution[0] / 500,
                    np.array([255, 255, 255]).astype(np.int32).tolist(), 2)
            frame_path = os.path.join(temp_dir, "frame_%06d.png" % frame_index)
            cv2.imwrite(frame_path, resized_mat)

    def __del__(self):
        self.remove_temp_frames()

    def remove_temp_frames(self):
        if len(self.frames_dir_path) > 0 and \
                os.path.exists(self.frames_dir_path) and \
                os.path.isdir(self.frames_dir_path):
            shutil.rmtree(self.frames_dir_path)


def _set_new_pose(pose_np, sign, axis):
    for dim_index in range(pose_np.shape[-1]):
        pose_np[..., dim_index] = \
            sign[dim_index] * pose_np[..., dim_index]
    pose_rearrange_axis_result = pose_np.copy()
    target_axis = ["x", "y", "z"]
    for axis_index, axis_name in enumerate(target_axis):
        src_axis_index = axis.index(axis_name)
        pose_rearrange_axis_result[..., axis_index] = \
            pose_np[..., src_axis_index]
    return pose_rearrange_axis_result


def _plot_line_on_fig(ax,
                      point1_location,
                      point2_location,
                      color,
                      linewidth=1):
    ax.plot([point1_location[0], point2_location[0]],
            [point1_location[1], point2_location[1]],
            [point1_location[2], point2_location[2]],
            color=color,
            linewidth=linewidth)
    return ax


def _get_cv2mat_from_buf(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
