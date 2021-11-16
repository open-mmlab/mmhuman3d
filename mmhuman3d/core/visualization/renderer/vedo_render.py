import numpy as np
import vedo
from scipy.spatial.transform import Rotation as scipy_Rotation


class VedoRenderer(object):
    """An interactive renderer for camera visualization."""

    def __init__(self, scale=0.03):
        """Visualize cameras in an interactive scene supported by vedo.

        Args:
            scale (float, optional):
                Scale factor. Defaults to 0.03.
        """
        self.scale = scale
        self.axis_list = self.__init_axis()
        self.camera_list = []
        self.frames_dir_path = ''
        self.y_reverse = False

    def __init_axis(self, axis_len=80):
        """Prepare arrows for axis.

        Args:
            axis_len (int, optional):
                Length of each axis.
                Defaults to 80.

        Returns:
            List[Arrows]:
                A list of three arrows.
        """
        arrow_end_np = np.eye(3) * axis_len * self.scale
        colors = ['r', 'g', 'b']  # r-x, g-y, b-z
        ret_list = []
        for axis_index in range(3):
            ret_list.append(
                vedo.Arrows([[0, 0, 0]],
                            [arrow_end_np[axis_index]]).c(colors[axis_index]))
        return ret_list

    def set_y_reverse(self):
        """Set y reverse before add_camera if it is needed.

        Vedo defines y+ as up direction. When visualizing kinect cameras, y- is
        up, call set_y_reverse in this situation to make text in correct
        direction.
        """
        self.y_reverse = True
        self.y_reverse_rotation = \
            scipy_Rotation.from_euler('z', 180, degrees=True)

    def add_camera(self, camera_parameter, arrow_len=30):
        """Add an camera to the scene.

        Args:
            camera_parameter (CameraParameter):
                An instance of class CameraParameter which stores
                rotation, translation and name of a camera.
            arrow_len (int, optional):
                Length of the arrow. Defaults to 30.

        Returns:
            list:
                A list of vedo items related to the input camera.
        """
        rot_mat = np.asarray(camera_parameter.get_value('rotation_mat'))
        translation = np.asarray(camera_parameter.get_value('translation'))
        cam_center = -np.linalg.inv(rot_mat).dot(translation)
        arrow_end_origin = np.eye(3) * arrow_len * self.scale
        colors = ['r', 'g', 'b']  # r-x, g-y, b-z
        arrow_end_camera = \
            np.einsum('ij,kj->ki', np.linalg.inv(rot_mat), arrow_end_origin)
        if self.y_reverse:
            cam_center = self.y_reverse_rotation.apply(cam_center)
            for axis_index in range(3):
                arrow_end_camera[axis_index, :] = \
                    self.y_reverse_rotation.apply(
                        arrow_end_camera[axis_index, :]
                    )
        vedo_list = []
        for i in range(3):
            vedo_list.append(
                vedo.Arrows([cam_center],
                            [cam_center + arrow_end_camera[i]]).c(colors[i]))
        vedo_list.append(
            vedo.Text3D(camera_parameter.name, cam_center, s=self.scale * 10))
        self.camera_list += vedo_list
        return vedo_list

    def show(self, with_axis=True, interactive=True):
        """Show cameras as well as axis arrow by vedo.show()

        Args:
            with_axis (bool, optional):
                Whether to show the axis arrow. Defaults to True.
            interactive (bool, optional):
                Pause and interact with window (True) or
                continue execution (False).
                Defaults to True.
        """
        list_to_show = []
        list_to_show += self.camera_list
        if with_axis:
            list_to_show += self.axis_list
        vedo.show(*list_to_show, interactive=interactive, axes=1)
        vedo.clear()
