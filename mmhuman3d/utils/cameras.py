import json

import numpy as np
import torch
from pytorch3d.common.types import Device
from pytorch3d.renderer import cameras
from pytorch3d.transforms import Transform3d


class WeakPerspectiveCameras(cameras.CamerasBase):
    """Inherited from https://github.com/facebookresearch/
    pytorch3d/blob/main/pytorch3d/renderer/cameras.py and mimiced the code
    style.

    And reinplemented functions: compute_projection_matrix,
    get_projection_transform, unproject_points, is_perspective, in_ndc.
    """

    def __init__(
        self,
        scale_x,
        scale_y,
        trans_x,
        trans_y,
        trans_z,
        device: Device = 'cpu',
    ):
        super().__init__(
            device=device,
            scale_x=scale_x,
            scale_y=scale_y,
            trans_x=trans_x,
            trans_y=trans_y,
        )
        K = torch.zeros((self._N, 4, 4),
                        dtype=torch.float32,
                        device=self.device)
        ones = torch.ones(self._N, dtype=torch.float32, device=self.device)

        K[:, 0, 0] = scale_x
        K[:, 1, 1] = scale_y
        K[:, 3, 3] = ones

        K[:, 2, 2] = -1
        self.K = K.to(device)
        self.R = torch.eye(3, 3)[None].repeat(self._N, 1, 1).to(device)
        self.T = torch.zeros(self._N, 3).to(device)
        self.T[:, 0] = trans_x
        self.T[:, 1] = trans_y
        self.T[:, 2] = trans_z

    def compute_projection_matrix(self, scale_x, scale_y, trans_x,
                                  trans_y) -> torch.Tensor:
        K = torch.zeros((self._N, 4, 4),
                        dtype=torch.float32,
                        device=self.device)
        ones = torch.ones(self._N, dtype=torch.float32, device=self.device)

        K[:, 0, 0] = scale_x
        K[:, 1, 1] = scale_y
        K[:, 3, 3] = ones

        K[:, 2, 2] = -1
        return K

    def get_projection_transform(self, **kwargs) -> Transform3d:
        K = kwargs.get('K', self.K)
        if K is not None:
            if K.shape != (self._N, 4, 4):
                msg = 'Expected K to have shape of (%r, 4, 4)'
                raise ValueError(msg % (self._N))
        else:
            K = self.compute_projection_matrix(
                kwargs.get('scale_x', self.scale_x),
                kwargs.get('scale_y', self.scale_y),
                kwargs.get('trans_x', self.trans_x),
                kwargs.get('trans_y', self.trans_y))

        transform = Transform3d(
            matrix=K.transpose(1, 2).contiguous(), device=self.device)
        return transform

    def unproject_points(self,
                         xy_depth: torch.Tensor,
                         world_coordinates: bool = True,
                         scaled_depth_input: bool = False,
                         **kwargs) -> torch.Tensor:
        if world_coordinates:
            to_ndc_transform = self.get_full_projection_transform(
                **kwargs.copy())
        else:
            to_ndc_transform = self.get_projection_transform(**kwargs.copy())

        if scaled_depth_input:
            # the input depth is already scaled
            xy_sdepth = xy_depth
        else:
            K = self.get_projection_transform(**kwargs).get_matrix()
            unsqueeze_shape = [1] * K.dim()
            unsqueeze_shape[0] = K.shape[0]
            mid_z = K[:, 3, 2].reshape(unsqueeze_shape)
            scale_z = K[:, 2, 2].reshape(unsqueeze_shape)
            scaled_depth = scale_z * xy_depth[..., 2:3] + mid_z
            # cat xy and scaled depth
            xy_sdepth = torch.cat((xy_depth[..., :2], scaled_depth), dim=-1)
        # finally invert the transform
        unprojection_transform = to_ndc_transform.inverse()
        return unprojection_transform.transform_points(xy_sdepth)

    def is_perspective(self):
        return False

    def in_ndc(self):
        return True


class CameraParameter:

    def __init__(self, name='default', H=1080, W=1920):
        """
        Args:
            name (str, optional):
                Name of this camera. Defaults to "default".
            H (int, optional):
                Height of a frame, in pixel. Defaults to 1080.
            W (int, optional):
                Width of a frame, in pixel. Defaults to 1920.
        """
        self.name = name
        self.parameters_dict = {}
        in_mat = __zero_mat_list__(3)
        self.parameters_dict['in_mat'] = in_mat
        for distort_name in __distort_coefficient_names__:
            self.parameters_dict[distort_name] = 0.0
        self.parameters_dict['H'] = H
        self.parameters_dict['W'] = W
        r_mat = __zero_mat_list__(3)
        self.parameters_dict['rotation_mat'] = r_mat
        t_list = [0.0, 0.0, 0.0]
        self.parameters_dict['translation'] = t_list

    def reset_distort(self):
        """Reset all distort coefficients to zero."""
        for distort_name in __distort_coefficient_names__:
            self.parameters_dict[distort_name] = 0.0

    def get_opencv_distort_mat(self):
        """Get a numpy array of 8 distort coefficients, which is the distCoeffs
        arg of cv2.undistort.

        Returns:
            ndarray:
                (k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6) of 8 elements.
        """
        dist_coeffs = [
            self.getValue('k1'),
            self.getValue('k2'),
            self.getValue('p1'),
            self.getValue('p2'),
            self.getValue('k3'),
            self.getValue('k4'),
            self.getValue('k5'),
            self.getValue('k6'),
        ]
        dist_coeffs = np.array(dist_coeffs)
        return dist_coeffs

    def set_mat_np(self, mat_key, mat_numpy):
        """Set a matrix-type parameter to mat_numpy.

        Args:
            mat_key (str):
                Key of the target matrix. in_mat or rotation_mat.
            mat_numpy (ndarray):
                Matrix in numpy format.
        Raises:
            KeyError: mat_key not in self.parameters_dict
        """
        if mat_key not in self.parameters_dict:
            raise KeyError(mat_key)
        else:
            self.parameters_dict[mat_key] = mat_numpy.tolist()

    def set_mat_list(self, mat_key, mat_list):
        """Set a matrix-type parameter to mat_list.

        Args:
            mat_key (str):
                Key of the target matrix. in_mat or rotation_mat.
            mat_numpy (ndarray):
                Matrix in list format.
        Raises:
            KeyError: mat_key not in self.parameters_dict
        """
        if mat_key not in self.parameters_dict:
            raise KeyError(mat_key)
        else:
            self.parameters_dict[mat_key] = mat_list

    def set_value(self, key, value):
        """Set a parameter to value.

        Args:
            key (str):
                Name of the parameter.
            value (object):
                New value of the parameter.

        Raises:
            KeyError: key not in self.parameters_dict
        """
        if key not in self.parameters_dict:
            raise KeyError(key)
        else:
            self.parameters_dict[key] = value

    def get_value(self, key):
        """Get a parameter by key.

        Args:
            key (str):
                Name of the parameter.
        Raises:
            KeyError: key not in self.parameters_dict

        Returns:
            object:
                Value of the parameter.
        """
        if key not in self.parameters_dict:
            raise KeyError(key)
        else:
            return self.parameters_dict[key]

    def to_string(self):
        """Convert self.to_dict() to a string.

        Returns:
            str:
                A dict in json string format.
        """
        dump_dict = self.to_dict()
        ret_str = json.dumps(dump_dict)
        return ret_str

    def to_dict(self):
        """Dump camera name and parameters to dict.

        Returns:
            dict:
                Put self.name and self.parameters_dict
                in one dict.
        """
        dump_dict = self.parameters_dict.copy()
        dump_dict['name'] = self.name
        return dump_dict

    def load_from_smc(self, smc_reader, kinect_id):
        """Load name and parameters from an SmcReader instance.

        Args:
            smc_reader (mocap.data_collection.smc_reader.SMCReader):
                An SmcReader instance containing kinect camera parameters.
            kinect_id (int):
                Id of the target kinect.
        """
        name = kinect_id
        extrinsics_dict = \
            smc_reader.get_kinect_color_extrinsics(
                kinect_id, homogeneous=False
            )
        rot_np = extrinsics_dict['R']
        trans_np = extrinsics_dict['T']
        intrinsics_np = \
            smc_reader.get_kinect_color_intrinsics(
                kinect_id
            )
        resolution = \
            smc_reader.get_kinect_color_resolution(
                kinect_id
            )
        rmatrix = np.linalg.inv(rot_np).reshape(3, 3)
        tvec = -np.dot(rmatrix, trans_np)
        self.name = name
        self.set_mat_np('in_mat', intrinsics_np)
        self.set_mat_np('rotation_mat', rmatrix)
        self.set_value('translation', tvec.tolist())
        self.set_value('H', resolution[1])
        self.set_value('W', resolution[0])

    def load_from_dict(self, json_dict):
        """Load name and parameters from a dict.

        Args:
            json_dict (dict):
                A dict comes from self.to_dict().
        """
        for key in json_dict.keys():
            if key == 'name':
                self.name = json_dict[key]
            elif key == 'rotation':
                self.parameters_dict['rotation_mat'] = np.array(
                    json_dict[key]).reshape(3, 3).tolist()
            elif key == 'translation':
                self.parameters_dict[key] = np.array(json_dict[key]).reshape(
                    (3)).tolist()
            else:
                self.parameters_dict[key] = json_dict[key]
                if '_mat' in key:
                    self.parameters_dict[key] = np.array(
                        self.parameters_dict[key]).reshape(3, 3).tolist()

    def load_from_chessboard(self, chessboard_dict, name, inverse=True):
        """Load name and parameters from a dict.

        Args:
            chessboard_dict (dict):
                A dict loaded from json.load(chessboard_file).
            name (str):
                Name of this camera.
            inverse (bool, optional):
                Whether to inverse rotation and translation mat.
                Defaults to False.
        """
        camera_param_dict = \
            __parse_chessboard_param__(chessboard_dict, name, inverse=inverse)
        self.load_from_dict(camera_param_dict)


def __parse_chessboard_param__(chessboard_camera_param, name, inverse=True):
    camera_param_dict = {}
    camera_param_dict['H'] = chessboard_camera_param['imgSize'][1]
    camera_param_dict['W'] = chessboard_camera_param['imgSize'][0]
    camera_param_dict['in_mat'] = chessboard_camera_param['K']
    camera_param_dict['k1'] = 0
    camera_param_dict['k2'] = 0
    camera_param_dict['k3'] = 0
    camera_param_dict['k4'] = 0
    camera_param_dict['k5'] = 0
    camera_param_dict['p1'] = 0
    camera_param_dict['p2'] = 0
    camera_param_dict['name'] = name
    camera_param_dict['rotation'] = chessboard_camera_param['R']
    camera_param_dict['translation'] = chessboard_camera_param['T']
    if inverse:
        rmatrix = np.linalg.inv(
            np.array(camera_param_dict['rotation']).reshape(3, 3))
        camera_param_dict['rotation'] = rmatrix.tolist()
        tmatrix = np.array(camera_param_dict['translation']).reshape((3, 1))
        tvec = -np.dot(rmatrix, tmatrix)
        camera_param_dict['translation'] = tvec.reshape((3)).tolist()
    return camera_param_dict


__distort_coefficient_names__ = [
    'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2'
]


def __zero_mat_list__(n=3):
    ret_list = [[0] * n for _ in range(n)]
    return ret_list
