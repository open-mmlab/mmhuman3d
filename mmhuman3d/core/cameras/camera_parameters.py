import json
import warnings
from typing import Any, List, Tuple, Union

import numpy as np
import torch

from mmhuman3d.core.cameras.cameras import PerspectiveCameras
from mmhuman3d.core.conventions.cameras import (
    convert_cameras,
    convert_K_3x3_to_4x4,
    convert_K_4x4_to_3x3,
)
from .builder import build_cameras

_CAMERA_PARAMETER_SUPPORTED_KEYS_ = {
    'H': {
        'type': int,
    },
    'W': {
        'type': int,
    },
    'in_mat': {
        'type': list,
        'len': 3,
    },
    'rotation_mat': {
        'type': list,
        'len': 3,
    },
    'translation': {
        'type': list,
        'len': 3,
    },
    'k1': {
        'type': float,
    },
    'k2': {
        'type': float,
    },
    'k3': {
        'type': float,
    },
    'k4': {
        'type': float,
    },
    'k5': {
        'type': float,
    },
    'k6': {
        'type': float,
    },
    'p1': {
        'type': float,
    },
    'p2': {
        'type': float,
    },
}


class CameraParameter:
    logger = None
    SUPPORTED_KEYS = _CAMERA_PARAMETER_SUPPORTED_KEYS_

    def __init__(self,
                 name: str = 'default',
                 H: int = 1080,
                 W: int = 1920) -> None:
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
        self.check_item('H', H)
        self.parameters_dict['H'] = H
        self.check_item('W', W)
        self.parameters_dict['W'] = W
        r_mat = __zero_mat_list__(3)
        self.parameters_dict['rotation_mat'] = r_mat
        t_list = [0.0, 0.0, 0.0]
        self.parameters_dict['translation'] = t_list

    def reset_distort(self) -> None:
        """Reset all distort coefficients to zero."""
        for distort_name in __distort_coefficient_names__:
            self.parameters_dict[distort_name] = 0.0

    def get_opencv_distort_mat(self) -> np.ndarray:
        """Get a numpy array of 8 distort coefficients, which is the distCoeffs
        arg of cv2.undistort.

        Returns:
            ndarray:
                (k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6) of 8 elements.
        """
        dist_coeffs = [
            self.get_value('k1'),
            self.get_value('k2'),
            self.get_value('p1'),
            self.get_value('p2'),
            self.get_value('k3'),
            self.get_value('k4'),
            self.get_value('k5'),
            self.get_value('k6'),
        ]
        dist_coeffs = np.array(dist_coeffs)
        return dist_coeffs

    def set_KRT(self,
                K_mat: np.ndarray,
                R_mat: np.ndarray,
                T_vec: np.ndarray,
                inverse_extrinsic: bool = False) -> None:
        """Set intrinsic and extrinsic of a camera.

        Args:
            K_mat (np.ndarray):
                In shape [3, 3].
            R_mat (np.ndarray):
                Rotation from world to view in default.
                In shape [3, 3].
            T_vec (np.ndarray):
                Translation from world to view in default.
                In shape [3,].
            inverse_extrinsic (bool, optional):
                If true, R_mat and T_vec transform a point
                from view to world. Defaults to False.
        """
        k_shape = K_mat.shape
        assert k_shape[0] == k_shape[1] == 3
        r_shape = R_mat.shape
        assert r_shape[0] == r_shape[1] == 3
        assert T_vec.ndim == 1 and T_vec.shape[0] == 3
        self.set_mat_np('in_mat', K_mat)
        if inverse_extrinsic:
            R_mat = np.linalg.inv(R_mat)
            T_vec = -np.dot(R_mat, T_vec).reshape((3))
        self.set_mat_np('rotation_mat', R_mat)
        self.set_value('translation', T_vec.tolist())

    def get_KRT(self, k_dim=3) -> List[np.ndarray]:
        """Get intrinsic and extrinsic of a camera.

        Args:
            k_dim (int, optional):
                Dimension of the returned mat K.
                Defaults to 3.

        Raises:
            ValueError: k_dim is neither 3 nor 4.

        Returns:
            List[np.ndarray]:
                K_mat (np.ndarray):
                    In shape [3, 3].
                R_mat (np.ndarray):
                    Rotation from world to view in default.
                    In shape [3, 3].
                T_vec (np.ndarray):
                    Translation from world to view in default.
                    In shape [3,].
        """
        K_3x3 = self.get_mat_np('in_mat')
        R_mat = self.get_mat_np('rotation_mat')
        T_vec = np.asarray(self.get_value('translation'))
        if k_dim == 3:
            return [K_3x3, R_mat, T_vec]
        elif k_dim == 4:
            K_3x3 = np.expand_dims(K_3x3, 0)  # shape (1, 3, 3)
            K_4x4 = convert_K_3x3_to_4x4(
                K=K_3x3, is_perspective=True)  # shape (1, 4, 4)
            K_4x4 = K_4x4[0, :, :]
            return [K_4x4, R_mat, T_vec]
        else:
            raise ValueError(f'K mat cannot be converted to {k_dim}x{k_dim}')

    def set_mat_np(self, mat_key: str, mat_numpy: np.ndarray) -> None:
        """Set a matrix-type parameter to mat_numpy.

        Args:
            mat_key (str):
                Key of the target matrix. in_mat or rotation_mat.
            mat_numpy (ndarray):
                Matrix in numpy format.

            Raises:
                TypeError:
                    mat_numpy is not an np.ndarray.
        """
        if not isinstance(mat_numpy, np.ndarray):
            raise TypeError
        self.set_mat_list(mat_key, mat_numpy.tolist())

    def set_mat_list(self, mat_key: str, mat_list: List[list]) -> None:
        """Set a matrix-type parameter to mat_list.

        Args:
            mat_key (str):
                Key of the target matrix. in_mat or rotation_mat.
            mat_list (List[list]):
                Matrix in list format.
        """
        self.check_item(mat_key, mat_list)
        self.parameters_dict[mat_key] = mat_list

    def set_value(self, key: str, value: Any) -> None:
        """Set a parameter to value.

        Args:
            key (str):
                Name of the parameter.
            value (object):
                New value of the parameter.
        """
        self.check_item(key, value)
        self.parameters_dict[key] = value

    def get_value(self, key: str) -> Any:
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

    def get_mat_np(self, key: str) -> np.ndarray:
        """Get a a matrix-type parameter by key.

        Args:
            key (str):
                Name of the parameter.
        Raises:
            KeyError: key not in self.parameters_dict

        Returns:
            ndarray:
                Value of the parameter.
        """
        if key not in self.parameters_dict:
            raise KeyError(key)
        else:
            mat_list = self.parameters_dict[key]
            mat_np = np.array(mat_list).reshape((3, 3))
            return mat_np

    def to_string(self) -> str:
        """Convert self.to_dict() to a string.

        Returns:
            str:
                A dict in json string format.
        """
        dump_dict = self.to_dict()
        ret_str = json.dumps(dump_dict)
        return ret_str

    def to_dict(self) -> dict:
        """Dump camera name and parameters to dict.

        Returns:
            dict:
                Put self.name and self.parameters_dict
                in one dict.
        """
        dump_dict = self.parameters_dict.copy()
        dump_dict['name'] = self.name
        return dump_dict

    def dump(self, json_path: str) -> None:
        """Dump camera name and parameters to a file.

        Returns:
            dict:
                Put self.name and self.parameters_dict
                in one dict, and dump them to a json file.
        """
        dump_dict = self.to_dict()
        with open(json_path, 'w') as f_write:
            json.dump(dump_dict, f_write)

    def load(self, json_path: str) -> None:
        """Load camera name and parameters from a file."""
        with open(json_path, 'r') as f_read:
            dumped_dict = json.load(f_read)
        self.load_from_dict(dumped_dict)

    def load_from_dict(self, json_dict: dict) -> None:
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

    def load_from_chessboard(self,
                             chessboard_dict: dict,
                             name: str,
                             inverse: bool = True) -> None:
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

    @classmethod
    def load_from_perspective_cameras(cls,
                                      cam,
                                      name: str,
                                      resolution: Union[List, Tuple] = None):
        """Load parameters from a PerspectiveCameras and return a
        CameraParameter.

        Args:
            cam (mmhuman3d.core.cameras.cameras.PerspectiveCameras):
                An instance.
            name (str):
                Name of this camera.
        """
        assert isinstance(cam, PerspectiveCameras
                          ), 'Wrong input, support PerspectiveCameras only!'
        if len(cam) > 1:
            warnings.warn('Will only use the first camera in the batch.')
        cam = cam[0]

        resolution = resolution if resolution is not None else cam.resolution[
            0].tolist()

        height, width = int(resolution[0]), int(resolution[1])

        cam_param = CameraParameter()
        cam_param.__init__(H=height, W=width, name=name)

        k_4x4 = cam.K  # shape (1, 4, 4)
        r_3x3 = cam.R  # shape (1, 3, 3)
        t_3 = cam.T  # shape (1, 3)
        is_perspective = cam.is_perspective()
        in_ndc = cam.in_ndc()

        k_4x4, r_3x3, t_3 = convert_cameras(
            K=k_4x4,
            R=r_3x3,
            T=t_3,
            is_perspective=False,
            in_ndc_dst=False,
            in_ndc_src=in_ndc,
            convention_src='pytorch3d',
            convention_dst='opencv',
            resolution_src=(height, width),
            resolution_dst=(height, width))

        k_3x3 = \
            convert_K_4x4_to_3x3(k_4x4, is_perspective=is_perspective)

        k_3x3 = k_3x3.numpy()[0]
        r_3x3 = r_3x3.numpy()[0]
        t_3 = t_3.numpy()[0]
        cam_param.name = name
        cam_param.set_mat_np('in_mat', k_3x3)
        cam_param.set_mat_np('rotation_mat', r_3x3)
        cam_param.set_value('translation', t_3.tolist())
        cam_param.parameters_dict.update(H=height)
        cam_param.parameters_dict.update(W=width)
        return cam_param

    def export_to_perspective_cameras(self) -> PerspectiveCameras:
        """Export to a opencv defined screen space PerspectiveCameras.

        Returns:
            Same defined PerspectiveCameras of batch_size 1.
        """
        height = self.parameters_dict['H']
        width = self.parameters_dict['W']
        k_4x4, rotation, translation = self.get_KRT(k_dim=4)
        k_4x4 = np.expand_dims(k_4x4, 0)  # shape (1, 3, 3)
        rotation = np.expand_dims(rotation, 0)  # shape (1, 3, 3)
        translation = np.expand_dims(translation, 0)  # shape (1, 3)
        new_K = torch.from_numpy(k_4x4)
        new_R = torch.from_numpy(rotation)
        new_T = torch.from_numpy(translation)
        cam = build_cameras(
            dict(
                type='PerspectiveCameras',
                K=new_K.float(),
                R=new_R.float(),
                T=new_T.float(),
                convention='opencv',
                in_ndc=False,
                resolution=(height, width)))
        return cam

    def check_item(self, key: Any, val: Any) -> None:
        """Check whether the key and its value matches definition in
        CameraParameter.SUPPORTED_KEYS.

        Args:
            key (Any):
                Key in CameraParameter.
            val (Any):
                Value to the key.

        Raises:
            KeyError:
                key cannot be found in
                CameraParameter.SUPPORTED_KEYS.
            TypeError:
                Value's type doesn't match definition.
        """
        self.__check_key__(key)
        self.__check_value_type__(key, val)

    def __check_key__(self, key: Any) -> None:
        """Check whether the key matches definition in
        CameraParameter.SUPPORTED_KEYS.

        Args:
            key (Any):
                Key in CameraParameter.

        Raises:
            KeyError:
                key cannot be found in
                CameraParameter.SUPPORTED_KEYS.
        """
        if key not in self.__class__.SUPPORTED_KEYS:
            err_msg = 'Key check failed in CameraParameter:\n'
            err_msg += f'key={str(key)}\n'
            raise KeyError(err_msg)

    def __check_value_type__(self, key: Any, val: Any) -> None:
        """Check whether the type of value matches definition in
        CameraParameter.SUPPORTED_KEYS.

        Args:
            key (Any):
                Key in CameraParameter.
            val (Any):
                Value to the key.

        Raises:
            TypeError:
                Value is supported but doesn't match definition.
        """
        if type(val) != self.__class__.SUPPORTED_KEYS[key]['type']:
            err_msg = 'Type check failed in CameraParameter:\n'
            err_msg += f'key={str(key)}\n'
            err_msg += f'type(val)={type(val)}\n'
            raise TypeError(err_msg)


def __parse_chessboard_param__(chessboard_camera_param, name, inverse=True):
    """Parse a dict loaded from chessboard file into another dict needed by
    CameraParameter.

    Args:
        chessboard_camera_param (dict):
            A dict loaded from json.load(chessboard_file).
        name (str):
            Name of this camera.
        inverse (bool, optional):
            Whether to inverse rotation and translation mat.
            Defaults to True.

    Returns:
        dict:
            A dict of parameters in CameraParameter.to_dict() format.
    """
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
    """Return a zero mat in list format.

    Args:
        n (int, optional):
            Length of the edge.
            Defaults to 3.

    Returns:
        list:
            List[List[int]]
    """
    ret_list = [[0] * n for _ in range(n)]
    return ret_list
