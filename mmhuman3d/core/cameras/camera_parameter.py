import json
from typing import Any, List

import numpy as np
import torch

from mmhuman3d.core.conventions.cameras import (
    convert_cameras,
    convert_K_3x3_to_4x4,
    convert_K_4x4_to_3x3,
)


class CameraParameter:

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

    def set_K_R_T(self,
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

    def set_mat_np(self, mat_key: str, mat_numpy: np.ndarray) -> None:
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

    def set_mat_list(self, mat_key: str, mat_list: List[list]) -> None:
        """Set a matrix-type parameter to mat_list.

        Args:
            mat_key (str):
                Key of the target matrix. in_mat or rotation_mat.
            mat_list (List[list]):
                Matrix in list format.
        Raises:
            KeyError: mat_key not in self.parameters_dict
        """
        if mat_key not in self.parameters_dict:
            raise KeyError(mat_key)
        else:
            self.parameters_dict[mat_key] = mat_list

    def set_value(self, key: str, value: Any) -> None:
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

    def get_mat_np(self, key: str) -> Any:
        """Get a a matrix-type parameter by key.

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

    def load_from_vibe(self,
                       vibe_camera,
                       name: str,
                       batch_index: int = 0) -> None:
        """Load name and parameters from a dict.

        Args:
            vibe_camera (mmhuman3d.core.cameras.
            cameras.WeakPerspectiveCamerasVibe):
                An instance.
            name (str):
                Name of this camera.
        """
        height = self.parameters_dict['H']
        width = self.parameters_dict['W']
        k_4x4 = vibe_camera.K[batch_index:batch_index + 1]  # shape (1, 4, 4)
        r_3x3 = vibe_camera.R[batch_index:batch_index + 1]  # shape (1, 3, 3)
        t_3 = vibe_camera.T[batch_index:batch_index + 1]  # shape (1, 3)
        new_K, new_R, new_T = convert_cameras(
            K=k_4x4,
            R=r_3x3,
            T=t_3,
            is_perspective=False,
            convention_src='pytorch3d',
            convention_dst='opencv',
            resolution_src=(height, width),
            resolution_dst=(height, width))
        k_3x3 = \
            convert_K_4x4_to_3x3(new_K, is_perspective=False)
        k_3x3.numpy().squeeze(0)
        r_3x3 = new_R.numpy().squeeze(0)
        t_3 = new_T.numpy().squeeze(0)
        self.name = name
        self.set_mat_np('in_mat', k_3x3)
        self.set_mat_np('rotation_mat', r_3x3)
        self.set_value('translation', t_3.tolist())

    def get_vibe_dict(self) -> dict:
        """Get a dict of camera parameters, which contains all necessary args
        for mmhuman3d.core.cameras.cameras.WeakPerspectiveCamerasVibe(). Use mm
        human3d.core.cameras.cameras.WeakPerspectiveCamerasVibe(**return_dict)
        to construct a camera.

        Returns:
            dict:
                A dict of camera parameters: name, dist, size, matrix, etc.
        """
        height = self.parameters_dict['H']
        width = self.parameters_dict['W']
        k_3x3 = self.get_mat_np('in_mat')  # shape (3, 3)
        k_3x3 = np.expand_dims(k_3x3, 0)  # shape (1, 3, 3)
        k_4x4 = convert_K_3x3_to_4x4(
            K=k_3x3, is_perspective=False)  # shape (1, 4, 4)
        rotation = self.get_mat_np('rotation_mat')  # shape (3, 3)
        rotation = np.expand_dims(rotation, 0)  # shape (1, 3, 3)
        translation = self.get_value('translation')  # list, len==3
        translation = np.asarray(translation)
        translation = np.expand_dims(translation, 0)  # shape (1, 3)
        new_K, new_R, new_T = convert_cameras(
            K=k_4x4,
            R=rotation,
            T=translation,
            is_perspective=False,
            convention_src='opencv',
            convention_dst='pytorch3d',
            resolution_src=(height, width),
            resolution_dst=(height, width))
        new_K = torch.from_numpy(new_K)
        new_R = torch.from_numpy(new_R)
        new_T = torch.from_numpy(new_T)
        ret_dict = {
            'K': new_K,
            'R': new_R,
            'T': new_T,
        }
        return ret_dict


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
