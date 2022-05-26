from typing import Union

import numpy
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

from mmhuman3d.core.conventions.joints_mapping.standard_joint_angles import (
    TRANSFORMATION_AA_TO_SJA,
    TRANSFORMATION_SJA_TO_AA,
)


class Compose:

    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self,
                 rotation: Union[torch.Tensor, numpy.ndarray],
                 convention: str = 'xyz',
                 **kwargs):
        convention = convention.lower()
        if not (set(convention) == set('xyz') and len(convention) == 3):
            raise ValueError(f'Invalid convention {convention}.')
        if isinstance(rotation, numpy.ndarray):
            data_type = 'numpy'
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = 'tensor'
        else:
            raise TypeError(
                'Type of rotation should be torch.Tensor or numpy.ndarray')
        for t in self.transforms:
            if 'convention' in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == 'numpy':
            rotation = rotation.detach().cpu().numpy()
        return rotation


def aa_to_rotmat(
    axis_angle: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f'Invalid input axis angles shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)


def aa_to_quat(
    axis_angle: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """
    Convert axis_angle to quaternions.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis angles f{axis_angle.shape}.')
    t = Compose([axis_angle_to_quaternion])
    return t(axis_angle)


def ee_to_rotmat(euler_angle: Union[torch.Tensor, numpy.ndarray],
                 convention='xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert euler angle to rotation matrixs.

    Args:
        euler_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(
            f'Invalid input euler angles shape f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix])
    return t(euler_angle, convention.upper())


def rotmat_to_ee(
        matrix: Union[torch.Tensor, numpy.ndarray],
        convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation matrixs to euler angle.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix shape f{matrix.shape}.')
    t = Compose([matrix_to_euler_angles])
    return t(matrix, convention.upper())


def rotmat_to_quat(
    matrix: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation matrixs to quaternions.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion])
    return t(matrix)


def rotmat_to_rot6d(
    matrix: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation matrixs to rotation 6d representations.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)


def quat_to_aa(
    quaternions: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert quaternions to axis angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_axis_angle])
    return t(quaternions)


def quat_to_rotmat(
    quaternions: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert quaternions to rotation matrixs.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(
            f'Invalid input quaternions shape f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix])
    return t(quaternions)


def rot6d_to_rotmat(
    rotation_6d: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation 6d representations to rotation matrixs.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)


def aa_to_ee(axis_angle: Union[torch.Tensor, numpy.ndarray],
             convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert axis angles to euler angle.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f'Invalid input axis_angle shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix, matrix_to_euler_angles])
    return t(axis_angle, convention)


def aa_to_rot6d(
    axis_angle: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert axis angles to rotation 6d representations.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis_angle f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix, matrix_to_rotation_6d])
    return t(axis_angle)


def ee_to_aa(euler_angle: Union[torch.Tensor, numpy.ndarray],
             convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert euler angles to axis angles.

    Args:
        euler_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler_angle f{euler_angle.shape}.')
    t = Compose([
        euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle
    ])
    return t(euler_angle, convention)


def ee_to_quat(euler_angle: Union[torch.Tensor, numpy.ndarray],
               convention='xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert euler angles to quaternions.

    Args:
        euler_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler_angle f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix, matrix_to_quaternion])
    return t(euler_angle, convention)


def ee_to_rot6d(euler_angle: Union[torch.Tensor, numpy.ndarray],
                convention='xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert euler angles to rotation 6d representation.

    Args:
        euler_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler_angle f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix, matrix_to_rotation_6d])
    return t(euler_angle, convention)


def rotmat_to_aa(
    matrix: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)


def quat_to_ee(quaternions: Union[torch.Tensor, numpy.ndarray],
               convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert quaternions to euler angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix, matrix_to_euler_angles])
    return t(quaternions, convention)


def quat_to_rot6d(
    quaternions: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert quaternions to rotation 6d representations.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix, matrix_to_rotation_6d])
    return t(quaternions)


def rot6d_to_aa(
    rotation_6d: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation 6d representations to axis angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([
        rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle
    ])
    return t(rotation_6d)


def rot6d_to_ee(rotation_6d: Union[torch.Tensor, numpy.ndarray],
                convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation 6d representations to euler angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_euler_angles])
    return t(rotation_6d, convention)


def rot6d_to_quat(
    rotation_6d: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation 6d representations to quaternions.

    Args:
        rotation (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(
            f'Invalid input rotation_6d shape f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion])
    return t(rotation_6d)


def aa_to_sja(
    axis_angle: Union[torch.Tensor, numpy.ndarray],
    R_t: Union[torch.Tensor, numpy.ndarray] = TRANSFORMATION_AA_TO_SJA,
    R_t_inv: Union[torch.Tensor, numpy.ndarray] = TRANSFORMATION_SJA_TO_AA
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert axis-angles to standard joint angles.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 21, 3), ndim of input is unlimited.
        R_t (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 21, 3, 3). Transformation matrices from
                original axis-angle coordinate system to
                standard joint angle coordinate system,
                ndim of input is unlimited.
        R_t_inv (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 21, 3, 3). Transformation matrices from
                standard joint angle coordinate system to
                original axis-angle coordinate system,
                ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """

    def _aa_to_sja(aa, R_t, R_t_inv):
        R_aa = axis_angle_to_matrix(aa)
        R_sja = R_t @ R_aa @ R_t_inv
        sja = matrix_to_euler_angles(R_sja, convention='XYZ')
        return sja

    if axis_angle.shape[-2:] != (21, 3):
        raise ValueError(
            f'Invalid input axis angles shape f{axis_angle.shape}.')
    if R_t.shape[-3:] != (21, 3, 3):
        raise ValueError(f'Invalid input R_t shape f{R_t.shape}.')
    if R_t_inv.shape[-3:] != (21, 3, 3):
        raise ValueError(f'Invalid input R_t_inv shape f{R_t.shape}.')
    t = Compose([_aa_to_sja])
    return t(axis_angle, R_t=R_t, R_t_inv=R_t_inv)


def sja_to_aa(
    sja: Union[torch.Tensor, numpy.ndarray],
    R_t: Union[torch.Tensor, numpy.ndarray] = TRANSFORMATION_AA_TO_SJA,
    R_t_inv: Union[torch.Tensor, numpy.ndarray] = TRANSFORMATION_SJA_TO_AA
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert standard joint angles to axis angles.

    Args:
        sja (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 21, 3). ndim of input is unlimited.
        R_t (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 21, 3, 3). Transformation matrices from
                original axis-angle coordinate system to
                standard joint angle coordinate system
        R_t_inv (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 21, 3, 3). Transformation matrices from
                standard joint angle coordinate system to
                original axis-angle coordinate system

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """

    def _sja_to_aa(sja, R_t, R_t_inv):
        R_sja = euler_angles_to_matrix(sja, convention='XYZ')
        R_aa = R_t_inv @ R_sja @ R_t
        aa = quaternion_to_axis_angle(matrix_to_quaternion(R_aa))
        return aa

    if sja.shape[-2:] != (21, 3):
        raise ValueError(f'Invalid input axis angles shape f{sja.shape}.')
    if R_t.shape[-3:] != (21, 3, 3):
        raise ValueError(f'Invalid input R_t shape f{R_t.shape}.')
    if R_t_inv.shape[-3:] != (21, 3, 3):
        raise ValueError(f'Invalid input R_t_inv shape f{R_t.shape}.')
    t = Compose([_sja_to_aa])
    return t(sja, R_t=R_t, R_t_inv=R_t_inv)


def batch_transform_to_camera_frame(global_orient, transl, pelvis, extrinsic):
    """Transform body model parameters to camera frame by batch.

    Args:
        global_orient (np.ndarray): shape (N, 3). Only global_orient and
            transl needs to be updated in the rigid transformation
        transl (np.ndarray): shape (N, 3).
        pelvis (np.ndarray): shape (N, 3). 3D joint location of pelvis
            This is necessary to eliminate the offset from SMPL
            canonical space origin to pelvis, because the global orient
            is conducted around the pelvis, not the canonical space origin
        extrinsic (np.ndarray): shape (4, 4). Transformation matrix
            from world frame to camera frame
    Returns:
        (new_gloabl_orient, new_transl)
            new_gloabl_orient: transformed global orient
            new_transl: transformed transl
    """
    N = len(global_orient)
    assert global_orient.shape == (N, 3)
    assert transl.shape == (N, 3)
    assert pelvis.shape == (N, 3)

    # take out the small offset from smpl origin to pelvis
    transl_offset = pelvis - transl
    T_p2w = numpy.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    T_p2w[:, :3, 3] = transl_offset

    # camera extrinsic: transformation from world frame to camera frame
    T_w2c = extrinsic

    # smpl transformation: from vertex frame to world frame
    T_v2p = numpy.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    global_orient_mat = aa_to_rotmat(global_orient)
    T_v2p[:, :3, :3] = global_orient_mat
    T_v2p[:, :3, 3] = transl

    # compute combined transformation from vertex to world
    T_v2w = T_p2w @ T_v2p

    # compute transformation from vertex to camera
    T_v2c = T_w2c @ T_v2w

    # decompose vertex to camera transformation
    # np: new pelvis frame
    # T_v2c = T_np2c x T_v2np
    T_np2c = T_p2w
    T_v2np = numpy.linalg.inv(T_np2c) @ T_v2c

    # decompose into new global orient and new transl
    new_global_orient_mat = T_v2np[:, :3, :3]
    new_gloabl_orient = rotmat_to_aa(new_global_orient_mat)
    new_transl = T_v2np[:, :3, 3]

    assert new_gloabl_orient.shape == (N, 3)
    assert new_transl.shape == (N, 3)

    return new_gloabl_orient, new_transl


def quat_feat(theta: torch.Tensor) -> torch.Tensor:
    """Computes a normalized quaternion ([0,0,0,0]  when the body is in rest
    pose) given joint angles.

    Args:
        theta (torch.Tensor): A tensor of joints axis angles,
            batch size x number of joints x 3

    Returns:
        quat (torch.Tensor)
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_sin * normalized, v_cos - 1], dim=1)
    return quat


def _quat2mat(quat: torch.Tensor) -> torch.Tensor:
    """Converts a quaternion to a rotation matrix.

    Args:
        quat (torch.Tensor)

    Returns:
        rotMat (torch.Tensor)
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], \
        norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(B, 3, 3)
    return rotMat


def rodrigues(theta: torch.Tensor) -> torch.Tensor:
    """Computes the rodrigues representation given joint angles.

    Parameters
    ----------
    :param theta: batch_size x number of joints x 3
    :return: batch_size x number of joints x 3 x 4
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return _quat2mat(quat)


def with_zeros(input: torch.Tensor) -> torch.Tensor:
    """Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor.

    Parameters
    ----------
    :param input: A tensor of dimensions batch size x 3 x 4
    :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
    """
    batch_size = input.shape[0]
    row_append = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float)
    row_append.requires_grad = False
    padded_tensor = torch.cat(
        [input, row_append.view(1, 1, 4).repeat(batch_size, 1, 1)], 1)
    return padded_tensor
