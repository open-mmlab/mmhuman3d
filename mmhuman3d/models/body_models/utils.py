# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch

from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa


def transform_to_camera_frame(global_orient, transl, pelvis, extrinsic):
    """Transform body model parameters to camera frame.

    Args:
        global_orient (np.ndarray): shape (3, ). Only global_orient and
            transl needs to be updated in the rigid transformation
        transl (np.ndarray): shape (3, ).
        pelvis (np.ndarray): shape (3, ). 3D joint location of pelvis
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

    # take out the small offset from smpl origin to pelvis
    transl_offset = pelvis - transl
    T_p2w = np.eye(4)
    T_p2w[:3, 3] = transl_offset

    # camera extrinsic: transformation from world frame to camera frame
    T_w2c = extrinsic

    # smpl transformation: from vertex frame to world frame
    T_v2p = np.eye(4)
    global_orient_mat = aa_to_rotmat(global_orient)
    T_v2p[:3, :3] = global_orient_mat
    T_v2p[:3, 3] = transl

    # compute combined transformation from vertex to world
    T_v2w = T_p2w @ T_v2p

    # compute transformation from vertex to camera
    T_v2c = T_w2c @ T_v2w

    # decompose vertex to camera transformation
    # np: new pelvis frame
    # T_v2c = T_np2c x T_v2np
    T_np2c = T_p2w
    T_v2np = np.linalg.inv(T_np2c) @ T_v2c

    # decompose into new global orient and new transl
    new_global_orient_mat = T_v2np[:3, :3]
    new_gloabl_orient = rotmat_to_aa(new_global_orient_mat)
    new_transl = T_v2np[:3, 3]

    return new_gloabl_orient, new_transl


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
    T_p2w = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    T_p2w[:, :3, 3] = transl_offset

    # camera extrinsic: transformation from world frame to camera frame
    T_w2c = extrinsic

    # smpl transformation: from vertex frame to world frame
    T_v2p = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
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
    T_v2np = np.linalg.inv(T_np2c) @ T_v2c

    # decompose into new global orient and new transl
    new_global_orient_mat = T_v2np[:, :3, :3]
    new_gloabl_orient = rotmat_to_aa(new_global_orient_mat)
    new_transl = T_v2np[:, :3, 3]

    assert new_gloabl_orient.shape == (N, 3)
    assert new_transl.shape == (N, 3)

    return new_gloabl_orient, new_transl


# Adapted from:
#
# https://github.com/ahmedosman/STAR/blob/master/star/pytorch/star.py
#
#


def quat_feat(theta):
    """Computes a normalized quaternion ([0,0,0,0]  when the body is in rest
    pose) given joint angles.

    :param theta: A tensor of joints axis angles,
        batch size x number of joints x 3
    :return:
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_sin * normalized, v_cos - 1], dim=1)
    return quat


def quat2mat(quat):
    """Converts a quaternion to a rotation matrix.

    :param quat:
    :return:
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]
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


def rodrigues(theta):
    """Computes the rodrigues representation given joint angles.

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
    return quat2mat(quat)


def with_zeros(input):
    """Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor.

    :param input: A tensor of dimensions batch size x 3 x 4
    :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
    """
    batch_size = input.shape[0]
    row_append = torch.cuda.FloatTensor(([0.0, 0.0, 0.0, 1.0]))
    row_append.requires_grad = False
    padded_tensor = torch.cat(
        [input, row_append.view(1, 1, 4).repeat(batch_size, 1, 1)], 1)
    return padded_tensor