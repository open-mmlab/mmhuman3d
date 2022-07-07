import numpy as np

from mmhuman3d.utils.transforms import aa_to_rotmat, rotmat_to_aa


def transform_to_camera_frame(global_orient, transl, pelvis, extrinsic):
    """Transform body model parameters to camera frame.

    Args:
        global_orient (numpy.ndarray): shape (3, ). Only global_orient and
            transl needs to be updated in the rigid transformation
        transl (numpy.ndarray): shape (3, ).
        pelvis (numpy.ndarray): shape (3, ). 3D joint location of pelvis
            This is necessary to eliminate the offset from SMPL
            canonical space origin to pelvis, because the global orient
            is conducted around the pelvis, not the canonical space origin
        extrinsic (numpy.ndarray): shape (4, 4). Transformation matrix
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
    new_global_orient = rotmat_to_aa(new_global_orient_mat)
    new_transl = T_v2np[:3, 3]

    return new_global_orient, new_transl


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
    new_global_orient = rotmat_to_aa(new_global_orient_mat)
    new_transl = T_v2np[:, :3, 3]

    assert new_global_orient.shape == (N, 3)
    assert new_transl.shape == (N, 3)

    return new_global_orient, new_transl
