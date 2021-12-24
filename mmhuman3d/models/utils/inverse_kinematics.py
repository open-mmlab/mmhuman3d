"""This script is based on the release codes:

"HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human
Pose and Shape Estimation. CVPR 2021"
(https://github.com/Jeff-sjtu/HybrIK).
"""

from __future__ import absolute_import, division, print_function

import torch

from mmhuman3d.utils.transforms import aa_to_rotmat


def batch_inverse_kinematics_transform(pose_skeleton,
                                       global_orient,
                                       phis,
                                       rest_pose,
                                       children,
                                       parents,
                                       dtype=torch.float32,
                                       train=False,
                                       leaf_thetas=None):
    """Applies inverse kinematics transform to joints in a batch.

    Args:
        pose_skeleton (torch.tensor):
            Locations of estimated pose skeleton with shape (Bx29x3)
        global_orient (torch.tensor|none):
            Tensor of global rotation matrices with shape (Bx1x3x3)
        phis (torch.tensor):
            Rotation on bone axis parameters with shape (Bx23x2)
        rest_pose (torch.tensor):
            Locations of rest (Template) pose with shape (Bx29x3)
        children (List[int]): list of indexes of kinematic children with len 29
        parents (List[int]): list of indexes of kinematic parents with len 29
        dtype (torch.dtype, optional):
            Data type of the created tensors. Default: torch.float32
        train (bool):
            Store True in train mode. Default: False
        leaf_thetas (torch.tensor, optional):
            Rotation matrixes for 5 leaf joints (Bx5x3x3). Default: None


    Returns:
        rot_mats (torch.tensor):
            Rotation matrics of all joints with shape (Bx29x3x3)
        rotate_rest_pose (torch.tensor):
            Locations of rotated rest/ template pose with shape (Bx29x3)
    """
    batch_size = pose_skeleton.shape[0]
    device = pose_skeleton.device

    rel_rest_pose = rest_pose.clone()
    rel_rest_pose[:, 1:] -= rest_pose[:, parents[1:]].clone()
    rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

    # rotate the T pose
    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    # set up the root
    rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

    rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - \
        rel_pose_skeleton[:, parents[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    # the predicted final pose
    final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
    final_pose_skeleton = final_pose_skeleton - \
        final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    rel_rest_pose = rel_rest_pose
    rel_pose_skeleton = rel_pose_skeleton
    final_pose_skeleton = final_pose_skeleton
    rotate_rest_pose = rotate_rest_pose

    assert phis.dim() == 3
    phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

    # TODO
    if train:
        global_orient_mat = batch_get_pelvis_orient(rel_pose_skeleton.clone(),
                                                    rel_rest_pose.clone(),
                                                    parents, children, dtype)
    else:
        global_orient_mat = batch_get_pelvis_orient_svd(
            rel_pose_skeleton.clone(), rel_rest_pose.clone(), parents,
            children, dtype)

    rot_mat_chain = [global_orient_mat]
    rot_mat_local = [global_orient_mat]
    # leaf nodes rot_mats
    if leaf_thetas is not None:
        leaf_cnt = 0
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

    for i in range(1, parents.shape[0]):
        if children[i] == -1:
            # leaf nodes
            if leaf_thetas is not None:
                rot_mat = leaf_rot_mats[:, leaf_cnt, :, :]
                leaf_cnt += 1

                rotate_rest_pose[:, i] = rotate_rest_pose[:, parents[
                    i]] + torch.matmul(rot_mat_chain[parents[i]],
                                       rel_rest_pose[:, i])

                rot_mat_chain.append(
                    torch.matmul(rot_mat_chain[parents[i]], rot_mat))
                rot_mat_local.append(rot_mat)
        elif children[i] == -3:
            # three children
            rotate_rest_pose[:,
                             i] = rotate_rest_pose[:,
                                                   parents[i]] + torch.matmul(
                                                       rot_mat_chain[
                                                           parents[i]],
                                                       rel_rest_pose[:, i])

            spine_child = []
            for c in range(1, parents.shape[0]):
                if parents[c] == i and c not in spine_child:
                    spine_child.append(c)

            # original
            spine_child = []
            for c in range(1, parents.shape[0]):
                if parents[c] == i and c not in spine_child:
                    spine_child.append(c)

            children_final_loc = []
            children_rest_loc = []
            for c in spine_child:
                temp = final_pose_skeleton[:, c] - rotate_rest_pose[:, i]
                children_final_loc.append(temp)

                children_rest_loc.append(rel_rest_pose[:, c].clone())

            rot_mat = batch_get_3children_orient_svd(children_final_loc,
                                                     children_rest_loc,
                                                     rot_mat_chain[parents[i]],
                                                     spine_child, dtype)

            rot_mat_chain.append(
                torch.matmul(rot_mat_chain[parents[i]], rot_mat))
            rot_mat_local.append(rot_mat)
        else:
            # (B, 3, 1)
            rotate_rest_pose[:,
                             i] = rotate_rest_pose[:,
                                                   parents[i]] + torch.matmul(
                                                       rot_mat_chain[
                                                           parents[i]],
                                                       rel_rest_pose[:, i])
            # (B, 3, 1)
            child_final_loc = final_pose_skeleton[:, children[
                i]] - rotate_rest_pose[:, i]

            if not train:
                orig_vec = rel_pose_skeleton[:, children[i]]
                template_vec = rel_rest_pose[:, children[i]]
                norm_t = torch.norm(template_vec, dim=1, keepdim=True)
                orig_vec = orig_vec * norm_t / torch.norm(
                    orig_vec, dim=1, keepdim=True)

                diff = torch.norm(
                    child_final_loc - orig_vec, dim=1, keepdim=True)
                big_diff_idx = torch.where(diff > 15 / 1000)[0]

                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]

            child_final_loc = torch.matmul(
                rot_mat_chain[parents[i]].transpose(1, 2), child_final_loc)

            child_rest_loc = rel_rest_pose[:, children[i]]
            # (B, 1, 1)
            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)

            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)

            # (B, 3, 1)
            axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
            axis_norm = torch.norm(axis, dim=1, keepdim=True)

            # (B, 1, 1)
            cos = torch.sum(
                child_rest_loc * child_final_loc, dim=1, keepdim=True) / (
                    child_rest_norm * child_final_norm + 1e-8)
            sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            # (B, 3, 1)
            axis = axis / (axis_norm + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                          dim=1).view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            # Convert spin to rot_mat
            # (B, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                          dim=1).view((batch_size, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            # (B, 1, 1)
            cos, sin = torch.split(phis[:, i - 1], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain.append(
                torch.matmul(rot_mat_chain[parents[i]], rot_mat))
            rot_mat_local.append(rot_mat)

    # (B, K + 1, 3, 3)
    rot_mats = torch.stack(rot_mat_local, dim=1)

    return rot_mats, rotate_rest_pose.squeeze(-1)


def batch_get_pelvis_orient_svd(rel_pose_skeleton, rel_rest_pose, parents,
                                children, dtype):
    """Get pelvis orientation svd for batch data.

    Args:
        rel_pose_skeleton (torch.tensor):
            Locations of root-normalized pose skeleton with shape (Bx29x3)
        rel_rest_pose (torch.tensor):
            Locations of rest/ template pose with shape (Bx29x3)
        parents (List[int]): list of indexes of kinematic parents with len 29
        children (List[int]): list of indexes of kinematic children with len 29
        dtype (torch.dtype, optional):
            Data type of the created tensors, the default is torch.float32

    Returns:
        rot_mat (torch.tensor):
            Rotation matrix of pelvis with shape (Bx3x3)
    """
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

    rest_mat = []
    target_mat = []
    for child in pelvis_child:
        rest_mat.append(rel_rest_pose[:, child].clone())
        target_mat.append(rel_pose_skeleton[:, child].clone())

    rest_mat = torch.cat(rest_mat, dim=2)
    target_mat = torch.cat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    mask_zero = S.sum(dim=(1, 2))

    S_non_zero = S[mask_zero != 0].reshape(-1, 3, 3)

    U, _, V = torch.svd(S_non_zero)

    rot_mat = torch.zeros_like(S)
    rot_mat[mask_zero == 0] = torch.eye(3, device=S.device)

    rot_mat_non_zero = torch.bmm(V, U.transpose(1, 2))
    rot_mat[mask_zero != 0] = rot_mat_non_zero

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('rot_mat', rot_mat)

    return rot_mat


def batch_get_pelvis_orient(rel_pose_skeleton, rel_rest_pose, parents,
                            children, dtype):
    """Get pelvis orientation for batch data.

    Args:
        rel_pose_skeleton (torch.tensor):
            Locations of root-normalized pose skeleton with shape (Bx29x3)
        rel_rest_pose (torch.tensor):
            Locations of rest/ template pose with shape (Bx29x3)
        parents (List[int]): list of indexes of kinematic parents with len 29
        children (List[int]): list of indexes of kinematic children with len 29
        dtype (torch.dtype, optional):
            Data type of the created tensors, the default is torch.float32

    Returns:
        rot_mat (torch.tensor):
            Rotation matrix of pelvis with shape (Bx3x3)
    """
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device

    assert children[0] == 3
    pelvis_child = [int(children[0])]
    for i in range(1, parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

    spine_final_loc = rel_pose_skeleton[:, int(children[0])].clone()
    spine_rest_loc = rel_rest_pose[:, int(children[0])].clone()
    # spine_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    # spine_norm = spine_final_loc / (spine_norm + 1e-8)

    # rot_mat_spine = vectors2rotmat(spine_rest_loc, spine_final_loc, dtype)

    # (B, 1, 1)
    vec_final_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    vec_rest_norm = torch.norm(spine_rest_loc, dim=1, keepdim=True)

    spine_norm = spine_final_loc / (vec_final_norm + 1e-8)

    # (B, 3, 1)
    axis = torch.cross(spine_rest_loc, spine_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)
    axis = axis / (axis_norm + 1e-8)
    angle = torch.arccos(
        torch.sum(spine_rest_loc * spine_final_loc, dim=1, keepdim=True) /
        (vec_rest_norm * vec_final_norm + 1e-8))
    axis_angle = (angle * axis).squeeze()
    # aa to rotmat
    rot_mat_spine = aa_to_rotmat(axis_angle)

    assert torch.sum(torch.isnan(rot_mat_spine)) == 0, ('rot_mat_spine',
                                                        rot_mat_spine)
    center_final_loc = 0
    center_rest_loc = 0
    for child in pelvis_child:
        if child == int(children[0]):
            continue
        center_final_loc = center_final_loc + rel_pose_skeleton[:,
                                                                child].clone()
        center_rest_loc = center_rest_loc + rel_rest_pose[:, child].clone()
    center_final_loc = center_final_loc / (len(pelvis_child) - 1)
    center_rest_loc = center_rest_loc / (len(pelvis_child) - 1)

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc)

    center_final_loc = center_final_loc - torch.sum(
        center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(
        center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm

    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(
        center_rest_loc * center_final_loc, dim=1, keepdim=True) / (
            center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat


def batch_get_3children_orient_svd(rel_pose_skeleton, rel_rest_pose,
                                   rot_mat_chain_parent, children_list, dtype):
    """Get pelvis orientation for batch data.

    Args:
        rel_pose_skeleton (torch.tensor):
            Locations of root-normalized pose skeleton with shape (Bx29x3)
        rel_rest_pose (torch.tensor):
            Locations of rest/ template pose with shape (Bx29x3)
        rot_mat_chain_parents (torch.tensor):
            parent's rotation matrix with shape (Bx3x3)
        children (List[int]): list of indexes of kinematic children with len 29
        dtype (torch.dtype, optional):
            Data type of the created tensors, the default is torch.float32

    Returns:
        rot_mat (torch.tensor):
            Child's rotation matrix with shape (Bx3x3)
    """
    rest_mat = []
    target_mat = []
    for c, child in enumerate(children_list):
        if isinstance(rel_pose_skeleton, list):
            target = rel_pose_skeleton[c].clone()
            template = rel_rest_pose[c].clone()
        else:
            target = rel_pose_skeleton[:, child].clone()
            template = rel_rest_pose[:, child].clone()

        target = torch.matmul(rot_mat_chain_parent.transpose(1, 2), target)

        target_mat.append(target)
        rest_mat.append(template)

    rest_mat = torch.cat(rest_mat, dim=2)
    target_mat = torch.cat(target_mat, dim=2)
    S = rest_mat.bmm(target_mat.transpose(1, 2))

    U, _, V = torch.svd(S)

    rot_mat = torch.bmm(V, U.transpose(1, 2))
    assert torch.sum(torch.isnan(rot_mat)) == 0, ('3children rot_mat', rot_mat)
    return rot_mat
