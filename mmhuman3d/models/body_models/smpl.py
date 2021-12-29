# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional

import numpy as np
import torch
from smplx import SMPL as _SMPL
from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.models.utils import batch_inverse_kinematics_transform
from mmhuman3d.utils.transforms import quat_to_rotmat, rotmat_to_quat
from ..builder import BODY_MODELS


@BODY_MODELS.register_module(name=['SMPL', 'smpl'])
class SMPL(_SMPL):
    """Extension of the official SMPL implementation."""

    body_pose_keys = {
        'global_orient',
        'body_pose',
    }
    full_pose_keys = {
        'global_orient',
        'body_pose',
    }
    NUM_VERTS = 6890
    NUM_FACES = 13776

    def __init__(self,
                 *args,
                 keypoint_src: str = 'smpl_45',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 **kwargs) -> None:
        """
        Args:
            *args: extra arguments for SMPL initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            joints_regressor: path to joint regressor. Should be a .npy
                file. If provided, replaces the official J_regressor of SMPL.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(SMPL, self).__init__(*args, **kwargs)
        # joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate

        # override the default SMPL joint regressor if available
        if joints_regressor is not None:
            joints_regressor = torch.tensor(
                np.load(joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor', joints_regressor)

        # allow for extra joints to be regressed if available
        if extra_joints_regressor is not None:
            joints_regressor_extra = torch.tensor(
                np.load(extra_joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor_extra',
                                 joints_regressor_extra)

        self.num_verts = self.get_num_verts()
        self.num_joints = get_keypoint_num(convention=self.keypoint_dst)
        self.body_part_segmentation = body_segmentation('smpl')

    def forward(self,
                *args,
                return_verts: bool = True,
                return_full_pose: bool = False,
                **kwargs) -> dict:
        """Forward function.

        Args:
            *args: extra arguments for SMPL
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for SMPL

        Returns:
            output: contains output parameters and attributes
        """

        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)

        if not hasattr(self, 'joints_regressor'):
            joints = smpl_output.joints
        else:
            joints = vertices2joints(self.joints_regressor,
                                     smpl_output.vertices)

        if hasattr(self, 'joints_regressor_extra'):
            extra_joints = vertices2joints(self.joints_regressor_extra,
                                           smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        joints, joint_mask = convert_kps(
            joints,
            src=self.keypoint_src,
            dst=self.keypoint_dst,
            approximate=self.keypoint_approximate)
        if isinstance(joint_mask, np.ndarray):
            joint_mask = torch.tensor(
                joint_mask, dtype=torch.uint8, device=joints.device)

        batch_size = joints.shape[0]
        joint_mask = joint_mask.reshape(1, -1).expand(batch_size, -1)

        output = dict(
            global_orient=smpl_output.global_orient,
            body_pose=smpl_output.body_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=smpl_output.betas)

        if return_verts:
            output['vertices'] = smpl_output.vertices
        if return_full_pose:
            output['full_pose'] = smpl_output.full_pose

        return output

    @classmethod
    def tensor2dict(cls,
                    full_pose: torch.Tensor,
                    betas: Optional[torch.Tensor] = None,
                    transl: Optional[torch.Tensor] = None):
        """Convert full pose tensor to pose dict.

        Args:
            full_pose (torch.Tensor): shape should be (..., 165) or
                (..., 55, 3). All zeros for T-pose.
            betas (Optional[torch.Tensor], optional): shape should be
                (..., 10). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.
            transl (Optional[torch.Tensor], optional): shape should be
                (..., 3). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.
        Returns:
            dict: dict of smpl pose containing transl & betas.
        """
        full_pose = full_pose.view(-1, (cls.NUM_BODY_JOINTS + 1) * 3)
        body_pose = full_pose[:, 3:]
        global_orient = full_pose[:, :3]
        batch_size = full_pose.shape[0]
        betas = betas.view(batch_size, -1) if betas is not None else betas
        transl = transl.view(batch_size, -1) if transl is not None else transl
        return {
            'betas': betas,
            'body_pose': body_pose,
            'global_orient': global_orient,
            'transl': transl,
        }

    @classmethod
    def dict2tensor(cls, smpl_dict: dict) -> torch.Tensor:
        """Convert smpl pose dict to full pose tensor.

        Args:
            smpl_dict (dict): smpl pose dict.

        Returns:
            torch: full pose tensor.
        """
        assert cls.body_pose_keys.issubset(smpl_dict)
        for k in smpl_dict:
            if isinstance(smpl_dict[k], np.ndarray):
                smpl_dict[k] = torch.Tensor(smpl_dict[k])
        global_orient = smpl_dict['global_orient'].view(-1, 3)
        body_pose = smpl_dict['body_pose'].view(-1, 3 * cls.NUM_BODY_JOINTS)
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        return full_pose


@BODY_MODELS.register_module()
class GenderedSMPL(torch.nn.Module):
    """A wrapper of SMPL to handle gendered inputs."""

    def __init__(self,
                 *args,
                 keypoint_src: str = 'smpl_45',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 **kwargs) -> None:
        """
        Args:
            *args: extra arguments for SMPL initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            joints_regressor: path to joint regressor. Should be a .npy
                file. If provided, replaces the official J_regressor of SMPL.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(GenderedSMPL, self).__init__()

        assert 'gender' not in kwargs, \
            self.__class__.__name__ + \
            'does not need \'gender\' for initialization.'

        self.smpl_neutral = SMPL(
            *args,
            gender='neutral',
            keypoint_src=keypoint_src,
            keypoint_dst=keypoint_dst,
            keypoint_approximate=keypoint_approximate,
            joints_regressor=joints_regressor,
            extra_joints_regressor=extra_joints_regressor,
            **kwargs)

        self.smpl_male = SMPL(
            *args,
            gender='male',
            keypoint_src=keypoint_src,
            keypoint_dst=keypoint_dst,
            keypoint_approximate=keypoint_approximate,
            joints_regressor=joints_regressor,
            extra_joints_regressor=extra_joints_regressor,
            **kwargs)

        self.smpl_female = SMPL(
            *args,
            gender='female',
            keypoint_src=keypoint_src,
            keypoint_dst=keypoint_dst,
            keypoint_approximate=keypoint_approximate,
            joints_regressor=joints_regressor,
            extra_joints_regressor=extra_joints_regressor,
            **kwargs)

        self.num_verts = self.smpl_neutral.num_verts
        self.num_joints = self.smpl_neutral.num_joints
        self.faces = self.smpl_neutral.faces

    def forward(self,
                *args,
                betas: torch.Tensor = None,
                body_pose: torch.Tensor = None,
                global_orient: torch.Tensor = None,
                transl: torch.Tensor = None,
                return_verts: bool = True,
                return_full_pose: bool = False,
                gender: torch.Tensor = None,
                device=None,
                **kwargs):
        """Forward function.
        Note:
            B: batch size
            J: number of joints of model, J = 23 (SMPL)
            K: number of keypoints
        Args:
            *args: extra arguments
            betas: Tensor([B, 10]), human body shape parameters of SMPL model.
            body_pose: Tensor([B, J*3] or [B, J, 3, 3]), human body pose
                parameters of SMPL model. It should be axis-angle vector
                ([B, J*3]) or rotation matrix ([B, J, 3, 3)].
            global_orient: Tensor([B, 3] or [B, 1, 3, 3]), global orientation
                of human body. It should be axis-angle vector ([B, 3]) or
                rotation matrix ([B, 1, 3, 3)].
            transl: Tensor([B, 3]), global translation of human body.
            gender: Tensor([B]), gender parameters of human body. -1 for
                neutral, 0 for male , 1 for female.
            device: the device of the output
            **kwargs: extra keyword arguments
        Returns:
            outputs (dict): Dict with mesh vertices and joints.
                - vertices: Tensor([B, V, 3]), mesh vertices
                - joints: Tensor([B, K, 3]), 3d keypoints regressed from
                    mesh vertices.
        """

        batch_size = None
        for attr in [betas, body_pose, global_orient, transl]:
            if attr is not None:
                if device is None:
                    device = attr.device
                if batch_size is None:
                    batch_size = attr.shape[0]
                else:
                    assert batch_size == attr.shape[0]

        if gender is not None:
            output = {
                'vertices':
                torch.zeros([batch_size, self.num_verts, 3], device=device),
                'joints':
                torch.zeros([batch_size, self.num_joints, 3], device=device),
                'joint_mask':
                torch.zeros([batch_size, self.num_joints],
                            dtype=torch.uint8,
                            device=device)
            }

            for body_model, gender_label in \
                    [(self.smpl_neutral, -1),
                     (self.smpl_male, 0),
                     (self.smpl_female, 1)]:
                gender_idxs = gender == gender_label

                # skip if no such gender is present
                if gender_idxs.sum() == 0:
                    continue

                output_model = body_model(
                    betas=betas[gender_idxs] if betas is not None else None,
                    body_pose=body_pose[gender_idxs]
                    if body_pose is not None else None,
                    global_orient=global_orient[gender_idxs]
                    if global_orient is not None else None,
                    transl=transl[gender_idxs] if transl is not None else None,
                    **kwargs)

                output['joints'][gender_idxs] = output_model['joints']

                # TODO: quick fix
                if 'joint_mask' in output_model:
                    output['joint_mask'][gender_idxs] = output_model[
                        'joint_mask']

                if return_verts:
                    output['vertices'][gender_idxs] = output_model['vertices']
                if return_full_pose:
                    output['full_pose'][gender_idxs] = output_model[
                        'full_pose']
        else:
            output = self.smpl_neutral(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                **kwargs)

        return output


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


@BODY_MODELS.register_module(
    name=['HybrIKSMPL', 'HybrIKsmpl', 'hybriksmpl', 'hybrik', 'hybrIK'])
class HybrIKSMPL(SMPL):
    """Extension of the SMPL for HybrIK."""

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis',
        'left_hip',
        'right_hip',  # 2
        'spine1',
        'left_knee',
        'right_knee',  # 5
        'spine2',
        'left_ankle',
        'right_ankle',  # 8
        'spine3',
        'left_foot',
        'right_foot',  # 11
        'neck',
        'left_collar',
        'right_collar',  # 14
        'jaw',  # 15
        'left_shoulder',
        'right_shoulder',  # 17
        'left_elbow',
        'right_elbow',  # 19
        'left_wrist',
        'right_wrist',  # 21
        'left_thumb',
        'right_thumb',  # 23
        'head',
        'left_middle',
        'right_middle',  # 26
        'left_bigtoe',
        'right_bigtoe'  # 28
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self, *args, extra_joints_regressor=None, **kwargs):
        """
        Args:
            *args: extra arguments for SMPL initialization.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(HybrIKSMPL, self).__init__(
            *args,
            extra_joints_regressor=extra_joints_regressor,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            **kwargs)

        self.dtype = torch.float32
        self.num_joints = 29

        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [
            self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES
        ]
        self.SPINE3_IDX = 9
        # # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        # extend kinematic tree
        parents[:24] = self.parents
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]
        self.register_buffer('children_map',
                             self._parents_to_children(parents))
        self.parents = parents

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def forward(self,
                pose_skeleton,
                betas,
                phis,
                global_orient,
                transl=None,
                return_verts=True,
                leaf_thetas=None):
        """Inverse pass for the SMPL model.

        Args:
            pose_skeleton: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint locations in
                (img, Y, Z) format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            phis: torch.tensor, shape Bx23x2
                Rotation on bone axis parameters
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)
            leaf_thetas: torch.tensor, optional, shape Bx5x4
                Quaternions of 5 leaf joints. (default=None)

        Returns
            outputs: output dictionary.
        """
        batch_size = pose_skeleton.shape[0]

        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)

        batch_size = max(betas.shape[0], pose_skeleton.shape[0])
        device = betas.device

        # 1. Add shape contribution
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)

        # 2. Get the rest joints
        # NxJx3 array
        if leaf_thetas is not None:
            rest_J = vertices2joints(self.J_regressor, v_shaped)
        else:
            rest_J = torch.zeros((v_shaped.shape[0], 29, 3),
                                 dtype=self.dtype,
                                 device=device)
            rest_J[:, :24] = vertices2joints(self.J_regressor, v_shaped)

            leaf_number = [411, 2445, 5905, 3216, 6617]
            leaf_vertices = v_shaped[:, leaf_number].clone()
            rest_J[:, 24:] = leaf_vertices

        # 3. Get the rotation matrics
        rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform(
            pose_skeleton,
            global_orient,
            phis,
            rest_J.clone(),
            self.children_map,
            self.parents,
            dtype=self.dtype,
            train=self.training,
            leaf_thetas=leaf_thetas)

        test_joints = True
        if test_joints:
            new_joints, A = batch_rigid_transform(
                rot_mats,
                rest_J[:, :24].clone(),
                self.parents[:24],
                dtype=self.dtype)
        else:
            new_joints = None

        # assert torch.mean(torch.abs(rotate_rest_pose - new_joints)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs) \
            .view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                   dtype=self.dtype,
                                   device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.joints_regressor_extra,
                                            vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.
                                                    root_idx_17, :].unsqueeze(
                                                        1).detach()
            new_joints = new_joints - new_joints[:, self.
                                                 root_idx_smpl, :].unsqueeze(
                                                     1).detach()
            joints_from_verts = joints_from_verts - \
                joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = {
            'vertices': vertices,
            'joints': new_joints,
            'rot_mats': rot_mats,
            'joints_from_verts': joints_from_verts,
        }
        return output
