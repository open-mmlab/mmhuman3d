# Copyright (c) OpenMMLab. All rights reserved.
# TODO:
# 1. the use of mask in output
from typing import Optional

import numpy as np
import torch
from smplx import SMPL as _SMPL
from smplx import SMPLX as _SMPLX
from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.models.builder import BODY_MODELS
from mmhuman3d.utils.transforms import quat_to_rotmat, rotmat_to_quat
from .inverse_kinematics import batch_inverse_kinematics_transform


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
                 keypoint_src='smpl_45',
                 keypoint_dst='human_data',
                 keypoint_approximate=False,
                 joints_regressor=None,
                 extra_joints_regressor=None,
                 **kwargs):
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
                return_verts=True,
                return_full_pose=False,
                **kwargs):
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

        output = dict(
            global_orient=smpl_output.global_orient,
            body_pose=smpl_output.body_pose,
            joints=joints,
            joint_mask=joint_mask,
            betas=smpl_output.betas)

        if return_verts:
            output['vertices'] = smpl_output.vertices
        if return_full_pose:
            output['full_pose'] = smpl_output.full_pose

        return output

    @classmethod
    def tensor2dict(cls,
                    full_pose: torch.torch.Tensor,
                    betas: Optional[torch.torch.Tensor] = None,
                    transl: Optional[torch.torch.Tensor] = None):
        """Convert full pose tensor to pose dict.

        Args:
            full_pose (torch.torch.Tensor): shape should be (..., 165) or
                (..., 55, 3). All zeros for T-pose.
            betas (Optional[torch.torch.Tensor], optional): shape should be
                (..., 10). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.
            transl (Optional[torch.torch.Tensor], optional): shape should be
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


@BODY_MODELS.register_module(name=['SMPLX', 'smplx'])
class SMPLX(_SMPLX):
    body_pose_keys = {'global_orient', 'body_pose'}
    full_pose_keys = {
        'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose'
    }
    NUM_VERTS = 10475
    NUM_FACES = 20908

    def __init__(self,
                 *args,
                 keypoint_src='smplx',
                 keypoint_dst='human_data',
                 keypoint_approximate=False,
                 joints_regressor=None,
                 extra_joints_regressor=None,
                 **kwargs):
        super(SMPLX, self).__init__(*args, **kwargs)
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
        self.body_part_segmentation = body_segmentation('smplx')

    def forward(self,
                *args,
                return_verts=True,
                return_full_pose=False,
                **kwargs):
        kwargs['get_skin'] = True
        smplx_output = super(SMPLX, self).forward(*args, **kwargs)

        if not hasattr(self, 'joints_regressor'):
            joints = smplx_output.joints
        else:
            joints = vertices2joints(self.joints_regressor,
                                     smplx_output.vertices)

        if hasattr(self, 'joints_regressor_extra'):
            extra_joints = vertices2joints(self.joints_regressor_extra,
                                           smplx_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        joints, joint_mask = convert_kps(
            joints,
            src=self.keypoint_src,
            dst=self.keypoint_dst,
            approximate=self.keypoint_approximate)
        if isinstance(joint_mask, np.ndarray):
            joint_mask = torch.tensor(
                joint_mask, dtype=torch.uint8, device=joints.device)

        output = dict(
            global_orient=smplx_output.global_orient,
            body_pose=smplx_output.body_pose,
            joints=joints,
            joint_mask=joint_mask,
            betas=smplx_output.betas)

        if return_verts:
            output['vertices'] = smplx_output.vertices
        if return_full_pose:
            output['full_pose'] = smplx_output.full_pose

        return output

    @classmethod
    def tensor2dict(cls,
                    full_pose: torch.torch.Tensor,
                    betas: Optional[torch.torch.Tensor] = None,
                    transl: Optional[torch.torch.Tensor] = None,
                    expression: Optional[torch.torch.Tensor] = None) -> dict:
        """Convert full pose tensor to pose dict.

        Args:
            full_pose (torch.torch.Tensor): shape should be (..., 165) or
                (..., 55, 3). All zeros for T-pose.
            betas (Optional[torch.torch.Tensor], optional): shape should be
                (..., 10). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.
            transl (Optional[torch.torch.Tensor], optional): shape should be
                (..., 3). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.
            expression (Optional[torch.torch.Tensor], optional): shape should
                be (..., 10). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.

        Returns:
            dict: dict of smplx pose containing transl & betas.
        """
        NUM_BODY_JOINTS = cls.NUM_BODY_JOINTS
        NUM_HAND_JOINTS = cls.NUM_HAND_JOINTS
        NUM_FACE_JOINTS = cls.NUM_FACE_JOINTS
        NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
        full_pose = full_pose.view(-1, (NUM_JOINTS + 1), 3)
        global_orient = full_pose[:, :1]
        body_pose = full_pose[:, 1:NUM_BODY_JOINTS + 1]
        jaw_pose = full_pose[:, NUM_BODY_JOINTS + 1:NUM_BODY_JOINTS + 2]
        leye_pose = full_pose[:, NUM_BODY_JOINTS + 2:NUM_BODY_JOINTS + 3]
        reye_pose = full_pose[:, NUM_BODY_JOINTS + 3:NUM_BODY_JOINTS + 4]
        left_hand_pose = full_pose[:, NUM_BODY_JOINTS + 4:NUM_BODY_JOINTS + 19]
        right_hand_pose = full_pose[:,
                                    NUM_BODY_JOINTS + 19:NUM_BODY_JOINTS + 34]
        batch_size = body_pose.shape[0]
        betas = betas.view(batch_size, -1) if betas is not None else betas
        transl = transl.view(batch_size, -1) if transl is not None else transl
        expression = expression.view(
            batch_size, -1) if expression is not None else torch.zeros(
                batch_size, 10)
        return {
            'betas':
            betas,
            'global_orient':
            global_orient.view(batch_size, 3),
            'body_pose':
            body_pose.view(batch_size, NUM_BODY_JOINTS * 3),
            'left_hand_pose':
            left_hand_pose.view(batch_size, NUM_HAND_JOINTS * 3),
            'right_hand_pose':
            right_hand_pose.view(batch_size, NUM_HAND_JOINTS * 3),
            'transl':
            transl,
            'expression':
            expression,
            'jaw_pose':
            jaw_pose.view(batch_size, 3),
            'leye_pose':
            leye_pose.view(batch_size, 3),
            'reye_pose':
            reye_pose.view(batch_size, 3),
        }

    @classmethod
    def dict2tensor(cls, smplx_dict: dict) -> torch.Tensor:
        """Convert smplx pose dict to full pose tensor.

        Args:
            smplx_dict (dict): smplx pose dict.

        Returns:
            torch: full pose tensor.
        """
        assert cls.body_pose_keys.issubset(smplx_dict)
        for k in smplx_dict:
            if isinstance(smplx_dict[k], np.ndarray):
                smplx_dict[k] = torch.Tensor(smplx_dict[k])
        NUM_BODY_JOINTS = cls.NUM_BODY_JOINTS
        NUM_HAND_JOINTS = cls.NUM_HAND_JOINTS
        NUM_FACE_JOINTS = cls.NUM_FACE_JOINTS
        NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
        global_orient = smplx_dict['global_orient'].reshape(-1, 1, 3)
        body_pose = smplx_dict['body_pose'].reshape(-1, NUM_BODY_JOINTS, 3)
        batch_size = global_orient.shape[0]
        jaw_pose = smplx_dict.get('jaw_pose', torch.zeros((batch_size, 1, 3)))
        leye_pose = smplx_dict.get('leye_pose', torch.zeros(
            (batch_size, 1, 3)))
        reye_pose = smplx_dict.get('reye_pose', torch.zeros(
            (batch_size, 1, 3)))
        left_hand_pose = smplx_dict.get(
            'left_hand_pose', torch.zeros((batch_size, NUM_HAND_JOINTS, 3)))
        right_hand_pose = smplx_dict.get(
            'right_hand_pose', torch.zeros((batch_size, NUM_HAND_JOINTS, 3)))
        full_pose = torch.cat([
            global_orient, body_pose,
            jaw_pose.reshape(-1, 1, 3),
            leye_pose.reshape(-1, 1, 3),
            reye_pose.reshape(-1, 1, 3),
            left_hand_pose.reshape(-1, 15, 3),
            right_hand_pose.reshape(-1, 15, 3)
        ],
                              dim=1).reshape(-1, (NUM_JOINTS + 1) * 3)
        return full_pose


@BODY_MODELS.register_module()
class GenderedSMPL(torch.nn.Module):
    """SMPL 3d human mesh model of paper ref: Matthew Loper.

    ``SMPL: A skinned
    multi-person linear model''. This module is based on the smplx project
    (https://github.com/vchoutas/smplx).
    Args:
        model_path (str): The path to the folder where the model weights are
            stored.
        joints_regressor (str): The path to the file where the joint
            regressor weight are stored.
        extra_joints_regressor (str): The path to the file where the extra
            joint regressor weight are stored.
    """

    def __init__(self,
                 *args,
                 keypoint_src='smpl_45',
                 keypoint_dst='human_data',
                 keypoint_approximate=False,
                 joints_regressor=None,
                 extra_joints_regressor=None,
                 **kwargs):
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
                betas=None,
                body_pose=None,
                global_orient=None,
                transl=None,
                return_verts=True,
                return_full_pose=False,
                gender=None,
                device=None,
                **kwargs):
        """Forward function.
        Note:
            B: batch size
            J: number of joints of model, J = 23 (SMPL)
            K: number of keypoints
        Args:
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


FOCAL_LENGTH = 5000.
IMG_RES = 224

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
"""
We create a superset of joints containing the OpenPose joints together with
the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from
every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply
ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',
    'Left Hip',
    'Left Knee',
    'Left Ankle',
    'Right Wrist',
    'Right Elbow',
    'Right Shoulder',
    'Left Shoulder',
    'Left Elbow',
    'Left Wrist',
    'Neck (LSP)',
    'Top of Head (LSP)',
    'Pelvis (MPII)',
    'Thorax (MPII)',
    'Spine (H36M)',
    'Jaw (H36M)',
    'Head (H36M)',
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24,
    'OP Neck': 12,
    'OP RShoulder': 17,
    'OP RElbow': 19,
    'OP RWrist': 21,
    'OP LShoulder': 16,
    'OP LElbow': 18,
    'OP LWrist': 20,
    'OP MidHip': 0,
    'OP RHip': 2,
    'OP RKnee': 5,
    'OP RAnkle': 8,
    'OP LHip': 1,
    'OP LKnee': 4,
    'OP LAnkle': 7,
    'OP REye': 25,
    'OP LEye': 26,
    'OP REar': 27,
    'OP LEar': 28,
    'OP LBigToe': 29,
    'OP LSmallToe': 30,
    'OP LHeel': 31,
    'OP RBigToe': 32,
    'OP RSmallToe': 33,
    'OP RHeel': 34,
    'Right Ankle': 8,
    'Right Knee': 5,
    'Right Hip': 45,
    'Left Hip': 46,
    'Left Knee': 4,
    'Left Ankle': 7,
    'Right Wrist': 21,
    'Right Elbow': 19,
    'Right Shoulder': 17,
    'Left Shoulder': 16,
    'Left Elbow': 18,
    'Left Wrist': 20,
    'Neck (LSP)': 47,
    'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49,
    'Thorax (MPII)': 50,
    'Spine (H36M)': 51,
    'Jaw (H36M)': 52,
    'Head (H36M)': 53,
    'Nose': 24,
    'Left Eye': 26,
    'Right Eye': 25,
    'Left Ear': 28,
    'Right Ear': 27
}

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]

# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [
    0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21,
    20, 23, 22
]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3 * i)
    SMPL_POSE_FLIP_PERM.append(3 * i + 1)
    SMPL_POSE_FLIP_PERM.append(3 * i + 2)
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [
    5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21,
    20, 23, 22
]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15,
                 18, 17, 22, 23, 24, 19, 20, 21] + \
                [25+i for i in J24_FLIP_PERM]


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

        Parameters
        ----------
        pose_skeleton: torch.tensor, optional, shape Bx(J*3)
            It should be a tensor that contains joint locations in
            (img, Y, Z) format. (default=None)
        betas: torch.tensor, optional, shape Bx10
            It can used if shape parameters
            `betas` are predicted from some external model.
            (default=None)
        global_orient: torch.tensor, optional, shape Bx3
            Global Orientations.
        transl: torch.tensor, optional, shape Bx3
            Global Translations.
        return_verts: bool, optional
            Return the vertices. (default=True)

        Returns
        -------
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
