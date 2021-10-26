# Copyright (c) OpenMMLab. All rights reserved.
# TODO:
# 1. the use of mask in output

import numpy as np
import torch
from smplx import SMPL as _SMPL
from smplx.lbs import vertices2joints

from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.models.builder import BODY_MODELS


@BODY_MODELS.register_module()
class SMPL(_SMPL):
    """Extension of the official SMPL implementation."""

    def __init__(self,
                 *args,
                 keypoint_src='smpl_45',
                 keypoint_dst='human_data_1.0',
                 joints_regressor=None,
                 extra_joints_regressor=None,
                 **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        # joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst

        # update the default SMPL joint regressor if available
        if joints_regressor is not None:
            joints_regressor = torch.tensor(
                np.load(joints_regressor), dtype=torch.float)[None, ...]
            self.register_buffer('J_regressor', joints_regressor)

        # allow for extra joints to be regressed if available
        if extra_joints_regressor is not None:
            J_regressor_extra = np.load(extra_joints_regressor)
            self.register_buffer(
                'J_regressor_extra',
                torch.tensor(J_regressor_extra, dtype=torch.float32))

        # self.joint_map = torch.tensor(joints, dtype=torch.long)
        self.num_verts = self.get_num_verts()
        self.num_joints = self.J_regressor.shape[1]

    def forward(self,
                *args,
                return_verts=True,
                return_full_pose=False,
                **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = smpl_output.joints

        if hasattr(self, 'J_regressor_extra'):
            extra_joints = vertices2joints(self.J_regressor_extra,
                                           smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        # convert keypoints if needed
        joints, joint_mask = convert_kps(
            joints, src=self.keypoint_src, dst=self.keypoint_dst)

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


@BODY_MODELS.register_module()
class GenderedSMPL(SMPL):
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
                 keypoint_dst='human_data_1.0',
                 joints_regressor=None,
                 extra_joints_regressor=None,
                 **kwargs):

        assert 'gender' not in kwargs, \
            self.__class__.__name__ + \
            'does not need \'gender\' for initialization.'

        self.smpl_neutral = SMPL(
            *args,
            gender='neutral',
            keypoint_src=keypoint_src,
            keypoint_dst=keypoint_dst,
            joints_regressor=joints_regressor,
            extra_joints_regressor=extra_joints_regressor,
            **kwargs)

        self.smpl_male = SMPL(
            *args,
            gender='male',
            keypoint_src=keypoint_src,
            keypoint_dst=keypoint_dst,
            joints_regressor=joints_regressor,
            extra_joints_regressor=extra_joints_regressor,
            **kwargs)

        self.smpl_female = SMPL(
            *args,
            gender='female',
            keypoint_src=keypoint_src,
            keypoint_dst=keypoint_dst,
            joints_regressor=joints_regressor,
            extra_joints_regressor=extra_joints_regressor,
            **kwargs)

        self.num_verts = self.smpl_neutral.get_num_verts()
        self.num_joints = self.smpl_neutral.J_regressor.shape[1]
        self.faces = self.smpl_neutral.faces

    def forward(self,
                *args,
                betas,
                body_pose,
                global_orient,
                transl=None,
                return_verts=True,
                return_full_pose=False,
                gender=None,
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
        Returns:
            outputs (dict): Dict with mesh vertices and joints.
                - vertices: Tensor([B, V, 3]), mesh vertices
                - joints: Tensor([B, K, 3]), 3d keypoints regressed from
                    mesh vertices.
        """

        batch_size = betas.shape[0]
        if gender is not None:
            output = {
                'vertices': betas.new_zeros([batch_size, self.num_verts, 3]),
                'joints': betas.new_zeros([batch_size, self.num_joints, 3]),
                'joint_mask': betas.new_zeros([batch_size, self.num_joints])
            }

            for body_model, gender_idx in \
                    [(self.smpl_neutral, -1),
                     (self.smpl_male, 0),
                     (self.smpl_female, 1)]:

                mask = gender == gender_idx

                # skip if no such gender is present
                if mask.sum() == 0:
                    continue

                output_model = body_model(
                    betas=betas[mask],
                    body_pose=body_pose[mask],
                    global_orient=global_orient[mask],
                    transl=transl[mask] if transl is not None else None,
                    **kwargs)

                output['joints'][mask] = output_model['joints']
                output['joint_mask'][mask] = output_model['joint_mask']

                if return_verts:
                    output['vertices'][mask] = output_model['vertices']
                if return_full_pose:
                    output['full_pose'][mask] = output_model['full_pose']

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
