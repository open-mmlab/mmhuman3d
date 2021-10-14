# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from smplx import SMPL as _SMPL
# TODO: temporary solution
from smplx.body_models import SMPLOutput
from smplx.lbs import vertices2joints

from ..builder import BODY_MODELS

try:
    from smplx import SMPL as SMPL_
    has_smpl = True
except (ImportError, ModuleNotFoundError):
    has_smpl = False


@BODY_MODELS.register_module()
class SMPL(nn.Module):
    """SMPL 3d human mesh model of paper ref: Matthew Loper.

    ``SMPL: A skinned
    multi-person linear model''. This module is based on the smplx project
    (https://github.com/vchoutas/smplx).
    Args:
        smpl_path (str): The path to the folder where the model weights are
            stored.
        joints_regressor (str): The path to the file where the joints
            regressor weight are stored.
    """

    def __init__(self,
                 smpl_path,
                 joints_regressor,
                 extra_joints_regressor=None):
        super().__init__()

        assert has_smpl, 'Please install smplx to use SMPL.'

        self.smpl_neutral = SMPL_(
            model_path=smpl_path,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            gender='neutral')

        self.smpl_male = SMPL_(
            model_path=smpl_path,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            gender='male')

        self.smpl_female = SMPL_(
            model_path=smpl_path,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            gender='female')

        joints_regressor = torch.tensor(
            np.load(joints_regressor), dtype=torch.float)[None, ...]
        self.register_buffer('joints_regressor', joints_regressor)
        if extra_joints_regressor is not None:
            extra_joints_regressor = torch.tensor(
                np.load(extra_joints_regressor), dtype=torch.float)
        self.register_buffer('extra_joints_regressor', extra_joints_regressor)

        self.num_verts = self.smpl_neutral.get_num_verts()
        self.num_joints = self.joints_regressor.shape[1]

    def smpl_forward(self, model, num_joints, **kwargs):
        """Apply a specific SMPL model with given model parameters.
        Note:
            B: batch size
            V: number of vertices
            K: number of joints
        Returns:
            outputs (dict): Dict with mesh vertices and joints.
                - vertices: Tensor([B, V, 3]), mesh vertices
                - joints: Tensor([B, K, 3]), 3d joints regressed
                    from mesh vertices.
        """

        betas = kwargs['betas']
        batch_size = betas.shape[0]
        device = betas.device
        output = {}
        if batch_size == 0:
            output['vertices'] = betas.new_zeros([0, self.num_verts, 3])
            output['joints'] = betas.new_zeros([0, num_joints, 3])
        else:
            smpl_out = model(**kwargs)
            output['vertices'] = smpl_out.vertices
            if num_joints == 24:
                output['joints'] = torch.matmul(
                    self.joints_regressor.to(device), output['vertices'])
            else:
                joints_mapper = [
                    24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25,
                    26, 27, 28, 29, 30, 31, 32, 33, 34, 8, 5, 45, 46, 4, 7, 21,
                    19, 17, 16, 18, 20, 47, 48, 49, 50, 51, 52, 53, 24, 26, 25,
                    28, 27
                ]
                extra_joints = vertices2joints(self.extra_joints_regressor,
                                               smpl_out.vertices)
                joints = torch.cat([smpl_out.joints, extra_joints], dim=1)
                joints = joints[:, joints_mapper, :]
                output['joints'] = joints

        return output

    def get_faces(self):
        """Return mesh faces.

        Note:
            F: number of faces
        Returns:
            faces: np.ndarray([F, 3]), mesh faces
        """
        return self.smpl_neutral.faces

    def forward(self,
                betas,
                body_pose,
                global_orient,
                num_joints=24,
                pose2rot=True,
                transl=None,
                gender=None):
        """Forward function.
        Note:
            B: batch size
            J: number of controllable joints of model, for smpl model J=23
            K: number of joints
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
                - joints: Tensor([B, K, 3]), 3d joints regressed from
                    mesh vertices.
        """

        batch_size = betas.shape[0]
        if batch_size > 0 and gender is not None:
            output = {
                'vertices': betas.new_zeros([batch_size, self.num_verts, 3]),
                'joints': betas.new_zeros([batch_size, self.num_joints, 3])
            }

            mask = gender < 0
            _out = self.smpl_forward(
                self.smpl_neutral,
                betas=betas[mask],
                body_pose=body_pose[mask],
                global_orient=global_orient[mask],
                transl=transl[mask] if transl is not None else None,
                pose2rot=pose2rot,
                num_joints=num_joints)
            output['vertices'][mask] = _out['vertices']
            output['joints'][mask] = _out['joints']

            mask = gender == 0
            _out = self.smpl_forward(
                self.smpl_male,
                betas=betas[mask],
                body_pose=body_pose[mask],
                global_orient=global_orient[mask],
                transl=transl[mask] if transl is not None else None,
                pose2rot=pose2rot,
                num_joints=num_joints)
            output['vertices'][mask] = _out['vertices']
            output['joints'][mask] = _out['joints']

            mask = gender == 1
            _out = self.smpl_forward(
                self.smpl_male,
                betas=betas[mask],
                body_pose=body_pose[mask],
                global_orient=global_orient[mask],
                transl=transl[mask] if transl is not None else None,
                pose2rot=pose2rot,
                num_joints=num_joints)
            output['vertices'][mask] = _out['vertices']
            output['joints'][mask] = _out['joints']
        else:
            return self.smpl_forward(
                self.smpl_neutral,
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                pose2rot=pose2rot,
                num_joints=num_joints)

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


@BODY_MODELS.register_module()
class SMPL49(_SMPL):
    """Extension of the official SMPL implementation to support more joints."""

    def __init__(self, *args, **kwargs):
        super(SMPL49, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(kwargs['extra_joints_regressor'])
        self.register_buffer(
            'J_regressor_extra',
            torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL49, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra,
                                       smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(
            vertices=smpl_output.vertices,
            global_orient=smpl_output.global_orient,
            body_pose=smpl_output.body_pose,
            joints=joints,
            betas=smpl_output.betas,
            full_pose=smpl_output.full_pose)
        return output
