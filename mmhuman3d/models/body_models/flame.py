import numpy as np
import torch
from smplx import FLAME as _FLAME
from smplx import FLAMELayer as _FLAMELayer

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)


class FLAME(_FLAME):
    """Extension of the official FLAME implementation."""
    head_pose_keys = {'global_orient', 'jaw_pose'}
    full_pose_keys = {
        'global_orient', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose'
    }

    NUM_VERTS = 5023
    NUM_FACES = 9976

    def __init__(self,
                 *args,
                 keypoint_src: str = 'flame',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 **kwargs):
        """
        Args:
            *args: extra arguments for FLAME initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            **kwargs: extra keyword arguments for FLAME initialization.

        Returns:
            None
        """
        super(FLAME, self).__init__(*args, **kwargs)
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate

        self.num_verts = self.get_num_verts()
        self.num_faces = self.get_num_faces()
        self.num_joints = get_keypoint_num(convention=self.keypoint_dst)

    def forward(self,
                *args,
                return_verts: bool = True,
                return_full_pose: bool = False,
                **kwargs) -> dict:
        """Forward function.

        Args:
            *args: extra arguments for FLAME
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for FLAME

        Returns:
            output: contains output parameters and attributes
        """
        flame_output = super(FLAME, self).forward(*args, **kwargs)
        joints = flame_output.joints
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
            global_orient=flame_output.global_orient,
            neck_pose=flame_output.neck_pose,
            jaw_pose=flame_output.jaw_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=flame_output.betas,
            expression=flame_output.expression)

        if return_verts:
            output['vertices'] = flame_output.vertices
        if return_full_pose:
            output['full_pose'] = flame_output.full_pose

        return output


class FLAMELayer(_FLAMELayer):
    """Extension of the official FLAME implementation."""
    head_pose_keys = {'global_orient', 'jaw_pose'}
    full_pose_keys = {
        'global_orient', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose'
    }

    NUM_VERTS = 5023
    NUM_FACES = 9976

    def __init__(self,
                 *args,
                 keypoint_src: str = 'flame',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 **kwargs):
        """
        Args:
            *args: extra arguments for FLAME initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            **kwargs: extra keyword arguments for FLAME initialization.

        Returns:
            None
        """
        super(FLAMELayer, self).__init__(*args, **kwargs)
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate

        self.num_verts = self.get_num_verts()
        self.num_faces = self.get_num_faces()
        self.num_joints = get_keypoint_num(convention=self.keypoint_dst)

    def forward(self,
                *args,
                return_verts: bool = True,
                return_full_pose: bool = False,
                **kwargs) -> dict:
        """Forward function.

        Args:
            *args: extra arguments for FLAME
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for FLAME

        Returns:
            output: contains output parameters and attributes
        """
        flame_output = super(FLAMELayer, self).forward(*args, **kwargs)
        joints = flame_output.joints
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
            global_orient=flame_output.global_orient,
            neck_pose=flame_output.neck_pose,
            jaw_pose=flame_output.jaw_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=flame_output.betas,
            expression=flame_output.expression)

        if return_verts:
            output['vertices'] = flame_output.vertices
        if return_full_pose:
            output['full_pose'] = flame_output.full_pose

        return output
