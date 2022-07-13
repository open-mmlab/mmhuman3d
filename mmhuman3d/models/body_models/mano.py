import numpy as np
import torch
from smplx import MANO as _MANO
from smplx import MANOLayer as _MANOLayer

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)


class MANO(_MANO):
    """Extension of the official MANO implementation."""
    full_pose_keys = {'global_orient', 'hand_pose'}

    NUM_VERTS = 776
    NUM_FACES = 9976

    KpId2manokps = {
        0: 0,  # Wrist
        1: 5,
        2: 6,
        3: 7,  # Index
        4: 9,
        5: 10,
        6: 11,  # Middle
        7: 17,
        8: 18,
        9: 19,  # Pinky
        10: 13,
        11: 14,
        12: 15,  # Ring
        13: 1,
        14: 2,
        15: 3
    }  # Thumb
    kpId2vertices = {
        4: 744,  # Thumb
        8: 320,  # Index
        12: 443,  # Middle
        16: 555,  # Ring
        20: 672  # Pink
    }

    def __init__(self,
                 *args,
                 keypoint_src: str = 'mano',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 **kwargs):
        """
        Args:
            *args: extra arguments for MANO initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            **kwargs: extra keyword arguments for MANO initialization.

        Returns:
            None
        """
        super(MANO, self).__init__(*args, **kwargs)
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
            *args: extra arguments for MANO
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for MANO

        Returns:
            output: contains output parameters and attributes
        """
        if 'right_hand_pose' in kwargs:
            kwargs['hand_pose'] = kwargs['right_hand_pose']
        mano_output = super(MANO, self).forward(*args, **kwargs)
        joints = mano_output.joints

        joints = self.get_keypoints_from_mesh(mano_output.vertices, joints)

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
            global_orient=mano_output.global_orient,
            hand_pose=mano_output.hand_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=mano_output.betas,
        )

        if return_verts:
            output['vertices'] = mano_output.vertices
        if return_full_pose:
            output['full_pose'] = mano_output.full_pose

        return output

    def get_keypoints_from_mesh(self, mesh_vertices, keypoints_regressed):
        """Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5
        mesh vertices for the fingers."""
        batch_size = keypoints_regressed.shape[0]
        keypoints = torch.zeros((batch_size, 21, 3)).cuda()

        # fill keypoints which are regressed
        for manoId, myId in self.KpId2manokps.items():
            keypoints[:, myId, :] = keypoints_regressed[:, manoId, :]
        # get other keypoints from mesh
        for myId, meshId in self.kpId2vertices.items():
            keypoints[:, myId, :] = mesh_vertices[:, meshId, :]

        return keypoints


class MANOLayer(_MANOLayer):
    """Extension of the official MANO implementation."""
    full_pose_keys = {'global_orient', 'hand_pose'}

    NUM_VERTS = 776
    NUM_FACES = 9976

    KpId2manokps = {
        0: 0,  # Wrist
        1: 5,
        2: 6,
        3: 7,  # Index
        4: 9,
        5: 10,
        6: 11,  # Middle
        7: 17,
        8: 18,
        9: 19,  # Pinky
        10: 13,
        11: 14,
        12: 15,  # Ring
        13: 1,
        14: 2,
        15: 3
    }  # Thumb
    kpId2vertices = {
        4: 744,  # Thumb
        8: 320,  # Index
        12: 443,  # Middle
        16: 555,  # Ring
        20: 672  # Pink
    }

    def __init__(self,
                 *args,
                 keypoint_src: str = 'mano',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 **kwargs):
        """
        Args:
            *args: extra arguments for MANO initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            **kwargs: extra keyword arguments for MANO initialization.

        Returns:
            None
        """
        super(MANOLayer, self).__init__(*args, **kwargs)
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
            *args: extra arguments for MANO
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for MANO

        Returns:
            output: contains output parameters and attributes
        """
        if 'right_hand_pose' in kwargs:
            kwargs['hand_pose'] = kwargs['right_hand_pose']
        mano_output = super(MANOLayer, self).forward(*args, **kwargs)
        joints = mano_output.joints

        joints = self.get_keypoints_from_mesh(mano_output.vertices, joints)

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
            global_orient=mano_output.global_orient,
            hand_pose=mano_output.hand_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=mano_output.betas,
        )

        if return_verts:
            output['vertices'] = mano_output.vertices
        if return_full_pose:
            output['full_pose'] = mano_output.full_pose

        return output

    def get_keypoints_from_mesh(self, mesh_vertices, keypoints_regressed):
        """Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5
        mesh vertices for the fingers."""
        batch_size = keypoints_regressed.shape[0]
        keypoints = torch.zeros((batch_size, 21, 3)).cuda()

        # fill keypoints which are regressed
        for manoId, myId in self.KpId2manokps.items():
            keypoints[:, myId, :] = keypoints_regressed[:, manoId, :]
        # get other keypoints from mesh
        for myId, meshId in self.kpId2vertices.items():
            keypoints[:, myId, :] = mesh_vertices[:, meshId, :]

        return keypoints
