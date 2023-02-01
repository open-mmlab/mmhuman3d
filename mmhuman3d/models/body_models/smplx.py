# yapf: disable
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from smplx import SMPLX as _SMPLX
from smplx import SMPLXLayer as _SMPLXLayer
from smplx.lbs import blend_shapes, vertices2joints

from mmhuman3d.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from mmhuman3d.core.conventions.keypoints_mapping.mano import (
    MANO_LEFT_REORDER_KEYPOINTS,
    MANO_RIGHT_REORDER_KEYPOINTS,
)
from mmhuman3d.core.conventions.keypoints_mapping.smplx import SMPLX_KEYPOINTS
from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.models.utils.SMPLX import get_partial_smpl

# yapf: enable


class SMPLX(_SMPLX):
    """Extension of the official SMPL-X implementation."""

    body_pose_keys = {'global_orient', 'body_pose'}
    full_pose_keys = {
        'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose'
    }
    NUM_VERTS = 10475
    NUM_FACES = 20908

    def __init__(self,
                 *args,
                 keypoint_src: str = 'smplx',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 **kwargs):
        """
        Args:
            *args: extra arguments for SMPLX initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            joints_regressor: path to joint regressor. Should be a .npy
                file. If provided, replaces the official J_regressor of SMPLX.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPLX initialization.

        Returns:
            None
        """
        super(SMPLX, self).__init__(*args, **kwargs)
        # joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate

        # override the default SMPLX joint regressor if available
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
                return_verts: bool = True,
                return_full_pose: bool = False,
                **kwargs) -> dict:
        """Forward function.

        Args:
            *args: extra arguments for SMPLX
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for SMPLX

        Returns:
            output: contains output parameters and attributes
        """

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

        batch_size = joints.shape[0]
        joint_mask = joint_mask.reshape(1, -1).expand(batch_size, -1)

        output = dict(
            global_orient=smplx_output.global_orient,
            body_pose=smplx_output.body_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=smplx_output.betas)

        if return_verts:
            output['vertices'] = smplx_output.vertices
        if return_full_pose:
            output['full_pose'] = smplx_output.full_pose

        return output

    @classmethod
    def tensor2dict(cls,
                    full_pose: torch.Tensor,
                    betas: Optional[torch.Tensor] = None,
                    transl: Optional[torch.Tensor] = None,
                    expression: Optional[torch.Tensor] = None) -> dict:
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
            expression (Optional[torch.Tensor], optional): shape should
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
        if betas is not None:
            # squeeze or unsqueeze betas to 2 dims
            betas = betas.view(-1, betas.shape[-1])
            if betas.shape[0] == 1:
                betas = betas.repeat(batch_size, 1)
        else:
            betas = betas
        transl = transl.view(batch_size, -1) if transl is not None else transl
        expression = expression.view(
            batch_size, -1) if expression is not None else torch.zeros(
                batch_size, 10).to(body_pose.device)
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


class SMPLXLayer(_SMPLXLayer):
    """Extension of the official SMPL-X implementation."""

    body_pose_keys = {'global_orient', 'body_pose'}
    full_pose_keys = {
        'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
        'jaw_pose', 'leye_pose', 'reye_pose'
    }
    NUM_VERTS = 10475
    NUM_FACES = 20908

    def __init__(self,
                 *args,
                 keypoint_src: str = 'smplx',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 **kwargs):
        """
        Args:
            *args: extra arguments for SMPLX initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            joints_regressor: path to joint regressor. Should be a .npy
                file. If provided, replaces the official J_regressor of SMPLX.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPLX initialization.

        Returns:
            None
        """
        super(SMPLXLayer, self).__init__(*args, **kwargs)
        # joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate

        # override the default SMPLX joint regressor if available
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
                return_verts: bool = True,
                return_full_pose: bool = False,
                **kwargs) -> dict:
        """Forward function.

        Args:
            *args: extra arguments for SMPLX
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for SMPLX

        Returns:
            output: contains output parameters and attributes
        """

        kwargs['get_skin'] = True
        smplx_output = super(SMPLXLayer, self).forward(*args, **kwargs)

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

        batch_size = joints.shape[0]
        joint_mask = joint_mask.reshape(1, -1).expand(batch_size, -1)

        output = dict(
            global_orient=smplx_output.global_orient,
            body_pose=smplx_output.body_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=smplx_output.betas)

        if return_verts:
            output['vertices'] = smplx_output.vertices
        if return_full_pose:
            output['full_pose'] = smplx_output.full_pose
        return output


class GenderedSMPLX(nn.Module):
    """A wrapper of SMPL-X implementation to handle gendered inputs, and
    support more joints.."""

    def __init__(self,
                 smplx_model_dir: str,
                 gender: str = 'neutral',
                 keypoint_src: str = 'smpl_54',
                 keypoint_dst: str = 'pymafx_49',
                 keypoint_approximate: bool = True,
                 extra_joints_regressor: str = None,
                 smplx_to_smpl: str = None,
                 partial_mesh_path: str = 'data/partial_mesh/',
                 batch_size: int = 1,
                 use_face_contour: bool = True,
                 **kwargs) -> None:
        """
        Args:
            smplx_model_dir (str): The path of the SMPLX model.
            gender (str, optional):
                You can choose in ['all', 'male', 'female', 'neutral'].
                If you choose 'all', it will create male, female and neutral
                models. Defaults to 'neutral'.
            keypoint_src (str, optional):
                source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention. Defaults to 'smpl_54'.
            keypoint_dst (str, optional):
                destination convention of keypoints. This convention
                is used for keypoints in the output. Defaults to 'pymafx_49'.
            keypoint_approximate (bool, optional):
                whether to use approximate matching in convention conversion
                for keypoints. Defaults to True.
            extra_joints_regressor (str, optional):
                path to extra joint regressor. Should be a .npy file.
                If provided, extra joints are regressed and concatenated
                after the joints regressed with the official J_regressor
                or joints_regressor. Defaults to None.
            smplx_to_smpl (str, optional):
                The path of the matrix that turns SMPLX into SMPL.
                Defaults to None.
            partial_mesh_path (str, optional):
                The path of the smpl/smplx partial mesh.
            batch_size (int, optional): Defaults to 1.
            use_face_contour (bool, optional): Defaults to True.
        """
        super(GenderedSMPLX, self).__init__()

        if gender == 'all':
            self.genders = ['male', 'female', 'neutral']
        else:
            self.genders = [gender]

        self.model_dict = nn.ModuleDict({
            gender: SMPLXLayer(
                smplx_model_dir,
                gender=gender,
                ext='npz',
                num_betas=10,
                use_pca=False,
                batch_size=batch_size,
                use_face_contour=use_face_contour,
                num_pca_comps=45,
                keypoint_src='smplx',
                keypoint_dst='smplx',
                **kwargs)
            for gender in self.genders
        })
        self.model_neutral = self.model_dict['neutral']

        self.use_face_contour = use_face_contour
        self.keypoint_dst = keypoint_dst
        self.keypoint_src = keypoint_src
        self.keypoint_approximate = keypoint_approximate

        if extra_joints_regressor is not None:
            self.register_buffer(
                'J_regressor_extra',
                torch.tensor(
                    np.load(extra_joints_regressor), dtype=torch.float32))
        if smplx_to_smpl is not None:
            smplx_to_smpl = dict(np.load(smplx_to_smpl))
            self.register_buffer(
                'smplx2smpl',
                torch.tensor(
                    smplx_to_smpl['matrix'][None], dtype=torch.float32))
        if partial_mesh_path is not None:
            smpl2limb_vert_faces = get_partial_smpl(partial_mesh_path)
            self.smpl2lhand = torch.from_numpy(
                smpl2limb_vert_faces['lhand']['vids']).long()
            self.smpl2rhand = torch.from_numpy(
                smpl2limb_vert_faces['rhand']['vids']).long()
        # left and right hand joint mapping
        SMPLX_JOINT_IDS = {
            SMPLX_KEYPOINTS[i]: i
            for i in range(len(SMPLX_KEYPOINTS))
        }
        smplx2lhand_joints = [
            SMPLX_JOINT_IDS[name] for name in MANO_LEFT_REORDER_KEYPOINTS
        ]
        smplx2rhand_joints = [
            SMPLX_JOINT_IDS[name] for name in MANO_RIGHT_REORDER_KEYPOINTS
        ]
        self.smplx2lh_joint_map = torch.tensor(
            smplx2lhand_joints, dtype=torch.long)
        self.smplx2rh_joint_map = torch.tensor(
            smplx2rhand_joints, dtype=torch.long)
        # left and right foot joint mapping
        FOOT_NAMES = ['bigtoe', 'smalltoe', 'heel']
        smplx2lfoot_joints = [
            SMPLX_JOINT_IDS[f'left_{name}'] for name in FOOT_NAMES
        ]
        smplx2rfoot_joints = [
            SMPLX_JOINT_IDS[f'right_{name}'] for name in FOOT_NAMES
        ]
        self.smplx2lf_joint_map = torch.tensor(
            smplx2lfoot_joints, dtype=torch.long)
        self.smplx2rf_joint_map = torch.tensor(
            smplx2rfoot_joints, dtype=torch.long)

        for g in self.genders:
            J_template = torch.einsum('ji,ik->jk', [
                self.model_dict[g].J_regressor[:24],
                self.model_dict[g].v_template
            ])
            J_dirs = torch.einsum('ji,ikl->jkl', [
                self.model_dict[g].J_regressor[:24],
                self.model_dict[g].shapedirs
            ])

            self.register_buffer(f'{g}_J_template', J_template)
            self.register_buffer(f'{g}_J_dirs', J_dirs)

    def forward(self,
                *args,
                betas: torch.Tensor = None,
                body_pose: torch.Tensor = None,
                global_orient: torch.Tensor = None,
                **kwargs):
        """Forward function.
        Note:
            B: batch size
            J: number of joints of model
        Args:
            *args: extra arguments
            betas: Tensor([B, 10]), human body shape parameters of SMPLX model.
            body_pose: Tensor([B, J, 3, 3]), human body pose parameters of
                SMPLX model. It should be rotation matrix ([B, J, 3, 3)].
            global_orient: Tensor([B, 1, 3, 3]), global orientation
                of human body. It should be rotation matrix ([B, 1, 3, 3)].
            **kwargs: extra keyword arguments
        Returns:
            outputs (dict): Dict with mesh vertices and joints.
                - vertices: Tensor([B, V, 3]), mesh vertices
                - joints: Tensor([B, 49, 3]), 3d keypoints regressed from
                    mesh vertices.
                - lhand_joints: Tensor([B, 21, 3]), 3d left hand keypoints.
                - rhand_joints: Tensor([B, 21, 3]), 3d right hand keypoints.
                - lfoot_joints: Tensor([B, 3, 3]), 3d left foot keypoints.
                - rfoot_joints: Tensor([B, 3, 3]), 3d right foot keypoints.
                - face_joints: Tensor([B, 68, 3]), 3d face keypoints.
        """

        batch_size = body_pose.shape[0]
        pose_keys = [
            'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
            'jaw_pose', 'leye_pose', 'reye_pose'
        ]
        param_keys = ['betas'] + pose_keys
        smplx_kwargs = dict(
            betas=betas, body_pose=body_pose, global_orient=global_orient)
        for k in param_keys:
            if k in kwargs:
                smplx_kwargs[k] = kwargs[k]

        if 'gender' not in kwargs:
            smplx_kwargs['gender'] = 2 * torch.ones(batch_size).to(
                body_pose.device)

        if body_pose.shape[1] == 23:
            # remove hand pose in the body_pose
            smplx_kwargs['body_pose'] = smplx_kwargs['body_pose'][:, :21]
        gender_idx_list = []
        smplx_vertices, smplx_joints = [], []
        for gi, g in enumerate(['male', 'female', 'neutral']):
            gender_idx = ((smplx_kwargs['gender'] == gi).nonzero(
                as_tuple=True)[0])
            if len(gender_idx) == 0:
                continue
            gender_idx_list.extend([int(idx) for idx in gender_idx])
            gender_kwargs = {
                k: smplx_kwargs[k][gender_idx]
                for k in param_keys if k in smplx_kwargs
            }
            gender_smplx_output = self.model_dict[g].forward(
                *args, **gender_kwargs)
            smplx_vertices.append(gender_smplx_output['vertices'])
            smplx_joints.append(gender_smplx_output['joints'])

        idx_rearrange = [
            gender_idx_list.index(i)
            for i in range(len(list(gender_idx_list)))
        ]
        idx_rearrange = torch.tensor(idx_rearrange).long().to(body_pose.device)

        smplx_vertices = torch.cat(smplx_vertices)[idx_rearrange]
        smplx_joints = torch.cat(smplx_joints)[idx_rearrange]

        lhand_joints = smplx_joints[:, self.smplx2lh_joint_map]
        rhand_joints = smplx_joints[:, self.smplx2rh_joint_map]
        face_joints = smplx_joints[:, -68:] if self.use_face_contour \
            else smplx_joints[:, -51:]
        lfoot_joints = smplx_joints[:, self.smplx2lf_joint_map]
        rfoot_joints = smplx_joints[:, self.smplx2rf_joint_map]

        smpl_vertices = torch.bmm(
            self.smplx2smpl.expand(batch_size, -1, -1), smplx_vertices)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_vertices)
        # smpl_output.joints: [B, 45, 3]  extra_joints: [B, 9, 3]
        smplx_j45, _ = convert_kps(smplx_joints, src='smplx', dst='smpl_45')
        smpl_54 = torch.cat([smplx_j45, extra_joints], dim=1)
        joints, _ = convert_kps(
            smpl_54,
            src=self.keypoint_src,
            dst=self.keypoint_dst,
            approximate=self.keypoint_approximate)

        output = dict(
            vertices=smpl_vertices,
            joints=joints,
            lhand_joints=lhand_joints,
            rhand_joints=rhand_joints,
            lfoot_joints=lfoot_joints,
            rfoot_joints=rfoot_joints,
            face_joints=face_joints,
        )
        return output

    def get_tpose(self, betas=None, gender=None):
        """Get tpose joints.

        Args:
            betas (betas, optional): Defaults to None.
            gender (str, optional): Defaults to None.

        Returns:
            smplx_joints (torch.Tensor): smplx joints in shape [1, n_joints, 3]
        """
        kwargs = {}
        if betas is None:
            betas = torch.zeros(1, 10).to(self.J_regressor_extra.device)
        kwargs['betas'] = betas

        batch_size = kwargs['betas'].shape[0]
        device = kwargs['betas'].device

        if gender is None:
            kwargs['gender'] = 2 * torch.ones(batch_size).to(device)
        else:
            kwargs['gender'] = gender

        param_keys = ['betas']

        gender_idx_list = []
        smplx_joints = []
        for gi, g in enumerate(['male', 'female', 'neutral']):
            gender_idx = ((kwargs['gender'] == gi).nonzero(as_tuple=True)[0])
            if len(gender_idx) == 0:
                continue
            gender_idx_list.extend([int(idx) for idx in gender_idx])
            gender_kwargs = {}
            gender_kwargs.update(
                {k: kwargs[k][gender_idx]
                 for k in param_keys if k in kwargs})

            J = getattr(self, f'{g}_J_template').unsqueeze(0) + blend_shapes(
                gender_kwargs['betas'], getattr(self, f'{g}_J_dirs'))

            smplx_joints.append(J)

        idx_rearrange = [
            gender_idx_list.index(i)
            for i in range(len(list(gender_idx_list)))
        ]
        idx_rearrange = torch.tensor(idx_rearrange).long().to(device)

        smplx_joints = torch.cat(smplx_joints)[idx_rearrange]
        return smplx_joints
