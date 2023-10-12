from abc import ABCMeta
from typing import Optional, Tuple, Union
import torchgeometry as tgm

import torch
import torch.nn as nn
from torch.nn import functional as F
from ..heads.smplerx_head import PositionNet, HandRotationNet, FaceRegressor, BoxNet, HandRoI, BodyRotationNet

from mmhuman3d.models.utils.human_models import smpl_x
from mmhuman3d.utils.transforms import rot6d_to_aa, rotmat_to_aa
import math
import mmcv
import copy

from ..backbones.builder import build_backbone
from .base_architecture import BaseArchitecture


class SMPLer_X(BaseArchitecture, metaclass=ABCMeta):

    def __init__(self,
                 backbone = None,
                 focal = (5000, 5000) ,
                 camera_3d_size = 2.5,
                 input_img_shape = (384, 512),
                 input_body_shape = (256, 192),
                 input_hand_shape = (256, 256),
                 input_face_shape =  (192, 192),
                 output_hm_shape = (16, 16, 12),
                 output_hand_hm_shape = (16, 16, 16),
                 output_face_hm_shape = (8, 8, 8),
                 testset = 'EHF',
                 princpt = (96, 128),
                 feat_dim = 1280,
                 upscale = 4,
                 device: str = 'cpu',
                 init_cfg: Optional[Union[list, dict, None]] = None):
        super(SMPLer_X, self).__init__(init_cfg)
        self.backbone = backbone
        self.focal = focal
        self.camera_3d_size = camera_3d_size
        self.input_img_shape = input_img_shape
        self.input_body_shape = input_body_shape
        self.input_hand_shape = input_hand_shape
        self.input_face_shape = input_face_shape
        self.output_hm_shape = output_hm_shape
        self.output_hand_hm_shape = output_hand_hm_shape
        self.output_face_hm_shape = output_face_hm_shape
        self.testset = testset
        self.princpt = princpt
        self.feat_dim = feat_dim
        self.upscale = upscale
        self.device = torch.device(device)
        self._create_encoder()
        self._prepare_body_module()
        self._prepare_hand_module()
        self._prepare_face_module()

        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
        self.body_num_joints = len(smpl_x.pos_joint_part['body'])
        self.hand_joint_num = len(smpl_x.pos_joint_part['rhand'])

        self.neck = [self.box_net, self.hand_roi_net]

        self.head = [self.body_position_net, self.body_regressor,
                    self.hand_position_net, self.hand_regressor,
                    self.face_regressor]

        self.trainable_modules = [self.encoder, self.body_position_net, self.body_regressor,
                                  self.box_net, self.hand_position_net,
                                  self.hand_roi_net, self.hand_regressor, self.face_regressor]
        self.special_trainable_modules = []

        # backbone:
        param_bb = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        # neck
        param_neck = 0
        for module in self.neck:
            param_neck += sum(p.numel() for p in module.parameters() if p.requires_grad)
        # head
        param_head = 0
        for module in self.head:
            param_head += sum(p.numel() for p in module.parameters() if p.requires_grad)

        param_net = param_bb + param_neck + param_head

        # print('#parameters:')
        # print(f'{param_bb}, {param_neck}, {param_head}, {param_net}')

    def _create_encoder(self):
        self.encoder = build_backbone(self.backbone)

    def _prepare_body_module(self):
        self.body_position_net = PositionNet('body', feat_dim=self.feat_dim, output_hm_shape=self.output_hm_shape, output_hand_hm_shape=self.output_hand_hm_shape)
        self.body_regressor = BodyRotationNet(feat_dim=self.feat_dim)
        self.box_net = BoxNet(feat_dim=self.feat_dim, output_hm_shape=self.output_hm_shape)


    def _prepare_hand_module(self):
        self.hand_position_net = PositionNet('hand', feat_dim=self.feat_dim, output_hm_shape=self.output_hm_shape, output_hand_hm_shape=self.output_hand_hm_shape)
        self.hand_roi_net = HandRoI(feat_dim=self.feat_dim, upscale=self.upscale, input_body_shape=self.input_body_shape, output_hm_shape=self.output_hm_shape, output_hand_hm_shape=self.output_hand_hm_shape)
        self.hand_regressor = HandRotationNet('hand', feat_dim=self.feat_dim)

    def _prepare_face_module(self):
        self.face_regressor = FaceRegressor(feat_dim=self.feat_dim)

    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.focal[0] * self.focal[1] * self.camera_3d_size * self.camera_3d_size / (
                self.input_body_shape[0] * self.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        mesh_cam = output.vertices
        if self.testset == 'AGORA':  # use 144 joints for AGORA evaluation
            joint_cam = output.joints
        else:
            joint_cam = output.joints[:, smpl_x.joint_idx, :]

        # project 3D coordinates to 2D space

        x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
            self.focal[0] + self.princpt[0]
        y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
            self.focal[1] + self.princpt[1]
        x = x / self.input_body_shape[1] * self.output_hm_shape[2]
        y = y / self.input_body_shape[0] * self.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam + cam_trans[:, None, :]  # for rendering
        joint_cam_wo_ra = joint_cam.clone()

        # left hand root (left wrist)-relative 3D coordinatese
        lhand_idx = smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinatese
        rhand_idx = smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam

    def generate_mesh_gt(self, targets):
        if 'smplx_mesh_cam' in targets:
            return targets['smplx_mesh_cam']
        nums = [3, 63, 45, 45, 3]
        accu = []
        temp = 0
        for num in nums:
            temp += num
            accu.append(temp)
        pose = targets['smplx_pose']
        root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose = \
            pose[:, :accu[0]], pose[:, accu[0]:accu[1]], pose[:, accu[1]:accu[2]], pose[:, accu[2]:accu[3]], pose[:,
                                                                                                             accu[3]:
                                                                                                             accu[4]]
        # print(lhand_pose)
        shape = targets['smplx_shape']
        expr = targets['smplx_expr']
        cam_trans = targets['smplx_cam_trans']

        # final output
        joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape,
                                                         expr, cam_trans)

        return mesh_cam

    def bbox_split(self, bbox):
        # bbox:[bs, 3, 3]
        lhand_bbox_center, rhand_bbox_center, face_bbox_center = \
            bbox[:, 0, :2], bbox[:, 1, :2], bbox[:, 2, :2]
        return lhand_bbox_center, rhand_bbox_center, face_bbox_center
    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        pass

    def forward_test(self,  img: torch.Tensor, img_metas: dict, **kwargs):

        body_img = F.interpolate(img, self.input_body_shape)

        # 1. Encoder
        # import pdb; pdb.set_trace()
        body_img = body_img.float()
        img_feat, task_tokens = self.encoder(body_img)  # task_token:[bs, N, c]
        shape_token, cam_token, expr_token, jaw_pose_token, hand_token, body_pose_token = \
            task_tokens[:, 0], task_tokens[:, 1], task_tokens[:, 2], task_tokens[:, 3], task_tokens[:, 4:6], task_tokens[:, 6:]

        # 2. Body Regressor
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape, cam_param, = self.body_regressor(body_pose_token, shape_token, cam_token, body_joint_img.detach())
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)
        cam_trans = self.get_camera_trans(cam_param)

        # 3. Hand and Face BBox Estimation
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size, face_bbox_center, face_bbox_size = self.box_net(img_feat, body_joint_hm.detach())
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, self.input_hand_shape[1] / self.input_hand_shape[0], 2.0, input_body_shape=self.input_body_shape, output_hm_shape=self.output_hm_shape).detach()  # xyxy in (self.input_body_shape[1], self.input_body_shape[0]) space
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, self.input_hand_shape[1] / self.input_hand_shape[0], 2.0, input_body_shape=self.input_body_shape, output_hm_shape=self.output_hm_shape).detach()  # xyxy in (self.input_body_shape[1], self.input_body_shape[0]) space
        face_bbox = restore_bbox(face_bbox_center, face_bbox_size, self.input_face_shape[1] / self.input_face_shape[0], 1.5, input_body_shape=self.input_body_shape, output_hm_shape=self.output_hm_shape).detach()  # xyxy in (self.input_body_shape[1], self.input_body_shape[0]) space

        # 4. Differentiable Feature-level Hand Crop-Upsample
        # hand_feat: list, [bsx2, c, cfg.output_hm_shape[1]*scale, cfg.output_hm_shape[2]*scale]
        hand_feat = self.hand_roi_net(img_feat, lhand_bbox, rhand_bbox)  # hand_feat: flipped left hand + right hand

        # 5. Hand/Face Regressor
        # hand regressor
        _, hand_joint_img = self.hand_position_net(hand_feat)  # (2N, J_P, 3)
        hand_pose = self.hand_regressor(hand_feat, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(hand_feat.shape[0], -1)
        # restore flipped left hand joint coordinates
        batch_size = hand_joint_img.shape[0] // 2
        lhand_joint_img = hand_joint_img[:batch_size, :, :]
        lhand_joint_img = torch.cat((self.output_hand_hm_shape[2] - 1 - lhand_joint_img[:, :, 0:1], lhand_joint_img[:, :, 1:]), 2)
        rhand_joint_img = hand_joint_img[batch_size:, :, :]
        # restore flipped left hand joint rotations
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(-1, len(smpl_x.orig_joint_part['lhand']), 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:, :]

        # hand regressor
        expr, jaw_pose = self.face_regressor(expr_token, jaw_pose_token)
        jaw_pose = rot6d_to_axis_angle(jaw_pose)

        # final output
        joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans)
        joint_img = torch.cat((body_joint_img, lhand_joint_img, rhand_joint_img), 1)

        # change hand output joint_img according to hand bbox
        for part_name, bbox in (('lhand', lhand_bbox), ('rhand', rhand_bbox)):
            joint_img[:, smpl_x.pos_joint_part[part_name], 0] *= (
                    ((bbox[:, None, 2] - bbox[:, None, 0]) / self.input_body_shape[1] * self.output_hm_shape[2]) /
                    self.output_hand_hm_shape[2])
            joint_img[:, smpl_x.pos_joint_part[part_name], 0] += (
                    bbox[:, None, 0] / self.input_body_shape[1] * self.output_hm_shape[2])
            joint_img[:, smpl_x.pos_joint_part[part_name], 1] *= (
                    ((bbox[:, None, 3] - bbox[:, None, 1]) / self.input_body_shape[0] * self.output_hm_shape[1]) /
                    self.output_hand_hm_shape[1])
            joint_img[:, smpl_x.pos_joint_part[part_name], 1] += (
                    bbox[:, None, 1] / self.input_body_shape[0] * self.output_hm_shape[1])

        # change input_body_shape to input_img_shape
        for bbox in (lhand_bbox, rhand_bbox, face_bbox):
            bbox[:, 0] *= self.input_img_shape[0] / self.input_body_shape[0]
            bbox[:, 1] *= self.input_img_shape[1] / self.input_body_shape[1]
            bbox[:, 2] *= self.input_img_shape[0] / self.input_body_shape[0]
            bbox[:, 3] *= self.input_img_shape[1] / self.input_body_shape[1]

            # test output
            out = {}
            out['joint_img'] = joint_img
            out['smplx_joint_proj'] = joint_proj
            out['smplx_mesh_cam'] = mesh_cam
            out['smplx_root_pose'] = root_pose
            out['smplx_body_pose'] = body_pose
            out['smplx_lhand_pose'] = lhand_pose
            out['smplx_rhand_pose'] = rhand_pose
            out['smplx_jaw_pose'] = jaw_pose
            out['smplx_shape'] = shape
            out['smplx_expr'] = expr
            out['cam_trans'] = cam_trans
            out['lhand_bbox'] = lhand_bbox
            out['rhand_bbox'] = rhand_bbox
            out['face_bbox'] = face_bbox
            out['camera'] = cam_param
            return out



def restore_bbox(bbox_center, bbox_size, aspect_ratio, extension_ratio, input_body_shape, output_hm_shape):
    bbox = bbox_center.view(-1, 1, 2) + torch.cat((-bbox_size.view(-1, 1, 2) / 2., bbox_size.view(-1, 1, 2) / 2.),
                                                  1)  # xyxy in (cfg.output_hm_shape[2], cfg.output_hm_shape[1]) space
    bbox[:, :, 0] = bbox[:, :, 0] / output_hm_shape[2] * input_body_shape[1]
    bbox[:, :, 1] = bbox[:, :, 1] / output_hm_shape[1] * input_body_shape[0]
    bbox = bbox.view(-1, 4)

    # xyxy -> xywh
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1]

    # aspect ratio preserving bbox
    w = bbox[:, 2]
    h = bbox[:, 3]
    c_x = bbox[:, 0] + w / 2.
    c_y = bbox[:, 1] + h / 2.

    mask1 = w > (aspect_ratio * h)
    mask2 = w < (aspect_ratio * h)
    h[mask1] = w[mask1] / aspect_ratio
    w[mask2] = h[mask2] * aspect_ratio

    bbox[:, 2] = w * extension_ratio
    bbox[:, 3] = h * extension_ratio
    bbox[:, 0] = c_x - bbox[:, 2] / 2.
    bbox[:, 1] = c_y - bbox[:, 3] / 2.

    # xywh -> xyxy
    bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
    return bbox

def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    # import pdb; pdb.set_trace()
    # axis_angle = rotmat_to_aa(rot_mat[...,:3, :3]).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle



