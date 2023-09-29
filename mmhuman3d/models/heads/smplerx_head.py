import torch
import torch.nn as nn
from torch.nn import functional as F

#from utils.transforms import sample_joint_features, soft_argmax_2d, soft_argmax_3d
from mmhuman3d.models.utils.human_models import smpl_x
from mmcv.ops.roi_align import roi_align

class PositionNet(nn.Module):
    def __init__(self, part, feat_dim, output_hm_shape, output_hand_hm_shape):
        super(PositionNet, self).__init__()
        if part == 'body':
            self.joint_num = len(smpl_x.pos_joint_part['body'])
            self.hm_shape = output_hm_shape
        elif part == 'hand':
            self.joint_num = len(smpl_x.pos_joint_part['rhand'])
            self.hm_shape = output_hand_hm_shape
        self.conv = make_conv_layers([feat_dim, self.joint_num * self.hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        joint_hm = F.softmax(joint_hm.view(-1, self.joint_num, self.hm_shape[0] * self.hm_shape[1] * self.hm_shape[2]), 2)
        joint_hm = joint_hm.view(-1, self.joint_num, self.hm_shape[0], self.hm_shape[1], self.hm_shape[2])
        return joint_hm, joint_coord

class HandRotationNet(nn.Module):
    def __init__(self, part, feat_dim = 768):
        super(HandRotationNet, self).__init__()
        self.part = part
        self.joint_num = len(smpl_x.pos_joint_part['rhand'])
        self.hand_conv = make_conv_layers([feat_dim, 512], kernel=1, stride=1, padding=0)
        self.hand_pose_out = make_linear_layers([self.joint_num * 515, len(smpl_x.orig_joint_part['rhand']) * 6], relu_final=False)
        self.feat_dim = feat_dim

    def forward(self, img_feat, joint_coord_img):
        batch_size = img_feat.shape[0]
        img_feat = self.hand_conv(img_feat)
        img_feat_joints = sample_joint_features(img_feat, joint_coord_img[:, :, :2])
        feat = torch.cat((img_feat_joints, joint_coord_img), 2)  # batch_size, joint_num, 512+3
        hand_pose = self.hand_pose_out(feat.view(batch_size, -1))
        return hand_pose

class BodyRotationNet(nn.Module):
    def __init__(self, feat_dim = 768):
        super(BodyRotationNet, self).__init__()
        self.joint_num = len(smpl_x.pos_joint_part['body'])
        self.body_conv = make_linear_layers([feat_dim, 512], relu_final=False)
        self.root_pose_out = make_linear_layers([self.joint_num * (512+3), 6], relu_final=False)
        self.body_pose_out = make_linear_layers(
            [self.joint_num * (512+3), (len(smpl_x.orig_joint_part['body']) - 1) * 6], relu_final=False)  # without root
        self.shape_out = make_linear_layers([feat_dim, smpl_x.shape_param_dim], relu_final=False)
        self.cam_out = make_linear_layers([feat_dim, 3], relu_final=False)
        self.feat_dim = feat_dim

    def forward(self, body_pose_token, shape_token, cam_token, body_joint_img):
        batch_size = body_pose_token.shape[0]

        # shape parameter
        shape_param = self.shape_out(shape_token)

        # camera parameter
        cam_param = self.cam_out(cam_token)

        # body pose parameter
        body_pose_token = self.body_conv(body_pose_token)
        body_pose_token = torch.cat((body_pose_token, body_joint_img), 2)
        root_pose = self.root_pose_out(body_pose_token.view(batch_size, -1))
        body_pose = self.body_pose_out(body_pose_token.view(batch_size, -1))

        return root_pose, body_pose, shape_param, cam_param

class FaceRegressor(nn.Module):
    def __init__(self, feat_dim=768):
        super(FaceRegressor, self).__init__()
        self.expr_out = make_linear_layers([feat_dim, smpl_x.expr_code_dim], relu_final=False)
        self.jaw_pose_out = make_linear_layers([feat_dim, 6], relu_final=False)

    def forward(self, expr_token, jaw_pose_token):
        expr_param = self.expr_out(expr_token)  # expression parameter
        jaw_pose = self.jaw_pose_out(jaw_pose_token)  # jaw pose parameter
        return expr_param, jaw_pose

class BoxNet(nn.Module):
    def __init__(self, feat_dim, output_hm_shape):
        super(BoxNet, self).__init__()
        self.output_hm_shape = output_hm_shape
        self.joint_num = len(smpl_x.pos_joint_part['body'])
        self.deconv = make_deconv_layers([feat_dim + self.joint_num * self.output_hm_shape[0], 256, 256, 256])
        self.bbox_center = make_conv_layers([256, 3], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, img_feat, joint_hm):
        joint_hm = joint_hm.view(joint_hm.shape[0], joint_hm.shape[1] * self.output_hm_shape[0], self.output_hm_shape[1], self.output_hm_shape[2])
        img_feat = torch.cat((img_feat, joint_hm), 1)
        img_feat = self.deconv(img_feat)

        # bbox center
        bbox_center_hm = self.bbox_center(img_feat)
        bbox_center = soft_argmax_2d(bbox_center_hm)
        lhand_center, rhand_center, face_center = bbox_center[:, 0, :], bbox_center[:, 1, :], bbox_center[:, 2, :]

        # bbox size
        lhand_feat = sample_joint_features(img_feat, lhand_center[:, None, :].detach())[:, 0, :]
        lhand_size = self.lhand_size(lhand_feat)
        rhand_feat = sample_joint_features(img_feat, rhand_center[:, None, :].detach())[:, 0, :]
        rhand_size = self.rhand_size(rhand_feat)
        face_feat = sample_joint_features(img_feat, face_center[:, None, :].detach())[:, 0, :]
        face_size = self.face_size(face_feat)

        lhand_center = lhand_center / 8
        rhand_center = rhand_center / 8
        face_center = face_center / 8
        return lhand_center, lhand_size, rhand_center, rhand_size, face_center, face_size

class BoxSizeNet(nn.Module):
    def __init__(self):
        super(BoxSizeNet, self).__init__()
        self.lhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.rhand_size = make_linear_layers([256, 256, 2], relu_final=False)
        self.face_size = make_linear_layers([256, 256, 2], relu_final=False)

    def forward(self, box_fea):
        # box_fea: [bs, 3, C]
        lhand_size = self.lhand_size(box_fea[:, 0])
        rhand_size = self.rhand_size(box_fea[:, 1])
        face_size = self.face_size(box_fea[:, 2])
        return lhand_size, rhand_size, face_size

class HandRoI(nn.Module):
    def __init__(self, feat_dim, upscale, input_body_shape, output_hm_shape, output_hand_hm_shape):
        super(HandRoI, self).__init__()
        self.upscale = upscale
        if upscale==1:
            self.deconv = make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv = make_conv_layers([feat_dim, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        elif upscale==2:
            self.deconv = make_deconv_layers([feat_dim, feat_dim//2])
            self.conv = make_conv_layers([feat_dim//2, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        elif upscale==4:
            self.deconv = make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4])
            self.conv = make_conv_layers([feat_dim//4, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        elif upscale==8:
            self.deconv = make_deconv_layers([feat_dim, feat_dim//2, feat_dim//4, feat_dim//8])
            self.conv = make_conv_layers([feat_dim//8, feat_dim], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.input_body_shape = input_body_shape
        self.output_hm_shape = output_hm_shape
        self.output_hand_hm_shape = output_hand_hm_shape
    def forward(self, img_feat, lhand_bbox, rhand_bbox):
        lhand_bbox = torch.cat((torch.arange(lhand_bbox.shape[0]).float().cuda()[:, None], lhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        rhand_bbox = torch.cat((torch.arange(rhand_bbox.shape[0]).float().cuda()[:, None], rhand_bbox),
                               1)  # batch_idx, xmin, ymin, xmax, ymax
        img_feat = self.deconv(img_feat)
        lhand_bbox_roi = lhand_bbox.clone()
        lhand_bbox_roi[:, 1] = lhand_bbox_roi[:, 1] / self.input_body_shape[1] * self.output_hm_shape[2] * self.upscale
        lhand_bbox_roi[:, 2] = lhand_bbox_roi[:, 2] / self.input_body_shape[0] * self.output_hm_shape[1] * self.upscale
        lhand_bbox_roi[:, 3] = lhand_bbox_roi[:, 3] / self.input_body_shape[1] * self.output_hm_shape[2] * self.upscale
        lhand_bbox_roi[:, 4] = lhand_bbox_roi[:, 4] / self.input_body_shape[0] * self.output_hm_shape[1] * self.upscale
        assert (self.output_hm_shape[1]*self.upscale, self.output_hm_shape[2]*self.upscale) == (img_feat.shape[2], img_feat.shape[3])
        lhand_img_feat = roi_align(img_feat, lhand_bbox_roi, (self.output_hand_hm_shape[1], self.output_hand_hm_shape[2]), 1.0, 0, 'avg', False)
        lhand_img_feat = torch.flip(lhand_img_feat, [3])  # flip to the right hand

        rhand_bbox_roi = rhand_bbox.clone()
        rhand_bbox_roi[:, 1] = rhand_bbox_roi[:, 1] / self.input_body_shape[1] * self.output_hm_shape[2] * self.upscale
        rhand_bbox_roi[:, 2] = rhand_bbox_roi[:, 2] / self.input_body_shape[0] * self.output_hm_shape[1] * self.upscale
        rhand_bbox_roi[:, 3] = rhand_bbox_roi[:, 3] / self.input_body_shape[1] * self.output_hm_shape[2] * self.upscale
        rhand_bbox_roi[:, 4] = rhand_bbox_roi[:, 4] / self.input_body_shape[0] * self.output_hm_shape[1] * self.upscale
        rhand_img_feat = roi_align(img_feat, rhand_bbox_roi, (self.output_hand_hm_shape[1], self.output_hand_hm_shape[2]), 1.0, 0, 'avg', False)
        hand_img_feat = torch.cat((lhand_img_feat, rhand_img_feat))  # [bs, c, self.output_hand_hm_shape[2]*scale, self.output_hand_hm_shape[1]*scale]
        hand_img_feat = self.conv(hand_img_feat)
        return hand_img_feat

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def sample_joint_features(img_feat, joint_xy):
    height, width = img_feat.shape[2:]
    x = joint_xy[:, :, 0] / (width - 1) * 2 - 1
    y = joint_xy[:, :, 1] / (height - 1) * 2 - 1
    grid = torch.stack((x, y), 2)[:, :, None, :]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:, :, :, 0]  # batch_size, channel_dim, joint_num
    img_feat = img_feat.permute(0, 2, 1).contiguous()  # batch_size, joint_num, channel_dim
    return img_feat


def soft_argmax_2d(heatmap2d):
    batch_size = heatmap2d.shape[0]
    height, width = heatmap2d.shape[2:]
    heatmap2d = heatmap2d.reshape((batch_size, -1, height * width))
    heatmap2d = F.softmax(heatmap2d, 2)
    heatmap2d = heatmap2d.reshape((batch_size, -1, height, width))

    accu_x = heatmap2d.sum(dim=(2))
    accu_y = heatmap2d.sum(dim=(3))

    accu_x = accu_x * torch.arange(width).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(height).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y), dim=2)
    return coord_out


def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(width).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(height).float().cuda()[None, None, :]
    accu_z = accu_z * torch.arange(depth).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out
