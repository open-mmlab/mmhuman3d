import numpy as np
import torch
import torch.nn as nn
from mmcv.runner.base_module import BaseModule

from mmhuman3d.utils.geometry import rot6d_to_rotmat
from ..builder import HEADS


@HEADS.register_module()
class HMRHead(BaseModule):

    def __init__(self,
                 feat_dim,
                 smpl_mean_params=None,
                 npose=144,
                 nbeta=10,
                 ncam=3,
                 hdim=1024,
                 init_cfg=None):
        super(HMRHead, self).__init__(init_cfg=init_cfg)
        self.fc1 = nn.Linear(feat_dim + npose + nbeta + ncam, hdim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hdim, hdim)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(hdim, npose)
        self.decshape = nn.Linear(hdim, nbeta)
        self.deccam = nn.Linear(hdim, ncam)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        if smpl_mean_params is None:
            init_pose = torch.zeros([1, npose])
            init_shape = torch.zeros([1, nbeta])
            init_cam = torch.FloatTensor([[1, 0, 0]])
        else:
            mean_params = np.load(smpl_mean_params)
            init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
            init_shape = torch.from_numpy(
                mean_params['shape'][:].astype('float32')).unsqueeze(0)
            init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self,
                x,
                init_pose=None,
                init_shape=None,
                init_cam=None,
                n_iter=3):

        # hmr head only support one layer feature
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[-1]

        output_seq = False
        if len(x.shape) == 4:
            # use feature from the last layer of the backbone
            # apply global average pooling on the feature map
            x = x.mean(dim=-1).mean(dim=-1)
        elif len(x.shape) == 3:
            # temporal feature
            output_seq = True
            B, T, L = x.shape
            x = x.view(-1, L)

        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        if output_seq:
            pred_rotmat = pred_rotmat.view(B, T, 24, 3, 3)
            pred_shape = pred_shape.view(B, T, 10)
            pred_cam = pred_cam.view(B, T, 3)
        output = {
            'pred_pose': pred_rotmat,
            'pred_shape': pred_shape,
            'pred_cam': pred_cam
        }
        return output
