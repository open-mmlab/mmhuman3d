from abc import abstractmethod
import os
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from torch.nn.modules.utils import _pair
from typing import Optional, Tuple, List
import torch.nn.init as nninit
import pickle
from mmhuman3d.utils.transforms import rot6d_to_aa
from mmhuman3d.utils.geometry import rot6d_to_rotmat
from mmcv.cnn import build_activation_layer
from mmcv.cnn import initialize

class IterativeRegression(nn.Module):
    def __init__(self, module, mean_param, num_stages=1, append_params=True, learn_mean=False,detach_mean=False, dim=1, **kwargs):
        super(IterativeRegression, self).__init__()
        self.module = module
        self._num_stages = num_stages
        self.dim = dim

        if learn_mean:
            self.register_parameter('mean_param',nn.Parameter(mean_param, requires_grad=True))
        else:
            self.register_buffer('mean_param', mean_param)

        self.append_params = append_params
        self.detach_mean = detach_mean

    def get_mean(self):
        return self.mean_param.clone()

    @property
    def num_stages(self):
        return self._num_stages

    def forward(self, features: torch.Tensor, cond: Optional[torch.Tensor] = None):
        ''' Computes deltas on top of condition iteratively
            Parameters
            ----------
                features: torch.Tensor
                    Input features
        '''
        batch_size = features.shape[0]
        expand_shape = [batch_size] + [-1] * len(features.shape[1:])

        parameters = []
        deltas = []
        module_input = features
        if cond is None:
            cond = self.mean_param.expand(*expand_shape).clone()

        # Detach mean
        if self.detach_mean:
            cond = cond.detach()

        if self.append_params:
            assert features is not None, (
                'Features are none even though append_params is True')
            module_input = torch.cat([module_input,cond],dim=self.dim)

        deltas.append(self.module(module_input))
        num_params = deltas[-1].shape[1]
        parameters.append(cond[:, :num_params].clone() + deltas[-1])

        for stage_idx in range(1, self.num_stages):
            module_input = torch.cat([features, parameters[stage_idx - 1]], dim=-1)
            params_upd = self.module(module_input)
            deltas.append(params_upd)
            parameters.append(parameters[stage_idx - 1] + params_upd)

        return parameters

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: List[int] = [],
        activ_type: str = 'relu',
        dropout: float = 0.5,
        gain: float = 0.01,
    ):
        super(MLP, self).__init__()
        curr_input_dim = input_dim
        self.num_layers = len(layers)

        self.blocks = nn.ModuleList()
        for layer_idx, layer_dim in enumerate(layers):
            if activ_type == 'none':
                activ = None
            else:
                activ = build_activation_layer(cfg = dict(
                    type = activ_type,
                    inplace = True
                ))
            linear = nn.Linear(curr_input_dim, layer_dim, bias=True)
            curr_input_dim = layer_dim

            layer = []
            layer.append(linear)

            if activ is not None:
                layer.append(activ)

            if dropout > 0.0:
                layer.append(nn.Dropout(dropout))

            block = nn.Sequential(*layer)
            self.add_module('layer_{:03d}'.format(layer_idx), block)
            self.blocks.append(block)
        
        self.output_layer = nn.Linear(curr_input_dim, output_dim)
        initialize(self.output_layer, 
                   init_cfg = dict(
                       type = 'Xavier', gain = gain, distribution = 'uniform'
                       )
                    )

    def forward(self, module_input):
        curr_input = module_input
        for block in self.blocks:
            curr_input = block(curr_input)
        return self.output_layer(curr_input)

class ContinuousRotReprDecoder:
    def __init__(self, num_angles, dtype=torch.float32, mean=None):
        self.num_angles = num_angles
        self.dtype = dtype

        if isinstance(mean, dict):
            mean = mean.get('cont_rot_repr', None)
        if mean is None:
            mean = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0],dtype=self.dtype).unsqueeze(dim=0).expand(self.num_angles, -1).contiguous().view(-1)
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        mean = mean.reshape(-1, 6)

        if mean.shape[0] < self.num_angles:
            mean = mean.repeat(self.num_angles // mean.shape[0] + 1, 1).contiguous()
            mean = mean[:self.num_angles]
        elif mean.shape[0] > self.num_angles:
            mean = mean[:self.num_angles]

        mean = mean.reshape(-1)
        self.mean = mean

    def get_type(self):
        return 'cont_rot_repr'

    def extra_repr(self):
        msg = 'Num angles: {}\n'.format(self.num_angles)
        msg += 'Mean: {}'.format(self.mean.shape)
        return msg

    def get_param_dim(self):
        return 6

    def get_dim_size(self):
        return self.num_angles * 6

    def get_mean(self):
        return self.mean.clone()

    def to_offsets(self, x):
        latent = x.reshape(-1, 3, 3)[:, :3, :2].reshape(-1, 6)
        return (latent - self.mean).reshape(x.shape[0], -1, 6)

    def encode(self, x, subtract_mean=False):
        orig_shape = x.shape
        if subtract_mean:
            raise NotImplementedError
        output = x.reshape(-1, 3, 3)[:, :3, :2].contiguous()
        return output.reshape(
            orig_shape[0], orig_shape[1], 3, 2)

    def __call__(self, module_input):
        batch_size = module_input.shape[0]
        reshaped_input = module_input.view(-1, 6)
        rot_mats = rot6d_to_rotmat(reshaped_input)
        # aa = rot6d_to_aa(reshaped_input)
        # return aa.view(batch_size,-1,3)
        return rot_mats.view(batch_size, -1, 3, 3)





class ExPoseHead(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        
    def load_regressor(self, input_feat_dim: int = 2048, param_mean: torch.Tensor = None, regressor_cfg: dict = None):   
        param_dim = param_mean.numel() 
        regressor = MLP(input_feat_dim + param_dim,
                        param_dim, **regressor_cfg)
        self.regressor = IterativeRegression(regressor, param_mean, num_stages=3)

    def load_param_decoder(self,mean_poses_dict):
        start = 0
        mean_lst = []
        self.pose_param_decoders = {}
        for pose_param in self.pose_param_conf:
            pose_name = pose_param['name']
            num_angles = pose_param['num_angles']
            if pose_param['use_mean']:
                pose_decoder = ContinuousRotReprDecoder(num_angles, dtype=torch.float32,mean = mean_poses_dict.get(pose_name, None))
            else:
                pose_decoder = ContinuousRotReprDecoder(num_angles, dtype=torch.float32,mean = None)
            self.pose_param_decoders['{}_decoder'.format(pose_name)] = pose_decoder
            pose_dim = pose_decoder.get_dim_size()
            pose_mean = pose_decoder.get_mean()
            if pose_param['rotate_axis_x']:
                pose_mean[3] = -1
            idxs = list(range(start, start + pose_dim))
            idxs = torch.tensor(idxs, dtype=torch.long)
            self.register_buffer('{}_idxs'.format(pose_name), idxs)
            start += pose_dim
            mean_lst.append(pose_mean.view(-1))
        return start, mean_lst

    def get_camera_param(self, camera_cfg):
        camera_pos_scale = camera_cfg.get('pos_func')
        if camera_pos_scale == 'softplus':
            camera_scale_func = F.softplus
        elif camera_pos_scale == 'exp':
            camera_scale_func = torch.exp
        elif camera_pos_scale == 'none' or camera_pos_scale == 'None':
            def func(x):
                return x
            camera_scale_func = func
        mean_scale = camera_cfg.get('mean_scale', 0.9)
        if camera_pos_scale == 'softplus':
            mean_scale = np.log(np.exp(mean_scale) - 1)
        elif camera_pos_scale == 'exp':
            mean_scale = np.log(mean_scale)
        camera_mean = torch.tensor([mean_scale, 0.0, 0.0], dtype=torch.float32)
        camera_param_dim = 3
        return camera_mean, camera_param_dim, camera_scale_func

    def flat_params_to_dict(self, param_tensor):
        smplx_dict = {}
        raw_dict = {}
        for pose_param in self.pose_param_conf:
            pose_name = pose_param['name']
            pose_idxs = getattr(self,f'{pose_name}_idxs')
            decoder = self.pose_param_decoders[f'{pose_name}_decoder']
            pose = torch.index_select(param_tensor, 1 , pose_idxs)
            raw_dict[f'raw_{pose_name}'] = pose.clone()
            smplx_dict[pose_name] = decoder(pose)
        return smplx_dict, raw_dict

    @abstractmethod
    def forward(self, features):
        pass

    
class ExPoseBodyHead(ExPoseHead):
    def __init__(self, init_cfg=None, num_betas: int = 10, num_expression_coeffs: int = 10, mean_pose_path: str = '', shape_mean_path: str = '', input_feat_dim: int = 2048,  regressor_cfg: dict = None, camera_cfg: dict = None):
        super().__init__(init_cfg)
        # poses
        self.pose_param_conf = [
            dict(
                name = 'global_orient',
                num_angles = 1,
                use_mean = False,
                rotate_axis_x = True),
            dict(
                name = 'body_pose',
                num_angles = 21,
                use_mean = True,
                rotate_axis_x = False),
            dict(
                name = 'left_hand_pose',
                num_angles = 15,
                use_mean = True,
                rotate_axis_x = False),
            dict(
                name = 'right_hand_pose',
                num_angles = 15,
                use_mean = True,
                rotate_axis_x = False),
            dict(
                name = 'jaw_pose',
                num_angles = 1,
                use_mean = False,
                rotate_axis_x = False),
        ]
        mean_poses_dict = {}
        if os.path.exists(mean_pose_path):
            with open(mean_pose_path, 'rb') as f:
                mean_poses_dict = pickle.load(f)
        start, mean_lst = self.load_param_decoder(mean_poses_dict)

        # shape
        if os.path.exists(shape_mean_path):
            shape_mean = torch.from_numpy(
                np.load(shape_mean_path, allow_pickle=True)).to(
                dtype=torch.float32).reshape(1, -1)[:, :num_betas].reshape(-1)
        else:
            shape_mean = torch.zeros([num_betas], dtype=torch.float32)
        shape_idxs = list(range(start, start + num_betas))
        self.register_buffer('shape_idxs', torch.tensor(shape_idxs, dtype=torch.long))
        start += num_betas
        mean_lst.append(shape_mean.view(-1))

        # expression
        expression_mean = torch.zeros([num_expression_coeffs], dtype=torch.float32)
        expression_idxs = list(range(start, start + num_expression_coeffs))
        self.register_buffer('expression_idxs', torch.tensor(expression_idxs, dtype=torch.long))
        start += num_expression_coeffs
        mean_lst.append(expression_mean.view(-1))
        
        # camera
        mean, dim, scale_func = self.get_camera_param(camera_cfg)
        self.camera_scale_func = scale_func
        camera_idxs = list(range(start, start + dim))
        self.register_buffer('camera_idxs', torch.tensor(camera_idxs, dtype=torch.long))
        start += dim
        mean_lst.append(mean)

        param_mean = torch.cat(mean_lst).view(1, -1)
        self.load_regressor(input_feat_dim,  param_mean, regressor_cfg)
    
    def forward(self, features):
        body_parameters = self.regressor(features)[-1]
        params_dict, raw_dict = self.flat_params_to_dict(body_parameters)
        params_dict['betas'] = torch.index_select(body_parameters, 1, self.shape_idxs)
        params_dict['expression'] = torch.index_select(body_parameters, 1, self.expression_idxs)

        camera_params = torch.index_select(body_parameters, 1, self.camera_idxs)
        scale = camera_params[:, 0:1]
        translation = camera_params[:, 1:3]
        scale = self.camera_scale_func(scale)
        camera_params = torch.cat([scale,translation],dim=1)
        return {
            'pred_param': params_dict,
            'pred_cam': camera_params,
            'pred_raw': raw_dict
        }


class ExPoseHandHead(ExPoseHead):
    def __init__(self, init_cfg=None, num_betas: int = 10,  mean_pose_path: str = '',  input_feat_dim: int = 2048,  regressor_cfg: dict = None, camera_cfg: dict = None):
        super().__init__(init_cfg)
        # poses
        self.pose_param_conf = [
            dict(
                name = 'global_orient',
                num_angles = 1,
                use_mean = False,
                rotate_axis_x = False),
            dict(
                name = 'right_hand_pose',
                num_angles = 15,
                use_mean = True,
                rotate_axis_x = False),
        ]
        mean_poses_dict = {}
        if os.path.exists(mean_pose_path):
            with open(mean_pose_path, 'rb') as f:
                mean_poses_dict = pickle.load(f)
        start, mean_lst = self.load_param_decoder(mean_poses_dict)

        shape_mean = torch.zeros([num_betas], dtype=torch.float32)
        shape_idxs = list(range(start, start + num_betas))
        self.register_buffer('shape_idxs', torch.tensor(shape_idxs, dtype=torch.long))
        start += num_betas
        mean_lst.append(shape_mean.view(-1))
        
        # camera
        mean, dim, scale_func = self.get_camera_param(camera_cfg)
        self.camera_scale_func = scale_func
        camera_idxs = list(range(start, start + dim))
        self.register_buffer('camera_idxs', torch.tensor(camera_idxs, dtype=torch.long))
        start += dim
        mean_lst.append(mean)

        param_mean = torch.cat(mean_lst).view(1, -1)
        self.load_regressor(input_feat_dim,  param_mean, regressor_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, features):
        batch_size = features[-1].size(0)
        features = self.avgpool(features[-1]).view(batch_size, -1)
        hand_parameters = self.regressor(features)[-1]
        params_dict, raw_dict = self.flat_params_to_dict(hand_parameters)
        params_dict['betas'] = torch.index_select(hand_parameters, 1, self.shape_idxs)

        camera_params = torch.index_select(hand_parameters, 1, self.camera_idxs)
        scale = camera_params[:, 0:1]
        translation = camera_params[:, 1:3]
        scale = self.camera_scale_func(scale)
        camera_params = torch.cat([scale,translation],dim=1)
        return {
            'pred_param': params_dict,
            'pred_cam': camera_params,
            'pred_raw': raw_dict
        }
