# isort: skip_file
from abc import ABCMeta

import torch
import torch.nn as nn

from mmhuman3d.core.conventions.keypoints_mapping import coco
from mmhuman3d.models.body_models.smplx import SMPLX_ALL, get_partial_smpl
from mmhuman3d.models.heads.pymafx_head import (
    IUV_predict_layer,
    MAF_Extractor,
    Mesh_Sampler,
    Regressor,
    get_attention_modules,
)
from ...core import constants
from ..backbones.builder import build_backbone
from ..heads.builder import build_head
from .base_architecture import BaseArchitecture


def get_fusion_modules(module_keys, ma_feat_dim, grid_feat_dim, n_iter,
                       out_feat_len):

    feat_fusion = nn.ModuleDict()
    for k in module_keys:
        feat_fusion[k] = nn.ModuleList()
        for i in range(n_iter):
            feat_fusion[k].append(
                nn.Linear(grid_feat_dim + ma_feat_dim[k], out_feat_len[k]))

    return feat_fusion


class PyMAFX(BaseArchitecture, metaclass=ABCMeta):
    """PyMAFX Architecture.

    Args:
        backbone (dict | None, optional): Backbone config dict. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 backbone,
                 head,
                 attention_config,
                 joint_regressor_train_extra,
                 smpl_model_dir,
                 mesh_model,
                 bhf_mode,
                 maf_on,
                 body_sfeat_dim,
                 hf_sfeat_dim,
                 grid_align,
                 n_iter=3,
                 grid_feat=False,
                 aux_supv_on=True,
                 hf_aux_supv_on=False,
                 mlp_dim=[256, 128, 64, 5],
                 hf_mlp_dim=[256, 128, 64, 5],
                 loss_uv_regression_weight=0.5,
                 hf_model_cfg=None,
                 device=torch.device('cuda'),
                 init_cfg=None):
        super(PyMAFX, self).__init__(init_cfg)
        assert bhf_mode in ['body_hand', 'full_body']
        self.opt_wrist = True
        self.use_iwp_cam = True
        self.pred_vis_h = True
        self.hand_vis_th = 0.1
        self.adapt_integr = True
        self.use_cam_feat = True

        self.backbone = backbone
        self.attention_config = attention_config
        self.joint_regressor_train_extra = joint_regressor_train_extra
        self.smpl_model_dir = smpl_model_dir
        self.hf_model_cfg = hf_model_cfg
        self.mesh_model = mesh_model
        self.body_sfeat_dim = body_sfeat_dim
        self.hf_sfeat_dim = hf_sfeat_dim
        self.bhf_mode = bhf_mode
        self.grid_align = grid_align
        self.grid_feat = grid_feat
        self.n_iter = n_iter
        self.with_uv = loss_uv_regression_weight > 0
        self.mlp_dim = mlp_dim
        self.aux_supv_on = aux_supv_on
        self.hf_aux_supv_on = hf_aux_supv_on
        self.hf_mlp_dim = hf_mlp_dim
        self.device = device
        self.global_feat_dim = 2048
        self.global_mode = not maf_on
        self.maf_on = maf_on
        self.prepare_body_module()
        self.encoders = nn.ModuleDict()
        self.part_module_names = {
            'body': {},
            'hand': {},
            'face': {},
            'link': {}
        }
        self.bhf_ma_feat_dim = {}
        self.build_encoders()
        self.grid_feature()
        self.regressor = nn.ModuleList()
        self.build_regressor()
        self.maf_extractor = nn.ModuleDict()
        self.build_maf_extractor()
        self.hf_box_center = True
        head['smpl2lhand'] = self.smpl2lhand
        head['smpl2rhand'] = self.smpl2rhand
        head['hf_root_idx'] = self.hf_root_idx
        head['mano_ds_len'] = self.mano_ds_len
        self.head = build_head(head)

    def prepare_body_module(self):
        self.smpl_mode = (self.mesh_model['name'] == 'smpl')
        self.smplx_mode = (self.mesh_model['name'] == 'smplx')
        self.smpl_mean_params = self.mesh_model['smpl_mean_params']
        self.body_hand_mode = (self.bhf_mode == 'body_hand')
        self.full_body_mode = (self.bhf_mode == 'full_body')

        self.bhf_names = []
        if self.bhf_mode in ['body_hand', 'full_body']:
            self.bhf_names.append('body')
        if self.bhf_mode in ['body_hand', 'full_body']:
            self.bhf_names.append('hand')
        if self.bhf_mode in ['full_body']:
            self.bhf_names.append('face')

        # the limb parts need to be handled
        if self.body_hand_mode:
            self.part_names = ['lhand', 'rhand']
        elif self.full_body_mode:
            self.part_names = ['lhand', 'rhand', 'face']
        else:
            self.part_names = []
        # joint index info
        if not self.smpl_mode:
            h_root_idx = constants.HAND_NAMES.index('wrist')
            h_idx = constants.HAND_NAMES.index('middle1')
            f_idx = constants.FACIAL_LANDMARKS.index('nose_middle')
            self.hf_center_idx = {
                'lhand': h_idx,
                'rhand': h_idx,
                'face': f_idx
            }
            self.hf_root_idx = {
                'lhand': h_root_idx,
                'rhand': h_root_idx,
                'face': f_idx
            }

            lh_idx_coco = coco.COCO_KEYPOINTS.index('left_wrist')
            rh_idx_coco = coco.COCO_KEYPOINTS.index('right_wrist')
            f_idx_coco = coco.COCO_KEYPOINTS.index('nose')
            self.hf_root_idx_coco = {
                'lhand': lh_idx_coco,
                'rhand': rh_idx_coco,
                'face': f_idx_coco
            }

        # create parametric mesh models
        self.smpl_family = {}
        self.smpl_family['body'] = SMPLX_ALL(
            gender=self.mesh_model['gender'],
            joint_regressor_train_extra=self.joint_regressor_train_extra,
            smpl_model_dir=self.smpl_model_dir)

    def build_encoders(self):
        # encoder for the body part
        if 'body' in self.bhf_names and self.backbone is not None:
            self.encoders['body'] = build_backbone(self.backbone)
            self.part_module_names['body'].update(
                {'encoders.body': self.encoders['body']})

            self.mesh_sampler = Mesh_Sampler(type='smpl')
            self.part_module_names['body'].update(
                {'mesh_sampler': self.mesh_sampler})
            if not self.grid_feat:
                self.ma_feat_dim = self.mesh_sampler.Dmap.shape[
                    0] * self.mlp_dim[-1]
            else:
                self.ma_feat_dim = 0
            self.bhf_ma_feat_dim['body'] = self.ma_feat_dim
            dp_feat_dim = self.body_sfeat_dim[-1]
            if self.aux_supv_on:
                assert self.maf_on
                self.dp_head = IUV_predict_layer(feat_dim=dp_feat_dim)
                self.part_module_names['body'].update(
                    {'dp_head': self.dp_head})
        # encoders for the hand / face parts
        if 'hand' in self.bhf_names or 'face' in self.bhf_names and \
           self.hf_model_cfg is not None:
            for hf in ['hand', 'face']:
                if hf in self.bhf_names:
                    self.encoders[hf] = build_backbone(
                        self.hf_model_cfg['backbone'])
                    self.part_module_names[hf].update(
                        {f'encoders.{hf}': self.encoders[hf]})
            if self.hf_aux_supv_on:
                assert self.maf_on
                self.dp_head_hf = nn.ModuleDict()
                if 'hand' in self.bhf_names:
                    self.dp_head_hf['hand'] = IUV_predict_layer(
                        feat_dim=self.hf_sfeat_dim[-1], mode='pncc')
                    self.part_module_names['hand'].update(
                        {'dp_head_hf.hand': self.dp_head_hf['hand']})
                if 'face' in self.bhf_names:
                    self.dp_head_hf['face'] = IUV_predict_layer(
                        feat_dim=self.hf_sfeat_dim[-1], mode='pncc')
                    self.part_module_names['face'].update(
                        {'dp_head_hf.face': self.dp_head_hf['face']})

            smpl2limb_vert_faces = get_partial_smpl()
            self.smpl2lhand = torch.from_numpy(
                smpl2limb_vert_faces['lhand']['vids']).long()
            self.smpl2rhand = torch.from_numpy(
                smpl2limb_vert_faces['rhand']['vids']).long()

    def grid_feature(self):
        # grid points for grid feature extraction
        grid_size = 21
        xv, yv = torch.meshgrid([
            torch.linspace(-1, 1, grid_size),
            torch.linspace(-1, 1, grid_size)
        ])
        grid_points = torch.stack([xv.reshape(-1),
                                   yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('grid_points', grid_points)
        self.grid_feat_dim = grid_size * grid_size * self.mlp_dim[-1]

        # the fusion of grid and mesh-aligned features
        self.fuse_grid_align = self.grid_align['use_att'] or self.grid_align[
            'use_fc']
        assert not (self.grid_align['use_att'] and self.grid_align['use_fc'])

        if self.fuse_grid_align:
            self.att_starts = self.grid_align['att_starts']
            n_iter_att = self.n_iter - self.att_starts
            self.att_feat_dim_idx = -self.grid_align['att_feat_idx']
            num_att_heads = self.grid_align['att_head']
            hidden_feat_dim = self.mlp_dim[self.att_feat_dim_idx]
            self.bhf_att_feat_dim = {'body': 2048}
        if 'hand' in self.bhf_names:
            self.mano_sampler = Mesh_Sampler(type='mano', level=1)
            self.mano_ds_len = self.mano_sampler.Dmap.shape[0]
            self.part_module_names['hand'].update(
                {'mano_sampler': self.mano_sampler})

            self.bhf_ma_feat_dim.update(
                {'hand': self.mano_ds_len * self.hf_mlp_dim[-1]})

            if self.fuse_grid_align:
                self.bhf_att_feat_dim.update({'hand': 1024})

        if 'face' in self.bhf_names:
            self.bhf_ma_feat_dim.update({
                'face':
                len(constants.FACIAL_LANDMARKS) * self.hf_mlp_dim[-1]
            })
            if self.fuse_grid_align:
                self.bhf_att_feat_dim.update({'face': 1024})
        # spatial alignment attention
        if self.grid_align['use_att']:
            hfimg_feat_dim_list = {}
            if 'body' in self.bhf_names:
                hfimg_feat_dim_list['body'] = self.body_sfeat_dim[-n_iter_att:]

            if 'hand' in self.bhf_names or 'face' in self.bhf_names:
                if 'hand' in self.bhf_names:
                    hfimg_feat_dim_list['hand'] = self.hf_sfeat_dim[
                        -n_iter_att:]
                if 'face' in self.bhf_names:
                    hfimg_feat_dim_list['face'] = self.hf_sfeat_dim[
                        -n_iter_att:]
            self.align_attention = get_attention_modules(
                self.attention_config,
                self.bhf_names,
                hfimg_feat_dim_list,
                hidden_feat_dim,
                n_iter=n_iter_att,
                num_attention_heads=num_att_heads)

            for part in self.bhf_names:
                self.part_module_names[part].update(
                    {f'align_attention.{part}': self.align_attention[part]})

        if self.fuse_grid_align:
            self.att_feat_reduce = get_fusion_modules(
                self.bhf_names,
                self.bhf_ma_feat_dim,
                self.grid_feat_dim,
                n_iter=n_iter_att,
                out_feat_len=self.bhf_att_feat_dim)
            for part in self.bhf_names:
                self.part_module_names[part].update(
                    {f'att_feat_reduce.{part}': self.att_feat_reduce[part]})

    def build_regressor(self):
        for i in range(self.n_iter):
            ref_infeat_dim = 0
            if 'body' in self.bhf_names:
                if self.maf_on:
                    if self.fuse_grid_align:
                        if i >= self.att_starts:
                            ref_infeat_dim = self.bhf_att_feat_dim['body']
                        elif i == 0 or self.grid_feat:
                            ref_infeat_dim = self.grid_feat_dim
                        else:
                            ref_infeat_dim = self.ma_feat_dim
                    else:
                        if i == 0 or self.grid_feat:
                            ref_infeat_dim = self.grid_feat_dim
                        else:
                            ref_infeat_dim = self.ma_feat_dim
                else:
                    ref_infeat_dim = self.global_feat_dim

            if self.smpl_mode:
                self.regressor.append(
                    Regressor(
                        self.mesh_model,
                        self.bhf_mode,
                        self.opt_wrist,
                        self.use_iwp_cam,
                        self.pred_vis_h,
                        self.hand_vis_th,
                        self.adapt_integr,
                        self.n_iter,
                        self.smpl_model_dir,
                        feat_dim=ref_infeat_dim,
                        smpl_mean_params=self.smpl_mean_params,
                        use_cam_feat=self.use_cam_feat,
                        smpl_models=self.smpl_family))
            else:
                if self.maf_on:
                    if 'hand' in self.bhf_names or 'face' in self.bhf_names:
                        if i == 0:
                            feat_dim_hand = self.grid_feat_dim if 'hand' in \
                                self.bhf_names else None
                            feat_dim_face = self.grid_feat_dim if 'face' in \
                                self.bhf_names else None
                        else:
                            if self.fuse_grid_align:
                                feat_dim_hand = self.bhf_att_feat_dim[
                                    'hand'] if 'hand' in self.bhf_names \
                                    else None
                                feat_dim_face = self.bhf_att_feat_dim[
                                    'face'] if 'face' in self.bhf_names \
                                    else None
                            else:
                                feat_dim_hand = self.bhf_ma_feat_dim[
                                    'hand'] if 'hand' in self.bhf_names \
                                    else None
                                feat_dim_face = self.bhf_ma_feat_dim[
                                    'face'] if 'face' in self.bhf_names \
                                    else None
                    else:
                        feat_dim_hand = ref_infeat_dim
                        feat_dim_face = ref_infeat_dim
                else:
                    ref_infeat_dim = self.global_feat_dim
                    feat_dim_hand = self.global_feat_dim
                    feat_dim_face = self.global_feat_dim

                self.regressor.append(
                    Regressor(
                        self.mesh_model,
                        self.bhf_mode,
                        self.opt_wrist,
                        self.use_iwp_cam,
                        self.pred_vis_h,
                        self.hand_vis_th,
                        self.adapt_integr,
                        self.n_iter,
                        self.smpl_model_dir,
                        feat_dim=ref_infeat_dim,
                        smpl_mean_params=self.smpl_mean_params,
                        use_cam_feat=self.use_cam_feat,
                        feat_dim_hand=feat_dim_hand,
                        feat_dim_face=feat_dim_face,
                        bhf_names=self.bhf_names,
                        smpl_models=self.smpl_family))

            # assign sub-regressor to each part
            for dec_name, dec_module in self.regressor[-1].named_children():
                if 'hand' in dec_name:
                    self.part_module_names['hand'].update({
                        'regressor.{}.{}.'.format(
                            len(self.regressor) - 1, dec_name):
                        dec_module
                    })
                elif 'face' in dec_name or 'head' in dec_name or \
                     'exp' in dec_name:
                    self.part_module_names['face'].update({
                        'regressor.{}.{}.'.format(
                            len(self.regressor) - 1, dec_name):
                        dec_module
                    })
                elif 'res' in dec_name or 'vis' in dec_name:
                    self.part_module_names['link'].update({
                        'regressor.{}.{}.'.format(
                            len(self.regressor) - 1, dec_name):
                        dec_module
                    })
                elif 'body' in self.part_module_names:
                    self.part_module_names['body'].update({
                        'regressor.{}.{}.'.format(
                            len(self.regressor) - 1, dec_name):
                        dec_module
                    })

    def build_maf_extractor(self):
        # mesh-aligned feature extractor
        for part in self.bhf_names:
            self.maf_extractor[part] = nn.ModuleList()
            filter_channels_default = self.mlp_dim if part == 'body' \
                else self.hf_mlp_dim
            sfeat_dim = self.body_sfeat_dim if part == 'body' \
                else self.hf_sfeat_dim
            for i in range(self.n_iter):
                for f_i, f_dim in enumerate(filter_channels_default):
                    if sfeat_dim[i] > f_dim:
                        filter_start = f_i
                        break
                filter_channels = [sfeat_dim[i]
                                   ] + filter_channels_default[filter_start:]

                if self.grid_align['use_att'] and i >= self.att_starts:
                    self.maf_extractor[part].append(
                        MAF_Extractor(
                            filter_channels=filter_channels_default[
                                self.att_feat_dim_idx:],
                            iwp_cam_mode=self.use_iwp_cam))
                else:
                    self.maf_extractor[part].append(
                        MAF_Extractor(
                            filter_channels=filter_channels,
                            iwp_cam_mode=self.use_iwp_cam))
            self.part_module_names[part].update(
                {f'maf_extractor.{part}': self.maf_extractor[part]})

        # check all modules have been added to part_module_names
        model_dict_all = dict.fromkeys(self.state_dict().keys())
        for key in self.part_module_names.keys():
            for name in list(model_dict_all.keys()):
                for k in self.part_module_names[key].keys():
                    if name.startswith(k):
                        del model_dict_all[name]
                # if name.startswith('regressor.') and '.smpl.' in name:
                #     del model_dict_all[name]
                # if name.startswith('regressor.') and '.mano.' in name:
                #     del model_dict_all[name]
                if name.startswith('regressor.') and '.init_' in name:
                    del model_dict_all[name]
                if name == 'grid_points':
                    del model_dict_all[name]
        assert (len(model_dict_all.keys()) == 0)

    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule. Please use '
                                  '`train_step` for training.')

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for m in ['body', 'hand', 'face']:
            if m in self.smpl_family:
                self.smpl_family[m].to(*args, **kwargs)
        return self

    def forward_test(self, batch={}, J_regressor=None, rw_cam={}, **kwargs):
        '''
        Args:
            batch:
                images: 'img_{part}', for part in body, hand, and face
                inversed affine transformation for cropped of hand/face img:
                '{part}_theta_inv' for part in lhand, rhand, and face
            J_regressor:
                joint regression matrix
            rw_cam:
                real-world camera information, applied when
                use_iwp_cam is False
        Returns:
            out_dict: the list containing the predicted parameters
        '''

        # extract spatial features or global features
        # run encoder for body
        if 'body' in self.bhf_names:
            img_body = batch['img_body']
            batch_size = img_body.shape[0]
            s_feat_body, g_feat = self.encoders['body'](batch['img_body'])
            if self.maf_on:
                assert len(s_feat_body) == self.n_iter

        # run encoders for hand / face
        if 'hand' in self.bhf_names or 'face' in self.bhf_names:
            limb_feat_dict = {}
            limb_gfeat_dict = {}
            if 'face' in self.bhf_names:
                img_face = batch['img_face']
                batch_size = img_face.shape[0]
                limb_feat_dict['face'], limb_gfeat_dict[
                    'face'] = self.encoders['face'](
                        img_face)

            if 'hand' in self.bhf_names:
                if 'lhand' in self.part_names:
                    img_rhand = batch['img_rhand']
                    batch_size = img_rhand.shape[0]
                    # flip left hand images
                    img_lhand = torch.flip(batch['img_lhand'], [3])
                    img_hands = torch.cat([img_rhand, img_lhand])
                    s_feat_hands, g_feat_hands = self.encoders['hand'](
                        img_hands)
                    limb_feat_dict['rhand'] = [
                        feat[:batch_size] for feat in s_feat_hands
                    ]
                    limb_feat_dict['lhand'] = [
                        feat[batch_size:] for feat in s_feat_hands
                    ]
                    if g_feat_hands is not None:
                        limb_gfeat_dict['rhand'] = g_feat_hands[:batch_size]
                        limb_gfeat_dict['lhand'] = g_feat_hands[batch_size:]
                else:
                    img_rhand = batch['img_rhand']
                    batch_size = img_rhand.shape[0]
                    limb_feat_dict['rhand'], limb_gfeat_dict[
                        'rhand'] = self.encoders['hand'](
                            img_rhand)

            if self.maf_on:
                for k in limb_feat_dict.keys():
                    assert len(limb_feat_dict[k]) == self.n_iter

        # grid-pattern points
        grid_points = torch.transpose(
            self.grid_points.expand(batch_size, -1, -1), 1, 2)

        # parameter predictions
        out_dict = self.head(batch, s_feat_body, limb_feat_dict, g_feat,
                             grid_points, J_regressor, self.att_feat_reduce,
                             self.fuse_grid_align, self.grid_feat,
                             self.align_attention, self.maf_extractor,
                             self.regressor, batch_size, limb_gfeat_dict,
                             self.bhf_names, self.part_names)

        return out_dict
