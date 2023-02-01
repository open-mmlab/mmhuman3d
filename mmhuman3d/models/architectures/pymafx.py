# yapf: disable
from abc import ABCMeta
from typing import Optional, Union

import torch
import torch.nn as nn

from mmhuman3d.core.conventions.keypoints_mapping.flame import (
    FLAME_73_KEYPOINTS,
)
from mmhuman3d.core.conventions.keypoints_mapping.mano import (
    MANO_RIGHT_REORDER_KEYPOINTS,
)
from mmhuman3d.models.body_models.smplx import GenderedSMPLX
from mmhuman3d.models.heads.pymafx_head import (
    IUV_predict_layer,
    MAF_Extractor,
    Mesh_Sampler,
    get_attention_modules,
)
from ..backbones.builder import build_backbone
from ..heads.builder import build_head
from .base_architecture import BaseArchitecture

# yapf: enable
GRID_SIZE = 21
GLOBAL_FEAT_DIM = 2048
FACIAL_LANDMARKS = FLAME_73_KEYPOINTS[5:]


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
        backbone (Optional[Union[dict, None]]): Backbone config dict.
        head (Optional[Union[dict, None]]): Head config dict.
        regressor (Optional[Union[dict, None]]): Regressor config dict.
        attention_config (str): Attention config path.
        extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
        smplx_to_smpl (str): The path of the matrix that turns SMPLX into SMPL.
        smplx_model_dir (str): The path of the SMPLX model.
        mesh_model (dict): Details of the SMPLX model.
        bhf_mode (str): The type of PyMAFX, ``full_body`` or ``body_hand``.
        maf_on (bool): Whether to use mesh alignment feedback.
        body_sfeat_dim (list): Dimension of the body feature.
        hf_sfeat_dim (list): Dimension of the hands and face feature.
        grid_align (dict): Details of the attention block.
        n_iter (int, optional): Number of iterations. Defaults to 3.
        grid_feat (bool, optional):
            Whether to use grid feature. Defaults to False.
        aux_supv_on (bool, optional): Defaults to True.
        hf_aux_supv_on (bool, optional): Defaults to False.
        mlp_dim (list, optional):
            Dimension of MLP for body. Defaults to [256, 128, 64, 5].
        hf_mlp_dim (list, optional):
            Dimension of MLP for hands and face. Defaults to [256, 128, 64, 5].
        loss_uv_regression_weight (float, optional):
            The loss of the UV regression weight. Defaults to 0.5.
        hf_model_cfg (Optional[Union[dict, None]], optional):
            Hands and face model config. Defaults to None.
        use_iwp_cam (bool, optional):
            Whether to use dentity rotation and Weak Perspective Camera.
            Defaults to True.
        device (str, optional): Defaults to 'cpu'.
        init_cfg (Optional[Union[list, dict, None]], optional):
            Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 backbone: Optional[Union[dict, None]],
                 head: Optional[Union[dict, None]],
                 regressor: Optional[Union[dict, None]],
                 attention_config: str,
                 extra_joints_regressor: str,
                 smplx_to_smpl: str,
                 smplx_model_dir: str,
                 mesh_model: dict,
                 bhf_mode: str,
                 maf_on: bool,
                 body_sfeat_dim: list,
                 hf_sfeat_dim: list,
                 grid_align: dict,
                 n_iter: int = 3,
                 grid_feat: bool = False,
                 aux_supv_on: bool = True,
                 hf_aux_supv_on: bool = False,
                 mlp_dim: list = [256, 128, 64, 5],
                 hf_mlp_dim: list = [256, 128, 64, 5],
                 loss_uv_regression_weight: float = 0.5,
                 hf_model_cfg: Optional[Union[dict, None]] = None,
                 use_iwp_cam: bool = True,
                 device: str = 'cpu',
                 init_cfg: Optional[Union[list, dict, None]] = None):

        super(PyMAFX, self).__init__(init_cfg)
        self.use_iwp_cam = use_iwp_cam
        self.backbone = backbone
        self.smplx_model_dir = smplx_model_dir
        self.smplx_to_smpl = smplx_to_smpl
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
        self.hf_mlp_dim = hf_mlp_dim
        self.aux_supv_on = aux_supv_on
        self.hf_aux_supv_on = hf_aux_supv_on
        self.device = torch.device(device)
        self.maf_on = maf_on
        self.fuse_grid_align = grid_align['use_att'] or grid_align['use_fc']
        assert not (grid_align['use_att'] and grid_align['use_fc'])
        self.grid_feat_dim = GRID_SIZE * GRID_SIZE * self.mlp_dim[-1]
        self.bhf_att_feat_dim = {}
        self._prepare_body_module(extra_joints_regressor)
        self._create_encoder()
        self._create_attention_modules(attention_config)
        self._create_regressor(regressor)
        self._create_maf_extractor()
        head['hf_root_idx'] = self.hf_root_idx
        head['mano_ds_len'] = self.mano_ds_len
        head['bhf_names'] = self.bhf_names
        self.head = build_head(head)

    def _prepare_body_module(self, extra_joints_regressor: str):
        """Prepare parametric mesh models.

        Args:
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
        """
        self.bhf_names = []
        if self.bhf_mode in ['body_hand', 'full_body']:
            self.bhf_names.append('body')
        if self.bhf_mode in ['body_hand', 'full_body']:
            self.bhf_names.append('hand')
        if self.bhf_mode in ['full_body']:
            self.bhf_names.append('face')

        # joint index info
        h_root_idx = MANO_RIGHT_REORDER_KEYPOINTS.index('right_wrist')
        f_idx = FACIAL_LANDMARKS.index('nose_middle')
        self.hf_root_idx = {
            'lhand': h_root_idx,
            'rhand': h_root_idx,
            'face': f_idx
        }
        # create parametric mesh models
        self.smpl_family = {}
        self.smpl_family['body'] = GenderedSMPLX(
            smplx_model_dir=self.smplx_model_dir,
            gender=self.mesh_model['gender'],
            extra_joints_regressor=extra_joints_regressor,
            smplx_to_smpl=self.smplx_to_smpl)

    def _create_encoder(self):
        """Create encoder for body, hands and head image."""
        self.encoders = nn.ModuleDict()
        self.bhf_ma_feat_dim = {}
        # encoder for the body part
        if 'body' in self.bhf_names and self.backbone is not None:
            self.encoders['body'] = build_backbone(self.backbone)

            self.mesh_sampler = Mesh_Sampler(type='smpl')
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
        # encoders for the hand / face parts
        if 'hand' in self.bhf_names or 'face' in self.bhf_names and \
           self.hf_model_cfg is not None:
            for hf in ['hand', 'face']:
                if hf in self.bhf_names:
                    self.encoders[hf] = build_backbone(
                        self.hf_model_cfg['backbone'])
            if self.hf_aux_supv_on:
                assert self.maf_on
                self.dp_head_hf = nn.ModuleDict()
                if 'hand' in self.bhf_names:
                    self.dp_head_hf['hand'] = IUV_predict_layer(
                        feat_dim=self.hf_sfeat_dim[-1], mode='pncc')
                if 'face' in self.bhf_names:
                    self.dp_head_hf['face'] = IUV_predict_layer(
                        feat_dim=self.hf_sfeat_dim[-1], mode='pncc')

    def _create_attention_modules(self, attention_config):
        """Create attention modules.

        Args:
            attention_config (str): Attention config path.
        """
        # the fusion of grid and mesh-aligned features
        if self.fuse_grid_align:
            n_iter_att = self.n_iter - self.grid_align['att_starts']
            self.att_feat_dim_idx = -self.grid_align['att_feat_idx']
            num_att_heads = self.grid_align['att_head']
            hidden_feat_dim = self.mlp_dim[self.att_feat_dim_idx]
            self.bhf_att_feat_dim.update({'body': 2048})
        if 'hand' in self.bhf_names:
            self.mano_sampler = Mesh_Sampler(type='mano', level=1)
            self.mano_ds_len = self.mano_sampler.Dmap.shape[0]

            self.bhf_ma_feat_dim.update(
                {'hand': self.mano_ds_len * self.hf_mlp_dim[-1]})

            if self.fuse_grid_align:
                self.bhf_att_feat_dim.update({'hand': 1024})

        if 'face' in self.bhf_names:
            self.bhf_ma_feat_dim.update(
                {'face': len(FACIAL_LANDMARKS) * self.hf_mlp_dim[-1]})
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
                attention_config,
                self.bhf_names,
                hfimg_feat_dim_list,
                hidden_feat_dim,
                n_iter=n_iter_att,
                num_attention_heads=num_att_heads)

        if self.fuse_grid_align:
            self.att_feat_reduce = get_fusion_modules(
                self.bhf_names,
                self.bhf_ma_feat_dim,
                self.grid_feat_dim,
                n_iter=n_iter_att,
                out_feat_len=self.bhf_att_feat_dim)

    def _create_regressor(self, regressor: Optional[Union[dict, None]]):
        """
        Args:
            regressor (Optional[Union[dict, None]]): Regressor config dict.
        """
        self.regressor = nn.ModuleList()
        for i in range(self.n_iter):
            ref_infeat_dim = 0
            if 'body' in self.bhf_names:
                if self.maf_on:
                    if self.fuse_grid_align:
                        if i >= self.grid_align['att_starts']:
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
                    ref_infeat_dim = GLOBAL_FEAT_DIM

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
                ref_infeat_dim = GLOBAL_FEAT_DIM
                feat_dim_hand = GLOBAL_FEAT_DIM
                feat_dim_face = GLOBAL_FEAT_DIM
            regressor['feat_dim'] = ref_infeat_dim
            regressor['feat_dim_hand'] = feat_dim_hand
            regressor['feat_dim_face'] = feat_dim_face
            regressor['bhf_names'] = self.bhf_names
            regressor['smpl_models'] = self.smpl_family

            self.regressor.append(build_head(regressor))

    def _create_maf_extractor(self):
        """mesh-aligned feature extractor."""
        self.maf_extractor = nn.ModuleDict()
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

                if self.grid_align[
                        'use_att'] and i >= self.grid_align['att_starts']:
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

    def forward_train(self, **kwargs):
        """Forward function for general training.

        For mesh estimation, we do not use this interface.
        """
        pass

    def forward_test(self, batch={}, rw_cam={}, **kwargs):
        """Test step function.

        Args:
            batch (dict, optional):
                'img_{part}': for part images in body, hand and face.
                '{part}_theta_inv': inversed affine transformation for cropped
                    of hand/face images, for part in lhand, rhand, and face.
            rw_cam (dict, optional):
                real-world camera information, applied when
                use_iwp_cam is False.

        Returns:
            out_dict: the list containing the predicted parameters.
        """
        # the limb parts need to be handled
        if self.bhf_mode == 'body_hand':
            part_names = ['lhand', 'rhand']
        if self.bhf_mode == 'full_body':
            part_names = ['lhand', 'rhand', 'face']

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
                if 'lhand' in part_names:
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

        # grid points for grid feature extraction
        xv, yv = torch.meshgrid([
            torch.linspace(-1, 1, GRID_SIZE),
            torch.linspace(-1, 1, GRID_SIZE)
        ])
        grid_points = torch.stack([xv.reshape(-1),
                                   yv.reshape(-1)]).unsqueeze(0)
        # grid-pattern points
        grid_points = torch.transpose(
            grid_points.expand(batch_size, -1, -1), 1, 2).to(self.device)

        # parameter predictions
        out_dict = self.head(batch, s_feat_body, limb_feat_dict, g_feat,
                             grid_points, self.att_feat_reduce,
                             self.align_attention, self.maf_extractor,
                             self.regressor, batch_size, limb_gfeat_dict,
                             part_names)

        return out_dict
