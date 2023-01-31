# yapf: disable
import math

import numpy as np
import scipy
import torch
import torch.cuda.comm
import torch.nn as nn
from mmcv.runner.base_module import BaseModule
from torch.nn import functional as F

from mmhuman3d.core.conventions.keypoints_mapping.flame import (
    FLAME_73_KEYPOINTS,
)
from mmhuman3d.core.conventions.keypoints_mapping.mano import (
    MANO_RIGHT_REORDER_KEYPOINTS,
)
from mmhuman3d.core.conventions.keypoints_mapping.openpose import (
    OPENPOSE_25_KEYPOINTS,
)
from mmhuman3d.core.conventions.keypoints_mapping.spin_smplx import (
    SPIN_SMPLX_KEYPOINTS,
)
from mmhuman3d.models.body_models.smpl import SMPL
from mmhuman3d.models.heads.bert.modeling_bert import (
    BertConfig,
    BertIntermediate,
    BertOutput,
    BertPreTrainedModel,
    BertSelfOutput,
)
from mmhuman3d.models.utils.SMPLX import get_partial_smpl
from mmhuman3d.utils.camera_utils import homo_vector
from mmhuman3d.utils.geometry import (
    compute_twist_rotation,
    projection,
    rot6d_to_rotmat,
    rotation_matrix_to_angle_axis,
)
from mmhuman3d.utils.keypoint_utils import transform_kps2d
from mmhuman3d.utils.transforms import aa_to_rotmat

# yapf: enable
FACIAL_LANDMARKS = FLAME_73_KEYPOINTS[5:]
JOINT_NAMES = OPENPOSE_25_KEYPOINTS + SPIN_SMPLX_KEYPOINTS

LayerNormClass = torch.nn.LayerNorm
BertLayerNorm = torch.nn.LayerNorm


class Mesh_Sampler(nn.Module):
    """Mesh Up/Down-sampling."""

    def __init__(self,
                 type='smpl',
                 level=2,
                 device=torch.device('cuda'),
                 option=None):
        super().__init__()

        # downsample SMPL mesh and assign part labels
        if type == 'smpl':
            smpl_mesh_graph = np.load(
                'data/smpl_downsampling.npz',
                allow_pickle=True,
                encoding='latin1')

            U = smpl_mesh_graph['U']
            D = smpl_mesh_graph['D']  # shape: (2,)
        elif type == 'mano':
            # TODO: replace path
            mano_mesh_graph = np.load(
                'data/mano_downsampling.npz',
                allow_pickle=True,
                encoding='latin1')

            U = mano_mesh_graph['U']
            D = mano_mesh_graph['D']  # shape: (2,)

        # downsampling
        point_downsample = []
        for lv in range(len(D)):
            d = scipy.sparse.coo_matrix(D[lv])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            point_downsample.append(torch.sparse.FloatTensor(i, v, d.shape))

        # downsampling mapping from 6890 points to 431 points
        if level == 2:
            Dmap = torch.matmul(point_downsample[1].to_dense(),
                                point_downsample[0].to_dense())  # 6890 -> 431
        elif level == 1:
            Dmap = point_downsample[0].to_dense()
        self.register_buffer('Dmap', Dmap)

        # upsampling
        ptU = []
        for lv in range(len(U)):
            d = scipy.sparse.coo_matrix(U[lv])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptU.append(torch.sparse.FloatTensor(i, v, d.shape))

        # upsampling mapping from 431 points to 6890 points
        if level == 2:
            Umap = torch.matmul(ptU[0].to_dense(),
                                ptU[1].to_dense())  # 431 -> 6890
        elif level == 1:
            Umap = ptU[0].to_dense()  #
        self.register_buffer('Umap', Umap)

    def downsample(self, x):
        """downsample function."""
        return torch.matmul(self.Dmap.unsqueeze(0), x)  # [B, 431, 3]

    def upsample(self, x):
        """upsample function."""
        return torch.matmul(self.Umap.unsqueeze(0), x)  # [B, 6890, 3]

    def forward(self, x: torch.Tensor, mode='downsample'):
        """Forward function.

        Args:
            x (torch.Tensor): Original point.
            mode (str, optional):Defaults to 'downsample'.

        Returns:
            torch.Tensor: The sampled point.
        """
        if mode == 'downsample':
            return self.downsample(x)
        elif mode == 'upsample':
            return self.upsample(x)


class MAF_Extractor(nn.Module):
    """Mesh-aligned Feature Extractor As discussed in the paper, we extract
    mesh-aligned features based on 2D projection of the mesh vertices.

    The features extracted from spatial feature maps will go through a MLP for
    dimension reduction.
    """

    def __init__(self,
                 filter_channels,
                 device=torch.device('cuda'),
                 iwp_cam_mode=True,
                 option=None):
        super().__init__()

        self.device = device
        self.filters = []
        self.num_views = 1
        self.last_op = nn.ReLU(True)

        self.iwp_cam_mode = iwp_cam_mode

        for ll in range(0, len(filter_channels) - 1):
            if 0 != ll:
                self.filters.append(
                    nn.Conv1d(filter_channels[ll] + filter_channels[0],
                              filter_channels[ll + 1], 1))
            else:
                self.filters.append(
                    nn.Conv1d(filter_channels[ll], filter_channels[ll + 1], 1))

            self.add_module('conv%d' % ll, self.filters[ll])

        # downsample SMPL mesh and assign part labels
        # https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
        smpl_mesh_graph = np.load(
            'data/smpl_downsampling.npz', allow_pickle=True, encoding='latin1')

        U = smpl_mesh_graph['U']
        D = smpl_mesh_graph['D']  # shape: (2,)

        # downsampling
        point_downsample = []
        for level in range(len(D)):
            d = scipy.sparse.coo_matrix(D[level])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            point_downsample.append(torch.sparse.FloatTensor(i, v, d.shape))

        # downsampling mapping from 6890 points to 431 points
        Dmap = torch.matmul(point_downsample[1].to_dense(),
                            point_downsample[0].to_dense())  # 6890 -> 431
        self.register_buffer('Dmap', Dmap)

        # upsampling
        ptU = []
        for level in range(len(U)):
            d = scipy.sparse.coo_matrix(U[level])
            i = torch.LongTensor(np.array([d.row, d.col]))
            v = torch.FloatTensor(d.data)
            ptU.append(torch.sparse.FloatTensor(i, v, d.shape))

        # upsampling mapping from 431 points to 6890 points
        Umap = torch.matmul(ptU[0].to_dense(),
                            ptU[1].to_dense())  # 431 -> 6890
        self.register_buffer('Umap', Umap)

    def reduce_dim(self, feature):
        """Dimension reduction by multi-layer perceptrons.

        Args:
            feature:
            list of [B, C_s, N] point-wise features before dimension reduction

        Returns:
            [B, C_p x N] concatantion of point-wise features after
            dimension reduction
        """
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            y = self._modules['conv' + str(i)](
                y if i == 0 else torch.cat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(-1, self.num_views, y.shape[1],
                           y.shape[2]).mean(dim=1)
                tmpy = feature.view(-1, self.num_views, feature.shape[1],
                                    feature.shape[2]).mean(dim=1)

        y = self.last_op(y)

        # y = y.view(y.shape[0], -1)

        return y

    def sampling(self,
                 points: torch.Tensor,
                 im_feat: torch.Tensor = None,
                 add_att: bool = False,
                 reduce_dim: bool = True):
        """Given 2D points, sample the point-wise features for each point, the
        dimension of point-wise features will be reduced from C_s to C_p by
        MLP. Image features should be pre-computed before this call.

        Args:
            points (torch.Tensor): [B, N, 2] image coordinates of points
            im_feat (torch.Tensor, optional):
                [B, C_s, H_s, W_s] spatial feature maps
            reduce_dim (bool, optional): _description_. Defaults to True.

        Returns:
            point_feat: [B, C_p x N] concatantion of point-wise features after
            dimension reduction
        """

        point_feat = torch.nn.functional.grid_sample(
            im_feat, points.unsqueeze(2), align_corners=False)[..., 0]

        if reduce_dim:
            mesh_align_feat = self.reduce_dim(point_feat)
            return mesh_align_feat
        else:
            return point_feat

    def forward(self,
                p: torch.Tensor,
                im_feat: torch.Tensor,
                cam: torch.Tensor = None,
                add_att=False,
                reduce_dim=True,
                **kwargs):
        """Returns mesh-aligned features for the 3D mesh points.

        Args:
            p (torch.Tensor): [B, N_m, 3] mesh vertices.
            im_feat (torch.Tensor): [B, C_s, H_s, W_s] spatial feature maps.
            cam (torch.Tensor, optional): [B, 3] camera. Defaults to None.
            add_att (bool, optional): Defaults to False.
            reduce_dim (bool, optional): Defaults to True.

        Returns:
            mesh_align_feat (torch.Tensor):
            [B, C_p x N_m] mesh-aligned features.
        """
        p_proj_2d = projection(p, cam, iwp_mode=self.iwp_cam_mode)
        if self.iwp_cam_mode:
            # Normalize keypoints to [-1,1]
            p_proj_2d = p_proj_2d / (224. / 2.)
        else:
            p_proj_2d = transform_kps2d(p_proj_2d, cam['kps_transf'])
        mesh_align_feat = self.sampling(
            p_proj_2d, im_feat, add_att=add_att, reduce_dim=reduce_dim)
        return mesh_align_feat


class IUV_predict_layer(nn.Module):

    def __init__(self,
                 feat_dim=256,
                 final_cov_k=3,
                 out_channels=25,
                 with_uv=True,
                 mode='iuv'):
        super().__init__()

        assert mode in ['iuv', 'seg', 'pncc']
        self.mode = mode

        if mode == 'seg':
            self.predict_ann_index = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=15,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0)

            self.predict_uv_index = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=25,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0)
        elif mode == 'iuv':
            self.predict_u = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=25,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0)

            self.predict_v = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=25,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0)

            self.predict_ann_index = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=15,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0)

            self.predict_uv_index = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=25,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0)
        elif mode in ['pncc']:
            self.predict_pncc = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=3,
                kernel_size=final_cov_k,
                stride=1,
                padding=1 if final_cov_k == 3 else 0)

        self.inplanes = feat_dim

    def forward(self, x):
        """Forward function."""
        return_dict = {}

        if self.mode in ['iuv', 'seg']:
            predict_uv_index = self.predict_uv_index(x)
            predict_ann_index = self.predict_ann_index(x)

            return_dict['predict_uv_index'] = predict_uv_index
            return_dict['predict_ann_index'] = predict_ann_index

            if self.mode == 'iuv':
                predict_u = self.predict_u(x)
                predict_v = self.predict_v(x)
                return_dict['predict_u'] = predict_u
                return_dict['predict_v'] = predict_v
            else:
                return_dict['predict_u'] = None
                return_dict['predict_v'] = None

        if self.mode == 'pncc':
            predict_pncc = self.predict_pncc(x)
            return_dict['predict_pncc'] = predict_pncc

        return return_dict


class Regressor(nn.Module):
    """Regressor for mesh model."""

    def __init__(self,
                 mesh_model,
                 bhf_mode,
                 use_iwp_cam,
                 n_iter,
                 smpl_model_dir,
                 feat_dim,
                 smpl_mean_params,
                 use_cam_feat=True,
                 feat_dim_hand=0,
                 feat_dim_face=0,
                 bhf_names=['body'],
                 smpl_models={},
                 hand_vis_th=0.1,
                 adapt_integr=True,
                 opt_wrist=True,
                 pred_vis_h=True):
        super().__init__()
        self.opt_wrist = opt_wrist
        self.use_iwp_cam = use_iwp_cam
        self.pred_vis_h = pred_vis_h
        self.hand_vis_th = hand_vis_th
        self.adapt_integr = adapt_integr
        self.n_iter = n_iter

        npose = 24 * 6
        shape_dim = 10
        cam_dim = 3
        hand_dim = 15 * 6
        face_dim = 3 * 6 + 10

        self.body_feat_dim = feat_dim
        self.mesh_model = mesh_model
        self.bhf_mode = bhf_mode

        self.smpl_mode = (self.mesh_model['name'] == 'smpl')
        self.smplx_mode = (self.mesh_model['name'] == 'smplx')
        self.use_cam_feat = use_cam_feat

        cam_feat_len = 4 if self.use_cam_feat else 0

        self.bhf_names = bhf_names
        self.body_hand_mode = (bhf_mode == 'body_hand')
        self.full_body_mode = (bhf_mode == 'full_body')

        if 'body' in self.bhf_names:
            self.fc1 = nn.Linear(
                feat_dim + npose + cam_feat_len + shape_dim + cam_dim, 1024)
            self.drop1 = nn.Dropout()
            self.fc2 = nn.Linear(1024, 1024)
            self.drop2 = nn.Dropout()
            self.decpose = nn.Linear(1024, npose)
            self.decshape = nn.Linear(1024, 10)
            self.deccam = nn.Linear(1024, 3)
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        if not self.smpl_mode:
            if self.body_hand_mode:
                self.part_names = ['lhand', 'rhand']
            elif self.full_body_mode:
                self.part_names = ['lhand', 'rhand', 'face']
            else:
                self.part_names = []

            if 'rhand' in self.part_names:
                self.fc1_hand = nn.Linear(feat_dim_hand + hand_dim, 1024)
                self.drop1_hand = nn.Dropout()
                self.fc2_hand = nn.Linear(1024, 1024)
                self.drop2_hand = nn.Dropout()

                # self.declhand = nn.Linear(1024, 15*6)
                self.decrhand = nn.Linear(1024, 15 * 6)
                # nn.init.xavier_uniform_(self.declhand.weight, gain=0.01)
                nn.init.xavier_uniform_(self.decrhand.weight, gain=0.01)

                if self.mesh_model['name'] == 'mano' or self.opt_wrist:
                    rh_cam_dim = 3
                    rh_orient_dim = 6
                    rh_shape_dim = 10
                    self.fc3_hand = nn.Linear(
                        1024 + rh_orient_dim + rh_shape_dim + rh_cam_dim, 1024)
                    self.drop3_hand = nn.Dropout()

                    self.decshape_rhand = nn.Linear(1024, 10)
                    self.decorient_rhand = nn.Linear(1024, 6)
                    self.deccam_rhand = nn.Linear(1024, 3)
                    nn.init.xavier_uniform_(
                        self.decshape_rhand.weight, gain=0.01)
                    nn.init.xavier_uniform_(
                        self.decorient_rhand.weight, gain=0.01)
                    nn.init.xavier_uniform_(
                        self.deccam_rhand.weight, gain=0.01)

            if 'face' in self.part_names:
                self.fc1_face = nn.Linear(feat_dim_face + face_dim, 1024)
                self.drop1_face = nn.Dropout()
                self.fc2_face = nn.Linear(1024, 1024)
                self.drop2_face = nn.Dropout()

                self.dechead = nn.Linear(1024, 3 * 6)
                self.decexp = nn.Linear(1024, 10)
                nn.init.xavier_uniform_(self.dechead.weight, gain=0.01)
                nn.init.xavier_uniform_(self.decexp.weight, gain=0.01)

                if self.mesh_model['name'] == 'flame':
                    rh_cam_dim = 3
                    rh_orient_dim = 6
                    rh_shape_dim = 10
                    self.fc3_face = nn.Linear(
                        1024 + rh_orient_dim + rh_shape_dim + rh_cam_dim, 1024)
                    self.drop3_face = nn.Dropout()

                    self.decshape_face = nn.Linear(1024, 10)
                    self.decorient_face = nn.Linear(1024, 6)
                    self.deccam_face = nn.Linear(1024, 3)
                    nn.init.xavier_uniform_(
                        self.decshape_face.weight, gain=0.01)
                    nn.init.xavier_uniform_(
                        self.decorient_face.weight, gain=0.01)
                    nn.init.xavier_uniform_(self.deccam_face.weight, gain=0.01)

            if self.smplx_mode and self.pred_vis_h:
                self.fc1_vis = nn.Linear(1024 + 1024 + 1024, 1024)
                self.drop1_vis = nn.Dropout()
                self.fc2_vis = nn.Linear(1024, 1024)
                self.drop2_vis = nn.Dropout()
                self.decvis = nn.Linear(1024, 2)
                nn.init.xavier_uniform_(self.decvis.weight, gain=0.01)

        if 'body' in smpl_models:
            self.smpl = smpl_models['body']

        if self.opt_wrist:
            self.body_model = SMPL(
                model_path=smpl_model_dir, batch_size=64, create_transl=False)
        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(
            mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_orient', init_pose[:, :6])

        self.flip_vector = torch.ones((1, 9), dtype=torch.float32)
        self.flip_vector[:, [1, 2, 3, 6]] *= -1
        self.flip_vector = self.flip_vector.reshape(1, 3, 3)

        if not self.smpl_mode:
            lhand_mean_rotmat = aa_to_rotmat(
                self.smpl.model_neutral.left_hand_mean.view(-1, 3))
            bs = lhand_mean_rotmat.shape[0]
            lhand_mean_rot6d = lhand_mean_rotmat[:, :, :2].reshape(bs, 6)

            rhand_mean_rotmat = aa_to_rotmat(
                self.smpl.model_neutral.right_hand_mean.view(-1, 3))
            rhand_mean_rot6d = rhand_mean_rotmat[:, :, :2].reshape(bs, 6)

            init_lhand = lhand_mean_rot6d.reshape(-1).unsqueeze(0)
            init_rhand = rhand_mean_rot6d.reshape(-1).unsqueeze(0)
            init_face_rotmat = torch.stack([torch.eye(3)] * 3)
            init_face = init_face_rotmat[:, :, :2].reshape(
                3, 6).reshape(-1).unsqueeze(0)
            init_exp = torch.zeros(10).unsqueeze(0)

        if self.smplx_mode or 'hand' in bhf_names:
            # init_hand = torch.cat([init_lhand, init_rhand]).unsqueeze(0)
            self.register_buffer('init_lhand', init_lhand)
            self.register_buffer('init_rhand', init_rhand)
        if self.smplx_mode or 'face' in bhf_names:
            self.register_buffer('init_face', init_face)
            self.register_buffer('init_exp', init_exp)

    def forward(self,
                x: torch.Tensor = None,
                n_iter: int = 1,
                rw_cam={},
                init_mode=False,
                global_iter=-1,
                **kwargs):
        """Forward function.

        Args:
            x (torch.Tensor, optional): Defaults to None.
            n_iter (int, optional): Defaults to 1.
            rw_cam (dict, optional): real-world camera information.
            init_mode (bool, optional): Defaults to False.
            global_iter (int, optional): Defaults to -1.

        Returns:
            dict: The parameters of mesh model.
        """
        FOOT_NAMES = ['bigtoe', 'smalltoe', 'heel']
        if x is not None:
            batch_size = x.shape[0]
        else:
            if 'xc_rhand' in kwargs:
                batch_size = kwargs['xc_rhand'].shape[0]
            elif 'xc_face' in kwargs:
                batch_size = kwargs['xc_face'].shape[0]

        if 'body' in self.bhf_names:
            if 'init_pose' not in kwargs:
                kwargs['init_pose'] = self.init_pose.expand(batch_size, -1)
            if 'init_shape' not in kwargs:
                kwargs['init_shape'] = self.init_shape.expand(batch_size, -1)
            if 'init_cam' not in kwargs:
                kwargs['init_cam'] = self.init_cam.expand(batch_size, -1)

            pred_cam = kwargs['init_cam']
            pred_pose = kwargs['init_pose']
            pred_shape = kwargs['init_shape']

        if self.full_body_mode or self.body_hand_mode:
            if self.opt_wrist:
                pred_rotmat_body = rot6d_to_rotmat(
                    pred_pose.reshape(batch_size, -1, 6))
            if self.pred_vis_h:
                pred_vis_hands = None

        if self.smplx_mode or 'hand' in self.bhf_names:
            if 'init_lhand' not in kwargs:
                kwargs['init_lhand'] = self.init_rhand.expand(batch_size, -1)
            if 'init_rhand' not in kwargs:
                kwargs['init_rhand'] = self.init_rhand.expand(batch_size, -1)

            pred_lhand, pred_rhand = kwargs['init_lhand'], kwargs['init_rhand']

            if self.mesh_model['name'] == 'mano' or self.opt_wrist:
                if 'init_orient_rh' not in kwargs:
                    kwargs['init_orient_rh'] = self.init_orient.expand(
                        batch_size, -1)
                if 'init_shape_rh' not in kwargs:
                    kwargs['init_shape_rh'] = self.init_shape.expand(
                        batch_size, -1)
                if 'init_cam_rh' not in kwargs:
                    kwargs['init_cam_rh'] = self.init_cam.expand(
                        batch_size, -1)
                pred_orient_rh = kwargs['init_orient_rh']
                pred_shape_rh = kwargs['init_shape_rh']
                pred_cam_rh = kwargs['init_cam_rh']
                if self.opt_wrist:
                    if 'init_orient_lh' not in kwargs:
                        kwargs['init_orient_lh'] = self.init_orient.expand(
                            batch_size, -1)
                    if 'init_shape_lh' not in kwargs:
                        kwargs['init_shape_lh'] = self.init_shape.expand(
                            batch_size, -1)
                    if 'init_cam_lh' not in kwargs:
                        kwargs['init_cam_lh'] = self.init_cam.expand(
                            batch_size, -1)
                    pred_orient_lh = kwargs['init_orient_lh']
                    pred_shape_lh = kwargs['init_shape_lh']
                    pred_cam_lh = kwargs['init_cam_lh']
                if self.mesh_model['name'] == 'mano':
                    pred_cam = torch.cat(
                        [pred_cam_rh[:, 0:1] * 10., pred_cam_rh[:, 1:]], dim=1)

        if self.smplx_mode or 'face' in self.bhf_names:
            if 'init_face' not in kwargs:
                kwargs['init_face'] = self.init_face.expand(batch_size, -1)
            if 'init_hand' not in kwargs:
                kwargs['init_exp'] = self.init_exp.expand(batch_size, -1)

            pred_face = kwargs['init_face']
            pred_exp = kwargs['init_exp']

            if self.mesh_model['name'] == 'flame' or self.opt_wrist:
                if 'init_orient_fa' not in kwargs:
                    kwargs['init_orient_fa'] = self.init_orient.expand(
                        batch_size, -1)
                pred_orient_fa = kwargs['init_orient_fa']
                if 'init_shape_fa' not in kwargs:
                    kwargs['init_shape_fa'] = self.init_shape.expand(
                        batch_size, -1)
                if 'init_cam_fa' not in kwargs:
                    kwargs['init_cam_fa'] = self.init_cam.expand(
                        batch_size, -1)
                pred_shape_fa = kwargs['init_shape_fa']
                pred_cam_fa = kwargs['init_cam_fa']
                if self.mesh_model['name'] == 'flame':
                    pred_cam = torch.cat(
                        [pred_cam_fa[:, 0:1] * 10., pred_cam_fa[:, 1:]], dim=1)

        if not init_mode:
            for i in range(n_iter):
                if 'body' in self.bhf_names:
                    xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
                    if self.use_cam_feat:
                        if self.use_iwp_cam:
                            # for IWP camera, simply use pre-defined values
                            vfov = torch.ones((batch_size, 1)).to(xc) * 0.8
                            crop_ratio = torch.ones(
                                (batch_size, 1)).to(xc) * 0.3
                            crop_center = torch.ones(
                                (batch_size, 2)).to(xc) * 0.5
                        else:
                            vfov = rw_cam['vfov'][:, None]
                            crop_ratio = rw_cam['crop_ratio'][:, None]
                            crop_center = rw_cam['bbox_center'] / torch.cat([
                                rw_cam['img_w'][:, None], rw_cam['img_h'][:,
                                                                          None]
                            ], 1)
                        xc = torch.cat([xc, vfov, crop_ratio, crop_center], 1)

                    xc = self.fc1(xc)
                    xc = self.drop1(xc)
                    xc = self.fc2(xc)
                    xc = self.drop2(xc)

                    pred_cam = self.deccam(xc) + pred_cam
                    pred_pose = self.decpose(xc) + pred_pose
                    pred_shape = self.decshape(xc) + pred_shape

                if not self.smpl_mode:
                    if self.body_hand_mode:
                        xc_lhand, xc_rhand = kwargs['xc_lhand'], kwargs[
                            'xc_rhand']
                        xc_lhand = torch.cat([xc_lhand, pred_lhand], 1)
                        xc_rhand = torch.cat([xc_rhand, pred_rhand], 1)
                    elif self.full_body_mode:
                        xc_lhand, xc_rhand, xc_face = kwargs[
                            'xc_lhand'], kwargs['xc_rhand'], kwargs['xc_face']
                        xc_lhand = torch.cat([xc_lhand, pred_lhand], 1)
                        xc_rhand = torch.cat([xc_rhand, pred_rhand], 1)
                        xc_face = torch.cat([xc_face, pred_face, pred_exp], 1)

                    if 'lhand' in self.part_names:
                        xc_lhand = self.drop1_hand(self.fc1_hand(xc_lhand))
                        xc_lhand = self.drop2_hand(self.fc2_hand(xc_lhand))
                        pred_lhand = self.decrhand(xc_lhand) + pred_lhand

                        if self.opt_wrist:
                            xc_lhand = torch.cat([
                                xc_lhand, pred_shape_lh, pred_orient_lh,
                                pred_cam_lh
                            ], 1)
                            xc_lhand = self.drop3_hand(self.fc3_hand(xc_lhand))

                            pred_shape_lh = self.decshape_rhand(
                                xc_lhand) + pred_shape_lh
                            pred_orient_lh = self.decorient_rhand(
                                xc_lhand) + pred_orient_lh
                            pred_cam_lh = self.deccam_rhand(
                                xc_lhand) + pred_cam_lh

                    if 'rhand' in self.part_names:
                        xc_rhand = self.drop1_hand(self.fc1_hand(xc_rhand))
                        xc_rhand = self.drop2_hand(self.fc2_hand(xc_rhand))
                        pred_rhand = self.decrhand(xc_rhand) + pred_rhand

                        if self.mesh_model['name'] == 'mano' or self.opt_wrist:
                            xc_rhand = torch.cat([
                                xc_rhand, pred_shape_rh, pred_orient_rh,
                                pred_cam_rh
                            ], 1)
                            xc_rhand = self.drop3_hand(self.fc3_hand(xc_rhand))

                            pred_shape_rh = self.decshape_rhand(
                                xc_rhand) + pred_shape_rh
                            pred_orient_rh = self.decorient_rhand(
                                xc_rhand) + pred_orient_rh
                            pred_cam_rh = self.deccam_rhand(
                                xc_rhand) + pred_cam_rh

                            if self.mesh_model['name'] == 'mano':
                                pred_cam = torch.cat([
                                    pred_cam_rh[:, 0:1] * 10.,
                                    pred_cam_rh[:, 1:] / 10.
                                ],
                                                     dim=1)

                    if 'face' in self.part_names:
                        xc_face = self.drop1_face(self.fc1_face(xc_face))
                        xc_face = self.drop2_face(self.fc2_face(xc_face))
                        pred_face = self.dechead(xc_face) + pred_face
                        pred_exp = self.decexp(xc_face) + pred_exp

                        if self.mesh_model['name'] == 'flame':
                            xc_face = torch.cat([
                                xc_face, pred_shape_fa, pred_orient_fa,
                                pred_cam_fa
                            ], 1)
                            xc_face = self.drop3_face(self.fc3_face(xc_face))

                            pred_shape_fa = self.decshape_face(
                                xc_face) + pred_shape_fa
                            pred_orient_fa = self.decorient_face(
                                xc_face) + pred_orient_fa
                            pred_cam_fa = self.deccam_face(
                                xc_face) + pred_cam_fa

                            if self.mesh_model['name'] == 'flame':
                                pred_cam = torch.cat([
                                    pred_cam_fa[:, 0:1] * 10.,
                                    pred_cam_fa[:, 1:] / 10.
                                ],
                                                     dim=1)

                    if self.full_body_mode or self.body_hand_mode:
                        if self.pred_vis_h:
                            xc_vis = torch.cat([xc, xc_lhand, xc_rhand], 1)

                            xc_vis = self.drop1_vis(self.fc1_vis(xc_vis))
                            xc_vis = self.drop2_vis(self.fc2_vis(xc_vis))
                            pred_vis_hands = self.decvis(xc_vis)

                            pred_vis_lhand = \
                                pred_vis_hands[:, 0] > self.hand_vis_th
                            pred_vis_rhand = \
                                pred_vis_hands[:, 1] > self.hand_vis_th

                        if self.opt_wrist:

                            pred_rotmat_body = rot6d_to_rotmat(
                                pred_pose.reshape(batch_size, -1, 6))
                            # pred_lwrist = pred_rotmat_body[:, 20]
                            # pred_rwrist = pred_rotmat_body[:, 21]

                            pred_gl_body, body_joints = \
                                self.body_model.get_global_rotation(
                                    global_orient=pred_rotmat_body[:, 0:1],
                                    body_pose=pred_rotmat_body[:, 1:])
                            pred_gl_lelbow = pred_gl_body[:, 18]
                            pred_gl_relbow = pred_gl_body[:, 19]

                            target_gl_lwrist = rot6d_to_rotmat(
                                pred_orient_lh.reshape(batch_size, -1, 6))
                            target_gl_lwrist *= self.flip_vector.to(
                                target_gl_lwrist.device)
                            target_gl_rwrist = rot6d_to_rotmat(
                                pred_orient_rh.reshape(batch_size, -1, 6))

                            opt_lwrist = torch.bmm(
                                pred_gl_lelbow.transpose(1, 2),
                                target_gl_lwrist)
                            opt_rwrist = torch.bmm(
                                pred_gl_relbow.transpose(1, 2),
                                target_gl_rwrist)

                            if self.adapt_integr:
                                tpose_joints = self.smpl.get_tpose(
                                    betas=pred_shape)
                                lelbow_twist_axis = nn.functional.normalize(
                                    tpose_joints[:, 20] - tpose_joints[:, 18],
                                    dim=1)
                                relbow_twist_axis = nn.functional.normalize(
                                    tpose_joints[:, 21] - tpose_joints[:, 19],
                                    dim=1)

                                lelbow_twist, lelbow_twist_angle = \
                                    compute_twist_rotation(
                                        opt_lwrist, lelbow_twist_axis)
                                relbow_twist, relbow_twist_angle = \
                                    compute_twist_rotation(
                                        opt_rwrist, relbow_twist_axis)

                                min_angle = -0.4 * float(np.pi)
                                max_angle = 0.4 * float(np.pi)

                                lelbow_twist_angle[
                                    lelbow_twist_angle == torch.clamp(
                                        lelbow_twist_angle, min_angle,
                                        max_angle)] = 0
                                relbow_twist_angle[
                                    relbow_twist_angle == torch.clamp(
                                        relbow_twist_angle, min_angle,
                                        max_angle)] = 0
                                lelbow_twist_angle[lelbow_twist_angle >
                                                   max_angle] -= max_angle
                                lelbow_twist_angle[lelbow_twist_angle <
                                                   min_angle] -= min_angle
                                relbow_twist_angle[relbow_twist_angle >
                                                   max_angle] -= max_angle
                                relbow_twist_angle[relbow_twist_angle <
                                                   min_angle] -= min_angle

                                lelbow_twist = aa_to_rotmat(lelbow_twist_axis *
                                                            lelbow_twist_angle)
                                relbow_twist = aa_to_rotmat(relbow_twist_axis *
                                                            relbow_twist_angle)

                                opt_lwrist = torch.bmm(
                                    lelbow_twist.transpose(1, 2), opt_lwrist)
                                opt_rwrist = torch.bmm(
                                    relbow_twist.transpose(1, 2), opt_rwrist)

                                # left elbow: 18
                                opt_lelbow = torch.bmm(pred_rotmat_body[:, 18],
                                                       lelbow_twist)
                                # right elbow: 19
                                opt_relbow = torch.bmm(pred_rotmat_body[:, 19],
                                                       relbow_twist)

                                if self.pred_vis_h and global_iter == (
                                        self.n_iter - 1):
                                    opt_lwrist_filtered = [
                                        opt_lwrist[_i] if pred_vis_lhand[_i]
                                        else pred_rotmat_body[_i, 20]
                                        for _i in range(batch_size)
                                    ]
                                    opt_rwrist_filtered = [
                                        opt_rwrist[_i] if pred_vis_rhand[_i]
                                        else pred_rotmat_body[_i, 21]
                                        for _i in range(batch_size)
                                    ]
                                    opt_lelbow_filtered = [
                                        opt_lelbow[_i] if pred_vis_lhand[_i]
                                        else pred_rotmat_body[_i, 18]
                                        for _i in range(batch_size)
                                    ]
                                    opt_relbow_filtered = [
                                        opt_relbow[_i] if pred_vis_rhand[_i]
                                        else pred_rotmat_body[_i, 19]
                                        for _i in range(batch_size)
                                    ]

                                    opt_lwrist = torch.stack(
                                        opt_lwrist_filtered)
                                    opt_rwrist = torch.stack(
                                        opt_rwrist_filtered)
                                    opt_lelbow = torch.stack(
                                        opt_lelbow_filtered)
                                    opt_relbow = torch.stack(
                                        opt_relbow_filtered)

                                pred_rotmat_body = torch.cat([
                                    pred_rotmat_body[:, :18],
                                    opt_lelbow.unsqueeze(1),
                                    opt_relbow.unsqueeze(1),
                                    opt_lwrist.unsqueeze(1),
                                    opt_rwrist.unsqueeze(1),
                                    pred_rotmat_body[:, 22:]
                                ], 1)
                            else:
                                if self.pred_vis_h and global_iter == (
                                        self.n_iter - 1):
                                    opt_lwrist_filtered = [
                                        opt_lwrist[_i] if pred_vis_lhand[_i]
                                        else pred_rotmat_body[_i, 20]
                                        for _i in range(batch_size)
                                    ]
                                    opt_rwrist_filtered = [
                                        opt_rwrist[_i] if pred_vis_rhand[_i]
                                        else pred_rotmat_body[_i, 21]
                                        for _i in range(batch_size)
                                    ]

                                    opt_lwrist = torch.stack(
                                        opt_lwrist_filtered)
                                    opt_rwrist = torch.stack(
                                        opt_rwrist_filtered)

                                pred_rotmat_body = torch.cat([
                                    pred_rotmat_body[:, :20],
                                    opt_lwrist.unsqueeze(1),
                                    opt_rwrist.unsqueeze(1),
                                    pred_rotmat_body[:, 22:]
                                ], 1)

        if self.full_body_mode or self.body_hand_mode:
            if self.opt_wrist:
                pred_rotmat = pred_rotmat_body
            else:
                pred_rotmat = rot6d_to_rotmat(
                    pred_pose.reshape(batch_size, -1, 6))
            assert pred_rotmat.shape[1] == 24
        else:
            pred_rotmat = rot6d_to_rotmat(pred_pose.reshape(batch_size, -1, 6))
            assert pred_rotmat.shape[1] == 24

        if self.smplx_mode:
            if self.pred_vis_h and global_iter == (self.n_iter - 1):
                pred_lhand_filtered = [
                    pred_lhand[_i]
                    if pred_vis_lhand[_i] else self.init_rhand[0]
                    for _i in range(batch_size)
                ]
                pred_rhand_filtered = [
                    pred_rhand[_i]
                    if pred_vis_rhand[_i] else self.init_rhand[0]
                    for _i in range(batch_size)
                ]
                pred_lhand_filtered = torch.stack(pred_lhand_filtered)
                pred_rhand_filtered = torch.stack(pred_rhand_filtered)
                pred_hf6d = torch.cat(
                    [pred_lhand_filtered, pred_rhand_filtered, pred_face],
                    dim=1).reshape(batch_size, -1, 6)
            else:
                pred_hf6d = torch.cat([pred_lhand, pred_rhand, pred_face],
                                      dim=1).reshape(batch_size, -1, 6)
            pred_hfrotmat = rot6d_to_rotmat(pred_hf6d)
            assert pred_hfrotmat.shape[1] == (15 * 2 + 3)

            # flip left hand pose
            pred_lhand_rotmat = pred_hfrotmat[:, :15] * self.flip_vector.to(
                pred_hfrotmat.device).unsqueeze(0)
            pred_rhand_rotmat = pred_hfrotmat[:, 15:30]
            pred_face_rotmat = pred_hfrotmat[:, 30:]

        smplx_kwargs = {}
        if self.smplx_mode:
            smplx_kwargs['left_hand_pose'] = pred_lhand_rotmat
            smplx_kwargs['right_hand_pose'] = pred_rhand_rotmat
            smplx_kwargs['jaw_pose'] = pred_face_rotmat[:, 0:1]
            smplx_kwargs['leye_pose'] = pred_face_rotmat[:, 1:2]
            smplx_kwargs['reye_pose'] = pred_face_rotmat[:, 2:3]
            smplx_kwargs['expression'] = pred_exp

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            **smplx_kwargs,
        )

        pred_vertices = pred_output['vertices']
        pred_joints = pred_output['joints']

        if self.smplx_mode:
            pred_joints_full = torch.cat([
                pred_joints, pred_output['lhand_joints'],
                pred_output['rhand_joints'], pred_output['face_joints'],
                pred_output['lfoot_joints'], pred_output['rfoot_joints']
            ],
                                         dim=1)
        else:
            pred_joints_full = pred_joints
        pred_keypoints_2d = projection(
            pred_joints_full, {
                **rw_cam, 'cam_sxy': pred_cam
            },
            iwp_mode=self.use_iwp_cam)
        if self.use_iwp_cam:
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
        else:
            pred_keypoints_2d = transform_kps2d(pred_keypoints_2d,
                                                rw_cam['kps_transf'])

        len_b_kp = len(JOINT_NAMES)
        output = {}
        if self.smpl_mode or self.smplx_mode:
            pose = rotation_matrix_to_angle_axis(
                pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
            output.update({
                'theta':
                torch.cat([pred_cam, pred_shape, pose], dim=1),
                'verts':
                pred_vertices,
                'rotmat':
                pred_rotmat,
                'pred_cam':
                pred_cam,
                'pred_shape':
                pred_shape,
                'pred_pose':
                pred_pose,
            })
            if self.smplx_mode:
                len_h_kp = len(MANO_RIGHT_REORDER_KEYPOINTS)
                len_f_kp = len(FACIAL_LANDMARKS)
                len_feet_kp = 2 * len(FOOT_NAMES)
                output.update({
                    'pred_lhand':
                    pred_lhand,
                    'pred_rhand':
                    pred_rhand,
                    'pred_face':
                    pred_face,
                    'pred_exp':
                    pred_exp,
                    'pred_lhand_rotmat':
                    pred_lhand_rotmat,
                    'pred_rhand_rotmat':
                    pred_rhand_rotmat,
                    'pred_face_rotmat':
                    pred_face_rotmat,
                    'pred_lhand_kp2d':
                    pred_keypoints_2d[:, len_b_kp:len_b_kp + len_h_kp],
                    'pred_rhand_kp2d':
                    pred_keypoints_2d[:, len_b_kp + len_h_kp:len_b_kp +
                                      len_h_kp * 2],
                    'pred_face_kp2d':
                    pred_keypoints_2d[:, len_b_kp + len_h_kp * 2:len_b_kp +
                                      len_h_kp * 2 + len_f_kp],
                    'pred_feet_kp2d':
                    pred_keypoints_2d[:, len_b_kp + len_h_kp * 2 +
                                      len_f_kp:len_b_kp + len_h_kp * 2 +
                                      len_f_kp + len_feet_kp],
                })
                if self.opt_wrist:
                    output.update({
                        'pred_orient_lh': pred_orient_lh,
                        'pred_shape_lh': pred_shape_lh,
                        'pred_orient_rh': pred_orient_rh,
                        'pred_shape_rh': pred_shape_rh,
                        'pred_cam_fa': pred_cam_fa,
                        'pred_cam_lh': pred_cam_lh,
                        'pred_cam_rh': pred_cam_rh,
                    })
                if self.pred_vis_h:
                    output.update({'pred_vis_hands': pred_vis_hands})
        return output


class BertSelfAttention(nn.Module):
    """Bert self-attention block."""

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f'The hidden size {config.hidden_size} is not a multiple of '
                'the number of attention heads {config.num_attention_heads}')
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = \
            self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                head_mask=None,
                history_state=None):
        """Forward function."""
        if history_state is not None:
            raise NotImplementedError
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to
        # get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in
        # BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            raise
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,
                   attention_probs) if self.output_attentions else (
                       context_layer, )
        return outputs


class BertAttention(nn.Module):
    """Bert attention."""

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,
                input_tensor,
                attention_mask,
                head_mask=None,
                history_state=None):
        """Forward function."""
        self_outputs = self.self(input_tensor, attention_mask, head_mask,
                                 history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class AttLayer(nn.Module):
    """Build attention Layer."""

    def __init__(self, config):
        super(AttLayer, self).__init__()
        self.attention = BertAttention(config)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def MHA(self,
            hidden_states,
            attention_mask,
            head_mask=None,
            history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                                           head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + attention_outputs[
            1:]  # add attentions if we output them
        return outputs

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                history_state=None):
        """Forward function."""
        return self.MHA(hidden_states, attention_mask, head_mask,
                        history_state)


class AttEncoder(nn.Module):
    """Build attention encoder."""

    def __init__(self, config):
        super(AttEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [AttLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                encoder_history_states=None):
        """Forward function."""
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            history_state = None if encoder_history_states is None \
                else encoder_history_states[i]
            layer_outputs = layer_module(hidden_states, attention_mask,
                                         head_mask[i], history_state)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        outputs = (hidden_states, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if self.output_attentions:
            outputs = outputs + (all_attentions, )

        return outputs  # outputs, (hidden states), (attentions)


class EncoderBlock(BertPreTrainedModel):
    """Build encoder block."""

    def __init__(self, config):
        super(EncoderBlock, self).__init__(config)
        self.config = config
        self.encoder = AttEncoder(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.img_dim = config.img_feature_dim

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except Exception:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(
            self.img_dim, self.config.hidden_size, bias=True)
        if self.use_img_layernorm:
            self.LayerNorm = LayerNormClass(
                config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)

    def forward(self,
                img_feats: torch.Tensor,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None):
        """Forward function."""
        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        input_ids = torch.zeros([batch_size, seq_length],
                                dtype=torch.long).to(img_feats.device)

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            raise

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            raise

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(
            dtype=img_feats.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            raise
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                    -1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1,
                                             -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters(
            )).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)

        # We empirically observe that adding an additional learnable
        # position embedding leads to more stable training
        embeddings = position_embeddings + img_embedding_output

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings)

        encoder_outputs = self.encoder(
            embeddings, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output, )
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states, )
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions, )

        return outputs


def get_att_block(config_path: str,
                  img_feature_dim=2048,
                  output_feat_dim=512,
                  hidden_feat_dim=1024,
                  num_attention_heads=4,
                  num_hidden_layers=1):
    """Get attention block."""

    config_class = BertConfig
    config = config_class.from_pretrained(config_path)

    interm_size_scale = 2
    config.output_attentions = False
    config.img_feature_dim = img_feature_dim
    config.hidden_size = hidden_feat_dim
    config.intermediate_size = int(config.hidden_size * interm_size_scale)
    config.num_hidden_layers = num_hidden_layers
    config.num_attention_heads = num_attention_heads
    config.max_position_embeddings = 900

    # init a transformer encoder and append it to a list
    assert config.hidden_size % config.num_attention_heads == 0

    att_model = EncoderBlock(config=config)

    return att_model


def get_attention_modules(config_path: str,
                          module_keys: list,
                          img_feature_dim: dict,
                          hidden_feat_dim: int,
                          n_iter: int,
                          num_attention_heads: int = 1):
    """Get attention modules.

    Args:
        config_path (str): Attention config path.
        module_keys (list): Model name.
        img_feature_dim (dict): Image feature dimension.
        hidden_feat_dim (int): Attention feature dimension.
        n_iter (int): Number of iterations.
        num_attention_heads (int, optional): Defaults to 1.

    Returns:
        Attention modules
    """
    align_attention = nn.ModuleDict()
    for k in module_keys:
        align_attention[k] = nn.ModuleList()
        for i in range(n_iter):
            align_attention[k].append(
                get_att_block(
                    config_path,
                    img_feature_dim=img_feature_dim[k][i],
                    hidden_feat_dim=hidden_feat_dim,
                    num_attention_heads=num_attention_heads))

    return align_attention


class PyMAFXHead(BaseModule):
    """PyMAF-X parameters regressor head."""

    def __init__(self,
                 maf_on,
                 n_iter,
                 bhf_mode,
                 grid_align,
                 bhf_names,
                 hf_root_idx,
                 mano_ds_len,
                 grid_feat=False,
                 hf_box_center=True,
                 use_iwp_cam=True,
                 init_cfg=None):
        super(PyMAFXHead, self).__init__(init_cfg=init_cfg)
        self.maf_on = maf_on
        self.bhf_mode = bhf_mode
        self.use_iwp_cam = use_iwp_cam
        self.hf_root_idx = hf_root_idx
        self.mano_ds_len = mano_ds_len
        self.grid_feat = grid_feat
        self.hf_box_center = hf_box_center
        self.grid_align = grid_align
        self.bhf_names = bhf_names
        self.opt_wrist = True
        self.mano_sampler = Mesh_Sampler(type='mano', level=1)
        self.mesh_sampler = Mesh_Sampler(type='smpl')
        self.init_mesh_output = None
        self.batch_size = 1
        self.n_iter = n_iter
        smpl2limb_vert_faces = get_partial_smpl()
        self.smpl2lhand = torch.from_numpy(
            smpl2limb_vert_faces['lhand']['vids']).long()
        self.smpl2rhand = torch.from_numpy(
            smpl2limb_vert_faces['rhand']['vids']).long()

    def init_mesh(self, regressor, batch_size, rw_cam={}):
        """initialize the mesh model with default poses and shapes."""
        if self.init_mesh_output is None or self.batch_size != batch_size:
            self.init_mesh_output = regressor[0](
                torch.zeros(batch_size), rw_cam=rw_cam, init_mode=True)
            self.batch_size = batch_size
        return self.init_mesh_output

    def forward(self,
                batch: dict,
                s_feat_body: list,
                limb_feat_dict: dict,
                g_feat,
                grid_points: torch.Tensor,
                att_feat_reduce,
                align_attention,
                maf_extractor,
                regressor,
                batch_size,
                limb_gfeat_dict,
                part_names=None,
                rw_cam={}):
        """Forward function of PyMAF-X Head.

        Args:
            batch (dict, optional):
                'img_{part}': for part images in body, hand and face.
                '{part}_theta_inv': inversed affine transformation for cropped
                    of hand/face images, for part in lhand, rhand, and face.
            s_feat_body (list): Image feature for body.
            limb_feat_dict (dict): Cropped image feature for part.
            grid_points (torch.Tensor): Grid-pattern points.
            att_feat_reduce: Fusion_modules.
            align_attention: Attention_modules
            maf_extractor: Mesh-aligned feature extractor.
            regressor: Regressor for mesh model.
            batch_size (int): The batch size
            part_names (list, optional): The name of part. Defaults to None.
            rw_cam (dict, optional): real-world camera information.

        Returns:
            out_dict (dict): Dict containing model predictions.
        """
        out_dict = {}
        self.maf_extractor = maf_extractor
        fuse_grid_align = self.grid_align['use_att'] or self.grid_align[
            'use_fc']
        if fuse_grid_align:
            att_starts = self.grid_align['att_starts']
        # initial parameters
        mesh_output = self.init_mesh(regressor, batch_size, rw_cam)
        out_dict['mesh_out'] = [mesh_output]

        for rf_i in range(self.n_iter):
            current_states = {}
            if 'body' in self.bhf_names:
                pred_cam = mesh_output['pred_cam'].detach()
                pred_shape = mesh_output['pred_shape'].detach()
                pred_pose = mesh_output['pred_pose'].detach()

                current_states['init_cam'] = pred_cam
                current_states['init_shape'] = pred_shape
                current_states['init_pose'] = pred_pose

                pred_smpl_verts = mesh_output['verts'].detach()

                if self.maf_on:
                    s_feat_i = s_feat_body[rf_i]

            # re-project mesh on the image plane
            if self.bhf_mode == 'body_hand':
                pred_lhand_v = self.mano_sampler(
                    pred_smpl_verts[:, self.smpl2lhand])
                pred_rhand_v = self.mano_sampler(
                    pred_smpl_verts[:, self.smpl2rhand])
                pred_hand_v = torch.cat([pred_lhand_v, pred_rhand_v], dim=1)
                pred_hand_proj = projection(
                    pred_hand_v, {
                        **rw_cam, 'cam_sxy': pred_cam
                    },
                    iwp_mode=self.use_iwp_cam)
                if self.use_iwp_cam:
                    pred_hand_proj = pred_hand_proj / (224. / 2.)
                else:
                    pred_hand_proj = transform_kps2d(pred_hand_proj,
                                                     rw_cam['kps_transf'])

                proj_hf_center = {
                    'lhand':
                    mesh_output['pred_lhand_kp2d']
                    [:, self.hf_root_idx['lhand']].unsqueeze(1),
                    'rhand':
                    mesh_output['pred_rhand_kp2d']
                    [:, self.hf_root_idx['rhand']].unsqueeze(1),
                }
                proj_hf_pts = {
                    'lhand':
                    torch.cat([
                        proj_hf_center['lhand'],
                        pred_hand_proj[:, :self.mano_ds_len]
                    ],
                              dim=1),
                    'rhand':
                    torch.cat([
                        proj_hf_center['rhand'],
                        pred_hand_proj[:, self.mano_ds_len:]
                    ],
                              dim=1),
                }
            elif self.bhf_mode == 'full_body':
                pred_lhand_v = self.mano_sampler(
                    pred_smpl_verts[:, self.smpl2lhand])
                pred_rhand_v = self.mano_sampler(
                    pred_smpl_verts[:, self.smpl2rhand])
                pred_hand_v = torch.cat([pred_lhand_v, pred_rhand_v], dim=1)
                pred_hand_proj = projection(
                    pred_hand_v, {
                        **rw_cam, 'cam_sxy': pred_cam
                    },
                    iwp_mode=self.use_iwp_cam)
                if self.use_iwp_cam:
                    pred_hand_proj = pred_hand_proj / (224. / 2.)
                else:
                    pred_hand_proj = transform_kps2d(pred_hand_proj,
                                                     rw_cam['kps_transf'])

                proj_hf_center = {
                    'lhand':
                    mesh_output['pred_lhand_kp2d']
                    [:, self.hf_root_idx['lhand']].unsqueeze(1),
                    'rhand':
                    mesh_output['pred_rhand_kp2d']
                    [:, self.hf_root_idx['rhand']].unsqueeze(1),
                    'face':
                    mesh_output['pred_face_kp2d']
                    [:, self.hf_root_idx['face']].unsqueeze(1)
                }
                proj_hf_pts = {
                    'lhand':
                    torch.cat([
                        proj_hf_center['lhand'],
                        pred_hand_proj[:, :self.mano_ds_len]
                    ],
                              dim=1),
                    'rhand':
                    torch.cat([
                        proj_hf_center['rhand'],
                        pred_hand_proj[:, self.mano_ds_len:]
                    ],
                              dim=1),
                    'face':
                    torch.cat([
                        proj_hf_center['face'], mesh_output['pred_face_kp2d']
                    ],
                              dim=1)
                }
            # extract mesh-aligned features for the hand / face part
            if 'hand' in self.bhf_names or 'face' in self.bhf_names:
                limb_rf_i = rf_i
                hand_face_feat = {}

                for hf_i, part_name in enumerate(part_names):
                    if 'hand' in part_name:
                        hf_key = 'hand'
                    elif 'face' in part_name:
                        hf_key = 'face'

                    if self.maf_on:
                        limb_feat_i = limb_feat_dict[part_name][limb_rf_i]

                        limb_reduce_dim = (not fuse_grid_align) or (rf_i <
                                                                    att_starts)

                        if limb_rf_i == 0 or self.grid_feat:
                            limb_ref_feat_ctd = self.maf_extractor[hf_key][
                                limb_rf_i].sampling(
                                    grid_points,
                                    im_feat=limb_feat_i,
                                    reduce_dim=limb_reduce_dim)
                        else:
                            if self.bhf_mode == 'full_body' or \
                               self.bhf_mode == 'body_hand':
                                # convert projection points to the space of
                                # cropped hand/face images
                                theta_i_inv = batch[f'{part_name}_theta_inv']
                                proj_hf_pts_crop = torch.bmm(
                                    theta_i_inv,
                                    homo_vector(proj_hf_pts[part_name]
                                                [:, :, :2]).permute(
                                                    0, 2, 1)).permute(0, 2, 1)

                                if part_name == 'lhand':
                                    flip_x = torch.tensor(
                                        [-1, 1])[None,
                                                 None, :].to(proj_hf_pts_crop)
                                    proj_hf_pts_crop *= flip_x

                                if self.hf_box_center:
                                    # align projection points with the
                                    # cropped img center
                                    part_box_ul = torch.min(
                                        proj_hf_pts_crop,
                                        dim=1)[0].unsqueeze(1)
                                    part_box_br = torch.max(
                                        proj_hf_pts_crop,
                                        dim=1)[0].unsqueeze(1)
                                    part_box_center = (part_box_ul +
                                                       part_box_br) / 2.
                                    proj_hf_pts_crop_ctd = \
                                        proj_hf_pts_crop[:, 1:] - \
                                        part_box_center
                                else:
                                    proj_hf_pts_crop_ctd = proj_hf_pts_crop[:,
                                                                            1:]

                            limb_ref_feat_ctd = self.maf_extractor[hf_key][
                                limb_rf_i].sampling(
                                    proj_hf_pts_crop_ctd.detach(),
                                    im_feat=limb_feat_i,
                                    reduce_dim=limb_reduce_dim)

                        if fuse_grid_align and \
                           limb_rf_i >= att_starts:

                            limb_grid_feature_ctd = self.maf_extractor[hf_key][
                                limb_rf_i].sampling(
                                    grid_points,
                                    im_feat=limb_feat_i,
                                    reduce_dim=limb_reduce_dim)
                            limb_grid_ref_feat_ctd = torch.cat(
                                [limb_grid_feature_ctd, limb_ref_feat_ctd],
                                dim=-1).permute(0, 2, 1)

                            if self.grid_align['use_att']:
                                att_ref_feat_ctd = align_attention[hf_key][
                                    limb_rf_i -
                                    att_starts](limb_grid_ref_feat_ctd)[0]
                            elif self.grid_align['use_fc']:
                                att_ref_feat_ctd = limb_grid_ref_feat_ctd

                            att_ref_feat_ctd = self.maf_extractor[hf_key][
                                limb_rf_i].reduce_dim(
                                    att_ref_feat_ctd.permute(0, 2, 1)).view(
                                        batch_size, -1)
                            limb_ref_feat_ctd = att_feat_reduce[hf_key][
                                limb_rf_i - att_starts](
                                    att_ref_feat_ctd)

                        else:
                            limb_ref_feat_ctd = limb_ref_feat_ctd.view(
                                batch_size, -1)
                        hand_face_feat[part_name] = limb_ref_feat_ctd
                    else:
                        hand_face_feat[part_name] = limb_gfeat_dict[part_name]
            # extract mesh-aligned features for the body part
            if 'body' in self.bhf_names:
                if self.maf_on:
                    reduce_dim = (not fuse_grid_align) or (rf_i < att_starts)
                    if rf_i == 0 or self.grid_feat:
                        ref_feature = self.maf_extractor['body'][
                            rf_i].sampling(
                                grid_points,
                                im_feat=s_feat_i,
                                reduce_dim=reduce_dim)
                    else:
                        # TODO: use a more sparse SMPL implementation
                        # (with 431 vertices) for acceleration
                        pred_smpl_verts_ds = self.mesh_sampler.downsample(
                            pred_smpl_verts)  # [B, 431, 3]
                        ref_feature = self.maf_extractor['body'][rf_i](
                            pred_smpl_verts_ds,
                            im_feat=s_feat_i,
                            cam={
                                **rw_cam, 'cam_sxy': pred_cam
                            },
                            add_att=True,
                            reduce_dim=reduce_dim)  # [B, 431 * n_feat]

                    if fuse_grid_align and rf_i >= att_starts:
                        if rf_i > 0 and not self.grid_feat:
                            grid_feature = self.maf_extractor['body'][
                                rf_i].sampling(
                                    grid_points,
                                    im_feat=s_feat_i,
                                    reduce_dim=reduce_dim)
                            grid_ref_feat = torch.cat(
                                [grid_feature, ref_feature], dim=-1)
                        else:
                            grid_ref_feat = ref_feature
                        grid_ref_feat = grid_ref_feat.permute(0, 2, 1)

                        if self.grid_align['use_att']:
                            att_ref_feat = align_attention['body'][
                                rf_i - att_starts](grid_ref_feat)[0]
                        elif self.grid_align['use_fc']:
                            att_ref_feat = grid_ref_feat

                        att_ref_feat = self.maf_extractor['body'][
                            rf_i].reduce_dim(att_ref_feat.permute(0, 2, 1))
                        att_ref_feat = att_ref_feat.view(batch_size, -1)

                        ref_feature = att_feat_reduce['body'][rf_i -
                                                              att_starts](
                                                                  att_ref_feat)
                    else:
                        ref_feature = ref_feature.view(batch_size, -1)
                else:
                    ref_feature = g_feat
            else:
                ref_feature = None

            if self.bhf_mode == 'body_hand':
                current_states['xc_lhand'] = hand_face_feat['lhand']
                current_states['xc_rhand'] = hand_face_feat['rhand']
            elif self.bhf_mode == 'full_body':
                current_states['xc_lhand'] = hand_face_feat['lhand']
                current_states['xc_rhand'] = hand_face_feat['rhand']
                current_states['xc_face'] = hand_face_feat['face']

            if rf_i > 0:
                for part in part_names:
                    current_states[f'init_{part}'] = mesh_output[
                        f'pred_{part}'].detach()
                    if part == 'face':
                        current_states['init_exp'] = mesh_output[
                            'pred_exp'].detach()
                if self.bhf_mode == 'full_body' or \
                   self.bhf_mode == 'body_hand':
                    if self.opt_wrist:
                        current_states['init_shape_lh'] = mesh_output[
                            'pred_shape_lh'].detach()
                        current_states['init_orient_lh'] = mesh_output[
                            'pred_orient_lh'].detach()
                        current_states['init_cam_lh'] = mesh_output[
                            'pred_cam_lh'].detach()

                        current_states['init_shape_rh'] = mesh_output[
                            'pred_shape_rh'].detach()
                        current_states['init_orient_rh'] = mesh_output[
                            'pred_orient_rh'].detach()
                        current_states['init_cam_rh'] = mesh_output[
                            'pred_cam_rh'].detach()
            # update mesh parameters
            mesh_output = regressor[rf_i](
                ref_feature,
                n_iter=1,
                rw_cam=rw_cam,
                global_iter=rf_i,
                **current_states)

            out_dict['mesh_out'].append(mesh_output)
        return out_dict
