import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from torch import Tensor, nn

from mmhuman3d.utils.transforms import (
    aa_to_rotmat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_rot6d,
)
from ..builder import POST_PROCESSING


@POST_PROCESSING.register_module(name=['DeciWatchPostProcessing', 'deciwatch'])
class DeciWatchPostProcessing:
    """DeciWatchFilter lib is from: https://arxiv.org/abs/2203.08713.

    Args:
        interval (int): The interval of Visible frames.
        slide_window_q (int): frames per slide window contains + 1.
        checkpoint (str): model checkpoint path
        device (Union[torch.device, str], optional):
                You can pass a str or torch.device for cpu or gpu render.
                Defaults to 'cpu'.

    Returns:
        smoothed poses (np.ndarray, torch.tensor)
    """

    def __init__(self, interval, slide_window_q, checkpoint, device=None):
        super(DeciWatchPostProcessing, self).__init__()
        self.interval = interval
        self.slide_window_q = slide_window_q
        self.slide_window_size = self.slide_window_q * self.interval + 1
        self.device = device

        self.input_dimension = 24 * 6

        self.model = DeciWatch(sample_interval=self.interval).to(self.device)

        self.checkpoint_path = checkpoint

        print(f'load checkpoint from local path: {self.checkpoint_path}')
        load_checkpoint(
            self.model, self.checkpoint_path, map_location=self.device)

    def __call__(self, x=None):
        # x.shape: [t,24,3]
        seq_len = x.shape[0]
        assert seq_len > self.slide_window_size
        assert x.shape[1:] == (24, 3, 3) or x.shape[1:] == (
            self.input_dimension) or x.shape[1:] == (24, 3)

        if x.shape[1:] == (24, 3, 3):
            input_type = 'matrix'
            x = torch.tensor(x).to(self.device)
            x = rotmat_to_rot6d(x).reshape(-1, self.input_dimension)
        elif x.shape[1:] == (24, 3):
            input_type = 'axis_angles'
            x = torch.tensor(x).to(self.device)
            x = rotmat_to_rot6d(aa_to_rotmat(x.reshape(-1, 3))).reshape(
                -1, self.input_dimension)
        else:
            x = torch.tensor(x).to(self.device)
            x = x.reshape(-1, self.input_dimension)
            input_type = 'rotation_6d'

        input = x.clone()

        slide_window_x = torch.as_strided(
            input, ((seq_len - self.slide_window_size) // (self.interval) + 1,
                    self.slide_window_size, self.input_dimension),
            (self.interval * self.input_dimension, self.input_dimension, 1),
            storage_offset=0).reshape(-1, self.slide_window_size,
                                      self.input_dimension)

        smoothed_len = (
            seq_len - self.slide_window_size
        ) // self.interval * self.interval + self.slide_window_size

        with torch.no_grad():
            smooth_poses, _ = self.model(slide_window_x, self.device)

        output_poses = [[] for i in range(smoothed_len)]

        for i in range(smooth_poses.shape[0]):
            for j in range(self.slide_window_size):
                output_poses[i * self.interval + j].append(smooth_poses[i,
                                                                        j, :])

        smooth_poses = torch.cat(
            (smooth_poses[:, :self.slide_window_size - 1, :].reshape(
                -1, self.input_dimension), smooth_poses[-1, -1, :].reshape(
                    -1, self.input_dimension)),
            dim=0)

        for i in range(smoothed_len):
            output_poses[i] = torch.stack(output_poses[i]).mean(0)

        output_poses = torch.stack(output_poses)

        if smoothed_len < seq_len:
            output_poses = torch.cat((output_poses, x[smoothed_len:, :]),
                                     dim=0)

        if input_type == 'matrix':
            output_poses = rot6d_to_rotmat(output_poses.reshape(
                -1, 6)).reshape(-1, 24, 3, 3)
        elif input_type == 'axis_angles':
            output_poses = rotmat_to_aa(
                rot6d_to_rotmat(output_poses.reshape(-1,
                                                     6))).reshape(-1, 24, 3)

        return output_poses


class PositionEmbeddingSine_1D(nn.Module):
    """This is a more standard version of the position embedding, very similar
    to the one used by the Attention is all you need paper, generalized to work
    on images."""

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=True,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, B, L):

        position = torch.arange(0, L, dtype=torch.float32).unsqueeze(0)
        position = position.repeat(B, 1)

        if self.normalize:
            eps = 1e-6
            position = position / (position[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature**(2 * (torch.div(dim_t, 1)) /
                                   self.num_pos_feats)

        pe = torch.zeros(B, L, self.num_pos_feats * 2)
        pe[:, :, 0::2] = torch.sin(position[:, :, None] / dim_t)
        pe[:, :, 1::2] = torch.cos(position[:, :, None] / dim_t)

        pe = pe.permute(1, 0, 2)

        return pe


class DeciWatch(nn.Module):
    """Apply DeciWatch framework for 10x efficiency.
    "DeciWatch: A Simple Baseline for 10× Efficient 2D and 3D Pose Estimation",
    arXiv'2022. More details can be found in the `paper
    <https://arxiv.org/pdf/2203.08713>` .
    Args:
        input_dim (int): The size of input spatial dimension,
            e.g., 15*2 for 2d pose on the jhmdb dataset
        sample_interval (int): DeciWatch argument. See :class:`DeciWatch`
            for details. The intervals of the uniform sampling.
            The sampling ratio is: 1/sample_interval. Default: 10
        encoder_hidden_dim (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Hidden dimension in the encoder. Default: 64
        decoder_hidden_dim (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Hidden dimension in the decoder. Default: 64
        dropout (float): DeciWatch argument. See :class:`DeciWatch`
            for details. dropout probability. Default: 0.1
        nheads (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Default: 4
        dim_feedforward (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Dimension of feed forward layers.
        enc_layers (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Layers of the encoder. Default: 5
        dec_layers (int): DeciWatch argument. See :class:`DeciWatch`
            for details. Layers of the encoder. Default: 5
        activation (str): DeciWatch argument. See :class:`DeciWatch`
            for details. Activation function in deciwatch.
            Default: 'leaky_relu'
        pre_norm (bool): DeciWatch argument. See :class:`DeciWatch`
            for details. Whether to normalize before positional embedding.
            Default: False
    """

    def __init__(self,
                 input_dim=24 * 6,
                 sample_interval=10,
                 encoder_hidden_dim=16,
                 decoder_hidden_dim=16,
                 dropout=0.1,
                 nheads=4,
                 dim_feedforward=256,
                 enc_layers=3,
                 dec_layers=3,
                 activation='leaky_relu',
                 pre_norm=False):
        super(DeciWatch, self).__init__()
        self.pos_embed_dim = encoder_hidden_dim
        self.pos_embed = self.build_position_encoding(self.pos_embed_dim)

        self.sample_interval = sample_interval

        self.deciwatch_par = {
            'input_dim': input_dim,
            'encoder_hidden_dim': encoder_hidden_dim,
            'decoder_hidden_dim': decoder_hidden_dim,
            'dropout': dropout,
            'nheads': nheads,
            'dim_feedforward': dim_feedforward,
            'enc_layers': enc_layers,
            'dec_layers': dec_layers,
            'activation': activation,
            'pre_norm': pre_norm
        }

        self.transformer = build_model(self.deciwatch_par)

    def build_position_encoding(self, pos_embed_dim):
        N_steps = pos_embed_dim // 2
        position_embedding = PositionEmbeddingSine_1D(N_steps, normalize=True)
        return position_embedding

    def generate_unifrom_mask(self, L, sample_interval=10):
        # 1 unseen 0 see

        seq_len = L
        if (seq_len - 1) % sample_interval != 0:
            raise Exception(
                'The following equation should be satisfied: [Window size] \
                    = [sample interval] * Q + 1, where Q is an integer.')

        sample_mask = np.ones(seq_len, dtype=np.int32)
        sample_mask[::sample_interval] = 0

        encoder_mask = sample_mask
        decoder_mask = np.array([0] * L, dtype=np.int32)

        return torch.tensor(encoder_mask), torch.tensor(decoder_mask)

    def seqence_interpolation(self, motion, rate):

        seq_len = motion.shape[-1]
        indice = torch.arange(seq_len, dtype=int)
        chunk = torch.div(indice, rate).type(torch.long)
        remain = indice % rate

        prev = motion[:, :, chunk * rate]

        next = torch.cat([
            motion[:, :, (chunk[:-1] + 1) * rate], motion[:, :, -1, np.newaxis]
        ], -1)
        remain = remain.to(motion.device)

        interpolate = (prev / rate * (rate - remain)) + (next / rate * remain)

        return interpolate

    def forward(self, sequence, device):
        B, L, C = sequence.shape
        seq = sequence.permute(0, 2, 1)  # B,C,L

        encoder_mask, decoder_mask = self.generate_unifrom_mask(
            L, sample_interval=self.sample_interval)
        encoder_mask = encoder_mask.to(seq.device)
        decoder_mask = decoder_mask.to(seq.device)

        self.input_seq = seq * (1 - encoder_mask.int())
        self.input_seq_interp = self.seqence_interpolation(
            self.input_seq, self.sample_interval)
        # self.input_seq=self.input_seq.reshape(1,1,-1)
        self.encoder_mask = encoder_mask.unsqueeze(0).repeat(B, 1).to(device)
        self.decoder_mask = decoder_mask.unsqueeze(0).repeat(B, 1).to(device)

        self.encoder_pos_embed = self.pos_embed(B, L).to(device)
        self.decoder_pos_embed = self.encoder_pos_embed.clone().to(device)

        self.recover, self.denoise = self.transformer.forward(
            input_seq=self.input_seq.to(torch.float32),
            encoder_mask=self.encoder_mask,
            encoder_pos_embed=self.encoder_pos_embed,
            input_seq_interp=self.input_seq_interp,
            decoder_mask=self.decoder_mask,
            decoder_pos_embed=self.decoder_pos_embed,
            sample_interval=self.sample_interval,
            device=device)

        self.recover = self.recover.permute(1, 0, 2).reshape(B, L, C)
        self.denoise = self.denoise.permute(1, 0, 2).reshape(B, L, C)

        return self.recover, self.denoise


class DeciWatchTransformer(nn.Module):

    def __init__(self,
                 input_nc,
                 encoder_hidden_dim=512,
                 decoder_hidden_dim=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 pre_norm=False):
        super(DeciWatchTransformer, self).__init__()

        self.joints_dim = input_nc
        # bring in semantic (5 frames) temporal information into tokens
        self.decoder_embed = nn.Conv1d(
            self.joints_dim,
            decoder_hidden_dim,
            kernel_size=5,
            stride=1,
            padding=2)

        self.encoder_embed = nn.Linear(self.joints_dim, encoder_hidden_dim)

        encoder_layer = DeciWatchTransformerEncoderLayer(
            encoder_hidden_dim, nhead, dim_feedforward, dropout, activation,
            pre_norm)
        encoder_norm = nn.LayerNorm(encoder_hidden_dim) if pre_norm else None
        self.encoder = DeciWatchTransformerEncoder(encoder_layer,
                                                   num_encoder_layers,
                                                   encoder_norm)

        decoder_layer = DeciWatchTransformerDecoderLayer(
            decoder_hidden_dim, nhead, dim_feedforward, dropout, activation,
            pre_norm)
        decoder_norm = nn.LayerNorm(decoder_hidden_dim)
        self.decoder = DeciWatchTransformerDecoder(decoder_layer,
                                                   num_decoder_layers,
                                                   decoder_norm)

        self.decoder_joints_embed = nn.Linear(decoder_hidden_dim,
                                              self.joints_dim)
        self.encoder_joints_embed = nn.Linear(encoder_hidden_dim,
                                              self.joints_dim)

        # reset parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        self.nhead = nhead

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def interpolate_embedding(self, input, rate):

        tmp = input.clone()
        seq_len = input.shape[0]
        indice = torch.arange(seq_len, dtype=int).to(self.device)
        chunk = torch.div(indice, rate).type(torch.long)
        remain = indice % rate

        prev = tmp[chunk * rate]

        next = torch.cat([tmp[(chunk[:-1] + 1) * rate], tmp[-1].unsqueeze(0)],
                         dim=0)

        interpolate = (prev / rate * (rate - remain.view(-1, 1, 1))) + (
            next / rate * remain.view(-1, 1, 1))

        return interpolate

    def forward(self, input_seq, encoder_mask, encoder_pos_embed,
                input_seq_interp, decoder_mask, decoder_pos_embed,
                sample_interval, device):

        self.device = device

        # flatten NxCxL to LxNxC
        bs, c, _ = input_seq.shape
        input_seq = input_seq.permute(2, 0, 1)
        input_seq_interp = input_seq_interp.permute(2, 0, 1)

        input = input_seq.clone()

        # mask on all sequences:
        trans_src = self.encoder_embed(input_seq)
        mem = self.encode(trans_src, encoder_mask, encoder_pos_embed)
        reco = self.encoder_joints_embed(mem) + input

        interp = self.interpolate_embedding(reco, sample_interval)
        center = interp.clone()
        trans_tgt = self.decoder_embed(interp.permute(1, 2,
                                                      0)).permute(2, 0, 1)

        output = self.decode(mem, encoder_mask, encoder_pos_embed, trans_tgt,
                             decoder_mask, decoder_pos_embed)

        joints = self.decoder_joints_embed(output) + center
        return joints, reco

    def encode(self, src, src_mask, pos_embed):

        mask = torch.eye(src.shape[0]).bool().to(src.device)
        memory = self.encoder(
            src, mask=mask, src_key_padding_mask=src_mask, pos=pos_embed)

        return memory

    def decode(self, memory, memory_mask, memory_pos, tgt, tgt_mask, tgt_pos):
        hs = self.decoder(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
            pos=memory_pos,
            query_pos=tgt_pos)
        return hs


class DeciWatchTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DeciWatchTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DeciWatchTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 encoder_hidden_dim,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 pre_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            encoder_hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(encoder_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, encoder_hidden_dim)

        self.norm1 = nn.LayerNorm(encoder_hidden_dim)
        self.norm2 = nn.LayerNorm(encoder_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask.bool())[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.pre_norm:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class DeciWatchTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 decoder_hidden_dim,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 pre_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            decoder_hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            decoder_hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(decoder_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, decoder_hidden_dim)

        self.norm1 = nn.LayerNorm(decoder_hidden_dim)
        self.norm2 = nn.LayerNorm(decoder_hidden_dim)
        self.norm3 = nn.LayerNorm(decoder_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask.bool())[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask.bool())[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self,
                    tgt,
                    memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask.bool())[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.pre_norm:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask,
                                 pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_model(args):
    return DeciWatchTransformer(
        input_nc=args['input_dim'],
        decoder_hidden_dim=args['decoder_hidden_dim'],
        encoder_hidden_dim=args['encoder_hidden_dim'],
        dropout=args['dropout'],
        nhead=args['nheads'],
        dim_feedforward=args['dim_feedforward'],
        num_encoder_layers=args['enc_layers'],
        num_decoder_layers=args['dec_layers'],
        activation=args['activation'],
        pre_norm=args['pre_norm'],
    )


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    if activation == 'leaky_relu':
        return F.leaky_relu
    raise RuntimeError(F'activation should be relu/gelu, not {activation}.')
