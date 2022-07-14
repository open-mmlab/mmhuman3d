# ------------------------------------------------------------------------------
# Adapted from https://github.com/jiawei-ren/BalancedMSE
# Original licence: Copyright (c) 2022 Jiawei Ren, under the MIT License.
# ------------------------------------------------------------------------------

from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info
from torch.nn.modules.loss import _Loss

from .utils import weighted_loss


@weighted_loss
def bmc_loss_md(pred: torch.Tensor, target: torch.Tensor,
                noise_var: torch.Tensor, all_gather: bool,
                loss_mse_weight: float,
                loss_debias_weight: float) -> torch.Tensor:
    """
    Args:
        pred (torch.Tensor): The prediction. Shape should be (N, L).
        target (torch.Tensor): The learning target of the prediction.
        noise_var (torch.Tensor): Noise var of ground truth distribution.
        all_gather (bool): Whether gather tensors across all sub-processes.
            Only used in DDP training scheme.
        loss_mse_weight (float, optional): The weight of the mse term.
        loss_debias_weight (float, optional): The weight of the debiased term.

    Returns:
            torch.Tensor: The calculated loss
    """
    N = pred.shape[0]
    L = pred.shape[1]
    device = pred.device

    loss_mse = F.mse_loss(pred, target, reduction='none').sum(-1)
    loss_mse = loss_mse / noise_var

    if all_gather:
        rank, world_size = get_dist_info()
        bs, length = target.shape
        all_bs = [torch.zeros(1).to(device) for _ in range(world_size)]
        dist.all_gather(all_bs, torch.Tensor([bs]).to(device))
        all_bs_int = [int(v.item()) for v in all_bs]
        max_bs_int = max(all_bs_int)
        target_padding = torch.zeros(max_bs_int, length).to(device)
        target_padding[:bs] = target
        all_tensor = []
        for _ in range(world_size):
            all_tensor.append(torch.zeros(max_bs_int, length).type_as(target))
        dist.all_gather(all_tensor, target_padding)
        # remove padding
        for i in range(world_size):
            all_tensor[i] = all_tensor[i][:all_bs_int[i]]
        target = torch.cat(all_tensor, dim=0)

    # Debias term
    target = target.unsqueeze(0).repeat(N, 1, 1)
    pred = pred.unsqueeze(1).expand_as(target)
    debias_term = F.mse_loss(pred, target, reduction='none').sum(-1)
    debias_term = -0.5 * debias_term / noise_var
    loss_debias = torch.logsumexp(debias_term, dim=1).squeeze(-1)
    loss = loss_mse * loss_mse_weight + loss_debias * loss_debias_weight
    # recover loss scale of mse_loss
    loss = loss / L * noise_var.detach()
    return loss


class BMCLossMD(_Loss):
    """Balanced MSE loss, use batch monte-carlo to estimate distribution.
    https://arxiv.org/abs/2203.16427.

    Args:
        init_noise_sigma (float, optional): The initial value of noise sigma.
            This sigma is used to represent ground truth distribution.
            Defaults to 1.0.
        all_gather (bool, optional): Whether gather tensors across all
            sub-processes. If set True, BMC will have more precise estimation
            with more time cost. Default: False.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_mse_weight (float, optional): The weight of the mse term.
            Defaults to 1.0.
        loss_debias_weight (float, optional): The weight of the debiased term.
            Defaults to 1.0.
    """

    def __init__(self,
                 init_noise_sigma: Optional[float] = 1.0,
                 all_gather: Optional[bool] = False,
                 reduction: Optional[str] = 'mean',
                 loss_mse_weight: Optional[float] = 1.0,
                 loss_debias_weight: Optional[float] = 1.0):
        super(BMCLossMD, self).__init__()
        self.noise_sigma = torch.nn.Parameter(
            torch.tensor(init_noise_sigma).float())
        self.all_gather = all_gather
        assert reduction in (None, 'none', 'mean', 'sum')
        reduction = 'none' if reduction is None else reduction
        self.reduction = reduction
        self.loss_mse_weight = loss_mse_weight
        self.loss_debias_weight = loss_debias_weight

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            weight: Optional[Union[torch.Tensor, None]] = None,
            avg_factor: Optional[Union[int, None]] = None,
            reduction_override: Optional[Union[str,
                                               None]] = None) -> torch.Tensor:
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        noise_var = (self.noise_sigma**2).type_as(pred)
        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)
        loss = bmc_loss_md(
            pred,
            target,
            noise_var=noise_var,
            all_gather=self.all_gather,
            loss_mse_weight=self.loss_mse_weight,
            loss_debias_weight=self.loss_debias_weight,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss
