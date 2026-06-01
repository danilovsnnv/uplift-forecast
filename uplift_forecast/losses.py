__all__ = [
    'MAELoss',
    'MSELoss',
    'PseudoPEHE',
    'DragonNetLoss',
    'RERUMLoss',
    'CFRLoss',
    'zero_inflated_lognormal_pred',
    'zero_inflated_lognormal_loss',
    'compute_imbalance',
    'safe_sqrt',
    'mmd2_rbf',
    'mmd2_lin',
    'lindisc',
    'wasserstein',
]


from collections.abc import Iterable
from typing import Literal

import torch
import torch.distributions as tdist
import torch.nn.functional as F
from torch import nn

_REDUCTIONS = {
    'mean': torch.mean,
    'sum': torch.sum,
    'none': lambda t: t,
}


class _PointDecodeMixin:
    """Loss whose outcome head emits a single point estimate per arm.

    `outputsize` is the per-arm head width; `decode` maps raw head output to the
    predicted outcome. Models read both to size their heads and decode predictions,
    so they stay agnostic to the specific loss.
    """

    outputsize = 1

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        return logits


class _ZilnDecodeMixin:
    """Loss whose outcome head emits zero-inflated-lognormal params `(logit, loc, scale)`."""

    outputsize = 3

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        return zero_inflated_lognormal_pred(logits)


class MSELoss(_PointDecodeMixin, nn.Module):
    """Factual MSE over the two outcome heads (selects the head matching `t_true`)."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        **kwargs,
    ) -> dict:
        del kwargs
        factual = t_true * y1_pred + (1 - t_true) * y0_pred
        return {'loss': F.mse_loss(factual, y_true, reduction=self.reduction)}


class MAELoss(_PointDecodeMixin, nn.Module):
    """Factual MAE over the two outcome heads (selects the head matching `t_true`)."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        **kwargs,
    ) -> dict:
        del kwargs
        factual = t_true * y1_pred + (1 - t_true) * y0_pred
        return {'loss': F.l1_loss(factual, y_true, reduction=self.reduction)}


class PseudoPEHE(_PointDecodeMixin, nn.Module):
    """IPW pseudo-PEHE for validation.

    Compares the average predicted uplift against an IPW estimate of the true
    ATE on the batch. Useful when the true individual treatment effect is
    unobserved.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        y1_pred: torch.Tensor,
        y0_pred: torch.Tensor,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        t_pred: torch.Tensor,
        *args,
        **kwargs,
    ) -> dict:
        del args, kwargs
        y1_pred = y1_pred.float()
        y0_pred = y0_pred.float()
        y_true = y_true.float()
        t_true = t_true.float()
        t_pred = torch.clamp(t_pred.float(), self.eps, 1.0 - self.eps)

        w_t = t_true / t_pred
        w_c = (1.0 - t_true) / (1.0 - t_pred)

        y_true_t = torch.sum(w_t * y_true) / (torch.sum(w_t) + self.eps)
        y_true_c = torch.sum(w_c * y_true) / (torch.sum(w_c) + self.eps)

        uplift_true = y_true_t - y_true_c
        uplift_pred_mean = (y1_pred - y0_pred).mean()
        pehe = (uplift_pred_mean - uplift_true) ** 2

        if self.reduction == 'sum':
            pehe = pehe.sum()
        elif self.reduction == 'mean':
            pehe = pehe.mean()
        return {'loss': pehe}


# ---------------------------------------------------------------------------
# Zero-inflated lognormal helpers (DragonNet output head)
# ---------------------------------------------------------------------------


def zero_inflated_lognormal_pred(logits: torch.Tensor) -> torch.Tensor:
    """Decode `[batch, 3]` ZILN logits into predicted mean.

    Args:
        logits: Tensor of shape `[batch, 3]`. Columns are `(positive_logit, loc, scale_raw)`.

    Returns:
        Tensor of shape `[batch, 1]` — `P(y > 0) * E[y | y > 0]`.
    """
    positive_probs = torch.sigmoid(logits[:, [0]])
    loc = torch.clip(logits[:, [1]], min=-10, max=10)
    scale = torch.clip(F.softplus(logits[:, [2]]) + 1e-6, max=5)
    return positive_probs * torch.exp(loc + 0.5 * scale**2)


def zero_inflated_lognormal_loss(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Per-sample ZILN NLL + BCE classification loss."""
    positive = (labels > 0).float()
    classification_loss = F.binary_cross_entropy_with_logits(
        logits[:, [0]],
        positive,
        reduction='mean',
    )
    loc = logits[..., 1:2]
    scale = torch.max(
        F.softplus(logits[..., 2:]),
        torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps)),
    )
    safe_labels = positive * labels + (1 - positive) * torch.ones_like(labels)
    log_prob = tdist.LogNormal(loc=loc, scale=scale).log_prob(safe_labels)
    regression_loss = -torch.mean(positive * log_prob, dim=-1)
    return classification_loss + regression_loss


# ---------------------------------------------------------------------------
# Shared ZILN / ranking helpers (used by DragonNetLoss and RERUMLoss)
# ---------------------------------------------------------------------------


def _get_log_normal_dist(loc: torch.Tensor, scale: torch.Tensor) -> tdist.Distribution:
    loc_upper_bound = 11
    scale_upper_bound = 5
    eps = 1e-6
    scale = F.softplus(scale) + eps
    loc = torch.clip(loc, max=loc_upper_bound)
    scale = torch.clip(scale, max=scale_upper_bound)
    return tdist.LogNormal(loc=loc, scale=scale)


def _ziln_mean(logits: torch.Tensor) -> torch.Tensor:
    """Decode `[batch, 3]` ZILN logits into `P(y > 0) * E[y | y > 0]`."""
    return zero_inflated_lognormal_pred(logits)


def _ziln_component_loss(y_true: torch.Tensor, logits: torch.Tensor, reduction: str) -> torch.Tensor:
    """ZILN objective (Eq. 3-4): purchasing-propensity BCE + payer-expense lognormal NLL."""
    logit, loc, scale = torch.split(logits, 1, dim=1)
    is_positive = y_true > 0
    y_pos = y_true[is_positive]
    dist_pos = _get_log_normal_dist(loc=loc[is_positive], scale=scale[is_positive])
    regression_loss = -_REDUCTIONS[reduction](dist_pos.log_prob(y_pos))
    classification_loss = F.binary_cross_entropy_with_logits(logit, is_positive.float())
    return regression_loss + classification_loss


def _within_group_pair_loss(y_true: torch.Tensor, y_pred: torch.Tensor, reduction: str) -> torch.Tensor:
    """Within-group response ranking (Eq. 10): penalise discordant pairs only."""
    if y_true.numel() == 0:
        return torch.tensor(0.0, device=y_true.device)
    y_true = y_true.unsqueeze(1)
    y_pred = y_pred.unsqueeze(1)
    y_true_matrix = y_true - y_true.T
    y_pred_matrix = y_pred - y_pred.T
    loss = (y_pred_matrix - y_true_matrix) ** 2
    # zero-out pairs already ordered correctly
    mask = (y_true_matrix * y_pred_matrix) >= 0
    loss = loss.masked_fill(mask, 0.0)
    return _REDUCTIONS[reduction](loss)


def _cross_group_pair_loss(
    obs_t: torch.Tensor,
    pred_t: torch.Tensor,
    obs_c: torch.Tensor,
    pred_c: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Cross-group response ranking (Eq. 12-13).

    Args:
        obs_t: Observed treated responses `y1_i`, shape `[Nt, 1]`.
        pred_t: Predicted treated means `ŷ1_i`, shape `[Nt, 1]`.
        obs_c: Observed control responses `y0_j`, shape `[Nc, 1]`.
        pred_c: Predicted control means `ŷ0_j`, shape `[Nc, 1]`.
        reduction: `'mean'`, `'sum'`, or `'none'`.
    """
    if obs_t.numel() == 0 or obs_c.numel() == 0:
        return torch.tensor(0.0, device=obs_t.device)

    obs_t = obs_t.reshape(-1, 1)
    pred_t = pred_t.reshape(-1, 1)
    obs_c = obs_c.reshape(1, -1)
    pred_c = pred_c.reshape(1, -1)

    # treatment-control pairs (i in Dt, j in Dc)
    m1_tc = pred_t - obs_c
    m2_tc = obs_t - pred_c
    loss_tc = (m1_tc - m2_tc) ** 2
    loss_tc = loss_tc.masked_fill((m1_tc * m2_tc) >= 0, 0.0)

    # control-treatment pairs (i in Dc, j in Dt): swap the 1/0 roles
    m1_ct = pred_c.reshape(-1, 1) - obs_t.reshape(1, -1)
    m2_ct = obs_c.reshape(-1, 1) - pred_t.reshape(1, -1)
    loss_ct = (m1_ct - m2_ct) ** 2
    loss_ct = loss_ct.masked_fill((m1_ct * m2_ct) >= 0, 0.0)

    return _REDUCTIONS[reduction](loss_tc) + _REDUCTIONS[reduction](loss_ct)


def _listwise_uplift_rank_loss(
    y_true: torch.Tensor,
    is_tr: torch.Tensor,
    is_ct: torch.Tensor,
    uplift: torch.Tensor,
) -> torch.Tensor:
    """Listwise uplift ranking (Eq. 20): softmax of `τ̂` over the whole batch."""
    log_softmax = F.log_softmax(uplift, dim=0)
    loss = torch.tensor(0.0, device=uplift.device)
    if is_tr.any():
        loss = loss - torch.sum(y_true[is_tr] * log_softmax[is_tr]) / is_tr.sum()
    if is_ct.any():
        loss = loss + torch.sum(y_true[is_ct] * log_softmax[is_ct]) / is_ct.sum()
    return loss


# ---------------------------------------------------------------------------
# DragonNet loss (RERUM-style: ZILN heads + treatment BCE + ranking + tarreg)
# ---------------------------------------------------------------------------


class DragonNetLoss(_ZilnDecodeMixin, nn.Module):
    """Composite DragonNet objective.

    Args:
        uplift_ranking_weight: Weight on the listwise uplift-ranking term.
        outcome_ranking_weight: Weight on the within-group outcome-ranking term.
        alpha: Treatment-classification weight.
        beta: Targeted-regularization weight (set 0 to disable).
        reduction: `'mean'`, `'sum'`, or `'none'` — passed to the BCE term.
    """

    def __init__(
        self,
        uplift_ranking_weight: float = 1e-10,
        outcome_ranking_weight: float = 1e-20,
        alpha: float = 1.0,
        beta: float = 0.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.uplift_ranking_weight = uplift_ranking_weight
        self.outcome_ranking_weight = outcome_ranking_weight
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        t_pred: torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        eps: torch.Tensor,
        *args,
        **kwargs,
    ) -> dict:
        del args, kwargs
        is_ct = (t_true == 0.0).squeeze()
        is_tr = (t_true == 1.0).squeeze()

        loss_ziln_ct = _ziln_component_loss(y_true[is_ct], y0_pred[is_ct], self.reduction)
        loss_ziln_tr = _ziln_component_loss(y_true[is_tr], y1_pred[is_tr], self.reduction)

        loss_uplift_ranking = self._uplift_ranking_loss(y_true, t_true, y0_pred, y1_pred)
        loss_outcome_ranking = self._outcome_ranking_loss(y_true, t_true, y0_pred, y1_pred)

        loss_y = (
            loss_ziln_ct
            + loss_ziln_tr
            + self.uplift_ranking_weight * loss_uplift_ranking
            + self.outcome_ranking_weight * loss_outcome_ranking
        )
        loss_t = F.binary_cross_entropy(t_pred, t_true, reduction=self.reduction)
        loss = loss_y + self.alpha * loss_t

        if self.beta > 0:
            y_pred_ct = _ziln_mean(y0_pred)
            y_pred_tr = _ziln_mean(y1_pred)
            # workaround to avoid div by zero
            t_pred_safe = (t_pred + 0.01) / 1.02
            y_pred = (1 - t_true) * y_pred_ct + t_true * y_pred_tr
            h = t_true / t_pred_safe - (1 - t_true) / (1 - t_pred_safe)
            y_pert = y_pred + eps * h
            targeted_regularization = F.mse_loss(y_true, y_pert, reduction=self.reduction)
            loss = loss + self.beta * targeted_regularization

        return {
            'loss': loss,
            'loss_ziln_ct': loss_ziln_ct,
            'loss_ziln_tr': loss_ziln_tr,
            'loss_uplift_ranking': self.uplift_ranking_weight * loss_uplift_ranking,
            'loss_outcome_ranking': self.outcome_ranking_weight * loss_outcome_ranking,
        }

    def _outcome_ranking_loss(
        self,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
    ) -> torch.Tensor:
        is_ct = t_true == 0.0
        is_tr = t_true == 1.0
        y0 = _ziln_mean(y0_pred)
        y1 = _ziln_mean(y1_pred)
        loss_ct = _within_group_pair_loss(y_true[is_ct], y0[is_ct], self.reduction)
        loss_tr = _within_group_pair_loss(y_true[is_tr], y1[is_tr], self.reduction)
        return loss_ct + loss_tr

    def _uplift_ranking_loss(
        self,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
    ) -> torch.Tensor:
        is_ct = t_true == 0.0
        is_tr = t_true == 1.0
        uplift = _ziln_mean(y1_pred) - _ziln_mean(y0_pred)
        log_softmax_ct = F.log_softmax(uplift[is_ct], dim=0)
        log_softmax_tr = F.log_softmax(uplift[is_tr], dim=0)
        loss_ct = torch.mean(y_true[is_ct] * log_softmax_ct)
        loss_tr = torch.mean(y_true[is_tr] * log_softmax_tr)
        loss = loss_ct - loss_tr
        if self.reduction == 'sum':
            loss *= t_true.shape[0]
        return loss


# ---------------------------------------------------------------------------
# RERUM loss (Rankability-enhanced Revenue Uplift Modeling, He et al., KDD'24)
# ---------------------------------------------------------------------------


class RERUMLoss(_ZilnDecodeMixin, nn.Module):
    """Rankability-enhanced revenue uplift objective (paper Eq. 21).

    Combines the zero-inflated-lognormal response regression with the three
    rankability terms of the RERUM framework:

    ``L = L_ZILN + within_w * L_within + cross_w * L_cross + listwise_w * L_listwise``

    plus, when the base model provides them, a treatment-classification term
    (DragonNet) and a counterfactual-imbalance term (CFRNet). The L2 ``λ‖θ‖²``
    term of Eq. 21 is applied through the optimizer's ``weight_decay`` rather
    than here.

    Outcome predictions ``y0_pred`` / ``y1_pred`` are ZILN logits ``[batch, 3]``
    (``logit, loc, scale``); they are decoded to means before the ranking terms.

    Args:
        within_ranking_weight: Weight on the within-group response ranking (Eq. 10-11).
        cross_ranking_weight: Weight on the cross-group response ranking (Eq. 12-13).
        listwise_ranking_weight: Weight on the listwise uplift ranking (Eq. 20).
        ranking_sample_size: Number of individuals sampled per group for the pairwise
            (within / cross) terms (Algorithm 1). ``None`` uses every pair.
        treatment_bce_weight: Weight on the treatment-classification BCE (DragonNet's α).
            Requires ``t_pred``; set 0 to disable.
        tarreg_weight: Weight on the targeted regularization (DragonNet's β). Requires
            ``eps`` and ``t_pred``; set 0 to disable.
        imbalance: Optional dict configuring the CFRNet IPM term, with keys ``weight``
            and the ``compute_imbalance`` parameters (``imb_fun``, ``r_alpha``,
            ``rbf_sigma``, ``wass_lambda``, ``wass_iterations``, ``wass_bpt``,
            ``use_p_correction``). Requires ``representations_norm``.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        within_ranking_weight: float = 1e-4,
        cross_ranking_weight: float = 0.0,
        listwise_ranking_weight: float = 10.0,
        ranking_sample_size: int | None = None,
        treatment_bce_weight: float = 0.0,
        tarreg_weight: float = 0.0,
        imbalance: dict | None = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.within_ranking_weight = within_ranking_weight
        self.cross_ranking_weight = cross_ranking_weight
        self.listwise_ranking_weight = listwise_ranking_weight
        self.ranking_sample_size = ranking_sample_size
        self.treatment_bce_weight = treatment_bce_weight
        self.tarreg_weight = tarreg_weight
        self.imbalance = imbalance
        self.reduction = reduction

    def forward(
        self,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        t_pred: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
        representations_norm: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        del kwargs
        is_ct = (t_true == 0.0).squeeze(-1)
        is_tr = (t_true == 1.0).squeeze(-1)

        loss_ziln_ct = _ziln_component_loss(y_true[is_ct], y0_pred[is_ct], self.reduction)
        loss_ziln_tr = _ziln_component_loss(y_true[is_tr], y1_pred[is_tr], self.reduction)

        y0_mean = _ziln_mean(y0_pred)
        y1_mean = _ziln_mean(y1_pred)

        loss_within, loss_cross = self._response_ranking(y_true, is_tr, is_ct, y0_mean, y1_mean)
        loss_listwise = _listwise_uplift_rank_loss(y_true, is_tr, is_ct, y1_mean - y0_mean)

        loss = (
            loss_ziln_ct
            + loss_ziln_tr
            + self.within_ranking_weight * loss_within
            + self.cross_ranking_weight * loss_cross
            + self.listwise_ranking_weight * loss_listwise
        )

        if self.treatment_bce_weight > 0 and t_pred is not None:
            loss = loss + self.treatment_bce_weight * F.binary_cross_entropy(
                t_pred, t_true, reduction=self.reduction,
            )

        if self.tarreg_weight > 0 and eps is not None and t_pred is not None:
            t_pred_safe = (t_pred + 0.01) / 1.02
            y_pred = (1 - t_true) * y0_mean + t_true * y1_mean
            h = t_true / t_pred_safe - (1 - t_true) / (1 - t_pred_safe)
            y_pert = y_pred + eps * h
            loss = loss + self.tarreg_weight * F.mse_loss(y_true, y_pert, reduction=self.reduction)

        loss_imbalance = self._imbalance(t_true, t_pred, representations_norm)
        loss = loss + loss_imbalance

        return {
            'loss': loss,
            'loss_ziln_ct': loss_ziln_ct,
            'loss_ziln_tr': loss_ziln_tr,
            'loss_within_ranking': self.within_ranking_weight * loss_within,
            'loss_cross_ranking': self.cross_ranking_weight * loss_cross,
            'loss_listwise_ranking': self.listwise_ranking_weight * loss_listwise,
            'loss_imbalance': loss_imbalance,
        }

    def _response_ranking(
        self,
        y_true: torch.Tensor,
        is_tr: torch.Tensor,
        is_ct: torch.Tensor,
        y0_mean: torch.Tensor,
        y1_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_t = self._sample_idx(int(is_tr.sum()), y_true.device)
        idx_c = self._sample_idx(int(is_ct.sum()), y_true.device)
        obs_t, pred_t = y_true[is_tr][idx_t], y1_mean[is_tr][idx_t]
        obs_c, pred_c = y_true[is_ct][idx_c], y0_mean[is_ct][idx_c]

        loss_within = _within_group_pair_loss(
            obs_t.reshape(-1), pred_t.reshape(-1), self.reduction,
        ) + _within_group_pair_loss(obs_c.reshape(-1), pred_c.reshape(-1), self.reduction)
        loss_cross = _cross_group_pair_loss(obs_t, pred_t, obs_c, pred_c, self.reduction)
        return loss_within, loss_cross

    def _sample_idx(self, n: int, device: torch.device) -> torch.Tensor:
        if self.ranking_sample_size is None or n <= self.ranking_sample_size:
            return torch.arange(n, device=device)
        return torch.randperm(n, device=device)[: self.ranking_sample_size]

    def _imbalance(
        self,
        t_true: torch.Tensor,
        t_pred: torch.Tensor | None,
        representations_norm: torch.Tensor | None,
    ) -> torch.Tensor:
        weight = (self.imbalance or {}).get('weight', 0.0)
        if not weight or representations_norm is None:
            return torch.tensor(0.0, device=t_true.device)
        cfg = self.imbalance
        if cfg.get('use_p_correction', False) and t_pred is not None:
            propensity = t_pred
        else:
            propensity = torch.full_like(t_true, 0.5)
        imb_loss, _, _ = compute_imbalance(
            representations_norm,
            t_true,
            propensity,
            imb_fun=cfg.get('imb_fun', 'mmd2_lin'),
            r_alpha=cfg.get('r_alpha', 1.0),
            rbf_sigma=cfg.get('rbf_sigma', 0.1),
            wass_lambda=cfg.get('wass_lambda', 10.0),
            wass_iterations=cfg.get('wass_iterations', 50),
            wass_bpt=cfg.get('wass_bpt', False),
        )
        return weight * imb_loss


# ---------------------------------------------------------------------------
# CFRNet imbalance helpers
# ---------------------------------------------------------------------------


def safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable square root used in the IPM penalty."""
    return torch.sqrt(torch.clamp(x, min=eps))


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(1)
    return tensor


def _split_groups(
    representations: torch.Tensor,
    treatment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = treatment.view(-1) > 0.5
    return representations[mask], representations[~mask]


def _weight_groups(
    treatment: torch.Tensor,
    propensity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    propensity = _ensure_2d(propensity)
    treatment = _ensure_2d(treatment.float())
    treated_w = treatment / (2.0 * torch.clamp(propensity, min=1e-6))
    control_w = (1.0 - treatment) / (2.0 * torch.clamp(1.0 - propensity, min=1e-6))
    return treated_w, control_w


def _cdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return torch.sum(diff * diff, dim=-1)


def mmd2_rbf(
    representations: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor | float,
    sigma: float,
) -> torch.Tensor:
    """Squared MMD with an RBF kernel."""
    if isinstance(propensity, float):
        propensity = torch.full(
            treatment.shape,
            fill_value=propensity,
            device=representations.device,
            dtype=representations.dtype,
        )
    propensity = _ensure_2d(propensity)
    treatment = _ensure_2d(treatment.float())
    treated, control = _split_groups(representations, treatment)
    if treated.numel() == 0 or control.numel() == 0:
        return torch.tensor(0.0, device=representations.device)

    bandwidth = 2.0 * sigma**2
    kt = torch.exp(-_cdist2(treated, treated) / bandwidth)
    kc = torch.exp(-_cdist2(control, control) / bandwidth)
    kx = torch.exp(-_cdist2(treated, control) / bandwidth)

    wt, wc = _weight_groups(treatment, propensity)
    wt = wt[treatment.view(-1) > 0.5]
    wc = wc[treatment.view(-1) <= 0.5]
    wt = wt / (wt.sum(dim=0, keepdim=True) + 1e-12)
    wc = wc / (wc.sum(dim=0, keepdim=True) + 1e-12)

    mmd_tt = torch.sum(kt * (wt @ wt.t()))
    mmd_cc = torch.sum(kc * (wc @ wc.t()))
    mmd_tc = torch.sum(kx * (wt @ wc.t()))
    return mmd_tt + mmd_cc - 2.0 * mmd_tc


def mmd2_lin(
    representations: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor | float,
) -> torch.Tensor:
    """Linear-kernel MMD (matches feature means)."""
    if isinstance(propensity, float):
        propensity = torch.full(
            treatment.shape,
            fill_value=propensity,
            device=representations.device,
            dtype=representations.dtype,
        )
    propensity = _ensure_2d(propensity)
    treatment = _ensure_2d(treatment.float())
    weights_t, weights_c = _weight_groups(treatment, propensity)
    treated_mask = treatment.view(-1) > 0.5
    treated = representations[treated_mask]
    control = representations[~treated_mask]
    if treated.numel() == 0 or control.numel() == 0:
        return torch.tensor(0.0, device=representations.device)

    mean_t = torch.sum(treated * weights_t[treated_mask], dim=0) / (
        torch.sum(weights_t[treated_mask], dim=0) + 1e-12
    )
    mean_c = torch.sum(control * weights_c[~treated_mask], dim=0) / (
        torch.sum(weights_c[~treated_mask], dim=0) + 1e-12
    )
    diff = mean_t - mean_c
    return torch.sum(diff * diff)


def lindisc(
    representations: torch.Tensor,
    propensity: torch.Tensor | float,
    treatment: torch.Tensor,
) -> torch.Tensor:
    """Linear discrepancy: sqrt of `mmd2_lin`."""
    return safe_sqrt(mmd2_lin(representations, treatment, propensity))


def wasserstein(
    representations: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor | float,
    lam: float,
    its: int,
    *,
    sq: bool,
    backprop_t: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regularised Wasserstein distance via Sinkhorn iterations."""
    del propensity  # unused, kept for API parity
    treatment = _ensure_2d(treatment.float())
    treated, control = _split_groups(representations, treatment)
    if treated.numel() == 0 or control.numel() == 0:
        zero = torch.tensor(0.0, device=representations.device)
        return zero, zero

    cost = torch.cdist(treated, control, p=2)
    if sq:
        cost = cost.pow(2)
    kernel = torch.exp(-cost / lam)
    v = torch.ones(control.size(0), device=representations.device) / control.size(0)
    u = torch.ones(treated.size(0), device=representations.device) / treated.size(0)
    kernel_t = kernel.transpose(0, 1)

    for _ in range(max(its, 1)):
        u = u / (kernel @ v + 1e-9)
        v = v / (kernel_t @ u + 1e-9)

    transport = torch.diag(u) @ kernel @ torch.diag(v)
    distance = torch.sum(transport * cost)
    if not backprop_t:
        transport = transport.detach()
    return distance, transport


LossLiteral = Literal['mse', 'l1', 'log', 'smooth_l1']
ImbalanceLiteral = Literal['mmd2_rbf', 'mmd2_lin', 'mmd_rbf', 'mmd_lin', 'wass', 'wass2']


def compute_imbalance(
    representations: torch.Tensor,
    treatment: torch.Tensor,
    propensity: torch.Tensor,
    *,
    imb_fun: ImbalanceLiteral,
    r_alpha: float = 1.0,
    rbf_sigma: float = 0.1,
    wass_lambda: float = 10.0,
    wass_iterations: int = 50,
    wass_bpt: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Integral-probability-metric imbalance between treated/control representations.

    Returns `(imb_loss, imb_dist, imb_mat)` where `imb_mat` is the Sinkhorn transport
    plan for the Wasserstein families and `None` otherwise. `propensity` is expected to
    be already adjusted for any `use_p_correction` decision by the caller.
    """
    imb_mat: torch.Tensor | None = None

    if imb_fun == 'mmd2_rbf':
        imb_dist = mmd2_rbf(representations, treatment, propensity, rbf_sigma)
        imb_loss = r_alpha * imb_dist
    elif imb_fun == 'mmd2_lin':
        imb_dist = mmd2_lin(representations, treatment, propensity)
        imb_loss = r_alpha * imb_dist
    elif imb_fun == 'mmd_rbf':
        imb_dist = torch.abs(mmd2_rbf(representations, treatment, propensity, rbf_sigma))
        imb_loss = safe_sqrt((r_alpha**2) * imb_dist)
    elif imb_fun == 'mmd_lin':
        imb_dist = mmd2_lin(representations, treatment, propensity)
        imb_loss = safe_sqrt((r_alpha**2) * imb_dist)
    elif imb_fun in {'wass', 'wass2'}:
        imb_dist, imb_mat = wasserstein(
            representations,
            treatment,
            propensity,
            lam=wass_lambda,
            its=wass_iterations,
            sq=imb_fun == 'wass2',
            backprop_t=wass_bpt,
        )
        imb_loss = r_alpha * imb_dist
    else:
        imb_dist = lindisc(representations, propensity, treatment)
        imb_loss = r_alpha * imb_dist

    return imb_loss, imb_dist, imb_mat


class CFRLoss(_PointDecodeMixin, nn.Module):
    """Counterfactual Regression objective.

    Args:
        loss_type: Factual outcome loss — `'mse'`, `'l1'`, `'log'`, or `'smooth_l1'`.
        reweight_sample: Whether to apply IPW sample reweighting.
        p_alpha: Weight on the integral-probability-metric (IPM) imbalance term.
        p_lambda: Weight on L2 weight decay across heads (and representation if
            `rep_weight_decay`).
        rep_weight_decay: Include representation-network weights in the L2 penalty.
        use_p_correction: Use observed propensity for the IPM term rather than
            a fixed 0.5.
        imb_fun: IPM family — one of `mmd2_rbf`, `mmd2_lin`, `mmd_rbf`,
            `mmd_lin`, `wass`, `wass2`.
        r_alpha: Multiplier applied to the IPM distance.
        r_lambda: Multiplier applied to the weight-decay term.
        rbf_sigma: Bandwidth of the RBF kernel (when applicable).
        wass_lambda: Sinkhorn entropic regularisation.
        wass_iterations: Number of Sinkhorn iterations.
        wass_bpt: Whether to backprop through transport plan.
    """

    def __init__(
        self,
        loss_type: LossLiteral = 'mse',
        reweight_sample: bool = False,
        p_alpha: float = 0.0,
        p_lambda: float = 0.0,
        rep_weight_decay: bool = False,
        use_p_correction: bool = False,
        imb_fun: ImbalanceLiteral = 'mmd2_lin',
        r_alpha: float = 1.0,
        r_lambda: float = 1.0,
        rbf_sigma: float = 0.1,
        wass_lambda: float = 10.0,
        wass_iterations: int = 50,
        wass_bpt: bool = False,
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reweight_sample = reweight_sample
        self.p_alpha = p_alpha
        self.p_lambda = p_lambda
        self.rep_weight_decay = rep_weight_decay
        self.use_p_correction = use_p_correction
        self.imb_fun = imb_fun.lower()
        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.rbf_sigma = rbf_sigma
        self.wass_lambda = wass_lambda
        self.wass_iterations = wass_iterations
        self.wass_bpt = wass_bpt

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        t_pred: torch.Tensor,
        representations_norm: torch.Tensor,
        rep_layers: Iterable[nn.Linear],
        head_modules: Iterable[nn.ModuleDict],
        *args,
        **kwargs,
    ) -> dict:
        del args, kwargs
        device = t_true.device
        treatment = _ensure_2d(t_true)
        factual = _ensure_2d(y_true)
        propensity = self._prepare_propensity(t_pred, treatment, device)

        sample_weight = self._compute_sample_weight(treatment, propensity, device)
        risk_loss, pred_metric = self._prediction_loss(y_pred, factual, sample_weight)

        wd_loss = torch.tensor(0.0, device=device)
        if self.p_lambda > 0.0:
            wd_loss = self._weight_decay(rep_layers, head_modules, device)

        imb_loss, imb_dist, imb_mat = self._imbalance(representations_norm, treatment, propensity)

        imb_term = self.p_alpha * imb_loss if self.p_alpha > 0.0 else torch.tensor(0.0, device=device)
        wd_term = (
            self.p_lambda * self.r_lambda * wd_loss
            if self.p_lambda > 0.0
            else torch.tensor(0.0, device=device)
        )

        total_loss = risk_loss + imb_term + wd_term
        return {
            'loss': total_loss,
            'pred_loss': pred_metric,
            'imb_loss': imb_loss,
            'imb_dist': imb_dist,
            'wd_loss': wd_loss,
            'imb_term': imb_term,
            'wd_term': wd_term,
            'sample_weight': sample_weight,
            'imb_mat': imb_mat,
        }

    def _prepare_propensity(
        self,
        propensity: torch.Tensor,
        treatment: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(propensity):
            propensity = torch.tensor(propensity, device=device, dtype=treatment.dtype)
        if propensity.dim() == 0:
            propensity = propensity.view(1, 1).expand_as(treatment)
        elif propensity.dim() == 1:
            propensity = propensity.unsqueeze(1)
        return propensity

    def _compute_sample_weight(
        self,
        treatment: torch.Tensor,
        propensity: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.reweight_sample:
            return torch.ones_like(treatment, device=device, dtype=treatment.dtype)
        treat_w = treatment / (2.0 * torch.clamp(propensity, min=1e-6))
        control_w = (1.0 - treatment) / (2.0 * torch.clamp(1.0 - propensity, min=1e-6))
        return treat_w + control_w

    def _prediction_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = targets - predictions

        if self.loss_type == 'l1':
            weighted = sample_weight * residual.abs()
            loss = weighted.mean()
            return loss, loss

        if self.loss_type == 'log':
            probs = torch.clamp(torch.sigmoid(predictions), 0.0025, 0.9975)
            log_loss = -sample_weight * (
                targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs)
            )
            loss = log_loss.mean()
            return loss, loss

        if self.loss_type == 'mse':
            sq_error = residual.pow(2)
            weighted_mse = (sample_weight * sq_error).mean()
            rmse = safe_sqrt(sq_error.mean())
            return weighted_mse, rmse

        if self.loss_type == 'smooth_l1':
            beta = 0.01
            abs_res = residual.abs()
            loss_elements = torch.where(
                abs_res < beta,
                0.5 * (abs_res**2) / beta,
                abs_res - 0.5 * beta,
            )
            weighted = sample_weight * loss_elements
            loss = weighted.mean()
            rmse_like = safe_sqrt(loss_elements).mean()
            return loss, rmse_like

        raise ValueError(f'Unknown loss_type={self.loss_type!r}.')

    def _weight_decay(
        self,
        rep_layers: Iterable[nn.Linear],
        head_modules: Iterable[nn.ModuleDict],
        device: torch.device,
    ) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=device)
        if self.rep_weight_decay:
            for layer in rep_layers:
                penalty = penalty + layer.weight.pow(2).sum()
        for head in head_modules:
            for layer in head['fcs']:
                penalty = penalty + layer.weight.pow(2).sum()
            penalty = penalty + head['pred'].weight.pow(2).sum()
        return penalty

    def _imbalance(
        self,
        representations: torch.Tensor,
        treatment: torch.Tensor,
        propensity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        propensity_for_ipm = propensity if self.use_p_correction else torch.full_like(propensity, 0.5)
        return compute_imbalance(
            representations,
            treatment,
            propensity_for_ipm,
            imb_fun=self.imb_fun,
            r_alpha=self.r_alpha,
            rbf_sigma=self.rbf_sigma,
            wass_lambda=self.wass_lambda,
            wass_iterations=self.wass_iterations,
            wass_bpt=self.wass_bpt,
        )
