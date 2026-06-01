__all__ = ['build_rerum_loss']


from ..common._base_neural import BaseNeuralUpliftModel
from ..losses import CFRLoss, RERUMLoss
from ..models import CFRNet, DragonNet


def _cfr_imbalance_config(loss: object) -> dict:
    """Read a CFRNet's IPM configuration off its current `CFRLoss` (or defaults)."""
    base = loss if isinstance(loss, CFRLoss) else CFRLoss()
    return {
        'weight': base.p_alpha,
        'imb_fun': base.imb_fun,
        'r_alpha': base.r_alpha,
        'rbf_sigma': base.rbf_sigma,
        'wass_lambda': base.wass_lambda,
        'wass_iterations': base.wass_iterations,
        'wass_bpt': base.wass_bpt,
        'use_p_correction': base.use_p_correction,
    }


def build_rerum_loss(
    model: BaseNeuralUpliftModel,
    *,
    within_ranking_weight: float,
    cross_ranking_weight: float,
    listwise_ranking_weight: float,
    ranking_sample_size: int | None,
) -> RERUMLoss:
    """Build a `RERUMLoss` with the structural terms appropriate for `model`.

    This is where the framework — not the model — encodes its per-model knowledge:
    DragonNet contributes a treatment-classification term, CFRNet contributes its
    counterfactual-imbalance (IPM) term, other models contribute neither.
    """
    extras: dict = {}
    if isinstance(model, DragonNet):
        extras['treatment_bce_weight'] = 1.0
    if isinstance(model, CFRNet):
        extras['imbalance'] = _cfr_imbalance_config(model.loss)
    return RERUMLoss(
        within_ranking_weight=within_ranking_weight,
        cross_ranking_weight=cross_ranking_weight,
        listwise_ranking_weight=listwise_ranking_weight,
        ranking_sample_size=ranking_sample_size,
        **extras,
    )
