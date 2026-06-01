import torch

from uplift_forecast.losses import (
    RERUMLoss,
    _cross_group_pair_loss,
    _listwise_uplift_rank_loss,
    _within_group_pair_loss,
)


def test_within_group_perfect_ranking_is_zero():
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([0.5, 1.5, 2.5])  # same order as y_true
    assert _within_group_pair_loss(y_true, y_pred, 'sum').item() == 0.0


def test_within_group_penalises_discordant_pairs():
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([3.0, 2.0, 1.0])  # fully reversed
    assert _within_group_pair_loss(y_true, y_pred, 'sum').item() > 0.0


def test_within_group_empty_is_zero():
    empty = torch.empty(0)
    assert _within_group_pair_loss(empty, empty, 'mean').item() == 0.0


def test_cross_group_concordant_is_zero():
    obs_t = torch.tensor([[2.0]])
    pred_t = torch.tensor([[3.0]])
    obs_c = torch.tensor([[1.0]])
    pred_c = torch.tensor([[0.0]])
    # tc: (3-1)*(2-0)=4>=0 -> 0 ; ct: (0-2)*(1-3)=4>=0 -> 0
    assert _cross_group_pair_loss(obs_t, pred_t, obs_c, pred_c, 'sum').item() == 0.0


def test_cross_group_discordant_matches_hand_value():
    obs_t = torch.tensor([[1.0]])
    pred_t = torch.tensor([[5.0]])
    obs_c = torch.tensor([[1.0]])
    pred_c = torch.tensor([[5.0]])
    # tc: M1=5-1=4, M2=1-5=-4, product<0 -> (4-(-4))^2=64 ; ct symmetric -> 64
    assert _cross_group_pair_loss(obs_t, pred_t, obs_c, pred_c, 'sum').item() == 128.0


def test_cross_group_empty_is_zero():
    obs_t = torch.empty(0, 1)
    obs_c = torch.tensor([[1.0]])
    assert _cross_group_pair_loss(obs_t, obs_t, obs_c, obs_c, 'sum').item() == 0.0


def test_listwise_rewards_aligned_uplift():
    # Treated unit has a high response and should be ranked high by uplift.
    y_true = torch.tensor([[3.0], [0.0]])
    is_tr = torch.tensor([True, False])
    is_ct = torch.tensor([False, True])
    aligned = torch.tensor([[5.0], [0.0]])  # high uplift on the high-response treated unit
    reversed_ = torch.tensor([[0.0], [5.0]])
    loss_aligned = _listwise_uplift_rank_loss(y_true, is_tr, is_ct, aligned)
    loss_reversed = _listwise_uplift_rank_loss(y_true, is_tr, is_ct, reversed_)
    assert loss_aligned.item() < loss_reversed.item()


def _random_batch(batch_size=16, seed=0) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)
    y = torch.rand(batch_size, 1, generator=gen)
    y[y < 0.4] = 0.0
    t = (torch.rand(batch_size, 1, generator=gen) > 0.5).float()
    y0 = torch.randn(batch_size, 3, generator=gen, requires_grad=True)
    y1 = torch.randn(batch_size, 3, generator=gen, requires_grad=True)
    t_pred = torch.rand(batch_size, 1, generator=gen)
    eps = torch.randn(batch_size, 1, generator=gen)
    return y, t, y0, y1, t_pred, eps


def test_forward_returns_terms_and_is_differentiable():
    y, t, y0, y1, t_pred, eps = _random_batch()
    loss_fn = RERUMLoss(
        within_ranking_weight=1e-3,
        cross_ranking_weight=1e-3,
        listwise_ranking_weight=1.0,
        treatment_bce_weight=1.0,
        tarreg_weight=1e-3,
    )
    out = loss_fn(y_true=y, t_true=t, y0_pred=y0, y1_pred=y1, t_pred=t_pred, eps=eps)
    expected = {
        'loss',
        'loss_ziln_ct',
        'loss_ziln_tr',
        'loss_within_ranking',
        'loss_cross_ranking',
        'loss_listwise_ranking',
        'loss_imbalance',
    }
    assert expected <= set(out)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert y0.grad is not None
    assert y1.grad is not None


def test_imbalance_term_active_with_representations():
    y, t, y0, y1, t_pred, _ = _random_batch()
    reps = torch.randn(y.shape[0], 5)
    loss_fn = RERUMLoss(imbalance={'weight': 1.0, 'imb_fun': 'mmd2_lin'})
    out = loss_fn(y_true=y, t_true=t, y0_pred=y0, y1_pred=y1, t_pred=t_pred, representations_norm=reps)
    assert out['loss_imbalance'].item() != 0.0


def test_sampling_keeps_loss_finite():
    y, t, y0, y1, t_pred, _ = _random_batch(batch_size=32)
    loss_fn = RERUMLoss(within_ranking_weight=1.0, cross_ranking_weight=1.0, ranking_sample_size=4)
    out = loss_fn(y_true=y, t_true=t, y0_pred=y0, y1_pred=y1, t_pred=t_pred)
    assert torch.isfinite(out['loss'])
