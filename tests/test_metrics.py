import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc

from uplift_forecast.metrics import (
    arm_score_summary,
    auuc_score,
    cumulative_gain_curve,
    kendall_uplift,
    lift_at_k,
    pehe,
    qini_curve,
    qini_score,
    uplift_component_mae,
    uplift_component_mse,
)

# A tiny, fully hand-checkable example: 2 treated, 2 control.
# Ordered by descending PERFECT uplift the rows are [idx2, idx0, idx1, idx3].
Y = np.array([0.0, 1.0, 2.0, 3.0])
T = np.array([1, 0, 1, 0])
PERFECT = Y * T - Y * (1 - T)  # oracle individual effect, also the score normaliser


@pytest.fixture
def ranked_data():
    """Outcomes whose true uplift correlates with a known signal (seeded)."""
    rng = np.random.default_rng(0)
    n = 400
    uplift_true = rng.normal(size=n)
    t = rng.integers(0, 2, size=n)
    y = (uplift_true * t + rng.normal(size=n)).astype(float)
    return y, uplift_true, t


class TestCurves:
    def test_cumulative_gain_curve_shape_and_anchor(self):
        x, gain = cumulative_gain_curve(Y, PERFECT, T)
        assert x.shape == gain.shape == (len(Y) + 1,)
        assert np.array_equal(x, np.arange(len(Y) + 1))
        assert gain[0] == 0.0

    def test_cumulative_gain_curve_values_hand_computed(self):
        # k=1: (2-0)*1=2 ; k=2: (1-0)*2=2 ; k=3: (1-1)*3=0 ; k=4: (1-2)*4=-4
        _, gain = cumulative_gain_curve(Y, PERFECT, T)
        assert gain.tolist() == [0.0, 2.0, 2.0, 0.0, -4.0]

    def test_qini_curve_endpoint_matches_formula(self):
        _, qini = qini_curve(Y, PERFECT, T)
        expected = (Y * T).sum() - (Y * (1 - T)).sum() * (T.sum() / (1 - T).sum())
        assert qini[-1] == pytest.approx(expected)

    def test_ties_use_stable_ordering(self):
        # All-equal uplift must keep the input order (stable sort), not reshuffle.
        x, gain = cumulative_gain_curve(Y, np.zeros_like(Y), T)
        assert np.array_equal(x, np.arange(len(Y) + 1))
        assert np.isfinite(gain).all()


class TestScores:
    @pytest.mark.parametrize('score_fn', [auuc_score, qini_score])
    def test_perfect_ordering_scores_one(self, score_fn):
        # The oracle uplift makes the model curve coincide with the normaliser curve.
        assert score_fn(Y, PERFECT, T) == pytest.approx(1.0)

    @pytest.mark.parametrize('score_fn', [auuc_score, qini_score])
    def test_reversed_ordering_is_negative(self, score_fn):
        assert score_fn(Y, -PERFECT, T) < 0.0

    @pytest.mark.parametrize('score_fn', [auuc_score, qini_score])
    def test_informative_model_beats_its_reverse(self, score_fn, ranked_data):
        y, uplift_true, t = ranked_data
        assert score_fn(y, uplift_true, t) > score_fn(y, -uplift_true, t)

    def test_normalize_false_returns_raw_auc(self):
        x, gain = cumulative_gain_curve(Y, PERFECT, T)
        assert auuc_score(Y, PERFECT, T, normalize=False) == pytest.approx(auc(x, gain))

    def test_constant_uplift_keeps_score_finite(self, ranked_data):
        y, _, t = ranked_data
        assert np.isfinite(auuc_score(y, np.zeros_like(y), t))


class TestInputValidation:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match='same length'):
            auuc_score(np.zeros(3), np.zeros(4), np.zeros(3))

    @pytest.mark.parametrize('score_fn', [auuc_score, qini_score])
    def test_multi_arm_treatment_returns_per_arm_dict(self, score_fn):
        # Non-binary treatment dispatches to the per-arm path (one score per treated arm).
        rng = np.random.default_rng(0)
        n = 200
        t = rng.integers(0, 3, size=n)
        y = rng.normal(size=n)
        uplift = rng.normal(size=(n, 2))
        scores = score_fn(y, uplift, t)
        assert isinstance(scores, dict)
        assert set(scores) == {1, 2}
        assert all(np.isfinite(v) for v in scores.values())

    @pytest.mark.parametrize('score_fn', [auuc_score, qini_score])
    def test_accepts_list_inputs(self, score_fn):
        # ArrayLike (not only ndarray) must be accepted at the boundary.
        assert score_fn([0.0, 1.0, 2.0, 3.0], list(PERFECT), [1, 0, 1, 0]) == pytest.approx(1.0)


class TestMultiArm:
    @staticmethod
    def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        n = 400
        t = rng.integers(0, 3, size=n)
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        y = (x1 + (t == 1) * x1 + (t == 2) * 0.5 * x2 + rng.normal(size=n)).astype(float)
        uplift = np.column_stack([x1, 0.5 * x2])
        return y, uplift, t

    @pytest.mark.parametrize('curve_fn', [cumulative_gain_curve, qini_curve])
    def test_curves_are_per_arm_dicts(self, curve_fn):
        y, u, t = self._data()
        curves = curve_fn(y, u, t)
        assert set(curves) == {1, 2}
        x, gain = curves[1]
        assert x.shape == gain.shape
        assert gain[0] == 0.0

    def test_lift_at_k_per_arm(self):
        y, u, t = self._data()
        lift = lift_at_k(y, u, t, 0.3)
        assert set(lift) == {1, 2}
        assert all(np.isfinite(v) for v in lift.values())
        assert np.isfinite(lift_at_k(y[t < 2], u[t < 2, 0], t[t < 2]))  # binary -> float

    def test_lift_at_k_rejects_bad_fraction(self):
        y, u, t = self._data()
        with pytest.raises(ValueError, match='k must be'):
            lift_at_k(y, u, t, 0.0)

    def test_arm_score_summary(self):
        y, u, t = self._data()
        mean, std = arm_score_summary(qini_score(y, u, t))
        assert np.isfinite(mean)
        assert std >= 0.0
        assert arm_score_summary(0.5) == (0.5, 0.0)  # binary float collapses

    def test_pehe_and_kendall(self):
        y, u, t = self._data()
        true = np.column_stack([np.full(len(t), 0.3), np.full(len(t), 0.7)])
        assert set(pehe(true, u)) == {1, 2}
        assert set(kendall_uplift(true, u)) == {1, 2}
        assert isinstance(pehe(true[:, 0], u[:, 0]), float)
        assert isinstance(kendall_uplift(true[:, 0], u[:, 0]), float)


class TestComponentMetrics:
    @pytest.fixture
    def component_df(self):
        return pd.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0],
            'w': [0, 0, 1, 1],
            'y_pred_ct': [1.5, 2.5, 0.0, 0.0],
            'y_pred_tr': [0.0, 0.0, 3.5, 3.0],
        })

    def test_mae_components_and_weighted_total(self, component_df):
        mae_ct, mae_tr, mae_total = uplift_component_mae(component_df)
        assert mae_ct == pytest.approx(0.5)       # mean(|1-1.5|, |2-2.5|)
        assert mae_tr == pytest.approx(0.75)      # mean(|3-3.5|, |4-3|)
        assert mae_total == pytest.approx(0.625)  # (0.5*2 + 0.75*2) / 4

    def test_mse_components_and_weighted_total(self, component_df):
        mse_ct, mse_tr, mse_total = uplift_component_mse(component_df)
        assert mse_ct == pytest.approx(0.25)      # mean(0.25, 0.25)
        assert mse_tr == pytest.approx(0.625)     # mean(0.25, 1.0)
        assert mse_total == pytest.approx((0.25 * 2 + 0.625 * 2) / 4)

    def test_total_is_arm_size_weighted_not_plain_mean(self):
        # Unequal arm sizes: the total must weight by per-arm counts.
        df = pd.DataFrame({
            'y': [0.0, 0.0, 0.0, 10.0],
            'w': [0, 0, 0, 1],
            'y_pred_ct': [1.0, 1.0, 1.0, 0.0],
            'y_pred_tr': [0.0, 0.0, 0.0, 12.0],
        })
        mae_ct, mae_tr, mae_total = uplift_component_mae(df)
        assert mae_ct == pytest.approx(1.0)
        assert mae_tr == pytest.approx(2.0)
        assert mae_total == pytest.approx((1.0 * 3 + 2.0 * 1) / 4)
        # the naive unweighted mean would be 1.5 — guard against that regression
        assert mae_total != pytest.approx(1.5)
