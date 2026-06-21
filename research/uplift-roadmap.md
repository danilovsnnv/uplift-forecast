# Uplift-forecast — research roadmap & technical tasks

Content research for extending `uplift-forecast` with modern, robust uplift models and
frameworks. Each section gives: **arXiv papers**, **OSS reference implementations**, a
**minimal code sketch** against our `UpliftModel` / model↔loss contract, and a
**developer TODO** (technical task).

Method: deep-research fan-out (web search → fetch → 3-vote adversarial verification →
synthesis), 23 sources fetched, 25 claims verified 3-0, plus targeted follow-up for the
topics the verifier ran out of budget on (time-aware, OPE, validators, explainable,
AutoUplift). Verified claims are marked ✅; follow-up findings are marked 🔎.

> **Guiding caveat (verified ✅, KDD 2024 benchmark, [arXiv:2406.00335](https://arxiv.org/abs/2406.00335)):**
> there is **no stable state-of-the-art deep uplift model** across datasets and preprocessing
> after sufficient tuning, and recent models often differ little from traditional ones under
> distribution shift. **Prioritize breadth, robust baselines, and diagnostics over chasing a
> single "best" architecture.** Every new model should be benchmarked, not assumed superior.

## Current state of the repo (don't re-implement)

Already shipped, so several research topics are largely covered:

- **Meta-learners**: `SLearner`, `TLearner`, `XLearner`, `RLearner`, `DRLearner`,
  `ZLearner` (transformed-outcome / class-transformation, Athey–Imbens), `CausalForest`,
  `PolicyLearner`, `PolicyForest`.
- **Neural**: `DragonNet`, `CFRNet`, `TARNet` (+ `RERUM` framework, ZILN losses).
- **Matching** (`uplift_forecast.matching`): CEM, Mahalanobis, PSM, kernel, embedding,
  general NN, propensity, **+ diagnostics** (`standardized_mean_difference`,
  `covariate_balance`, `match_rate`).
- **Auto**: `Suggest*` declarative spaces + `OptunaHyperparameterTuner`.
- **Metrics**: `auuc_score`, `qini_score`, cumulative-gain / qini curves, component MAE/MSE/MAPE.

This means **Topic 10 (classical methods) is done** (X/R/DR/Z-learners, causal forest,
class-transformation all present) and **Topic 7 (validators)** is partly done (SMD + balance
exist; positivity/overlap and a unified validator API are the gap).

Library dependency rule (from `CLAUDE.md`): the library must **not** import `causalml`,
`econml`, `optuna` (except lazily in `auto.py`), etc. So CausalML / EconML / scikit-uplift /
CATENets are **references and `examples/` material**, never runtime deps — we re-implement
the architecture against our own API.

---

## Topic 1 — Modern neural uplift models

Highest-value additions. All map onto `BaseNeuralUpliftModel` and the loss `outputsize`/`decode`
contract. The KDD-2024 taxonomy ✅ splits these into "treatment-as-branch-switch" (TARNet,
CFRNet, FlexTENet, EUEN) and "treatment-as-feature" (DragonNet, SNet, DESCN, EFIN).

| Model | Paper | OSS | Why add |
|---|---|---|---|
| **EFIN** ✅ | KDD 2023, [arXiv:2306.00315](https://arxiv.org/abs/2306.00315) | [dgliu/KDD23_EFIN](https://github.com/dgliu/KDD23_EFIN) | Explicit feature-interaction; treatment-aware interaction module models ITE via interactions between treatment features and other features; intervention-constraint head balances ITE across groups. Strong on online-marketing data. |
| **DESCN** ✅ | KDD 2022, [arXiv:2207.09920](https://arxiv.org/abs/2207.09920) | authors released code + e-commerce dataset | Deep entire-space cross network; jointly models propensity + response + hidden treatment effect via a cross network in one multi-task model. Designed for treatment bias + group-size imbalance. |
| **FlexTENet / SNet** ✅ | NeurIPS 2021 [arXiv:2106.03765](https://arxiv.org/abs/2106.03765); AISTATS 2021 [arXiv:2101.10943](https://arxiv.org/abs/2101.10943) | [AliciaCurth/CATENets](https://github.com/AliciaCurth/CATENets) | FlexTENet = flexible shared/private structure for the two outcome heads (information-sharing regularizer). SNet = end-to-end shared+private representation with propensity. **CATENets is the API template**: `.fit(X, y, w)` / `.predict(X)` returning CATE — maps 1:1 onto our contract. |
| **EUEN** | Listed in the KDD-2024 benchmark ✅ as a branch-switch model | (in benchmark repos) | Explicit Uplift Effect Network — directly parameterizes the uplift term rather than differencing two outcome heads; lightweight, good baseline. |

**Code sketch** (FlexTENet-style head sharing, fits our heads-from-loss design):

```python
# uplift_forecast/models/flextenet.py
from ..common._base_neural import BaseNeuralUpliftModel

class FlexTENet(BaseNeuralUpliftModel):
    """FlexTENet (Curth & van der Schaar, NeurIPS 2021, arXiv:2106.03765).

    Two outcome heads with structured shared/private subspaces; an information-sharing
    penalty (ortho/L2 on private vs shared weights) regularizes how much the heads share.
    """
    def __init__(self, input_size, hidden_size=128, n_layers=2,
                 shared_ratio=0.5, ortho_lambda=1e-3, loss=None, valid_loss=None,
                 alias=None, **trainer_kwargs):
        super().__init__(input_size=input_size, hidden_size=hidden_size,
                         n_layers=n_layers, loss=loss, valid_loss=valid_loss,
                         alias=alias, **trainer_kwargs)
        ...  # save_hyperparameters handled by base

    def _build_output_heads(self):
        # read self._outcome_size (1 point / 3 ZILN) -> build shared + 2 private heads
        ...
    def forward(self, x): ...
    def _step(self, batch, stage):
        # add ortho_lambda * shared/private orthogonality penalty to loss.forward(...)
        ...
```

**TODO (Topic 1):**
1. Add `models/efin.py`, `models/descn.py`, `models/flextenet.py`, `models/euen.py`,
   each subclassing `BaseNeuralUpliftModel`, reading `self._outcome_size` /
   `self._decode_outcome` so they work with both point and ZILN losses. Register in
   `models/__init__.py`.
2. EFIN/DESCN need a **propensity / treatment-classification auxiliary head** — reuse the
   DragonNet pattern; keep the multi-task weighting inside the model's `_step`, not in a new
   loss, unless the auxiliary term is reusable (then add a loss to `losses.py`).
3. Architecture-only port — re-implement from the papers/CATENets as reference (check
   licenses before copying: CATENets is research code; EFIN repo license TBD). Do **not** add
   CATENets as a dependency.
4. Add each to `examples/` and benchmark against existing models on Criteo/Hillstrom +
   synthetic, reporting AUUC/Qini — per the "no stable SOTA" caveat, ship as peers not defaults.

---

## Topic 2 — Task-specific uplift (revenue / LTV, retention, pricing)

We already have **ZILN losses** for revenue/LTV (zero-inflated lognormal, the right tool for
spend with a point mass at 0) and **RERUM** for revenue-uplift rankability.

🔎 Gaps worth filling:
- **Two-stage conversion × value** uplift: P(convert) × E[value | convert]; common in
  e-commerce. Can be expressed as a composite head (binary head + ZILN value head) on a shared
  representation.
- **Survival/retention** outcomes → see Topic 5 (causal survival forest).
- The CLV/ZILN foundation is [arXiv:1912.07753](https://arxiv.org/abs/1912.07753) ("A Deep
  Probabilistic Model for Customer Lifetime Value Prediction", Wang/Liu/Miao) — already the
  basis of our ZILN loss; cite it in the losses docstring if not already.

**TODO (Topic 2):** add a `TwoStageUplift` composite (conversion gate × ZILN value) as either a
new neural model or a loss variant; document the revenue-uplift recipe (ZILN + RERUM + AUUC on
revenue) as a tutorial.

---

## Topic 3 — Multi-treatment uplift (multiple discrete arms)

Current API is binary (`treatment` 0/1, metrics validate `{0,1}`). Multi-arm is a real
extension.

| Method | Paper | OSS | Notes |
|---|---|---|---|
| **M3TN** ✅ | ICASSP 2024, [arXiv:2401.14426](https://arxiv.org/abs/2401.14426) | authors' repo | MMoE representation + uplift reparameterization; additive `μ_k(x) = μ_0(x) + τ_k(x)`. Built for K≥2 where many-head nets lose efficiency / accumulate error. |
| **Multi-treatment cost optimization** ✅ | [arXiv:1908.05372](https://arxiv.org/abs/1908.05372) (Zhao & Harinen, Uber) | **CausalML** | Multiple treatment groups with different costs (channels × promo types) + cost optimization. Foundational; the meta-learner generalization is straightforward. |
| Multi-arm meta-learners | — | CausalML, EconML | S/T/R/DR-learners extend to K arms (one-vs-base). |

**Design** (implemented): the arm decomposition lives in `common/_base_meta.py::_BaseMultiArmLearner`,
which `SLearner` / `TLearner` / `DRLearner` subclass — there are **no** separate `Multi*` classes.
`treatment` may be an integer in `{0..K-1}` (0 = control); `predict` returns `[n, K-1]` uplift columns,
collapsing to a flat `[n]` array in the binary (K=2) case, and `UpliftForecast` emits
`uplift_<model>_arm{k}`. Subclasses implement `_fit_arms` / `_predict_arm`; the binary
`_fit_estimators` / `_predict_components` template (`BaseMetaUpliftModel`) is reserved for the
binary-intrinsic learners (R/X/Z, policy, tree, forests).

```python
class TLearner(_BaseMultiArmLearner):
    def _fit_arms(self, X, treatment, y, eval_set=None, **fit_params):
        for arm in self.arms_:                         # 0 = control
            template = self.model if arm == 0 else (self.model_treated or self.model)
            self._models[arm] = deepcopy(template).fit(X[treatment == arm], y[treatment == arm])
    def _predict_arm(self, X, arm):
        return self._predict_outcome(self._models[arm], X)
```

**Status (Topic 3): done.**
1. ✅ Contract extension: integer `treatment ∈ {0..K-1}` (0 = control); `predict` returns
   `[n, K-1]` (binary collapses to `[n]`); `UpliftForecast` emits `uplift_<model>_arm{k}`.
2. ✅ Metrics generalized: `auuc_score` / `qini_score` return a `{arm: score}` dict for multi-arm
   treatment (a single float for binary), plus the Zhao–Harinen cost-aware targeting curve and
   `optimal_treatment_assignment`.
3. ✅ Multi-arm `SLearner` / `TLearner` / `DRLearner`; `M3TN` as the neural option.

---

## Topic 4 — Continuous treatment / dose-response

Entirely new capability (we're binary today). Treatment becomes a continuous dose `t ∈ ℝ`;
target is the **average/individual dose-response function (ADRF)** `μ(x, t)`.

| Method | Paper | OSS | Notes |
|---|---|---|---|
| **VCNet** ✅ | ICLR 2021 (oral), [arXiv:2103.07861](https://arxiv.org/abs/2103.07861) | [lushleaf/varying-coefficient-net-with-functional-tr](https://github.com/lushleaf/Varying-Coefficient-Net-with-Functional-TR) | Varying-coefficient net: head weights are smooth functions of dose `t` (spline basis), preserving ADRF continuity. Current best-in-class for continuous CATE. |
| **DRNet** ✅ | AAAI 2020, [arXiv:1902.00981](https://arxiv.org/abs/1902.00981) | [d909b/drnet](https://github.com/d909b/drnet) | Per-dosage-stratum heads (hierarchical); also handles **multiple parametric treatments** → bridges Topics 3 & 4. |
| SCIGAN | NeurIPS 2020 | (authors) | GAN for continuous-dose counterfactuals; heavier, lower priority. |

**Code sketch** (VCNet — dose-conditioned head via spline basis):

```python
# uplift_forecast/models/vcnet.py
class VCNet(BaseNeuralUpliftModel):
    """VCNet (Nie et al., ICLR 2021, arXiv:2103.07861) — continuous-dose ADRF.

    Shared representation z(x); a spline basis b(t) over the dose makes the outcome head
    coefficients vary smoothly with t: y_hat = head(z(x); W(t)). Returns mu(x, t).
    """
    def predict_dose_response(self, X, t_grid):  # [n, len(t_grid)]
        ...
```

**TODO (Topic 4):**
1. This needs a **contract extension**, not just a new model: `treatment` may be continuous;
   add `predict_dose_response(X, t_grid)` to the continuous models; uplift = `μ(x,t) − μ(x,t0)`
   for a reference dose `t0`.
2. Add continuous-treatment metrics: ADRF MISE / dose-policy value (AUUC/Qini are binary-only).
3. Ship `VCNet` first (cleaner, SOTA), `DRNet` second (also covers multi-parametric-treatment).
4. Gate behind a `treatment_type='continuous'` flag so binary paths stay untouched.

---

## Topic 5 — Time-aware / survival uplift

🔎 The mature, robust line here is **causal survival forests** (right-censored time-to-event
CATE), not deep time-series uplift (which is still nascent).

| Method | Paper | OSS | Notes |
|---|---|---|---|
| **Causal Survival Forest** | Cui et al., JRSS-B 2023, [arXiv:2001.09887](https://arxiv.org/abs/2001.09887) | [grf-labs/grf](https://grf-labs.github.io/grf/articles/survival.html) (`causal_survival_forest`) | CATE with right-censored outcomes; estimates **RMST difference** or **survival-probability difference at horizon h**. The standard for retention/churn-time uplift. |
| Practical recommendations | [arXiv:2501.05836](https://arxiv.org/abs/2501.05836) | — | Compares G-formula, AIPCW-AIPTW, Buckley-James, CSF — good for designing our estimator. |
| Time-varying causal survival | [arXiv:2503.00730](https://arxiv.org/abs/2503.00730) | — | For treatment-timing-varies settings; research-stage. |

**TODO (Topic 5):**
1. Add a survival meta-learner: define the outcome as RMST(h) or S(h), reduce to our existing
   `CausalForest` / DR-learner on the (pseudo-)outcome with **IPCW** (inverse-prob-of-censoring
   weights). Inputs gain `(time, event)` instead of a scalar `y`.
2. Metrics: time-dependent uplift curve at horizon h.
3. Lower priority than Topics 1/3/4 unless retention/churn is a target use case; pure
   deep-time-series uplift is not yet robust enough to recommend.

---

## Topic 6 — Off-policy evaluation (OPE) & policy learning

We have `PolicyLearner` / `PolicyForest` (policy *learning*); the gap is standalone **OPE
estimators** to *score* a learned policy offline.

| Tool | Source | Estimators / API |
|---|---|---|
| **Open Bandit Pipeline (OBP)** 🔎 | [st-tech/zr-obp](https://github.com/st-tech/zr-obp), [arXiv:1907.09623](https://arxiv.org/abs/1907.09623) | DM, IPW, **SNIPW**, **DR**, Switch-DR, MRDR, **DRos** (optimistic shrinkage), Sub-Gaussian IPW/DR, DML. Core class `OffPolicyEvaluation`; `RegressionModel` for DM; `IPWLearner`. |
| **policytree** 🔎 | [grf-labs/policytree](https://github.com/grf-labs/policytree) | Optimal shallow policy trees from doubly-robust reward scores (pairs with GRF). |
| **scikit-uplift** 🔎 | [maks-sh/scikit-uplift](https://github.com/maks-sh/scikit-uplift) | `uplift_at_k`, `qini_auc_score`, `uplift_auc_score`, `weighted_average_uplift` — uplift-flavored offline eval. |

**Code sketch** (the estimators are small, pure-numpy — fit our no-deps rule):

```python
# uplift_forecast/ope.py  (new module, __all__ = [...])
def ips(reward, action, propensity, policy_action):           # IPW / IPS
    w = (action == policy_action) / propensity
    return float(np.mean(w * reward))

def snips(reward, action, propensity, policy_action):         # self-normalized IPS
    w = (action == policy_action) / propensity
    return float(np.sum(w * reward) / np.sum(w))

def doubly_robust(reward, action, propensity, policy_action, q_hat):
    direct = q_hat[np.arange(len(reward)), policy_action]
    w = (action == policy_action) / propensity
    return float(np.mean(direct + w * (reward - q_hat[np.arange(len(reward)), action])))
```

**TODO (Topic 6):**
1. New module `uplift_forecast/ope.py` (pure numpy, no deps): `ips`, `snips`, `direct_method`,
   `doubly_robust`, `switch_dr`, `dr_os`. Mirror OBP's API names/semantics for familiarity.
2. A thin `evaluate_policy(model, X, treatment, y, propensity, *, estimator='dr')` helper that
   scores a fitted `UpliftModel`'s induced policy (treat-if-uplift>threshold).
3. Tutorial: learn with `PolicyForest`, evaluate with DR/SNIPS, compare to AUUC. OBP is the
   reference, not a dependency.

---

## Topic 7 — Validators & diagnostics (overlap, balance, SMD, positivity)

Partly done: `matching/_diagnostics.py` already has `standardized_mean_difference`,
`covariate_balance`, `match_rate`. 🔎 Gaps: **positivity / common-support / overlap** checks and
a single user-facing validator entry point.

Reference rules (verified 🔎): SMD threshold **0.1** for acceptable balance; **variance ratio**
acceptable in **[0.5, 2.0]**; positivity = propensity-score overlap, flag regions with only
treated or only control units (inference doesn't generalize beyond the overlap region).

**Code sketch:**

```python
# extend uplift_forecast/matching/_diagnostics.py (or a new diagnostics.py)
def variance_ratio(X, treatment): ...                 # per-feature var(t=1)/var(t=0)
def positivity_check(propensity, treatment, *, low=0.05, high=0.95):
    """Flag near-deterministic propensities (lack of common support)."""
    pct_outside = np.mean((propensity < low) | (propensity > high))
    return {'share_outside_overlap': float(pct_outside),
            'min_propensity': float(propensity.min()),
            'max_propensity': float(propensity.max())}
def overlap_report(X, treatment, propensity=None):    # SMD + var-ratio + positivity, one call
    ...
```

**TODO (Topic 7):**
1. Add `variance_ratio`, `positivity_check`, and a `overlap_report(...)` aggregator that
   returns SMD (existing) + variance ratio + positivity + a pass/fail vs the 0.1 / [0.5,2.0]
   thresholds.
2. Surface a top-level `uplift_forecast.diagnostics` (re-export from matching) so it's usable
   without the matching models.
3. Optionally let `UpliftForecast.fit` emit a balance warning when SMD>0.1 on key covariates.

---

## Topic 8 — Explainable uplift

🔎 We have no explainability layer; this is a clean, high-value add. CausalML's interpretation
API is the reference design.

| Tool | Source | What it gives |
|---|---|---|
| CausalML interpretation 🔎 | [docs](https://causalml.readthedocs.io/en/latest/interpretation.html) | `.get_importance()` (auto / permutation via eli5), `.get_shap_values()`, `.plot_shap_values()`, `.plot_shap_dependence()`, `uplift_tree_plot()`. |
| metalearners SHAP 🔎 | [metalearners docs](https://metalearners.readthedocs.io/en/stable/examples/example_feature_importance_shap/) | SHAP on CATE meta-learners. |

**Code sketch** (model-agnostic, SHAP applied to the uplift output):

```python
# uplift_forecast/explain.py  (lazy import shap, like auto.py does optuna)
def uplift_shap_values(model, X, *, background=None):
    import shap
    f = lambda data: model.predict(data)          # explain the uplift directly
    explainer = shap.Explainer(f, background if background is not None else X)
    return explainer(X)

def permutation_importance(model, X, treatment, y, *, metric='auuc', n_repeats=5):
    """Drop in AUUC when each feature is shuffled -> importance for the uplift signal."""
    ...
```

**TODO (Topic 8):**
1. New module `uplift_forecast/explain.py` with **lazy `shap` import** (mirror `auto.py`'s lazy
   optuna pattern; SHAP stays an optional extra, not a runtime dep).
2. `uplift_shap_values(model, X)` (explains `predict` = uplift directly) and a metric-based
   `permutation_importance` keyed on AUUC/Qini (the causally meaningful target, not MSE).
3. For tree models (`PolicyForest`/`CausalForest`) expose native `feature_importances_` and a
   tree-plot helper.

---

## Topic 9 — AutoUplift (automated model selection + Auto* API)

We have `OptunaHyperparameterTuner` (tunes one model's hyperparameters). 🔎 **AutoUplift** is
broader: *automatically evaluate/select across multiple uplift algorithms*.

- **autoum** 🔎 — [jroessler/autoum](https://github.com/jroessler/autoum) (Apache-2.0): "Python
  framework for automatically evaluating various uplift modeling algorithms to estimate ITE",
  associated with Rößler's benchmarking work ("Bridging the Gap: A Systematic Benchmarking of
  Uplift Modeling and HTE Methods"). This is the API template.
- Build on our existing `auto.py` infrastructure — add a model-selection layer on top of the
  per-model tuner.

**Code sketch:**

```python
# uplift_forecast/auto.py  — add an AutoUplift selector on top of the tuner
class AutoUplift:
    """Automatically tune & select across a candidate set of UpliftModels.

    Each candidate gets OptunaHyperparameterTuner over its own space; models are ranked by
    validation AUUC/Qini (or DR-OPE policy value); best (or top-k ensemble) is returned.
    """
    def __init__(self, candidates, spaces, *, metric='qini', n_trials=50, cv=3): ...
    def fit(self, X, treatment, y, eval_set=None):
        # for each candidate: tune -> refit -> score on val -> leaderboard
        self.leaderboard_ = ...      # DataFrame: model, best_params, val_metric
        self.best_model_ = ...
        return self
    def predict(self, X, *, return_components=False):
        return self.best_model_.predict(X, return_components=return_components)
```

**TODO (Topic 9):**
1. Add `AutoUplift` to `auto.py` (keeps optuna lazy). It implements the `UpliftModel` contract
   (so it slots into `UpliftForecast` and `save`/`load`), wraps a candidate list +
   per-candidate `Suggest*` spaces, tunes each, and ranks by a uplift metric (AUUC/Qini) or
   DR-OPE policy value (ties Topic 6 in).
2. Expose `.leaderboard_` (model × best_params × score) and `best_model_`; optional top-k
   averaging ensemble.
3. autoum (Apache-2.0) is the reference; re-implement the selection loop on our metrics —
   don't add it as a dep.
4. Per the "no stable SOTA" caveat, default the candidate set to a **diverse** mix
   (meta-learner + tree/forest + 1–2 neural) and report the full leaderboard, not just the winner.

---

## Topic 10 — Classical methods (status: mostly DONE)

Verified ✅ that CausalML/EconML cover these; **we already have them**:

- ✅ Meta-learners S/T/X/R/DR + transformed-outcome Z-learner (`ZLearner`, Athey–Imbens) —
  present.
- ✅ Class-transformation / revert-label — that's our `ZLearner`.
- ✅ Causal forest / GRF-style — `CausalForest` present (EconML's `CausalForestDML`,
  `ForestDRLearner`, `DMLOrthoForest`/`DROrthoForest` give **valid asymptotic confidence
  intervals** via honest splitting, [pywhy/EconML](https://www.pywhy.org/EconML/spec/estimation/forest.html)
  — a feature to consider adding to our forest).

Remaining classical gaps worth a small ticket:
- **Uplift trees with divergence split criteria** (KL, Euclidean/ED, Chi-square, DDP, IDDP, CIT,
  IT, CTS) — verified ✅ in CausalML's `UpliftTreeClassifier`/`UpliftRandomForestClassifier`.
  We have policy/causal forests but not a divergence-criterion uplift tree. Caveat ✅: those are
  **classification-outcome** (binary) and several criteria (DDP/IDDP/IT/CIT) support only two
  treatment groups.
- **Confidence intervals** on `CausalForest.predict` (honest splitting → asymptotic CIs).

**TODO (Topic 10):** (a) optional `UpliftTree` with KL/ED/Chi/DDP criteria for binary-outcome
uplift (low priority; mainly for parity/interpretability); (b) add `predict_interval` /
honest-splitting CIs to `CausalForest`.

---

## Suggested priority order

1. **Topic 7 + 6 + 8** (diagnostics completion, OPE estimators, SHAP) — small, pure-numpy /
   lazy-import, no contract changes, immediately useful for *any* model. Lowest risk, high value.
2. **Topic 9** — `AutoUplift` on top of existing `auto.py`; ties everything together via a
   leaderboard. Directly answers the "no stable SOTA" caveat.
3. **Topic 1** — EFIN, DESCN, FlexTENet, EUEN neural models (CATENets is the API template).
4. **Topic 3 / 4** — multi-treatment and continuous-dose; these need **contract extensions**
   (treatment dtype, new predict methods, new metrics) so plan them together.
5. **Topic 5 / 2** — survival/retention and two-stage revenue uplift, if those use cases are in
   scope.

## Sources

Verified primary (3-0): [2306.00315](https://arxiv.org/abs/2306.00315),
[2207.09920](https://arxiv.org/abs/2207.09920),
[CATENets](https://github.com/AliciaCurth/CATENets) +
[2101.10943](https://arxiv.org/abs/2101.10943) /
[2106.03765](https://arxiv.org/abs/2106.03765),
[2406.00335](https://arxiv.org/abs/2406.00335) (benchmark caveat),
[2401.14426](https://arxiv.org/abs/2401.14426) (M3TN),
[1908.05372](https://arxiv.org/abs/1908.05372) (multi-treatment cost),
[2103.07861](https://arxiv.org/abs/2103.07861) (VCNet),
[1902.00981](https://arxiv.org/abs/1902.00981) + [d909b/drnet](https://github.com/d909b/drnet) (DRNet),
[CausalML methodology](https://causalml.readthedocs.io/en/latest/methodology.html),
[EconML forests](https://www.pywhy.org/EconML/spec/estimation/forest.html).

Follow-up (single-fetch, lower assurance): [zr-obp](https://github.com/st-tech/zr-obp) +
[1907.09623](https://arxiv.org/abs/1907.09623) (OBP),
[policytree](https://github.com/grf-labs/policytree),
[scikit-uplift](https://github.com/maks-sh/scikit-uplift),
[CausalML interpretation](https://causalml.readthedocs.io/en/latest/interpretation.html),
[autoum](https://github.com/jroessler/autoum),
causal survival forest [2001.09887](https://arxiv.org/abs/2001.09887) +
[grf survival](https://grf-labs.github.io/grf/articles/survival.html),
balance/SMD [ehsanx psw](https://ehsanx.github.io/psw/balance.html).

> **Correction:** the deep-research run mis-attributed AutoUplift to
> [arXiv:1912.07753](https://arxiv.org/abs/1912.07753) — that ID is the **ZILN/CLV** paper
> (Wang, Liu & Miao), which is the basis of our existing ZILN loss, *not* an AutoUplift paper.
> The actual AutoUplift OSS is [jroessler/autoum](https://github.com/jroessler/autoum).
