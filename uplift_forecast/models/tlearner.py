__all__ = ['TLearner']


from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from ..common._base_meta import BaseMetaUpliftModel
from ..common._uplift_model import _row_subset


class TLearner(BaseMetaUpliftModel):
    """Two-model meta-learner (T-learner).

    Trains one estimator on the control group and another on the treated group;
    uplift = model_treated.predict(X) - model.predict(X).

    Args:
        model: Base estimator for the control arm.
        model_treated: Optional separate estimator for the treated arm.
            Defaults to a deepcopy of model.
        alias (str): Optional display name for UpliftForecast output columns.
    """

    def __init__(self, model: Any, model_treated: Any | None = None, alias: str | None = None):
        super(TLearner, self).__init__(alias=alias)
        self.model = model
        self.model_treated = model_treated
        self._model_ct = None
        self._model_tr = None

    def _fit_estimators(
        self,
        X: np.ndarray | pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None,
        **fit_params: Any,
    ) -> None:
        mask_ct = treatment.astype(int) == 0
        mask_tr = ~mask_ct

        self._model_ct = deepcopy(self.model)
        self._model_tr = deepcopy(self.model_treated or self.model)

        ct_kwargs = dict(fit_params)
        tr_kwargs = dict(fit_params)

        if eval_set is not None:
            x_val, t_val, y_val = eval_set
            val_ct = t_val.astype(int) == 0
            val_tr = ~val_ct
            ct_kwargs.setdefault('eval_set', (_row_subset(x_val, val_ct), y_val[val_ct]))
            tr_kwargs.setdefault('eval_set', (_row_subset(x_val, val_tr), y_val[val_tr]))

        self._model_ct.fit(_row_subset(X, mask_ct), y[mask_ct], **ct_kwargs)
        self._model_tr.fit(_row_subset(X, mask_tr), y[mask_tr], **tr_kwargs)

    def _predict_components(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y0 = np.asarray(self._model_ct.predict(X)).reshape(-1)
        y1 = np.asarray(self._model_tr.predict(X)).reshape(-1)
        return y0, y1
