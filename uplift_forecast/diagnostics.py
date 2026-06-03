"""Covariate-balance and overlap diagnostics.

Top-level re-export of the diagnostics implemented in
`uplift_forecast.matching._diagnostics`, so they can be used to validate a study
design without touching the matching models:

    from uplift_forecast.diagnostics import overlap_report
    report = overlap_report(X, treatment, propensity=e_hat)
"""

__all__ = [
    'covariate_balance',
    'match_rate',
    'overlap_report',
    'positivity_check',
    'standardized_mean_difference',
    'variance_ratio',
]


from .matching._diagnostics import (
    covariate_balance,
    match_rate,
    overlap_report,
    positivity_check,
    standardized_mean_difference,
    variance_ratio,
)
