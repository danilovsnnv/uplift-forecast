# Matching

Covariate matching estimators and balance diagnostics. Each matcher learns a metric on `fit`
and returns an ATT-weighted matched sample on `transform`; the diagnostics quantify covariate
balance before and after matching.

::: uplift_forecast.matching
