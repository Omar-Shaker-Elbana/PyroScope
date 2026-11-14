"""core.validation: forecast-vs-real-data confidence scoring (plan steps 4-6).

Not implemented yet — this is the scaffold the `validation` Django app
imports from. See confidence.py for the function signatures to fill in:

  1. harmonize real-time FIRMS data through core.src.harmonize for the same
     region/countries as core.forecast's model, instead of legacy_hotsync_data.
  2. join forecast_by_country.csv against that observed activity by
     (country, date) and compute error metrics (extends outputs/metrics.json's
     historical backtest into live data).
  3. attach a harmonization-confidence flag per (country, date) from
     core.src.harmonize's k_trust / bootstrap CI / anomaly-validation output,
     separate from the forecast model's own p10/p90 uncertainty.
"""
