"""Skeleton for plan steps 4-6. Every function below is a TODO stub —
signatures and docstrings only, so the `validation` app has something real
to import while the logic gets filled in next.
"""
from __future__ import annotations

import pandas as pd


def load_observed_activity(countries: list[str], start: str, end: str) -> pd.DataFrame:
    """Real-time FIRMS -> core.src.harmonize -> per-country wide activity table,
    for exactly the countries/date range core.forecast's model covers.

    Returns a DataFrame shaped like core.forecast.legacy_hotsync_data's
    `wide` (date index, one column per country code), but produced by
    core.src.harmonize's per-satellite k instead of the legacy global kM/kV.
    """
    raise NotImplementedError


def backtest_forecast(forecast_csv: str, observed: pd.DataFrame) -> pd.DataFrame:
    """Join forecast_by_country.csv against `observed` on (country, date).

    Returns one row per (country, date, horizon_days) with: predicted,
    low_p10, high_p90, observed, error, abs_error, and whether observed
    fell inside [low_p10, high_p90]. This is outputs/metrics.json's
    backtest, but against live data instead of a historical holdout.
    """
    raise NotImplementedError


def harmonization_confidence(frp_mapping: dict, countries: list[str]) -> pd.DataFrame:
    """Per-country harmonization-confidence flag, from core.src.harmonize's
    frp_mapping.json: k_trust grade, bootstrap CI width, and whether the
    window had to be widened. This is independent of the forecast model's
    own p10/p90 band — a country can have a tight forecast interval built
    on untrustworthy harmonization, and this is what should catch that.
    """
    raise NotImplementedError


def confidence_report(backtest: pd.DataFrame, confidence: pd.DataFrame) -> pd.DataFrame:
    """Merge backtest_forecast() + harmonization_confidence() into the single
    per-country/date table the `validation` app's view renders: forecast
    error AND whether that error is explained by weak harmonization trust.
    """
    raise NotImplementedError
