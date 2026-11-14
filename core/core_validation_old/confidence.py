"""Plan steps 4-6: forecast-vs-real-data confidence report.

Ports worldmap_forcase/jspyro's forecast-vs-observed comparison (the
forecast-section chart in HotSync_harmonized_Final.html: an orange
predicted band from forecast_by_country.csv overlaid with teal observed
activity from recent_by_country.csv, scored against a same-season baseline
in metrics.json) into core -- same idea, but computed as a real statistical
backtest here instead of eyeballed on a chart, and against
core.src.harmonize's per-satellite, trust-graded activity instead of
worldmap_forcase's single global kM/kV scale (see
core/forecast/legacy_hotsync_data.py's module docstring).

Everything here is read-only against whatever the management commands /
cron already wrote to disk (the plan's "pipelines run async, the web layer
just reads whatever they last wrote"):

  core.src.config.PROCESSED_DIR/calendar_cells.csv   per-cell-day harmonized FRP   (`manage.py harmonize`)
  core.src.config.PROCESSED_DIR/calendar_daily.csv   day-level harmonized_status   (`manage.py harmonize`)
  core.src.config.PROCESSED_DIR/frp_mapping.json     k_trust grades for that run  (`manage.py harmonize`)
  core.src.config.OUTPUTS_DIR/forecast_by_country.csv the model's 14-day forecast  (`manage.py predict_countries`)

Nothing here re-runs preprocess/harmonize/predict_countries itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.src.config import OUTPUTS_DIR, PROCESSED_DIR

# cell_id (core.src.harmonize's CELL_DEG grid) -> Natural Earth country code ("ADM0_A3"),
# filled in lazily by _cell_country_map and cached for the life of the process. core.src.harmonize's
# default 0.5 deg grid is coarser than legacy_hotsync_data's 0.02 deg one, so this is its own
# cell->country join, not a reuse of build_cell_country_map's cell keys.
_CELL_COUNTRY: dict[int, str] = {}
_BORDERS = None


def _country_borders():
    """Natural Earth admin-0 borders (country, name, iso_n3, geometry), cached.
    Reuses core.forecast.legacy_hotsync_data.load_countries so both layers agree on which
    polygon owns which coastline and on the on-disk cache under core.src.config.CACHE_DIR."""
    global _BORDERS
    if _BORDERS is None:
        from core.forecast.legacy_hotsync_data import load_countries

        _BORDERS = load_countries()
    return _BORDERS


def _cell_country_map(cells: pd.DataFrame) -> pd.Series:
    """cell_id -> country code for every cell_id in `cells`, filling `_CELL_COUNTRY` for any
    cell seen for the first time. Done on unique cells, not rows, so repeated calls over a growing
    calendar stay cheap. Cells just offshore are snapped to the nearest coast (<= 0.3 deg)."""
    unseen = cells.loc[~cells["cell_id"].isin(_CELL_COUNTRY), ["cell_id", "cell_lat", "cell_lon"]]
    unseen = unseen.drop_duplicates("cell_id")
    if len(unseen):
        import warnings

        import geopandas as gpd

        pts = gpd.GeoDataFrame(
            {"cell_id": unseen["cell_id"].to_numpy()},
            geometry=gpd.points_from_xy(unseen["cell_lon"], unseen["cell_lat"]),
            crs="EPSG:4326",
        )
        ref = _country_borders()[["country", "geometry"]]
        j = gpd.sjoin(pts, ref, how="left", predicate="within").drop_duplicates("cell_id")
        miss = j["country"].isna()
        if miss.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                near = gpd.sjoin_nearest(
                    pts[pts["cell_id"].isin(j.loc[miss, "cell_id"])], ref, how="left", max_distance=0.3
                ).drop_duplicates("cell_id")
            j = j.set_index("cell_id")
            j.loc[near["cell_id"].to_numpy(), "country"] = near["country"].to_numpy()
            j = j.reset_index()
        for cid, country in zip(j["cell_id"], j["country"]):
            if pd.notna(country):
                _CELL_COUNTRY[int(cid)] = country

    return cells["cell_id"].map(_CELL_COUNTRY)


def load_observed_activity(countries: list[str], start: str, end: str) -> pd.DataFrame:
    """Real-time FIRMS -> core.src.harmonize -> per-country wide activity table,
    for exactly the countries/date range core.forecast's model covers.

    Returns a DataFrame shaped like core.forecast.legacy_hotsync_data's
    `wide` (date index, one column per country code), but produced by
    core.src.harmonize's per-satellite k instead of the legacy global kM/kV.

    Reads calendar_cells.csv / calendar_daily.csv from PROCESSED_DIR as the
    harmonize management command last wrote them; it does not fetch FIRMS or
    run the pipeline itself. A date outside [start, end] or a day
    harmonize's calendar marked not `harmonized_is_measured` (cloud, sensor
    outage, partial window, ...) is NaN, never 0: a day without data is not
    a day without fire.
    """
    cells_path = PROCESSED_DIR / "calendar_cells.csv"
    cal_path = PROCESSED_DIR / "calendar_daily.csv"
    if not cells_path.exists() or not cal_path.exists():
        raise FileNotFoundError(
            f"{cells_path.name} / {cal_path.name} not found in {PROCESSED_DIR}. Run "
            "`python manage.py preprocess` then `python manage.py harmonize` first."
        )

    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range(t0, t1, freq="D")

    cal = pd.read_csv(cal_path, parse_dates=["date"], usecols=["date", "harmonized_is_measured"])
    cal = cal[cal["date"].between(t0, t1)]
    measured_days = set(cal.loc[cal["harmonized_is_measured"].astype(bool), "date"])

    cells = pd.read_csv(
        cells_path,
        parse_dates=["date"],
        usecols=["sensor_family", "date", "cell_id", "cell_lat", "cell_lon", "frp_mw", "frp_viirs_equiv_mw"],
    )
    cells = cells[cells["date"].between(t0, t1)]

    wide = pd.DataFrame(index=days, columns=list(countries) if countries else [])
    wide.index.name = "date"
    if cells.empty:
        return wide

    cells["country"] = _cell_country_map(cells)
    cells = cells.dropna(subset=["country"])
    if countries:
        cells = cells[cells["country"].isin(countries)]
    if cells.empty:
        return wide

    # VIIRS cells are already on the reference scale; MODIS cells contribute only the
    # VIIRS-equivalent FRP core.src.harmonize fitted for their own satellite (NaN -- dropped by the
    # sum below, not treated as 0 -- when that satellite has no validated k; see frp_mapping.json).
    cells["harmonized_frp"] = np.where(
        cells["sensor_family"] == "VIIRS", cells["frp_mw"], cells["frp_viirs_equiv_mw"]
    )
    cells = cells.dropna(subset=["harmonized_frp"])

    by_day = cells.groupby(["date", "country"])["harmonized_frp"].sum(min_count=1).unstack("country")
    by_day = by_day.reindex(index=days)
    if countries:
        by_day = by_day.reindex(columns=countries)
    by_day.index.name = "date"

    # A day the calendar did not score as measured is unknown for every country, not a measured zero.
    unmeasured = [d for d in days if d not in measured_days]
    by_day.loc[by_day.index.isin(unmeasured)] = np.nan
    return by_day


def backtest_forecast(forecast_csv: str, observed: pd.DataFrame) -> pd.DataFrame:
    """Join forecast_by_country.csv against `observed` on (country, date).

    Returns one row per (country, date, horizon_days) with: predicted,
    low_p10, high_p90, observed, error, abs_error, and whether observed
    fell inside [low_p10, high_p90]. This is outputs/metrics.json's
    backtest, but against live data instead of a historical holdout.

    `in_band` is NaN (not True/False) for a (country, date) that has no
    observed value yet -- either the forecast date is still in the future,
    or `load_observed_activity` marked it unmeasured.
    """
    path = Path(forecast_csv)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python manage.py predict_countries` first.")
    fc = pd.read_csv(path, parse_dates=["date"])
    needed = {"country", "date", "horizon_days", "predicted", "low_p10", "high_p90"}
    missing = needed - set(fc.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")

    # melt, not stack: keeps NaN rows (a country/date with no observed value yet) regardless of
    # pandas version, instead of relying on stack()'s dropna behaviour (removed in pandas >= 3).
    obs = observed.reset_index().melt(id_vars="date", var_name="country", value_name="observed")

    out = fc.merge(obs, on=["date", "country"], how="left")
    out["error"] = out["observed"] - out["predicted"]
    out["abs_error"] = out["error"].abs()
    # nullable "boolean" dtype: True/False where observed, <NA> (not NaN-in-a-bool-column, which
    # pandas refuses) where there is nothing to score yet.
    out["in_band"] = out["observed"].between(out["low_p10"], out["high_p90"]).astype("boolean")
    out.loc[out["observed"].isna(), "in_band"] = pd.NA

    cols = ["country", "date", "horizon_days", "predicted", "low_p10", "high_p90",
            "observed", "error", "abs_error", "in_band"]
    return out[cols].sort_values(["country", "date", "horizon_days"]).reset_index(drop=True)


# k_trust grades in increasing order of trust, for picking the worst satellite in a run.
_K_TRUST_ORDER = {"NOT_VALIDATED": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def harmonization_confidence(frp_mapping: dict, countries: list[str]) -> pd.DataFrame:
    """Per-country harmonization-confidence flag, from core.src.harmonize's
    frp_mapping.json: k_trust grade, bootstrap CI width, and whether the
    window had to be widened. This is independent of the forecast model's
    own p10/p90 band built on it -- a country can have a tight forecast
    interval built on untrustworthy harmonization, and this is what should
    catch that.

    One frp_mapping.json covers one harmonize run over a region, not one
    run per country, so every country in that run's coverage gets the SAME
    k_trust / CI-width / window columns here: the grade describes the
    harmonization the forecast's observed activity was built from, not a
    per-country sample size.
    """
    frp_mapping = frp_mapping or {}
    satellites = frp_mapping.get("modis_satellites") or {}
    window = (frp_mapping.get("scale_policy") or {}).get("k_window") or {}
    widened = bool(window.get("widened"))

    grades = {s: (info or {}).get("k_trust", "NOT_VALIDATED") for s, info in satellites.items()}
    worst_sat = min(grades, key=lambda s: _K_TRUST_ORDER.get(grades[s], 0)) if grades else None
    worst_grade = grades.get(worst_sat, "NOT_VALIDATED")

    def _ci_width_pct(info):
        ci, k = (info or {}).get("k_ci95"), (info or {}).get("k")
        if not ci or not k:
            return np.nan
        return float((ci[1] - ci[0]) / k) * 100.0

    ci_widths = [w for w in (_ci_width_pct(info) for info in satellites.values()) if np.isfinite(w)]
    worst_ci_pct = max(ci_widths) if ci_widths else np.nan

    problems = sorted({p for info in satellites.values() for p in (info or {}).get("k_trust_problems", [])})
    if widened:
        problems.append(f"k time-gap window widened to {window.get('used_min')} min "
                         f"(from {window.get('strict_min')} min)")

    if not satellites:
        flag = "UNKNOWN"
    elif worst_grade == "HIGH" and not widened:
        flag = "OK"
    elif worst_grade in ("HIGH", "MEDIUM"):
        flag = "CAUTION"
    else:
        flag = "LOW_TRUST"

    row = {
        "k_trust": worst_grade,
        "k_trust_satellite": worst_sat,
        "k_ci95_width_pct": worst_ci_pct,
        "k_window_widened": widened,
        "confidence_flag": flag,
        "harmonization_problems": "; ".join(problems),
    }
    return pd.DataFrame([{**{"country": c}, **row} for c in countries])


def confidence_report(backtest: pd.DataFrame, confidence: pd.DataFrame) -> pd.DataFrame:
    """Merge backtest_forecast() + harmonization_confidence() into the single
    per-country/date table the `validation` app's view renders: forecast
    error AND whether that error is explained by weak harmonization trust.
    """
    report = backtest.merge(confidence, on="country", how="left")
    # in_band is nullable ("boolean" dtype): <NA> means "not scored yet", not a miss, so a pending
    # row is never flagged as a harmonization-explained miss even though it isn't in_band either.
    missed = report["in_band"].notna() & ~report["in_band"]
    report["harmonization_explains_miss"] = missed & report["confidence_flag"].isin(
        ["CAUTION", "LOW_TRUST", "UNKNOWN"]
    )
    cols = ["country", "date", "horizon_days", "predicted", "low_p10", "high_p90", "observed",
            "error", "abs_error", "in_band", "k_trust", "k_trust_satellite", "k_ci95_width_pct",
            "k_window_widened", "confidence_flag", "harmonization_problems", "harmonization_explains_miss"]
    return report[cols].sort_values(["country", "date", "horizon_days"]).reset_index(drop=True)


def build_report(countries: list[str] | None = None, forecast_csv: str | Path | None = None) -> pd.DataFrame:
    """End-to-end entry point: wires load_observed_activity + backtest_forecast +
    harmonization_confidence + confidence_report together from whatever is already on disk.
    This is what the `validation` app's view and management command call.

    `countries` defaults to every country forecast_by_country.csv covers. The observed window is
    the forecast's own date range, so a date still in the future simply comes back with `observed`
    / `in_band` as NaN until a later harmonize run has data for it -- the same thing jspyro's chart
    did by only drawing teal "observed" points up to the last day recent_by_country.csv had.
    """
    forecast_csv = Path(forecast_csv) if forecast_csv else OUTPUTS_DIR / "forecast_by_country.csv"
    if not forecast_csv.exists():
        raise FileNotFoundError(f"{forecast_csv} not found. Run `python manage.py predict_countries` first.")

    fc_head = pd.read_csv(forecast_csv, usecols=["country", "date"], parse_dates=["date"])
    if not countries:
        countries = sorted(fc_head["country"].unique())
    start, end = fc_head["date"].min(), fc_head["date"].max()

    observed = load_observed_activity(countries, str(start.date()), str(end.date()))
    backtest = backtest_forecast(str(forecast_csv), observed)

    mapping_path = PROCESSED_DIR / "frp_mapping.json"
    frp_mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    confidence = harmonization_confidence(frp_mapping, countries)

    return confidence_report(backtest, confidence)
