"""core.forecast.harmonized_country_data
=========================================
Step 4 of the plan: the per-country wide-table output
core/forecast/legacy_hotsync_data.py's module docstring asks for --
"once core.validation wraps core.src.harmonize into a per-country wide-table
output shaped like harmonize_by_country() below, retrain core.forecast's
model on THAT instead of this file" -- built from core.src.harmonize's
per-satellite, trust-graded calendar instead of legacy's single global
kM/kV scale.

Shape matches legacy_hotsync_data.harmonize_by_country() exactly:
returns (activity, frp, info) where activity/frp are wide DataFrames
(date index x country columns). country borders/metadata
(load_countries, country_meta) are UNCHANGED and still come from
legacy_hotsync_data.py -- only the sensor-harmonization half of that
module is replaced here.

Nothing here re-runs preprocess/harmonize; it only reads whatever
`python manage.py harmonize` last wrote to
core.src.config.PROCESSED_DIR (same "pipelines run async, the web
layer just reads whatever they last wrote" rule as
core.validation.confidence, which reads the same three files for
backtesting).

IMPORTANT ASYMMETRY between the two outputs -- core.src.harmonize only
validates an FRP ratio (harmonize.fit_scale), never a count ratio, so:

  frp:      directly equivalent to legacy's frp target. Per cell-day,
            VIIRS is the reference value and MODIS contributes
            frp_viirs_equiv_mw ONLY where that satellite has a
            validated k (core.src.harmonize already drops unscaled
            MODIS frp_viirs_equiv_mw to NaN upstream). Summed per
            country/day. NaN where nothing was measured for ANY cell
            that country/day, never 0.

  activity: NOT a drop-in replacement for legacy's count-based,
            kM/kV-scaled "harmonized active cell-days". There is no
            validated count-ratio in core.src.harmonize's output to
            scale MODIS counts by. This instead returns VIIRS active-
            cell count on days VIIRS was observed, and FALLS BACK to
            the country's raw MODIS active-cell count, UNSCALED, on
            days VIIRS was not observed -- labelled per-day in
            info["activity_scale_basis_by_day"] as "reference" or
            ACTIVITY_SCALE_UNSCALED, mirroring core.src.harmonize's
            own rule of labelling rather than silently equating an
            unvalidated number to a validated one. Treat `activity`
            results from MODIS-only days as lower-confidence than
            `frp` results; prefer target="frp" where that's an option.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core.src.config import PROCESSED_DIR
from core.validation.geo import cell_country_map

ACTIVITY_SCALE_UNSCALED = "count_unscaled"


def _load_calendars():
    cells_path = PROCESSED_DIR / "calendar_cells.csv"
    daily_path = PROCESSED_DIR / "calendar_daily.csv"
    mapping_path = PROCESSED_DIR / "frp_mapping.json"
    for p in (cells_path, daily_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p.name} not found in {PROCESSED_DIR}. Run `python manage.py preprocess` then "
                "`python manage.py harmonize` first."
            )
    cells = pd.read_csv(cells_path, parse_dates=["date"])
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    return cells, daily, mapping


def _mean_or_nan(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float("nan")


def harmonize_by_country(countries, start: str | None = None, end: str | None = None):
    """(activity, frp, info) -- see module docstring for exactly what each
    wide table means and the activity/frp asymmetry. `countries` is a
    Natural-Earth-style GeoDataFrame (country, name, iso_n3, geometry),
    i.e. whatever legacy_hotsync_data.load_countries() returns.
    """
    cells, daily, mapping = _load_calendars()
    if start:
        cells = cells[cells["date"] >= pd.Timestamp(start)]
        daily = daily[daily["date"] >= pd.Timestamp(start)]
    if end:
        cells = cells[cells["date"] <= pd.Timestamp(end)]
        daily = daily[daily["date"] <= pd.Timestamp(end)]
    if daily.empty:
        raise ValueError("No rows in calendar_daily.csv for the requested [start, end] window.")

    days = pd.DatetimeIndex(sorted(daily["date"].unique()))
    codes = list(countries["country"])

    cells = cells.copy()
    cells["country"] = cell_country_map(cells, countries)
    cells = cells.dropna(subset=["country"])

    is_viirs = cells["sensor_family"] == "VIIRS"
    daily_idx = daily.set_index("date")
    viirs_obs = daily_idx["viirs_observed"].reindex(days).fillna(False).astype(bool)
    modis_obs = daily_idx["modis_observed"].reindex(days).fillna(False).astype(bool)
    any_obs = (viirs_obs | modis_obs).to_numpy()

    # ---- frp: per-cell-day harmonized FRP (VIIRS reference, validated-k MODIS), summed per country ----
    cells["harmonized_frp_cell"] = np.where(is_viirs, cells["frp_mw"], cells["frp_viirs_equiv_mw"])
    frp_rows = cells.dropna(subset=["harmonized_frp_cell"])
    frp = (
        frp_rows.groupby(["date", "country"])["harmonized_frp_cell"].sum(min_count=1)
        .unstack("country").reindex(index=days, columns=codes)
    )
    measured_mask = np.broadcast_to(any_obs[:, None], frp.shape)
    frp = frp.where(frp.notna() | ~measured_mask)  # still NaN if nothing measured that country/day
    frp.values[np.isnan(frp.values) & measured_mask] = 0.0  # measured, genuinely zero cells -> 0, not NaN

    # ---- activity: VIIRS active-cell count where observed, else unscaled MODIS active-cell count ----
    def _active_cells(mask):
        sub = cells[mask].drop_duplicates(["date", "country", "cell_id"])
        return (
            sub.groupby(["date", "country"]).size()
            .unstack("country").reindex(index=days, columns=codes).fillna(0.0)
        )

    viirs_active = _active_cells(is_viirs)
    modis_active = _active_cells(~is_viirs)
    viirs_obs_bcast = np.broadcast_to(viirs_obs.to_numpy()[:, None], viirs_active.shape)
    activity = pd.DataFrame(
        np.where(viirs_obs_bcast, viirs_active.to_numpy(), modis_active.to_numpy()),
        index=days, columns=codes,
    )
    activity.values[~np.broadcast_to(any_obs[:, None], activity.shape)] = np.nan

    activity_basis_by_day = {
        str(d.date()): ("reference" if is_ref else ACTIVITY_SCALE_UNSCALED)
        for d, is_ref in zip(days, viirs_obs.to_numpy())
    }

    k_by_sat = mapping.get("k_by_modis_satellite", {}) or {}
    info = {
        "source": "core.src.harmonize (per-satellite, trust-graded calendar)",
        "k_by_modis_satellite": k_by_sat,
        "scale_basis": mapping.get("scale_basis", {}),
        "reference_viirs_satellite": mapping.get("reference_viirs_satellite"),
        "window_utc": mapping.get("window_utc"),
        "activity_target_caveat": (
            "activity falls back to UNSCALED MODIS active-cell counts on days VIIRS was not "
            "observed -- see activity_scale_basis_by_day and this module's docstring; frp has no "
            "such gap and should be preferred where the choice is open"
        ),
        "activity_scale_basis_by_day": activity_basis_by_day,
        # kept only so existing console prints / joblib dumps expecting info["kM"]/["kV"] don't
        # crash: the mean of the per-satellite validated k's, NOT a single fitted scale the way
        # legacy's constant kM was -- see k_by_modis_satellite for the real, per-satellite values.
        "kM": _mean_or_nan(k_by_sat.values()),
        "kV": 1.0,
    }
    return activity, frp, info
