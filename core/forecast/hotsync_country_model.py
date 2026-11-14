"""One pooled model for many countries (country-by-country forecasts).

Instead of ~100 separate tiny models, a single gradient-boosting model learns
from every country at once. Each row is (country, day t, horizon h) and carries
that country's own lags / same-season history, plus its long-run activity level
(mean365) and centroid latitude/longitude so the model can tell apart, say, a
Northern-Hemisphere summer burner from a Southern-Hemisphere one.

Thin-data countries therefore borrow strength from similar countries instead
of being modelled in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import hotsync_model as hm

FEATURES_C = hm.FEATURES + ["mean365", "lat", "lon"]


def select_countries(wide: pd.DataFrame, min_active_days: int = 60, max_countries: int = 120):
    """Keep countries with enough fire history, biggest first."""
    active_days = (wide.fillna(0) > 0).sum()
    keep = active_days[active_days >= min_active_days].index
    order = wide[keep].sum().sort_values(ascending=False).index[:max_countries]
    return list(order)


def country_meta(countries_gdf) -> dict:
    pts = countries_gdf.geometry.representative_point()
    return {
        r.country: {"name": r.name, "iso_n3": int(r.iso_n3), "lat": float(p.y), "lon": float(p.x)}
        for r, p in zip(countries_gdf.itertuples(), pts)
    }


def thresholds(wide: pd.DataFrame, target: str) -> dict:
    return {c: hm.high_activity_threshold(wide[c], target) for c in wide.columns}


def build_panel(wide: pd.DataFrame, meta: dict, horizons=hm.HORIZONS, stride: int = 1,
                latest_only: bool = False) -> pd.DataFrame:
    """Stack rows for every country. `stride` thins training rows (every n-th day)."""
    frames = []
    for code in wide.columns:
        y = wide[code].dropna().asfreq("D")
        if len(y) < 120:
            continue
        rows = hm.build_rows(y, horizons)
        mean365 = y.rolling(365, min_periods=90).mean()
        rows["mean365"] = mean365.reindex(rows["date"]).values
        rows["lat"] = meta[code]["lat"]
        rows["lon"] = meta[code]["lon"]
        rows["country"] = code
        if latest_only:
            rows = rows[rows["date"] == y.index[-1]]
        elif stride > 1:
            day_no = rows["date"].values.astype("datetime64[D]").astype(np.int64)
            rows = rows[day_no % stride == 0]
        num = rows.select_dtypes("float64").columns
        rows[num] = rows[num].astype("float32")
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def evaluate_panel(wide, meta, target, test_days=365, stride=3, horizons=hm.HORIZONS, top=25):
    rows = build_panel(wide, meta, horizons, stride).dropna(subset=["target"])
    last = wide.index.max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    train = rows[rows["target_date"] < test_start]
    test = rows[rows["date"] >= test_start].copy()
    if len(train) < 2000 or test.empty:
        raise ValueError("Not enough history for a train/test split; try a smaller --test-days.")

    models = hm.fit(train, FEATURES_C)
    test = test.join(hm.predict(models, test, FEATURES_C))
    fallback = train.groupby("country")["target"].mean()
    base = test["clim_mean"].fillna(test["country"].map(fallback))

    metrics = {
        "target": target,
        "test_start": str(test_start.date()),
        "countries": int(wide.shape[1]),
        "test_rows": int(len(test)),
        "overall": {
            "model": hm._err(test["target"], test["pred"]),
            "persistence_baseline": hm._err(test["target"], test["lag0"].fillna(base)),
            "same_season_baseline": hm._err(test["target"], base),
        },
        "interval_80pct_coverage": float(((test["target"] >= test["low"]) & (test["target"] <= test["high"])).mean()),
        "by_country": [],
    }
    test["base"] = base
    big = test.groupby("country")["target"].sum().sort_values(ascending=False).index[:top]
    for c in big:
        s = test[test["country"] == c]
        metrics["by_country"].append({
            "country": c, "name": meta[c]["name"],
            "MAE_model": float((s["pred"] - s["target"]).abs().mean()),
            "MAE_same_season": float((s["base"] - s["target"]).abs().mean()),
            "mean_actual": float(s["target"].mean()),
        })
    return metrics, test, models


def forecast_latest(models, wide, meta, thr, target, horizons=hm.HORIZONS) -> pd.DataFrame:
    now = build_panel(wide, meta, horizons, latest_only=True)
    out = now.join(hm.predict(models, now, FEATURES_C))
    last_by_country = {c: wide[c].dropna().index[-1] for c in wide.columns}
    out["date"] = [last_by_country[c] + pd.Timedelta(days=int(h)) for c, h in zip(out["country"], out["h"])]
    return pd.DataFrame({
        "country": out["country"].values,
        "name": out["country"].map(lambda c: meta[c]["name"]).values,
        "iso_n3": out["country"].map(lambda c: meta[c]["iso_n3"]).values,
        "date": out["date"].dt.date.values,
        "horizon_days": out["h"].astype(int).values,
        "predicted": out["pred"].round(2).values,
        "low_p10": out["low"].round(2).values,
        "high_p90": out["high"].round(2).values,
        "typical": out["clim_mean"].round(2).values,
        "busy_day": [bool(p >= thr[c]) for c, p in zip(out["country"], out["pred"])],
        "target": target,
    }).sort_values(["country", "horizon_days"]).reset_index(drop=True)


def prepare_country_series(target="activity", countries_path=None, start=None, end=None,
                           min_active_days=60, max_countries=120):
    """core.src.harmonize's calendar -> (wide date x country series, meta, info).

    STEP 4: replaces the old raw-FIRMS-CSV-in / legacy_hotsync_data.harmonize_by_country
    path. `python manage.py harmonize` has already produced the per-satellite,
    trust-graded calendar for the AOI + date range configured in core.src.config;
    this only reads it (see core.forecast.harmonized_country_data for the
    activity/frp asymmetry that comes with that swap -- read its module docstring
    before picking `target` for a new training run).

    Country borders/metadata (load_countries, country_meta) are unchanged and
    still come from legacy_hotsync_data.py.
    """
    from . import harmonized_country_data as hcd
    from . import legacy_hotsync_data as hd  # country borders/metadata only now

    countries = hd.load_countries(countries_path)
    activity, frp, info = hcd.harmonize_by_country(countries, start, end)
    wide = activity if target == "activity" else frp
    keep = select_countries(wide, min_active_days, max_countries)
    meta = country_meta(countries)
    keep = [c for c in keep if c in meta]
    return wide[keep], meta, info
