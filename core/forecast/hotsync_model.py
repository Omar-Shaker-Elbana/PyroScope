"""Forecasting model for HotSync's daily harmonized fire activity (or FRP).

One "direct multi-horizon" model: every training row is (day t, horizon h) and
asks "what will the value be on day t+h?", using only information known at t
plus calendar features of the target day. Same-season history comes from
EARLIER years only, so there is no look-ahead leakage.

Three gradient-boosting models are fitted: the mean (Poisson loss, suited to
counts) and the 10% / 90% quantiles, which give an uncertainty band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZONS = list(range(1, 15))
CLIM_YEARS = 12
FEATURES = [
    "h", "lag0", "lag1", "lag2", "lag6",
    "mean3", "mean7", "mean14", "mean30", "mean90",
    "max7", "ratio7_30", "active14",
    "doy_sin", "doy_cos", "clim_mean", "clim_max",
]


def _base_features(y_ext: pd.Series):
    f = pd.DataFrame(index=y_ext.index)
    f["lag0"] = y_ext
    for k in (1, 2, 6):
        f[f"lag{k}"] = y_ext.shift(k)
    for w in (3, 7, 14, 30, 90):
        f[f"mean{w}"] = y_ext.rolling(w, min_periods=max(1, w // 2)).mean()
    f["max7"] = y_ext.rolling(7, min_periods=1).max()
    f["ratio7_30"] = f["mean7"] / (f["mean30"] + 1.0)
    f["active14"] = (y_ext.fillna(0) > 0).rolling(14, min_periods=1).sum()

    # Same-season level in earlier years: 15-day smoothed value 365*k days before.
    smooth = y_ext.rolling(15, center=True, min_periods=5).mean()
    past = pd.concat([smooth.shift(365 * k) for k in range(1, CLIM_YEARS + 1)], axis=1)
    clim = pd.DataFrame({"clim_mean": past.mean(axis=1), "clim_max": past.max(axis=1)})
    return f, clim


def build_rows(y: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    """Stack (t, h) rows. `y` must be a contiguous daily series."""
    y = y.astype(float).asfreq("D")
    end = y.index[-1]
    ext = pd.date_range(y.index[0], end + pd.Timedelta(days=max(horizons)), freq="D")
    y_ext = y.reindex(ext)
    f, clim = _base_features(y_ext)

    frames = []
    for h in horizons:
        X = f.loc[:end].copy()
        X["h"] = h
        X["clim_mean"] = clim["clim_mean"].shift(-h).loc[:end]
        X["clim_max"] = clim["clim_max"].shift(-h).loc[:end]
        tgt = X.index + pd.Timedelta(days=h)
        doy = tgt.dayofyear.values
        X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        X["target"] = y_ext.shift(-h).loc[:end]
        X["date"] = X.index
        X["target_date"] = tgt
        frames.append(X)
    return pd.concat(frames, ignore_index=True)


def high_activity_threshold(y: pd.Series, target: str) -> float:
    """HotSync-style 'busy day' cut-off: 75th percentile (at least 3 cell-days)."""
    q = float(y.dropna().quantile(0.75))
    return max(3.0, q) if target == "activity" else q


# --------------------------------------------------------------------------- #
def fit(rows: pd.DataFrame, features=None) -> dict:
    features = features or FEATURES
    rows = rows.dropna(subset=["target"])
    X, t = rows[features], rows["target"]
    common = dict(max_iter=300, learning_rate=0.05, max_depth=6,
                  min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    return {
        "mean": HistGradientBoostingRegressor(loss="poisson", **common).fit(X, t),
        "p10": HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **common).fit(X, t),
        "p90": HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **common).fit(X, t),
    }


def predict(models: dict, X: pd.DataFrame, features=None) -> pd.DataFrame:
    X = X[features or FEATURES]
    mean = np.clip(models["mean"].predict(X), 0, None)
    lo = np.minimum(np.clip(models["p10"].predict(X), 0, None), mean)
    hi = np.maximum(np.clip(models["p90"].predict(X), 0, None), mean)
    return pd.DataFrame({"pred": mean, "low": lo, "high": hi}, index=X.index)


# --------------------------------------------------------------------------- #
def _err(actual, pred):
    e = np.asarray(pred) - np.asarray(actual)
    a = np.asarray(actual)
    corr = float(np.corrcoef(a, pred)[0, 1]) if a.std() > 0 and np.std(pred) > 0 else float("nan")
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())), "corr": corr}


def _prf(actual_hi, pred_hi):
    tp = int((actual_hi & pred_hi).sum())
    prec = tp / pred_hi.sum() if pred_hi.sum() else float("nan")
    rec = tp / actual_hi.sum() if actual_hi.sum() else float("nan")
    return {"precision": float(prec), "recall": float(rec)}


def evaluate(y: pd.Series, target: str, test_days: int = 365, horizons=HORIZONS):
    """Time-split evaluation: train on the past, test on the last `test_days` days.

    Training rows are only those whose *target day* is before the test period.
    Returns (metrics dict, test-rows DataFrame with predictions).
    """
    rows = build_rows(y, horizons).dropna(subset=["target"])
    test_start = y.index[-1] - pd.Timedelta(days=test_days - 1)
    train = rows[rows["target_date"] < test_start]
    test = rows[rows["date"] >= test_start].copy()
    if len(train) < 500 or test.empty:
        raise ValueError("Not enough history for a train/test split; try --test-days smaller.")

    models = fit(train)
    p = predict(models, test)
    test = test.join(p)

    fallback = float(train["target"].mean())
    persistence = test["lag0"].fillna(fallback)
    seasonal = test["clim_mean"].fillna(fallback)
    thr = high_activity_threshold(y[y.index < test_start], target)

    metrics = {
        "target": target,
        "test_start": str(test_start.date()),
        "test_rows": int(len(test)),
        "high_activity_threshold": thr,
        "overall": {
            "model": _err(test["target"], test["pred"]),
            "persistence_baseline": _err(test["target"], persistence),
            "same_season_baseline": _err(test["target"], seasonal),
        },
        "by_horizon_MAE": {},
        "interval_80pct_coverage": float(((test["target"] >= test["low"]) & (test["target"] <= test["high"])).mean()),
        "busy_day_detection": {
            "model": _prf(test["target"] >= thr, test["pred"] >= thr),
            "same_season_baseline": _prf(test["target"] >= thr, seasonal >= thr),
        },
    }
    for h in (1, 3, 7, 14):
        s = test[test["h"] == h]
        if s.empty:
            continue
        metrics["by_horizon_MAE"][h] = {
            "model": float((s["pred"] - s["target"]).abs().mean()),
            "persistence": float((s["lag0"].fillna(fallback) - s["target"]).abs().mean()),
            "same_season": float((s["clim_mean"].fillna(fallback) - s["target"]).abs().mean()),
        }
    return metrics, test
