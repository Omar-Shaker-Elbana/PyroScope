"""Conditional MODIS -> VIIRS FRP scale: k as a function of fire size, day/night and time gap (no I/O, no pipeline state).

A single constant k assumes one number covers every fire size and every overpass gap. `k_sensitivity` already showed it
does not (k falls as MODIS FRP rises, differs by day / night, grows with the allowed time gap). This module turns that
table into a model:

        E[ VIIRS FRP in the footprint | MODIS FRP = f, night, dt ]  =  f * k(f, night, dt)
        log k(f, night, dt) = a + s * log(f / f_ref) + c * night + d * dt_hours

Estimator: quasi-Poisson regression (log link, MODIS FRP as an offset, cluster-robust by cell), fitted by IRLS.
Why this and not a regression of log(VIIRS) on log(MODIS):
  * The calendar SUMS FRP. A regression in logs targets the median pixel and needs a retransformation correction to
    predict totals; the log-link quasi-Poisson model targets the conditional MEAN directly and is consistent whatever the
    variance (Santos Silva & Tenreyro 2006). It weights pixels by their FRP, so large fires count in proportion to the
    FRP they contribute to a total.
  * With no covariate it IS the existing estimator: k = sum(VIIRS) / sum(MODIS). The constant k is the special case,
    the model is a strict generalisation (tested in selftest.py).
  * In-sample the fitted totals equal the observed totals exactly (score equation of the intercept).
  * `log_log_ols` reports the log-log regression that is the obvious first idea, as a cross-check of the size effect.

Which covariates are kept is decided by cross-validation, not by significance: all 8 subsets of {size, night, dt} are
scored by the held-out Poisson deviance with WHOLE CELLS held out (a model that is only good on places it has already
seen would be useless for a new region), and the simplest subset within one standard error of the best (paired over
folds) is kept. If that subset is empty the constant k stays.

Prediction. The model is fitted with the observed time gap `dt` and applied at dt = 0 (simultaneous overpass): the
VIIRS-equivalent FRP is what VIIRS would have seen at the MODIS overpass time, and a MODIS pixel with no VIIRS pixel has
no gap of its own. Fire size is clipped to the central (1 - 2 * clip_q) of the fitted data, beyond which the ratio is held
at its boundary value: log-linear extrapolation from a few very large fires is not trusted.

What it does NOT do: it is still conditional on a MODIS detection (small fires seen only by VIIRS are not recovered), and
it does not model the diurnal cycle, so a satellite that overpasses hours away from VIIRS (Terra) is not helped by it.
"""
from itertools import combinations

import numpy as np
import pandas as pd

TERMS = ("size", "night", "dt")
COEF_NAME = {"size": "log_frp", "night": "night", "dt": "dt_hours"}
MIN_GROUP = 20            # a day/night class with fewer pixels than this cannot be estimated
MIN_DT_RANGE_MIN = 5.0    # a time gap that varies by less than this over the fitted pixels cannot be estimated
MIN_CELLS_MODEL = 10      # fewer clusters than this: no cross-validation, no model
N_BINS = 5


# --------------------------------------------------------------------------- design
def night_flag(daynight) -> np.ndarray:
    """FIRMS day/night column -> 1.0 (night), 0.0 (day), NaN (unknown)."""
    s = pd.Series(np.asarray(daynight, dtype=object)).astype(str).str.strip().str.upper().to_numpy()
    out = np.full(len(s), np.nan)
    out[s == "N"] = 1.0
    out[s == "D"] = 0.0
    return out


def make_spec(frp, night, clip_q) -> dict:
    """Centring / clipping constants taken from the pixels a model is fitted on."""
    lo, hi = np.quantile(frp, [clip_q, 1.0 - clip_q])
    share = float(np.nanmean(night)) if np.isfinite(night).any() else 0.0
    return {"centre_log_frp": float(np.log(np.median(frp))), "frp_clip": [float(lo), float(hi)], "night_share": share}


def design(spec, terms, frp, night, dt_min) -> np.ndarray:
    """Columns: intercept, then one column per term. Unknown day/night -> the fitted night share."""
    frp = np.asarray(frp, float)
    n = len(frp)
    cols = [np.ones(n)]
    if "size" in terms:
        lo, hi = spec["frp_clip"]
        cols.append(np.log(np.clip(frp, lo, hi)) - spec["centre_log_frp"])
    if "night" in terms:
        cols.append(np.full(n, spec["night_share"]) if night is None
                    else np.where(np.isfinite(night), night, spec["night_share"]))
    if "dt" in terms:
        cols.append(np.asarray(dt_min, float) / 60.0)
    return np.column_stack(cols)


def estimable_terms(frp, night, dt_min) -> tuple:
    """Terms the data can actually identify (a constant column cannot be estimated)."""
    out = []
    if np.ptp(np.log(frp)) > 0:
        out.append("size")
    if np.isfinite(night).any():
        known = night[np.isfinite(night)]
        if min((known == 1).sum(), (known == 0).sum()) >= MIN_GROUP:
            out.append("night")
    if np.ptp(dt_min) >= MIN_DT_RANGE_MIN:
        out.append("dt")
    return tuple(out)


# --------------------------------------------------------------------------- estimator
def _fit(spec, terms, frp, y, night, dt_min, w=None, max_iter=50, tol=1e-9):
    """Quasi-Poisson IRLS with offset log(frp). w = case weights (bootstrap multiplicities). -> (beta, converged)."""
    X = design(spec, terms, frp, night, dt_min)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log((w * y).sum() / (w * frp).sum())      # start at the constant k
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = frp * np.exp(eta)
        z = eta + (y - mu) / mu
        wt = w * mu
        a, b = X.T @ (X * wt[:, None]), X.T @ (wt * z)
        try:
            new = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(a, b, rcond=None)[0]
        if not np.all(np.isfinite(new)):
            return beta, False
        done = np.max(np.abs(new - beta)) < tol
        beta = new
        if done:
            return beta, True
    return beta, False


def _mu(spec, terms, beta, frp, night, dt_min):
    return np.asarray(frp, float) * np.exp(np.clip(design(spec, terms, frp, night, dt_min) @ beta, -30.0, 30.0))


def deviance(y, mu) -> float:
    """Poisson deviance 2 * sum[y log(y/mu) - (y - mu)]: the loss this estimator minimises (y > 0)."""
    return float(2.0 * np.sum(y * np.log(y / mu) - (y - mu)))


def predict_ratio(model, frp, night=None, dt_min=None) -> np.ndarray:
    """k(f, night, dt) of a fitted `model` dict (as stored in frp_mapping.json). dt defaults to the model's reference
    gap (0 = simultaneous overpass); night = night_flag() output or None (unknown -> the fitted night share)."""
    terms = tuple(model["terms"])
    beta = np.array([model["coef"]["intercept"]] + [model["coef"][COEF_NAME[t]] for t in terms])
    frp = np.asarray(frp, float)
    dt = np.full(len(frp), float(model.get("reference_dt_min", 0.0))) if dt_min is None else dt_min
    return np.exp(np.clip(design(model, terms, frp, night, dt) @ beta, -30.0, 30.0))


def _model_dict(spec, terms, beta, n_pairs) -> dict:
    coef = {"intercept": float(beta[0])}
    coef.update({COEF_NAME[t]: float(b) for t, b in zip(terms, beta[1:])})
    return {"form": "quasi-Poisson, log link, offset log(MODIS FRP): VIIRS FRP = MODIS FRP * exp(coef . x)",
            "terms": list(terms), "coef": coef, "reference_dt_min": 0.0, "n_pairs": int(n_pairs), **spec}


# --------------------------------------------------------------------------- bootstrap
def _bootstrap(spec, terms, frp, y, night, dt, codes, rng, n_boot):
    """Cluster bootstrap of the coefficients: whole cells are resampled (as multiplicity weights)."""
    g = int(codes.max()) + 1
    out, failed = [], 0
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, g, g), minlength=g)
        beta, ok = _fit(spec, terms, frp, y, night, dt, cnt[codes].astype(float))
        if ok:
            out.append(beta)
        else:
            failed += 1
    return np.array(out), failed


def _coef_table(terms, beta, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0) if len(boot) else (np.full(len(beta), np.nan),) * 2
    names = ["intercept"] + [COEF_NAME[t] for t in terms]
    return {n: {"coef": float(b), "ci95": [float(l), float(h)],
                "ci_excludes_0": bool(l > 0 or h < 0) if n != "intercept" else None}
            for n, b, l, h in zip(names, beta, lo, hi)}


def _log_log_ols(frp, y, night, dt_min, terms, codes, rng, n_boot):
    """The log-log regression  log(VIIRS) ~ log(MODIS) [+ night + dt_hours]  (cross-check; NOT the model that is applied)."""
    cols = [np.ones(len(y)), np.log(frp)]
    names = ["intercept", "log_frp"]
    if "night" in terms:
        cols.append(np.where(np.isfinite(night), night, np.nanmean(night)))
        names.append("night")
    if "dt" in terms:
        cols.append(dt_min / 60.0)
        names.append("dt_hours")
    x, ly = np.column_stack(cols), np.log(y)
    beta = np.linalg.lstsq(x, ly, rcond=None)[0]
    g = int(codes.max()) + 1
    boot = []
    for _ in range(min(n_boot, 1000)):
        sw = np.sqrt(np.bincount(rng.integers(0, g, g), minlength=g)[codes].astype(float))
        boot.append(np.linalg.lstsq(x * sw[:, None], ly * sw, rcond=None)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"formula": "log(VIIRS_FRP) ~ " + " + ".join(names[1:]), "estimand": "median pixel (log scale), not total FRP",
            "coef": {n: {"coef": float(b), "ci95": [float(l), float(h)]} for n, b, l, h in zip(names, beta, lo, hi)},
            "note": "log_frp = 1 + size coefficient of the applied model when both are unbiased for the same relation"}


# --------------------------------------------------------------------------- cross-validation and selection
def cross_validate(frp, y, night, dt, folds, subsets, clip_q) -> dict:
    """Held-out Poisson deviance of every term subset. Centring / clipping are re-derived from each training part and
    the held-out pixels are predicted at their OBSERVED time gap. Also stores out-of-fold predictions."""
    n_f = int(folds.max()) + 1
    res = {s: {"dev": np.zeros(n_f), "pred": np.zeros(n_f), "obs": np.zeros(n_f), "oof": np.full(len(y), np.nan),
               "failed": 0} for s in subsets}
    for f in range(n_f):
        te = folds == f
        tr = ~te
        spec = make_spec(frp[tr], night[tr], clip_q)
        for s in subsets:
            beta, ok = _fit(spec, s, frp[tr], y[tr], night[tr], dt[tr])
            mu = _mu(spec, s, beta, frp[te], night[te], dt[te])
            res[s]["oof"][te] = mu
            res[s]["dev"][f] = deviance(y[te], mu) if ok else np.inf
            res[s]["failed"] += int(not ok)
            res[s]["pred"][f], res[s]["obs"][f] = mu.sum(), y[te].sum()
    return res


def select_terms(cv, subsets):
    """Simplest subset whose CV deviance is within one paired standard error of the best (the 1-SE rule).
    Returns (chosen, best, table)."""
    tot = {s: float(cv[s]["dev"].sum()) for s in subsets}
    best = min(subsets, key=lambda s: tot[s])
    rows, ok = [], []
    for s in subsets:
        d = cv[s]["dev"] - cv[best]["dev"]
        fin = np.isfinite(d)
        se = float(np.sqrt(fin.sum()) * np.std(d[fin], ddof=1)) if s != best and fin.sum() > 1 else 0.0
        within = bool(np.isfinite(tot[s]) and tot[s] - tot[best] <= se)
        if within:
            ok.append(s)
        rows.append({"terms": list(s), "n_params": len(s) + 1, "cv_deviance": tot[s], "delta_vs_best": tot[s] - tot[best],
                     "se_delta": se, "within_1se_of_best": within,
                     "total_pred_over_obs": float(cv[s]["pred"].sum() / cv[s]["obs"].sum())})
    chosen = min(ok, key=lambda s: (len(s), tot[s]))
    rows.sort(key=lambda r: (r["cv_deviance"]))
    return chosen, best, rows


def _bins(frp, y, oof_const, oof_sel):
    """Where does a constant k miss, and does the model fix it? Out-of-fold, by quintile of MODIS FRP."""
    edges = np.unique(np.quantile(frp, np.linspace(0, 1, N_BINS + 1)))
    idx = np.clip(np.searchsorted(edges, frp, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({"frp_min": float(frp[m].min()), "frp_max": float(frp[m].max()), "n_pairs": int(m.sum()),
                     "observed_ratio": float(y[m].sum() / frp[m].sum()),
                     "mean_frp": float(frp[m].mean()),
                     "observed_over_constant_k": float(y[m].sum() / oof_const[m].sum()),
                     "observed_over_selected_model": float(y[m].sum() / oof_sel[m].sum())})
    return rows


def _curve(spec, terms, beta, boot, n=25):
    lo, hi = spec["frp_clip"]
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    x = design(spec, terms, grid, None, np.zeros(n))
    r = np.exp(x @ beta)
    band = np.exp(x @ boot.T) if len(boot) else None
    plo, phi = (np.percentile(band, [2.5, 97.5], axis=1) if band is not None else (np.full(n, np.nan),) * 2)
    return [{"frp": float(f), "k": float(k), "k_lo": float(a), "k_hi": float(b)} for f, k, a, b in zip(grid, r, plo, phi)]


# --------------------------------------------------------------------------- driver
def fit_k_model(use, rng, n_boot=2000, clip_q=0.001, n_folds=10) -> dict:
    """Fit, select and validate the conditional scale for ONE MODIS satellite.

    use : usable matched pixels within the fit time gap; columns frp, viirs_frp_sum, abs_dt_min, cell_id,
          acq_datetime_utc and (optional) daynight.
    Returns a JSON-ready dict: `adopted` (a covariate model beat the constant k in cross-validation), `model` (what
    predict_ratio needs; None when the constant k stays), coefficients with cluster-bootstrap intervals, the CV table
    of all candidates, bias by fire-size bin, the fitted curve, and the log-log OLS cross-check."""
    frp = use["frp"].to_numpy(float)
    y = use["viirs_frp_sum"].to_numpy(float)
    dt = use["abs_dt_min"].to_numpy(float)
    night = night_flag(use["daynight"]) if "daynight" in use.columns else np.full(len(use), np.nan)
    codes, _ = pd.factorize(use["cell_id"])
    n_cells = int(codes.max()) + 1 if len(codes) else 0
    if n_cells < MIN_CELLS_MODEL:
        return {"available": False, "adopted": False, "model": None,
                "reason": f"only {n_cells} cells (need >= {MIN_CELLS_MODEL}) to cross-validate a covariate model"}

    full = estimable_terms(frp, night, dt)
    subsets = [s for r in range(len(full) + 1) for s in combinations(full, r)]
    f_n = min(int(n_folds), n_cells)
    folds = (rng.permutation(n_cells) % f_n)[codes]
    cv = cross_validate(frp, y, night, dt, folds, subsets, clip_q)
    chosen, best, table = select_terms(cv, subsets)
    spec = make_spec(frp, night, clip_q)

    out = {"available": True, "estimator": "quasi-Poisson regression, log link, offset log(MODIS FRP) (nests k = sum VIIRS / sum MODIS)",
           "terms_estimable": list(full),
           "terms_not_estimable": [t for t in TERMS if t not in full],
           "applied_at": "dt = 0 (simultaneous overpass); unknown day/night -> fitted night share; FRP clipped to "
                         f"[{spec['frp_clip'][0]:.3g}, {spec['frp_clip'][1]:.3g}] MW (central {1 - 2 * clip_q:.1%} of the fit)",
           "cv": {"scheme": f"{f_n}-fold, whole 0.5-degree cells held out (clusters = the bootstrap unit)", "n_folds": f_n,
                  "loss": "Poisson deviance of VIIRS footprint FRP; held-out pixels predicted at their observed time gap",
                  "selection_rule": "simplest term subset within one paired standard error of the best",
                  "best_subset": list(best), "selected_subset": list(chosen), "candidates": table,
                  "non_converged_fits": int(sum(c["failed"] for c in cv.values()))}}

    # leave-one-year-out view of the same comparison (temporal generalisation; same places as the training years)
    years = use["acq_datetime_utc"].dt.year.to_numpy()
    if len(np.unique(years)) >= 3:
        ycodes, yvals = pd.factorize(years, sort=True)
        cmp_subsets = [(), chosen] if chosen != () else [()]
        cy = cross_validate(frp, y, night, dt, ycodes, cmp_subsets, clip_q)
        out["cv_leave_one_year_out"] = {
            "years": [int(v) for v in yvals],
            "constant_k_cv_deviance": float(cy[()]["dev"].sum()),
            "selected_cv_deviance": float(cy[chosen]["dev"].sum()),
            "total_pred_over_obs_by_year": {
                "constant_k": {str(int(v)): float(cy[()]["pred"][i] / cy[()]["obs"][i]) for i, v in enumerate(yvals)},
                "selected": {str(int(v)): float(cy[chosen]["pred"][i] / cy[chosen]["obs"][i]) for i, v in enumerate(yvals)}}}

    # full model: every estimable term, reported with cluster-bootstrap intervals (post-selection CIs would be too narrow)
    beta_full, ok = _fit(spec, full, frp, y, night, dt)
    boot_full, failed = _bootstrap(spec, full, frp, y, night, dt, codes, rng, n_boot)
    out["coefficients_all_terms"] = _coef_table(full, beta_full, boot_full)
    out["bootstrap"] = {"n_boot": int(n_boot), "n_failed_fits": int(failed), "unit": "0.5-degree cell"}
    out["log_log_ols"] = _log_log_ols(frp, y, night, dt, full, codes, rng, n_boot)

    if chosen == ():
        out.update(adopted=False, model=None,
                   reason="no covariate improves on the constant k in cross-validation by more than one standard error")
        out["bias_by_frp_bin"] = _bins(frp, y, cv[()]["oof"], cv[()]["oof"])
        return out

    beta, ok_s = _fit(spec, chosen, frp, y, night, dt)
    boot_s = boot_full if chosen == full else _bootstrap(spec, chosen, frp, y, night, dt, codes, rng, n_boot)[0]
    model = _model_dict(spec, chosen, beta, len(y))
    mu_in = _mu(spec, chosen, beta, frp, night, dt)
    out.update(adopted=True, model=model, coefficients_selected=_coef_table(chosen, beta, boot_s),
               in_sample_total_pred_over_obs=float(mu_in.sum() / y.sum()),
               bias_by_frp_bin=_bins(frp, y, cv[()]["oof"], cv[chosen]["oof"]),
               curve=_curve(spec, chosen, beta, boot_s),
               cv_improvement_over_constant_k={
                   "deviance_reduction": float(1.0 - cv[chosen]["dev"].sum() / cv[()]["dev"].sum()),
                   "constant_k_cv_deviance": float(cv[()]["dev"].sum()),
                   "selected_cv_deviance": float(cv[chosen]["dev"].sum())})
    out["reason"] = "cross-validation prefers " + " + ".join(chosen) + " over the constant k"
    if "size" in chosen:
        s = beta[1 + chosen.index("size")]
        ci = out["coefficients_selected"]["log_frp"]["ci95"]
        f10 = lambda x: float(np.exp(x * np.log(10.0)) - 1.0)
        out["size_effect"] = {"k_change_per_10x_modis_frp": f10(s), "ci95": [f10(ci[0]), f10(ci[1])]}
    return out


def describe(km) -> str:
    """One console line."""
    if not km or not km.get("available"):
        return "k model: not fitted (" + (km or {}).get("reason", "disabled") + ")"
    if not km["adopted"]:
        return "k model: constant k kept (" + km["reason"] + ")"
    c = km["coefficients_selected"]
    parts = ", ".join(f"{n} {d['coef']:+.3f}" for n, d in c.items() if n != "intercept")
    gain = km["cv_improvement_over_constant_k"]["deviance_reduction"]
    s = ""
    if "size_effect" in km:
        e = km["size_effect"]
        s = (f"; k changes {e['k_change_per_10x_modis_frp']:+.0%} per 10x MODIS FRP "
             f"(95% CI {e['ci95'][0]:+.0%} to {e['ci95'][1]:+.0%})")
    return f"k model: {' + '.join(km['model']['terms'])} ({parts}){s}; held-out deviance {gain:.1%} below the constant k"
