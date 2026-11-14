"""Synthetic end-to-end check of matching, the FRP scale k, the observation-status logic, the grid diagnostics and the
anomaly / validation code, all with a KNOWN answer. Nothing is read from or written to data/.

Run:  python -m core.src.selftest      (must finish with SELFTEST PASSED)
"""
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.src import anomaly, anomaly_validation, external_validation, harmonize, harmonize_streaming, harmonize_validation as hv, hotsync_anomaly, kmodel, kstats
from core.src import (preprocess)
from core.src.harmonize import add_features, match_pair, fit_scale

TRUE_K = 1.6          # VIIRS-equivalent FRP = TRUE_K x MODIS FRP
NOISE = 0.25          # log-normal scatter of the summed VIIRS FRP
EXPECTED_K = TRUE_K * np.exp(NOISE ** 2 / 2)  # the mean of exp(N(0, NOISE)) is exp(NOISE^2/2), not 1
FIRES_PER_YEAR = 400
YEARS = (2015, 2016, 2017, 2018)
VIIRS_LEADS_MIN = 5.0  # S-NPP passes this many minutes before Aqua


def synthetic_overpasses(seed=1, year_k=None, size_exp=1.0, night_factor=1.0, dt_decay=0.0):
    """Aqua pixels + the S-NPP pixels inside their footprints (+ VIIRS-only clutter). year_k: {year: true k}.
    Planted dependences of the true ratio (defaults = none, and then the random stream is exactly the old one):
      size_exp     ratio ~ (FRP / 40) ** (size_exp - 1)
      night_factor ratio x night_factor on night-time fires (half of the fires, drawn from a separate generator)
      dt_decay     ratio x exp(-dt_decay * gap_hours), gap drawn uniformly from 4-26 min (else a fixed ~5 min lead)"""
    rng = np.random.default_rng(seed)
    rng_x = np.random.default_rng(seed + 12345)        # day/night and time-gap draws never touch the main stream
    mrows, vrows = [], []
    for year in YEARS:
        kk = TRUE_K if year_k is None else year_k[year]
        is_night = rng_x.random(FIRES_PER_YEAR) < 0.5
        gaps = rng_x.uniform(4, 26, FIRES_PER_YEAR) if dt_decay else np.full(FIRES_PER_YEAR, VIIRS_LEADS_MIN)
        days = rng.integers(0, 365, FIRES_PER_YEAR)
        t0 = pd.Timestamp(f"{year}-01-01 13:30", tz="UTC") + pd.to_timedelta(
            days * 1440 + rng.uniform(-8, 8, FIRES_PER_YEAR), unit="m")
        lat = rng.uniform(-30, 30, FIRES_PER_YEAR)
        lon = rng.uniform(-60, 60, FIRES_PER_YEAR)
        frp_m = np.exp(rng.normal(np.log(40), 0.9, FIRES_PER_YEAR))
        for i in range(FIRES_PER_YEAR):
            mrows.append(dict(sensor_family="MODIS", source_satellite="Aqua", latitude=lat[i],
                              longitude=lon[i], acq_datetime_utc=t0[i], frp=frp_m[i], scan=1.2, track=1.1,
                              daynight="N" if is_night[i] else "D"))
            n = int(rng.integers(2, 7))
            extra = (frp_m[i] / 40.0) ** (size_exp - 1.0) * (night_factor if is_night[i] else 1.0) \
                * np.exp(-dt_decay * gaps[i] / 60.0)
            share = rng.dirichlet(np.ones(n)) * frp_m[i] * kk * extra * np.exp(rng.normal(0, NOISE))
            for j in range(n):
                vrows.append(dict(
                    sensor_family="VIIRS", source_satellite="S-NPP",
                    latitude=lat[i] + rng.uniform(-0.3, 0.3) / 111.2,
                    longitude=lon[i] + rng.uniform(-0.3, 0.3) / (111.2 * np.cos(np.radians(lat[i]))),
                    acq_datetime_utc=t0[i] - pd.Timedelta(minutes=gaps[i] + rng.uniform(-2, 2)),
                    frp=share[j], scan=0.4, track=0.4))
    planted = len(vrows)
    for _ in range(1500):   # VIIRS-only clutter far from every MODIS detection
        vrows.append(dict(sensor_family="VIIRS", source_satellite="S-NPP",
                          latitude=rng.uniform(40, 60), longitude=rng.uniform(100, 140),
                          acq_datetime_utc=pd.Timestamp("2016-06-01 13:00", tz="UTC")
                          + pd.Timedelta(days=int(rng.integers(0, 300))),
                          frp=float(np.exp(rng.normal(0.3, 0.7))), scan=0.4, track=0.4))
    return pd.DataFrame(mrows), pd.DataFrame(vrows), planted


# --------------------------------------------------------------------------- A. scale k
def check_scale():
    M, V, planted = synthetic_overpasses()
    M, V = add_features(M), add_features(V)
    pairs = match_pair(M, V, 180)
    n_fires = len(YEARS) * FIRES_PER_YEAR
    hit = int((pairs["n_viirs"] > 0).sum())
    used = int(pairs["n_viirs"].sum())
    k, info = fit_scale(pairs, "Aqua")
    print(f"matched MODIS pixels {hit}/{n_fires}; VIIRS pixels assigned {used} (planted {planted})")
    print(f"k = {k:.3f} (expected {EXPECTED_K:.3f}), CI {info['k_ci95'][0]:.3f}-{info['k_ci95'][1]:.3f}, "
          f"trust {info['k_trust']}, years used {sorted(info['k_by_year'])}")
    assert hit >= 0.95 * n_fires, "too few planted fires matched"
    assert used == planted, "VIIRS pixel double counting or loss"
    assert k is not None and abs(k / EXPECTED_K - 1) < 0.05, "k not recovered"
    assert (info["k_ci95"][1] - info["k_ci95"][0]) / k < 0.12, "CI implausibly wide for 1600 clean pixels"
    lo, hi = info["k_prediction_interval95_new_year"]
    assert lo < info["k_ci95"][0] and hi > info["k_ci95"][1], "prediction interval must be wider than the CI"
    het = info["k_year_heterogeneity"]
    assert het is not None and het["p_Q"] > 0.01, f"homogeneous years must not look heterogeneous: {het}"
    assert info["k_trust"] == "HIGH", f"k should be trusted with 4 clean, homogeneous years: {info['k_trust_problems']}"

    # one year only -> must NOT be graded HIGH (multi-year requirement)
    one_year = pairs[pairs["acq_datetime_utc"].dt.year == YEARS[0]]
    _, info1 = fit_scale(one_year, "Aqua")
    assert info1["k_trust"] != "HIGH", "a single year must not give a HIGH-trust k"

    # a satellite that passes 3 h before S-NPP must get NO k, however many pixels overlap within the matching window
    Mt = M.copy()
    Mt["source_satellite"] = "Terra"
    Mt["acq_datetime_utc"] = Mt["acq_datetime_utc"] - pd.Timedelta(minutes=170)
    Mt = add_features(Mt.drop(columns=["date", "lst_h", "area_km2", "cell_id", "cell_lat", "cell_lon"]))
    pt = match_pair(Mt, V, 180)
    kt, it = fit_scale(pt, "Terra")
    print(f"Terra (170 min before S-NPP): k = {kt}, trust {it['k_trust']}, diagnostic ratio only: "
          f"{it.get('k_diagnostic_all_gaps', {}).get('k')}")
    assert kt is None and it["k_trust"] == "NOT_VALIDATED", "a satellite without close overpasses must not get a k"
    assert it["k_diagnostic_all_gaps"]["k"] > 0, "the all-gap ratio is kept as a labelled diagnostic"

    fits = {"Aqua": (k, info), "Terra": (kt, it)}
    scale, basis = harmonize.decide_scales(fits)
    assert "Terra" not in scale and basis["Terra"] == "none" and basis["Aqua"] == "own", "Terra must not inherit Aqua's k"
    both = pd.concat([M, Mt], ignore_index=True)
    eq = harmonize.apply_scale(both, scale, basis)
    assert eq.loc[eq["source_satellite"] == "Terra", "frp_equiv"].isna().all(), "Terra FRP must stay NaN, not 1 x or Aqua's k"
    assert np.allclose(eq.loc[eq["source_satellite"] == "Aqua", "frp_equiv"], eq.loc[eq["source_satellite"] == "Aqua", "frp"] * k)
    harmonize.APPLY_UNVALIDATED_K = True            # the opt-in: still labelled, never silent
    try:
        scale2, basis2 = harmonize.decide_scales(fits)
    finally:
        harmonize.APPLY_UNVALIDATED_K = False
    assert basis2["Terra"] == "own_low_trust" and "Terra" in scale2, "opt-in must apply AND label the unvalidated scale"
    return pairs, M, V


def _fit_arrays(pairs):
    use = harmonize._usable(pairs)
    use = use[use["abs_dt_min"] <= harmonize.K_FIT_MAX_DT_MIN]
    return (use, use["frp"].to_numpy(float), use["viirs_frp_sum"].to_numpy(float),
            kmodel.night_flag(use["daynight"]), use["abs_dt_min"].to_numpy(float))


def check_k_model():
    """k as a function of covariates (kmodel.py): nests the constant k, recovers planted effects, does not invent
    effects that are not there, never extrapolates, and is applied (and only applied) where it was fitted."""
    # 1. the estimator nests the constant k and preserves totals exactly
    M, V, _ = synthetic_overpasses()
    pairs = match_pair(add_features(M), add_features(V), 180)
    use, frp, y, night, dt = _fit_arrays(pairs)
    spec = kmodel.make_spec(frp, night, 0.005)
    beta, ok = kmodel._fit(spec, (), frp, y, night, dt)
    assert ok and np.isclose(np.exp(beta[0]), y.sum() / frp.sum(), rtol=1e-9), "intercept-only model must equal the ratio of sums"
    beta, ok = kmodel._fit(spec, kmodel.TERMS, frp, y, night, dt)
    mu = kmodel._mu(spec, kmodel.TERMS, beta, frp, night, dt)
    assert ok and abs(mu.sum() / y.sum() - 1) < 1e-9, "fitted totals must equal observed totals"

    # 2. placebo: the true k is constant, so no covariate may be adopted and the constant k must stay
    k, info = fit_scale(pairs, "Aqua")
    km = info["k_model"]
    print(f"placebo (constant truth): adopted = {km['adopted']}, selected {km['cv']['selected_subset']}")
    assert km["available"] and not km["adopted"] and km["model"] is None, "an effect that is not there must not be adopted"
    scale, basis = harmonize.decide_scales({"Aqua": (k, info)})
    assert basis["Aqua"] == "own" and "model" not in scale["Aqua"]
    assert all(c["terms"] != [] or c["delta_vs_best"] >= 0 for c in km["cv"]["candidates"])

    # 3. planted dependence on size, day/night and time gap is recovered, adopted and worth it
    truth = {"log_frp": -0.3, "night": float(np.log(0.75)), "dt_hours": -0.8}
    # seed 11: a typical draw. Across 24 seeds the cluster-bootstrap 95% CIs of the three coefficients covered the truth
    # 92-96 % of the time (mean size estimate -0.305, sd 0.010), so any single seed can still miss by chance.
    M2, V2, _ = synthetic_overpasses(seed=11, size_exp=0.7, night_factor=0.75, dt_decay=0.8)
    pairs2 = match_pair(add_features(M2), add_features(V2), 180)
    k2, info2 = fit_scale(pairs2, "Aqua")
    km2 = info2["k_model"]
    c = km2["coefficients_all_terms"]
    print("planted effects: " + ", ".join(f"{n} {c[n]['coef']:+.3f} (true {t:+.3f}, CI {c[n]['ci95'][0]:+.2f}..{c[n]['ci95'][1]:+.2f})"
                                          for n, t in truth.items()))
    for n, t in truth.items():
        assert c[n]["ci95"][0] <= t <= c[n]["ci95"][1], f"95% CI of {n} misses the planted value {t}"
    assert abs(c["log_frp"]["coef"] - truth["log_frp"]) < 0.05 and abs(c["night"]["coef"] - truth["night"]) < 0.05
    assert km2["adopted"] and km2["cv"]["selected_subset"] == ["size", "night", "dt"], km2["cv"]["selected_subset"]
    assert km2["cv_improvement_over_constant_k"]["deviance_reduction"] > 0.05, "planted effects must pay off out of sample"
    worst_const = max(abs(b["observed_over_constant_k"] - 1) for b in km2["bias_by_frp_bin"])
    worst_model = max(abs(b["observed_over_selected_model"] - 1) for b in km2["bias_by_frp_bin"])
    print(f"out-of-fold bias by fire-size bin: constant k up to {worst_const:.0%}, model up to {worst_model:.0%}")
    assert worst_const > 0.15 and worst_model < 0.08, "the model must remove the size-dependent bias of the constant k"
    assert abs(km2["in_sample_total_pred_over_obs"] - 1) < 1e-9
    assert km2["size_effect"]["ci95"][0] < km2["size_effect"]["k_change_per_10x_modis_frp"] < km2["size_effect"]["ci95"][1]
    # the log-log regression of the brief agrees: elasticity = 1 + size coefficient
    assert abs(km2["log_log_ols"]["coef"]["log_frp"]["coef"] - (1 + truth["log_frp"])) < 0.05

    # 4. evaluation: no extrapolation, dt = 0 reference, unknown day/night
    mod = km2["model"]
    hi = mod["frp_clip"][1]
    r = kmodel.predict_ratio(mod, [hi, 10 * hi, 1e6], night=np.zeros(3))
    assert np.allclose(r, r[0]), "above the clip the ratio must be held, not extrapolated"
    r_d = kmodel.predict_ratio(mod, [40.0], night=np.array([0.0]))[0]
    r_n = kmodel.predict_ratio(mod, [40.0], night=np.array([1.0]))[0]
    r_u = kmodel.predict_ratio(mod, [40.0], night=np.array([np.nan]))[0]
    assert abs(r_n / r_d - 0.75) < 0.05 and r_n < r_u < r_d, "unknown day/night must fall between day and night"
    r_gap = kmodel.predict_ratio(mod, [40.0], night=np.array([0.0]), dt_min=np.array([30.0]))[0]
    assert abs(r_gap / r_d - np.exp(-0.8 * 0.5)) < 0.1, "a 30 min gap must lower the ratio as planted; the default is dt = 0"

    # 5. application: conditional for the satellite it was fitted on, NaN for Terra, constants only when transferred
    fits = {"Aqua": (k2, info2), "Terra": (None, {"modis_satellite": "Terra", "k": None, "n_pairs": 0})}
    scale, basis = harmonize.decide_scales(fits)
    assert basis["Aqua"] == "own_conditional" and "model" in scale["Aqua"] and basis["Terra"] == "none"
    both = pd.concat([add_features(M2), add_features(M2).assign(source_satellite="Terra")], ignore_index=True)
    eq = harmonize.apply_scale(both, scale, basis)
    a = eq["source_satellite"] == "Aqua"
    exp_r = kmodel.predict_ratio(mod, eq.loc[a, "frp"].to_numpy(), kmodel.night_flag(eq.loc[a, "daynight"]))
    assert np.allclose(eq.loc[a, "frp_equiv"], eq.loc[a, "frp"] * exp_r) and eq.loc[~a, "frp_equiv"].isna().all()
    band = scale["Aqua"]["k_hi"] / scale["Aqua"]["k"]
    assert np.allclose(eq.loc[a, "frp_equiv_hi"], eq.loc[a, "frp_equiv"] * band), "interval keeps its relative width"
    eff = harmonize.applied_scale(eq, scale)
    assert eff["Terra"] is None and eff["Aqua"] > 0
    harmonize.APPLY_UNVALIDATED_K = True
    try:
        scale3, basis3 = harmonize.decide_scales(fits)
    finally:
        harmonize.APPLY_UNVALIDATED_K = False
    assert basis3["Terra"] == "transferred:Aqua" and "model" not in scale3["Terra"], "a conditional model is never transferred"

    # 6. switch-off and degenerate inputs
    harmonize.K_MODEL = False
    try:
        k_off, info_off = fit_scale(pairs2, "Aqua")
    finally:
        harmonize.K_MODEL = True
    assert not info_off["k_model"]["available"] and k_off == k2 and \
        harmonize.decide_scales({"Aqua": (k_off, info_off)})[1]["Aqua"] == "own", "PYRO_K_MODEL=0 keeps the constant k"
    allday = use.assign(daynight="D")
    km_d = kmodel.fit_k_model(allday, np.random.default_rng(1), n_boot=50)
    assert "night" in km_d["terms_not_estimable"] and km_d["available"], "a constant day/night column cannot be estimated"
    km_s = kmodel.fit_k_model(use[use["cell_id"].isin(use["cell_id"].unique()[:3])], np.random.default_rng(1), n_boot=50)
    assert not km_s["available"] and not km_s["adopted"], "too few cells to cross-validate: no model"
    print("k model: nests the constant k, recovers planted effects, rejects placebo, clips, labels, switches off")


def check_heterogeneity():
    """Between-year variation of the scale: invisible when absent, detected and sized when present."""
    ky = {2015: 1.1, 2016: 1.9, 2017: 1.3, 2018: 2.0}      # true year effect of ~25 % SD
    M, V, _ = synthetic_overpasses(seed=3, year_k=ky)
    pairs = match_pair(add_features(M), add_features(V), 180)
    k, info = fit_scale(pairs, "Aqua")
    het = info["k_year_heterogeneity"]
    print(f"heterogeneous years: Q p = {het['p_Q']:.2g}, tau = {het['tau']:.2f} (95% {het['tau_ci95'][0]:.2f}-{het['tau_ci95'][1]:.2f})")
    assert het["p_Q"] < 0.01 and het["tau_ci95"][0] > 0, "real between-year variation must be detected"
    assert info["k_trust"] != "HIGH" and any("differs between years" in p for p in info["k_trust_problems"])
    assert info["k_prediction_interval95_new_year"][1] / info["k_prediction_interval95_new_year"][0] > \
        info["k_ci95"][1] / info["k_ci95"][0] * 1.5, "the interval for an unseen year must reflect tau"
    # the max-min spread alone must NOT decide: noisy years with a common k look 'unstable' by range only
    rng = np.random.default_rng(0)
    y = np.log(0.8) + rng.normal(0, 0.15, 10)
    h = kstats.heterogeneity(y, np.full(10, 0.15))
    spread = (np.exp(y).max() - np.exp(y).min()) / np.median(np.exp(y))
    print(f"common k, SE 15% per year: max-min spread {spread:.0%} but Q p = {h['p_Q']:.2f}")
    assert spread > 0.25 and h["p_Q"] > 0.05, "range is a bad stability statistic; Q must not flag sampling noise"


def check_stats_helpers():
    rng = np.random.default_rng(2)
    arr = np.c_[np.full(60, 10.0), 10.0 * 0.8 * np.exp(rng.normal(0, 0.3, 60))]
    k, lo, hi, se = kstats.ratio_ci(arr, rng, 4000)
    assert lo < k < hi and abs(k - arr[:, 1].sum() / arr[:, 0].sum()) < 1e-12
    # coverage of the cluster-bootstrap 95 % CI over independent replicates (a single dataset proves nothing)
    truth = 0.8 * np.exp(0.3 ** 2 / 2)
    cover = 0
    for _ in range(150):
        a = np.c_[np.full(60, 10.0), 10.0 * 0.8 * np.exp(rng.normal(0, 0.3, 60))]
        _, l, h, _ = kstats.ratio_ci(a, rng, 600)
        cover += l <= truth <= h
    print(f"bootstrap CI coverage over 150 replicates: {cover / 150:.0%}")
    assert cover / 150 >= 0.88, "95% bootstrap CI under-covers"
    # the sphere is tiled exactly by the grid
    lats = np.arange(-89.5, 90, 1.0)
    assert abs((kstats.cell_area_km2(lats, 1.0) * 360).sum() / (4 * np.pi * kstats.R_KM ** 2) - 1) < 1e-9
    assert kstats.cell_area_km2(60.0, 0.5) < 0.55 * kstats.cell_area_km2(0.0, 0.5), "cells must shrink with cos(lat)"
    e = [kstats.edge_split_probability(1.0, d, 30.0) for d in (0.05, 0.25, 0.5, 2.0)]
    assert all(a > b for a, b in zip(e, e[1:])) and e[2] < 0.04 < e[0], "edge error must fall with cell size"
    # missing-day inference: dense file -> one empty day is a gap; sparse file -> only long runs; empty file -> never
    r = np.random.default_rng(4)
    dense = r.poisson(88, 400).astype(float)
    dense[100] = 0
    dense[200:208] = 0
    assert set(np.flatnonzero(kstats.infer_missing_days(dense))) == {100, *range(200, 208)}, "dense gaps not found"
    sparse = r.poisson(1.5, 1500).astype(float)
    assert kstats.infer_missing_days(sparse).sum() <= 8, "sparse file: ordinary empty days must not be gaps"
    sparse[500:520] = 0
    assert kstats.infer_missing_days(sparse)[505], "a 20-day run in a sparse file is a gap"
    nearly_empty = np.zeros(200)
    nearly_empty[100] = 1.0
    assert not kstats.infer_missing_days(nearly_empty).any(), "a file that is empty all around reveals no gap"
    # NaN is never valid JSON
    assert "NaN" not in hv.dumps({"a": float("nan"), "b": np.float64("inf"), "c": np.int64(3)})


# --------------------------------------------------------------------------- B. calendar status
def _calendar_case(n_viirs, clear=None, **cloud_kw):
    """12 days, VIIRS with `n_viirs` detections a day except on a gap day (not observed) and a zero day (observed, empty).
    clear / cloud_kw: optional cloud information passed to build_calendar (None = the calendar as it was without it)."""
    t0, t1 = pd.Timestamp("2016-03-01 00:00", tz="UTC"), pd.Timestamp("2016-03-12 23:59", tz="UTC")
    rng = np.random.default_rng(5)
    days = pd.date_range("2016-03-01", "2016-03-12", freq="D", tz="UTC")
    gap_day, zero_day = pd.Timestamp("2016-03-05"), pd.Timestamp("2016-03-08")
    rows = []
    for d in days:
        for _ in range(n_viirs):
            rows.append(dict(sensor_family="VIIRS", source_satellite="S-NPP", latitude=30 + rng.uniform(0, 1),
                             longitude=31 + rng.uniform(0, 1), frp=5.0, scan=0.4, track=0.4,
                             acq_datetime_utc=d + pd.Timedelta(hours=float(rng.choice([1.5, 11, 12.5, 23])))))
    v = pd.DataFrame(rows)
    v = v[~v["acq_datetime_utc"].dt.floor("D").dt.tz_localize(None).isin([gap_day, zero_day])]
    mrows = []                                  # Aqua: 2 detections every day (also on the VIIRS gap), Terra: 1 a day
    for d in days:
        for sat, n in (("Aqua", 2), ("Terra", 1)):
            for _ in range(n):
                mrows.append(dict(sensor_family="MODIS", source_satellite=sat, latitude=30 + rng.uniform(0, 1),
                                  longitude=31 + rng.uniform(0, 1), frp=10.0, scan=1.2, track=1.1,
                                  acq_datetime_utc=d + pd.Timedelta(hours=float(rng.choice([2.0, 12.0])))))
    m, v = add_features(pd.DataFrame(mrows)), add_features(v)
    v["i4_saturated"] = False
    scale = {"Aqua": {"k": 0.8, "k_lo": 0.6, "k_hi": 1.0}}
    basis = {"Aqua": "own", "Terra": "none"}
    m, v = harmonize.apply_scale(m, scale, basis), harmonize.apply_scale(v, scale, basis)
    all_days = pd.date_range("2016-03-01", "2016-03-12", freq="D")
    observed = {"viirs": set(all_days) - {gap_day}, "Aqua": set(all_days), "Terra": set(all_days)}   # gap_day: NOT observed
    cal, cells = harmonize.build_calendar(m, v, "S-NPP", t0, t1, observed, basis, clear, **cloud_kw)
    return cal.set_index("date"), cells, (m, v, t0, t1, observed, basis), gap_day, zero_day


def check_calendar(M, V):
    """A day with zero detections is not a day without observations; MODIS never fills a VIIRS gap."""
    cal, cells, args, gap_day, zero_day = _calendar_case(n_viirs=3)          # sparse: an observed empty day is a real zero
    g, z = cal.loc[gap_day], cal.loc[zero_day]
    print(f"gap day -> {g['harmonized_status']}, observed empty day (sparse series) -> {z['harmonized_status']}")
    assert g["harmonized_status"] == "viirs_not_observed" and np.isnan(g["harmonized_frp_mw_viirs_equiv"]), "gap must be NaN"
    assert pd.isna(g["viirs_detections"]) and np.isnan(g["viirs_frp_mw"]), "sensor numbers on a gap day are unknown, not 0"
    assert bool(g["modis_only_evidence"]) and g["modis_frp_viirs_equiv_mw"] > 0, "MODIS-only evidence flagged, never substituted"
    assert g["harmonized_source"] == "none", "MODIS must not be the source of a harmonized value"
    assert z["harmonized_status"] == "measured_zero" and z["harmonized_frp_mw_viirs_equiv"] == 0.0 and z["viirs_detections"] == 0, \
        "an observed empty day is a measured zero (0.0), not a missing value"
    ok = cal.loc[pd.Timestamp("2016-03-03")]
    assert ok["harmonized_status"] == "measured" and ok["harmonized_frp_mw_viirs_equiv"] == ok["viirs_frp_mw"]
    assert np.isclose(ok["modis_frp_viirs_equiv_mw"], 0.8 * ok["aqua_frp_mw"]), "equivalent uses Aqua only"
    assert ok["terra_frp_mw"] > 0 and ok["modis_frp_viirs_equiv_satellites"] == "Aqua", "Terra reported, never scaled"
    assert cells.loc[cells["source_satellite"] == "Terra", "frp_viirs_equiv_mw"].isna().all(), "Terra cells: NaN, not 0"
    assert cells["cell_area_km2"].gt(0).all()
    m, v, t0, t1, observed, basis = args
    cal2, _ = harmonize.build_calendar(m, v, "S-NPP", pd.Timestamp("2016-03-01 12:00", tz="UTC"), t1, observed, basis)
    assert cal2.set_index("date").loc[pd.Timestamp("2016-03-01"), "harmonized_status"] == "partial_day", "window edge"

    dense, _, _, gap_day, zero_day = _calendar_case(n_viirs=40)              # dense: an empty day is an outage suspect
    d = dense.loc[zero_day]
    print(f"empty day in a dense series (40/day) -> {d['harmonized_status']}")
    assert d["harmonized_status"] == "measured_low_coverage" and d["harmonized_is_measured"], "zero where 40 are normal"
    assert dense.loc[gap_day, "harmonized_status"] == "viirs_not_observed"

    # outage detector as a pure function
    det = pd.Series(np.r_[np.full(40, 80.0), 6.0, np.full(40, 80.0), 30.0, np.full(30, 80.0)])
    cl = pd.Series(np.r_[np.full(40, 3.0), 3.0, np.full(40, 3.0), 1.0, np.full(30, 3.0)])
    low, _ = harmonize.flag_low_coverage(det, cl)
    assert set(np.flatnonzero(low)) == {40, 81}, "80 -> 6 (any passes) and a one-pass 80 -> 30 day are outage-suspect"
    sparse = pd.Series(np.r_[np.full(40, 2.0), 0.0, np.full(40, 2.0)])
    assert not harmonize.flag_low_coverage(sparse, pd.Series(np.zeros(81)))[0].any(), "sparse series: no inference"


# --------------------------------------------------------------------------- B2. cloud obscuration
def _rejected(path):
    """Message of the SystemExit that cloud_mask.read_table raises for a bad file."""
    from core.src import cloud_mask
    try:
        cloud_mask.read_table(path)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{path} should have been rejected")


def check_cloud():
    """Third observation status, 'observed, obscured': an empty day under cloud is not a confirmed zero."""
    import tempfile
    from pathlib import Path
    from core.src import cloud_mask

    # 1. MOD35 Cloud_Mask byte 0: bit 0 = determined, bits 1-2 = 0 cloudy / 1 uncertain / 2 probably clear / 3 clear;
    #    the other bits (day/night, glint, snow, land/water) must not matter
    b = np.array([0b111, 0b101, 0b011, 0b001, 0b110, 0b000, 0b11001111, 0b00001001])
    assert np.isclose(cloud_mask.clear_share_from_mod35(b), 3 / 8), "clear + probably clear (determined only)"
    assert np.isclose(cloud_mask.clear_share_from_mod35(b, probably_clear=False), 2 / 8)
    area = np.array([True] * 6 + [False] * 2)
    assert np.isclose(cloud_mask.clear_share_from_mod35(b, area), 2 / 6), "only pixels inside the area count"
    assert np.isnan(cloud_mask.clear_share_from_mod35(b, np.zeros(8, bool))), "an area without pixels has no clear share"
    for bad in (lambda: cloud_mask.clear_share_from_mod35([300]), lambda: cloud_mask.clear_share_from_mod35(b, area[:3])):
        try:
            bad()
            raise AssertionError("invalid cloud-mask input accepted")
        except ValueError:
            pass

    # 2. the daily table: columns, units, satellite selection, unknown days, bad files
    days = pd.date_range("2016-03-01", "2016-03-06", freq="D")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def write(name, text):
            (tmp / name).write_bytes(text.encode("utf-8"))
            return tmp / name

        # Excel-style file: BOM, CRLF, capitalised headers, a timestamp, two rows for one day, an empty value
        p = write("excel.csv", "﻿Date,Cloud_Fraction\r\n2016-03-02,0.8\r\n2016-03-02,0.6\r\n2016-03-03,\r\n"
                               "2016-03-04 11:30:00,0.1\r\n")
        s, info = cloud_mask.clear_series(cloud_mask.read_table(p), "S-NPP", days)
        assert np.isclose(s[pd.Timestamp("2016-03-02")], 0.3), "cloud 0.8 and 0.6 -> clear 0.2 and 0.4 -> mean 0.3"
        assert np.isclose(s[pd.Timestamp("2016-03-04")], 0.9), "a timestamp is floored to its UTC day"
        assert s[[pd.Timestamp("2016-03-01"), pd.Timestamp("2016-03-03")]].isna().all(), "absent / empty = unknown, not clear"
        assert info["days_with_cloud_info"] == 2 and info["days_without_cloud_info"] == 4 and not info["problems"]
        for col, val in (("clear_percent", "80"), ("cloud_percent", "20")):
            s, _ = cloud_mask.clear_series(cloud_mask.read_table(write(f"{col}.csv", f"date,{col}\n2016-03-02,{val}\n")), "S-NPP", days)
            assert np.isclose(s[pd.Timestamp("2016-03-02")], 0.8), col
        sat = cloud_mask.read_table(write("sat.csv", "date,clear_fraction,satellite\n2016-03-02,0.9,S-NPP\n2016-03-02,0.1,Aqua\n"
                                                    "2016-03-03,0.2,aqua\n2016-03-04,0.4,\n"))
        s, info = cloud_mask.clear_series(sat, "S-NPP", days)
        assert s.notna().sum() == 1 and np.isclose(s[pd.Timestamp("2016-03-02")], 0.9), \
            "a satellite with rows of its own uses only those (not Aqua's, not the rows without a satellite)"
        s, _ = cloud_mask.clear_series(sat, "Aqua", days)
        assert np.isclose(s[pd.Timestamp("2016-03-02")], 0.1) and np.isclose(s[pd.Timestamp("2016-03-03")], 0.2), "case-insensitive"
        s, info = cloud_mask.clear_series(sat, "NOAA-20", days)
        assert s.notna().sum() == 1 and np.isclose(s[pd.Timestamp("2016-03-04")], 0.4) and info["rows_used"] == "rows without a satellite"
        only_aqua = cloud_mask.read_table(write("aqua.csv", "date,clear_fraction,satellite\n2016-03-02,0.1,Aqua\n"))
        s, info = cloud_mask.clear_series(only_aqua, "S-NPP", days)
        assert s.isna().all() and info["problems"], "no usable rows for the reference satellite: nothing applied, and said so"
        s, info = cloud_mask.clear_series(cloud_mask.read_table(write("late.csv", "date,clear_fraction\n2030-01-01,0.5\n")), "S-NPP", days)
        assert s.isna().all() and info["problems"], "a file outside the window is reported, not silently applied"
        assert "between 0 and 1" in _rejected(write("pct_in_fraction.csv", "date,clear_fraction\n2016-03-02,80\n")), \
            "a percentage in a fraction column is rejected, not guessed"
        for name, text in (("neg.csv", "date,cloud_fraction\n2016-03-02,-0.2\n"), ("pct150.csv", "date,clear_percent\n2016-03-02,150\n"),
                           ("text.csv", "date,clear_fraction\n2016-03-02,abc\n"), ("nodate.csv", "day,clear_fraction\n2016-03-02,0.5\n"),
                           ("noval.csv", "date,something\n2016-03-02,0.5\n"), ("baddate.csv", "date,clear_fraction\nsoon,0.5\n"),
                           ("empty.csv", "date,clear_fraction\n2016-03-02,\n"), ("blank.csv", "")):
            _rejected(write(name, text))
        _rejected(tmp / "does_not_exist.csv")

    # 3. the calendar: obscured = observed + empty + too little clear sky; everything else keeps its old meaning
    cdays = pd.date_range("2016-03-01", "2016-03-12", freq="D")

    def clr(over=None, base=0.9):
        c = pd.Series(base, index=cdays)
        for d, x in (over or {}).items():
            c[pd.Timestamp(d)] = x
        return c

    cloudy = {"2016-03-08": 0.2, "2016-03-05": 0.1, "2016-03-03": 0.1, "2016-03-10": np.nan}   # zero day, gap day, a day WITH detections, unknown
    cal, _, _, gap_day, zero_day = _calendar_case(3, clr(cloudy))
    z, g, n, u = cal.loc[zero_day], cal.loc[gap_day], cal.loc["2016-03-03"], cal.loc["2016-03-10"]
    print(f"cloudy empty day -> {z['harmonized_status']}, cloudy day with detections -> {n['harmonized_status']}, "
          f"unknown cloud -> {u['harmonized_status']}")
    assert z["harmonized_status"] == "observed_obscured" and np.isnan(z["harmonized_frp_mw_viirs_equiv"]), "not a confirmed zero"
    assert not bool(z["harmonized_is_measured"]) and z["harmonized_source"] == "none"
    assert z["viirs_detections"] == 0 and np.isclose(z["viirs_clear_fraction"], 0.2) and bool(z["viirs_cloud_limited"]), \
        "the sensor WAS observed: its own numbers stay 0, not NaN"
    assert g["harmonized_status"] == "viirs_not_observed" and not bool(g["viirs_cloud_limited"]), "a gap stays a gap"
    assert n["harmonized_status"] == "measured" and n["harmonized_frp_mw_viirs_equiv"] == n["viirs_frp_mw"] and bool(n["viirs_cloud_limited"]), \
        "detections are real: kept and flagged by default"
    assert u["harmonized_status"] == "measured" and np.isnan(u["viirs_clear_fraction"]) and not bool(u["viirs_cloud_limited"]), \
        "unknown cloud cover is neither clear nor cloudy"
    assert not bool(cal.loc["2016-03-04", "viirs_cloud_limited"])
    summ = harmonize.cloud_summary(cal.reset_index(), {}, 0.5, False)
    assert (summ["days_cloud_limited"], summ["days_marked_observed_obscured"], summ["cloud_limited_days_with_detections_kept"]) == (2, 1, 1)
    assert "observed_obscured" in cloud_mask.describe(dict(summ, calendar_days=12, days_with_cloud_info=11, days_without_cloud_info=1,
                                                           rows_used="x"))
    for label, c in (("clear sky", clr()), ("unknown everywhere", clr(base=np.nan)), ("exactly at the threshold", clr({zero_day: 0.5}))):
        zc = _calendar_case(3, c)[0].loc[zero_day]
        assert zc["harmonized_status"] == "measured_zero" and zc["harmonized_frp_mw_viirs_equiv"] == 0.0, f"{label}: a confirmed zero"
    assert _calendar_case(3, clr(cloudy), min_clear=0.15)[0].loc[zero_day, "harmonized_status"] == "measured_zero", "threshold 0.15 < clear 0.2"
    nz = _calendar_case(3, clr(cloudy), obscure_nonzero=True)[0].loc["2016-03-03"]
    assert nz["harmonized_status"] == "observed_obscured" and np.isnan(nz["harmonized_frp_mw_viirs_equiv"]) and nz["viirs_frp_mw"] > 0, \
        "PYRO_CLOUD_OBSCURE_NONZERO: harmonized value NaN, the sensor's own FRP stays in the calendar"
    dense_cloudy = _calendar_case(40, clr(cloudy))[0].loc[zero_day]
    dense_clear = _calendar_case(40, clr())[0].loc[zero_day]
    assert dense_cloudy["harmonized_status"] == "observed_obscured" and dense_clear["harmonized_status"] == "measured_low_coverage", \
        "dense series: cloud explains the empty day; on a clear day it is still an outage suspect"
    base = _calendar_case(3)[0]
    same = _calendar_case(3, clr(base=1.0))[0]
    extra = ["viirs_clear_fraction", "viirs_cloud_limited"]
    assert not set(extra) & set(base.columns) and set(same.columns) - set(base.columns) == set(extra), "no cloud file: no new columns"
    pd.testing.assert_frame_equal(base, same.drop(columns=extra))     # a fully clear sky changes nothing

    # 4. the anomaly detector reads the real calendar_daily.csv: obscured days stay unscored, even in the re-admission variant
    rng = np.random.default_rng(11)
    idx = pd.date_range("2012-01-01", "2019-12-31", freq="D")
    doy = anomaly.doy365(idx.to_series())
    val = (200 * np.exp(-0.5 * ((doy - 200) / 35.0) ** 2) + 5) * np.exp(rng.normal(0, 0.15, len(idx)))
    obsc = pd.DatetimeIndex(["2014-02-03", "2015-07-04", "2017-11-20"])
    status = pd.Series("measured", index=idx)
    status.loc[obsc] = "observed_obscured"
    table = pd.DataFrame({"date": idx, "harmonized_status": status.to_numpy(),
                          anomaly.VALUE_COL: np.where(status == "observed_obscured", np.nan, val)})
    with tempfile.TemporaryDirectory() as td:
        table.to_csv(Path(td) / "calendar_daily.csv", index=False)
        old = anomaly.PROCESSED_DIR
        anomaly.PROCESSED_DIR = Path(td)
        try:
            acal = anomaly.load_series()
        finally:
            anomaly.PROCESSED_DIR = old
    assert (acal.loc[obsc, "status"] == "observed_obscured").all() and not acal.loc[obsc, "valid"].any()
    assert acal.loc[obsc, "value"].isna().all() and acal.loc[obsc, "value_measured"].isna().all()
    assert acal["valid"].sum() == len(idx) - len(obsc)
    scored = anomaly.score(acal)
    readmit = anomaly.score(acal, valid=acal["status"].isin(("measured", "measured_zero", "measured_low_coverage")).to_numpy())
    assert scored.loc[obsc, "z"].isna().all() and readmit.loc[obsc, "z"].isna().all(), "obscured days are never scored"
    assert anomaly_validation.missing_observation_checks(acal, scored, {})["days_by_status"]["observed_obscured"] == len(obsc)
    print("cloud: MOD35 decoding, file checks, status logic (precedence, threshold, unknown days) and anomaly exclusion verified")


# --------------------------------------------------------------------------- C. grid
def check_k_window_and_demo():
    """(1) the k time-gap window widens only when needed, and says so; (2) PYRO_SINGLE_DAY_DEMO labels a partial day
    without ever making it a measured day."""
    saved = (harmonize.K_FIT_MAX_DT_MIN, harmonize.K_FIT_AUTO_WIDEN, dict(harmonize.K_WINDOW_NOTE), harmonize.SINGLE_DAY_DEMO)
    try:
        harmonize.K_FIT_AUTO_WIDEN = True
        harmonize.K_FIT_MAX_DT_MIN = 30.0
        assert harmonize.resolve_k_window(lambda w: w >= 30) == 30.0 and not harmonize.K_WINDOW_NOTE["widened"]
        harmonize.K_FIT_MAX_DT_MIN = 30.0
        assert harmonize.resolve_k_window(lambda w: w >= 60) == 60.0 and harmonize.K_WINDOW_NOTE["widened"]
        assert harmonize.K_WINDOW_NOTE["strict_min"] == 30.0 and harmonize.K_WINDOW_NOTE["used_min"] == 60.0
        harmonize.K_FIT_MAX_DT_MIN = 30.0
        assert harmonize.resolve_k_window(lambda w: False) == 30.0 and not harmonize.K_WINDOW_NOTE["widened"]
        harmonize.K_FIT_AUTO_WIDEN, harmonize.K_FIT_MAX_DT_MIN = False, 30.0     # a fixed window is never widened
        assert harmonize.resolve_k_window(lambda w: w >= 60) == 30.0
        print("k window: kept when it works, widened (and recorded) only when it does not, never when fixed")

        _, _, (m, v, t0, t1, observed, basis), _, _ = _calendar_case(5)
        t0p = pd.Timestamp("2016-03-01 06:00", tz="UTC")                          # the first day is only partly covered
        harmonize.SINGLE_DAY_DEMO = False
        off, _ = harmonize.build_calendar(m, v, "S-NPP", t0p, t1, observed, basis)
        off = off.set_index("date")
        assert off.loc["2016-03-01", "harmonized_status"] == "partial_day"
        assert np.isnan(off.loc["2016-03-01", "harmonized_frp_mw_viirs_equiv"])
        harmonize.SINGLE_DAY_DEMO = True
        on, _ = harmonize.build_calendar(m, v, "S-NPP", t0p, t1, observed, basis)
        on = on.set_index("date")
        row = on.loc["2016-03-01"]
        assert row["harmonized_status"] == "demo_partial_window"
        assert row["harmonized_frp_mw_viirs_equiv"] == row["viirs_frp_mw"] > 0 and not row["harmonized_is_measured"]
        assert "demo" in row["harmonized_source"]
        assert on.loc["2016-03-02":, "harmonized_status"].equals(off.loc["2016-03-02":, "harmonized_status"])
        assert "demo_partial_window" not in anomaly.SCORING_STATUS
        print("single-day demo: partial day labelled demo_partial_window with a lower-bound value, never measured or scored; other days unchanged")
    finally:
        (harmonize.K_FIT_MAX_DT_MIN, harmonize.K_FIT_AUTO_WIDEN, _n, harmonize.SINGLE_DAY_DEMO) = saved
        harmonize.K_WINDOW_NOTE.clear()
        harmonize.K_WINDOW_NOTE.update(_n)


def check_grid(M, V):
    pairs = match_pair(add_features(M), add_features(V), 180)
    use = harmonize._usable(pairs)
    use = use[use["abs_dt_min"] <= 30]
    Md, Vd = add_features(M), add_features(V)
    Md["frp_equiv"] = Md["frp"] * 1.6
    days = pd.DatetimeIndex(sorted(set(Md["date"]) | set(Vd["date"])))
    df, info = hv.grid_sensitivity(Md, Vd, days, use, [0.1, 0.5, 2.0], 0.5, 600, "Aqua")
    print("grid sensitivity (synthetic):", df[["cell_deg", "viirs_over_modis_active_cells", "k_ci_rel_width", "edge_split_probability_1km"]].round(3).values.tolist())
    assert info["k_point_estimate_identical_across_grids"], "k is a ratio of sums: it cannot depend on the grid"
    assert df["viirs_over_modis_active_cells"].is_monotonic_decreasing, "coarser cells merge VIIRS cells faster than MODIS cells"
    assert df["share_of_modis_active_cells_also_viirs"].is_monotonic_increasing
    assert df["edge_split_probability_1km"].is_monotonic_decreasing
    sens = hv.k_sensitivity(use, 30, 50)
    assert abs(sens["primary_k"] / info["at_calendar_grid"]["viirs_over_modis_scaled_total_frp"] - 1) > 0   # runs, keys exist
    assert sens["variants"] and "median of pixel ratios (typical pixel)" in sens["variants"]
    units = hv.cluster_unit_sensitivity(use, 400)
    assert {u["unit"] for u in units} >= {"pixel (no clustering)", "0.5 deg cell", "day"}
    assert [u for u in units if u["unit"] == "4 deg cell"][0]["reliable"] is False or len(use) > 0


# --------------------------------------------------------------------------- D. anomalies
def _cal(idx, value, status=None):
    doy = anomaly.doy365(idx.to_series())
    cal = pd.DataFrame({"value": value, "valid": True, "doy": doy, "year": idx.year}, index=idx)
    cal["value_measured"] = value
    cal["status"] = "measured" if status is None else status
    return cal


def check_anomalies():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2012-01-01", "2019-12-31", freq="D")
    doy = anomaly.doy365(idx.to_series())
    seasonal = 200 * np.exp(-0.5 * ((doy - 200) / 35.0) ** 2) + 5
    value = seasonal * np.exp(rng.normal(0, 0.15, len(idx)))
    spike = (idx >= "2017-06-01") & (idx <= "2017-06-07")      # off-peak burst, x8
    value[spike] *= 8
    cal = _cal(idx, value)
    df = anomaly.score(cal)
    events = anomaly.find_events(df)
    hit = events[(pd.to_datetime(events["start"]) <= "2017-06-07") & (pd.to_datetime(events["end"]) >= "2017-06-01")]
    print(f"episodes found: {len(events)} (planted 1), planted one detected: {len(hit) == 1}")
    assert len(hit) == 1, "planted burst not detected"
    assert len(events) <= 2, "too many false alarms"
    # missing days must give NaN scores, never zero-activity anomalies
    gap = cal.copy()
    gap.loc["2016-03-01":"2016-03-20", ["value", "valid"]] = [np.nan, False]
    assert anomaly.score(gap).loc["2016-03-01":"2016-03-20", "z"].isna().all(), "missing days must stay NaN"

    sm = anomaly.climatology(df)
    windows, start = anomaly.critical_windows(sm)
    inside = ((windows["start_doy"] <= 200) & (windows["end_doy"] >= 200)).any()
    print("critical window(s):", windows[["start", "end", "peak"]].values.tolist())
    assert inside, "critical window must contain the planted peak (day 200)"

    # the validation: the planted burst must be ROBUST, the classical z must also see it, a seasonal control must exist
    anomaly_validation.ANOM_REPLICATES, anomaly_validation.ANOM_NULL_SIMS = 4, 150
    v = anomaly_validation.validate(cal, df, events)
    ev = v["events"]
    row = ev[(pd.to_datetime(ev["start"]) <= "2017-06-07") & (pd.to_datetime(ev["end"]) >= "2017-06-01")].iloc[0]
    print(f"validation of the planted burst: {row['robustness']}, {row['variants_detected']}/{row['variants_testable']} variants")
    assert row["robustness"] == "ROBUST" and bool(row["detected_by_classical_z"]), "a x8 burst must be robust"
    assert row["null_p_family_t"] < 0.2 or row["null_p_family_gaussian"] < 0.05, "a x8 burst is not ordinary variability"
    rep = v["report"]
    assert rep["seasonality"]["max_abs_monthly_median_z"] < 0.3, "seasonal cycle not removed"
    assert rep["seasonality"]["high_flags_in_peak_months"]["seasonal_baseline"] is not None
    assert rep["missing_observations"]["scored_days_that_are_not_valid"] == 0
    json.dumps(anomaly_validation._jsonable(rep))        # serialisable

    # placebo: with NO planted event nothing may come out ROBUST
    clean = _cal(idx, seasonal * np.exp(rng.normal(0, 0.15, len(idx))))
    dfc = anomaly.score(clean)
    vc = anomaly_validation.validate(clean, dfc, anomaly.find_events(dfc))
    print(f"placebo (no planted event): {len(vc['events'])} episode(s), robustness {list(vc['events'].get('robustness', []))}")
    assert not (vc["events"].get("robustness", pd.Series(dtype=str)) == "ROBUST").any(), "noise must not give a ROBUST episode"

    # partial-outage days must not become 'low' anomalies: excluded when flagged, flagged when kept
    outage = value.copy()
    out_days = [pd.Timestamp(d) for d in ("2015-03-10", "2016-09-14", "2018-11-02")]
    for d in out_days:
        outage[idx.get_loc(d)] *= 0.04
    status = pd.Series("measured", index=idx)
    status.loc[out_days] = "measured_low_coverage"
    both = _cal(idx, outage, status)
    both["valid"] = both["status"].isin(("measured", "measured_zero"))
    both["value"] = both["value_measured"].where(both["valid"])
    kept = anomaly.score(both, valid=both["status"].isin(("measured", "measured_zero", "measured_low_coverage")).to_numpy())
    dropped = anomaly.score(both)
    assert (kept.loc[out_days, "flag"] == "low").all(), "an outage day left in the data is flagged 'low'"
    assert dropped.loc[out_days, "z"].isna().all() and not (dropped["flag"] == "low").loc[out_days].any(), "excluded outage days stay unscored"
    # a trend biases z by year (early years high, late years low) unless the annual level is removed
    trend = value * np.exp(-0.12 * (idx.year - 2015.5))
    ct = _cal(idx, trend)
    bias = lambda d: float(d["z"].groupby(d["year"]).median().abs().max())
    raw_b, adj_b = bias(anomaly.score(ct, level_adjust=False)), bias(anomaly.score(ct, level_adjust=True))
    print(f"largest yearly median |z| under a -12 %/yr trend: {raw_b:.2f} without, {adj_b:.2f} with the annual level removed")
    assert raw_b > 1.0 and adj_b < 0.7, "annual-level removal must cancel the trend-induced bias of z"
    tr = anomaly_validation.annual_level_trend(anomaly.score(ct, level_adjust=False))
    assert tr["p"] < 0.01 and tr["relative_change_per_year"] < -0.05, "the trend check must see a -12 %/yr decline"


def check_hotsync_activity_anomalies():
    """The HotSync-style active-cell Z-score detects a planted spike and respects VIIRS gaps."""
    idx = pd.date_range("2015-01-01", "2022-12-31", freq="D")
    cal = pd.DataFrame(index=idx)
    cal["valid"] = True
    cal["doy"] = anomaly.doy365(idx.to_series())
    cal["year"] = idx.year
    cal["viirs_active_cells"] = 10.0
    cal["modis_active_cells"] = 8.0
    cal["cells_active_in_both"] = 5.0
    cal["modis_observed"] = True
    cal["viirs_observed"] = True

    spike = pd.Timestamp("2019-06-15")
    cal.loc[spike, "modis_active_cells"] += 100.0
    viirs_only = pd.Timestamp("2021-08-15")
    cal.loc[viirs_only, ["modis_observed", "modis_active_cells", "cells_active_in_both"]] = [False, np.nan, np.nan]
    missing = pd.Timestamp("2020-03-15")
    cal.loc[missing, "valid"] = False
    cal.loc[missing, "viirs_active_cells"] = np.nan
    # Even strong MODIS-only evidence must not fill a VIIRS-unobserved day.
    cal.loc[missing, "modis_active_cells"] = 500.0

    scored, info = hotsync_anomaly.score_activity(cal)
    assert info["calibrated_to_sensor_union"], "overlap days should calibrate VIIRS activity to the sensor union"
    assert scored.loc[spike, "hotsync_activity_flag"] == "high", "planted active-cell spike not detected"
    assert scored.loc[spike, "hotsync_activity_z"] >= 2.5, "HotSync Z-score is below its default threshold"
    assert scored.loc[viirs_only, "hotsync_activity_source"] == "VIIRS scaled to union", "VIIRS-only day should use overlap scale"
    assert np.isfinite(scored.loc[viirs_only, "hotsync_activity_count"]), "VIIRS-only day should remain scoreable"
    assert pd.isna(scored.loc[missing, "hotsync_activity_z"]), "MODIS must not fill a VIIRS-unobserved day"
    events = hotsync_anomaly.top_anomalies(scored)
    assert not events.empty and events.iloc[0]["date"] == spike.date().isoformat(), "top anomaly list missed the planted spike"

    # Without enough overlap, the series deliberately falls back to VIIRS-only
    # rather than mixing uncalibrated sensor counts.
    no_overlap = cal.copy()
    no_overlap["modis_observed"] = False
    no_overlap["modis_active_cells"] = np.nan
    no_overlap["cells_active_in_both"] = np.nan
    raw, raw_info = hotsync_anomaly.score_activity(no_overlap)
    assert not raw_info["calibrated_to_sensor_union"], "no-overlap data must not claim a union calibration"
    assert raw.loc[spike, "hotsync_activity_source"] == "VIIRS only (uncalibrated)"
    print(f"HotSync-style activity anomaly: planted spike Z={scored.loc[spike, 'hotsync_activity_z']:.2f}; "
          f"VIIRS-to-union scale={info['viirs_to_union_scale']:.3f}")


def check_stub_season():
    """A record that ends a few days into a new season must not crash season_by_year (it did on real data)."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2012-01-01", "2019-12-31", freq="D")
    doy = anomaly.doy365(idx.to_series())
    value = (200 * np.exp(-0.5 * ((doy - 200) / 35.0) ** 2) + 5) * np.exp(rng.normal(0, 0.15, len(idx)))
    cal = _cal(idx, value)
    df = anomaly.score(cal)
    _, start = anomaly.critical_windows(anomaly.climatology(df))
    cut = pd.Timestamp("2019-01-01") + pd.Timedelta(days=start - 1 + 3)   # 4 days into the last season
    seasons = anomaly.season_by_year(df.loc[:cut], start)
    assert not seasons.empty, "no seasons returned"
    assert not bool(seasons.iloc[-1]["complete"]), "a 4-day stub season must not count as complete"
    print(f"stub season handled (last season has {int(seasons.iloc[-1]['valid_days'])} valid days)")



def check_streaming_helpers():
    """Exercise bounded staging, streamed calendar parity and exact disk-backed k statistics."""
    rng = np.random.default_rng(44)
    incoming = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=40),
                             "latitude": rng.uniform(-20, 20, 40), "frp": rng.uniform(1, 20, 40)})
    all_at_once = harmonize_streaming._sample_append(pd.DataFrame(), incoming, 12,
                                                     ["date", "latitude", "frp"])
    split = pd.DataFrame()
    for frame in (incoming.iloc[:13], incoming.iloc[13:29], incoming.iloc[29:]):
        split = harmonize_streaming._sample_append(split, frame, 12, ["date", "latitude", "frp"])
    assert len(all_at_once) == len(split) == 12
    assert set(all_at_once["_sample_hash"]) == set(split["_sample_hash"]), "bounded sample must be chunk-boundary invariant"

    # Preprocessing must preserve the list-of-count-frames interface while its dedupe history spills to disk.
    with tempfile.TemporaryDirectory(prefix="pyroscope_preprocess_test_") as td:
        root = Path(td)
        raw_dir, processed_dir = root / "raw", root / "processed"
        raw_dir.mkdir(); processed_dir.mkdir()
        raw_rows = pd.DataFrame([
            {"latitude": 30.0, "longitude": 31.0, "acq_date": "2020-01-01", "acq_time": "1200",
             "satellite": "A", "instrument": "MODIS", "confidence": "80", "frp": "10",
             "daynight": "D", "brightness": "300", "scan": "1", "track": "1", "version": "061", "type": "0"},
            {"latitude": 30.1, "longitude": 31.1, "acq_date": "2020-01-01", "acq_time": "1210",
             "satellite": "A", "instrument": "MODIS", "confidence": "80", "frp": "12",
             "daynight": "D", "brightness": "301", "scan": "1", "track": "1", "version": "061", "type": "0"},
        ])
        raw_rows.to_csv(raw_dir / "first_modis.csv", index=False)
        raw_rows.iloc[[1]].to_csv(raw_dir / "overlap_modis.csv", index=False)
        old = (preprocess.RAW_DIR, preprocess.PROCESSED_DIR, preprocess.CHUNK_ROWS, preprocess.DEDUPE)
        try:
            preprocess.RAW_DIR, preprocess.PROCESSED_DIR = raw_dir, processed_dir
            preprocess.CHUNK_ROWS, preprocess.DEDUPE = 1, True
            daily = preprocess.process_sensor("MODIS", "modis", processed_dir / "modis_standardized.csv")
            written = pd.read_csv(processed_dir / "modis_standardized.csv")
            assert isinstance(daily, list) and len(daily) == 1
            assert len(written) == 2 and int(daily[0]["detections"].sum()) == 2
            assert not list(processed_dir.glob("pyroscope_dedupe_*"))
        finally:
            (preprocess.RAW_DIR, preprocess.PROCESSED_DIR, preprocess.CHUNK_ROWS, preprocess.DEDUPE) = old

    # The streaming SQL calendar must match the legacy calendar from the same raw synthetic detections.
    _, _, args, _, _ = _calendar_case(n_viirs=3)
    m, v, t0, _, observed, basis = args
    t1 = pd.Timestamp("2016-03-03 23:59", tz="UTC")
    m = m[m["date"] <= pd.Timestamp("2016-03-03")].copy()
    v = v[v["date"] <= pd.Timestamp("2016-03-03")].copy()
    with tempfile.TemporaryDirectory(prefix="pyroscope_stream_test_") as td:
        db = sqlite3.connect(Path(td) / "calendar.sqlite")
        try:
            for d in sorted(m["date"].unique()):
                day = m[m["date"] == d]
                harmonize_streaming._aggregate_cells(day).to_sql("modis_cells", db, if_exists="append", index=False)
            for d in sorted(v["date"].unique()):
                day = v[v["date"] == d]
                nclus = int(harmonize.overpass_clusters(day).iloc[0])
                harmonize_streaming._aggregate_cells(day, nclus).to_sql("viirs_cells", db, if_exists="append", index=False)
            got = harmonize_streaming._calendar_from_sql(db, "S-NPP", ["Aqua", "Terra"], t0, t1,
                                                        observed, basis, None).set_index("date")
            expected, expected_cells = harmonize.build_calendar(m, v, "S-NPP", t0, t1, observed, basis)
            expected = expected.set_index("date")
            for col in ("modis_detections", "viirs_detections", "modis_frp_mw", "viirs_frp_mw",
                        "modis_active_cells", "viirs_active_cells", "viirs_overpass_clusters",
                        "cells_active_in_both", "harmonized_status", "harmonized_frp_mw_viirs_equiv"):
                a, b = got[col], expected[col]
                if col == "harmonized_status":
                    assert a.astype(str).equals(b.astype(str)), f"streamed calendar mismatch for {col}"
                else:
                    np.testing.assert_allclose(a.astype(float), b.astype(float), equal_nan=True,
                                               err_msg=f"streamed calendar mismatch for {col}")
            out_cells = Path(td) / "calendar_cells.csv"
            harmonize_streaming._write_calendar_cells(db, out_cells, basis)
            streamed_cells = pd.read_csv(out_cells).sort_values(
                ["sensor_family", "source_satellite", "date", "cell_id"]).reset_index(drop=True)
            expected_cells = expected_cells.sort_values(
                ["sensor_family", "source_satellite", "date", "cell_id"]).reset_index(drop=True)
            assert len(streamed_cells) == len(expected_cells)
            np.testing.assert_allclose(streamed_cells["frp_mw"], expected_cells["frp_mw"], equal_nan=True)
            np.testing.assert_allclose(streamed_cells["frp_viirs_equiv_mw"],
                                       expected_cells["frp_viirs_equiv_mw"], equal_nan=True)
        finally:
            db.close()

    # Full-stream k and its cluster bootstrap must remain exact while year-cell totals spill to SQLite.
    rows = []
    for year in (2015, 2016, 2017):
        for pair_i in range(60):
            day_i = pair_i % 20
            ts = pd.Timestamp(f"{year}-01-01", tz="UTC") + pd.Timedelta(days=day_i, minutes=pair_i)
            cell = 10_000_010 + (pair_i % 12)
            rows.append({"source_satellite": "Aqua", "acq_datetime_utc": ts,
                         "latitude": 0.5 + (cell % 12) * 0.1, "longitude": 1.5,
                         "frp": 10.0, "viirs_frp_sum": 20.0, "viirs_frp_max": 20.0,
                         "n_viirs": 1, "abs_dt_min": 5.0, "cell_id": cell,
                         "daynight": "D", "type": 0, "version": "061",
                         "confidence_class": "high", "viirs_satellite": "S-NPP"})
    pairs = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory(prefix="pyroscope_fit_test_") as td:
        pair_path = Path(td) / "pairs.csv"
        pairs.to_csv(pair_path, index=False)
        db = sqlite3.connect(Path(td) / "fit.sqlite")
        old = (harmonize_streaming.N_BOOT, harmonize_streaming.K_MODEL,
               harmonize.N_BOOT, harmonize.K_MODEL, harmonize.K_MODEL_BOOT)
        try:
            harmonize_streaming.N_BOOT = harmonize.N_BOOT = 40
            harmonize_streaming.K_MODEL = harmonize.K_MODEL = False
            harmonize.K_MODEL_BOOT = 40
            k, info = harmonize_streaming._exact_fit_summary(db, pair_path, pairs, "Aqua")
            assert k is not None and abs(k - 2.0) < 1e-12, f"streamed k should be exactly 2, got {k}"
            assert info["n_pairs"] == 180 and info["n_cells"] == 12
            assert set(info["k_by_year"]) == {"2015", "2016", "2017"}
            assert not db.execute("SELECT name FROM sqlite_master WHERE name='fit_summary'").fetchone(), \
                "temporary sufficient-statistics table should be removed"
        finally:
            (harmonize_streaming.N_BOOT, harmonize_streaming.K_MODEL,
             harmonize.N_BOOT, harmonize.K_MODEL, harmonize.K_MODEL_BOOT) = old
            db.close()
    print("out-of-core helpers: chunk-stable sampling, calendar parity, streamed cell output and exact k passed")

def main():
    check_stats_helpers()
    pairs, M, V = check_scale()
    check_heterogeneity()
    check_k_model()
    check_calendar(M, V)
    check_cloud()
    check_k_window_and_demo()
    check_grid(M, V)
    check_anomalies()
    check_hotsync_activity_anomalies()
    check_stub_season()
    external_validation.self_test()
    print("external burned-area validation helpers passed (synthetic arrays; no HDF files required)")
    check_streaming_helpers()
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
