"""Synthetic end-to-end check of matching, the FRP scale k, the observation-status logic, the grid diagnostics and the
anomaly / validation code, all with a KNOWN answer. Nothing is read from or written to data/.

Run:  python -m src.selftest      (must finish with SELFTEST PASSED)
"""
import json

import numpy as np
import pandas as pd

from src import anomaly, anomaly_validation, harmonize, harmonize_validation as hv, kstats
from src.harmonize import add_features, match_pair, fit_scale

TRUE_K = 1.6          # VIIRS-equivalent FRP = TRUE_K x MODIS FRP
NOISE = 0.25          # log-normal scatter of the summed VIIRS FRP
EXPECTED_K = TRUE_K * np.exp(NOISE ** 2 / 2)  # the mean of exp(N(0, NOISE)) is exp(NOISE^2/2), not 1
FIRES_PER_YEAR = 400
YEARS = (2015, 2016, 2017, 2018)
VIIRS_LEADS_MIN = 5.0  # S-NPP passes this many minutes before Aqua


def synthetic_overpasses(seed=1, year_k=None):
    """Aqua pixels + the S-NPP pixels inside their footprints (+ VIIRS-only clutter). year_k: {year: true k}."""
    rng = np.random.default_rng(seed)
    mrows, vrows = [], []
    for year in YEARS:
        kk = TRUE_K if year_k is None else year_k[year]
        days = rng.integers(0, 365, FIRES_PER_YEAR)
        t0 = pd.Timestamp(f"{year}-01-01 13:30", tz="UTC") + pd.to_timedelta(
            days * 1440 + rng.uniform(-8, 8, FIRES_PER_YEAR), unit="m")
        lat = rng.uniform(-30, 30, FIRES_PER_YEAR)
        lon = rng.uniform(-60, 60, FIRES_PER_YEAR)
        frp_m = np.exp(rng.normal(np.log(40), 0.9, FIRES_PER_YEAR))
        for i in range(FIRES_PER_YEAR):
            mrows.append(dict(sensor_family="MODIS", source_satellite="Aqua", latitude=lat[i],
                              longitude=lon[i], acq_datetime_utc=t0[i], frp=frp_m[i], scan=1.2, track=1.1))
            n = int(rng.integers(2, 7))
            share = rng.dirichlet(np.ones(n)) * frp_m[i] * kk * np.exp(rng.normal(0, NOISE))
            for j in range(n):
                vrows.append(dict(
                    sensor_family="VIIRS", source_satellite="S-NPP",
                    latitude=lat[i] + rng.uniform(-0.3, 0.3) / 111.2,
                    longitude=lon[i] + rng.uniform(-0.3, 0.3) / (111.2 * np.cos(np.radians(lat[i]))),
                    acq_datetime_utc=t0[i] - pd.Timedelta(minutes=VIIRS_LEADS_MIN + rng.uniform(-2, 2)),
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
def _calendar_case(n_viirs):
    """12 days, VIIRS with `n_viirs` detections a day except on a gap day (not observed) and a zero day (observed, empty)."""
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
    cal, cells = harmonize.build_calendar(m, v, "S-NPP", t0, t1, observed, basis)
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


# --------------------------------------------------------------------------- C. grid
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


def main():
    check_stats_helpers()
    pairs, M, V = check_scale()
    check_heterogeneity()
    check_calendar(M, V)
    check_grid(M, V)
    check_anomalies()
    check_stub_season()
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
