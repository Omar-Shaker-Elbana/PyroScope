"""Statistical validation of anomaly episodes (called by anomaly.py; also runnable on its own).

A robust z-score above 3 does not prove that an event is exceptional. This module asks, for every episode of the
production analysis, the questions a sceptical reviewer would:

 1. Seasonality      : is the seasonal cycle really removed?  (residual z by month; a no-season negative control)
 2. Extreme values   : do a few extreme observations distort the baseline?  (classical mean/SD vs median/MAD,
                       IQR scale, how far the baselines differ)
 3. Missing data     : do missing observations create artificial anomalies?  (days re-admitted, low flags next to gaps,
                       random data loss, random multi-day outages)
 4. Baseline period  : does the result survive a different baseline?  (window width, earlier / later half of the years,
                       neighbouring years skipped, causal past-only baseline, annual level removed)
 5. Robust alternative: same episodes under an IQR scale and under the classical z; where the peak day sits in the
                       distribution-free pool of same-season values of the other years
 6. Null model       : how often would ordinary variability produce an episode like this?  (empirical-null tail
                       counts and a Monte-Carlo AR(1) record with Gaussian and heavy-tailed (Student-t) marginals,
                       parameters estimated from the NON-event days only)

Each episode gets `robustness` = ROBUST / MODERATE / FRAGILE (share of the consensus variants and of the random
data-loss replicates in which it is still detected, whichever is lower). It is a statement about the statistics, not a
proof that the fire activity was exceptional: sensor, processing and land-cover changes are not testable from FRP alone.
"""
import json

import numpy as np
import pandas as pd
from scipy import signal, stats

from src import anomaly as A
from src.config import PROCESSED_DIR, ANOM_Z, ANOM_WINDOW_DAYS, ANOM_REPLICATES, ANOM_NULL_SIMS, ANOM_DETREND

EVENT_PAD = 2                      # a variant "detects" an episode if one of its episodes overlaps it +- this many days
ROBUST_FRAC, MODERATE_FRAC = 0.8, 0.5
NULL_THRESHOLDS = (2.0, 2.5, 3.0, 3.5, 4.0)
MIN_YEARS_FOR_PERIOD_SPLIT = 6
MIN_POOL_TESTABLE = 45             # a variant only tests an episode if its baseline holds >= 45 values (~3 years of +-7 d)


# --------------------------------------------------------------------------- helpers
def _event_positions(df, events):
    """[(first_pos, last_pos)] of every production episode in the daily index."""
    out = []
    for _, e in events.iterrows():
        out.append((df.index.get_loc(pd.Timestamp(e["start"])), df.index.get_loc(pd.Timestamp(e["end"]))))
    return out


def _overlaps(slices, a, b, pad=EVENT_PAD):
    return any(s <= b + pad and e >= a - pad for s, e in slices)


def _protect_mask(n, pos, margin):
    m = np.zeros(n, bool)
    for a, b in pos:
        m[max(0, a - margin): b + margin + 1] = True
    return m


def variant_specs(cal):
    """(name, family, score-kwargs, in_consensus). Consensus variants are legitimate alternative analyses of the same
    data; the classical z, the re-admission of outage-suspect days and the no-season control are reported but not
    counted (a method that is less sensitive by construction cannot make an episode 'fragile')."""
    years = sorted(cal.loc[cal["valid"], "year"].unique())
    specs = [
        ("baseline window +-3 days", "baseline window", dict(window=3), True),
        ("baseline window +-14 days", "baseline window", dict(window=14), True),
        ("baseline window +-21 days", "baseline window", dict(window=21), True),
        ("scale = IQR/1.349 instead of MAD", "scale estimator", dict(method="median_iqr"), True),
        ("baseline skips the two neighbouring years", "baseline period", dict(pool="skip_neighbours"), True),
        ("baseline = earlier years only (causal)", "baseline period", dict(pool="past"), True),
        (("annual level kept" if ANOM_DETREND else "annual level removed (trend-robust)"), "trend",
         dict(level_adjust=not ANOM_DETREND), True),
    ]
    if len(years) >= MIN_YEARS_FOR_PERIOD_SPLIT:
        h = len(years) // 2
        specs += [
            (f"baseline = {years[0]}-{years[h - 1]} only", "baseline period", dict(pool=set(years[:h])), True),
            (f"baseline = {years[h]}-{years[-1]} only", "baseline period", dict(pool=set(years[h:])), True),
        ]
    specs += [("classical mean / SD z-score", "classical z (not robust)", dict(method="mean_sd"), False)]
    if "value_measured" in cal.columns and "status" in cal.columns:
        with_low = cal["status"].isin(("measured", "measured_zero", "measured_low_coverage")).to_numpy()
        if (with_low != cal["valid"].to_numpy()).any():
            specs.append(("outage-suspect days re-admitted", "missing data", dict(valid=with_low), False))
    specs.append(("NO seasonal baseline (whole year pooled)", "negative control", dict(window=183), False))
    return specs


def _window(df, e, pad=0):
    a, b = df.index.get_loc(pd.Timestamp(e["start"])), df.index.get_loc(pd.Timestamp(e["end"]))
    return max(0, a - pad), min(len(df) - 1, b + pad)


# --------------------------------------------------------------------------- 1-2-4-5: variants
def run_variants(cal, df, events):
    """Every variant re-scores the whole record; returns (summary rows, long per-event rows, scored frames)."""
    pos = _event_positions(df, events)
    rows, long_rows, frames = [], [], {}
    specs = [("production analysis", "production", {}, False)] + variant_specs(cal)
    for name, family, kw, consensus in specs:
        d = df if not kw else A.score(cal, **kw)
        frames[name] = d
        z = d["z"].to_numpy()
        sl = A.episode_slices(z)
        rows.append({
            "variant": name, "family": family, "in_consensus": bool(consensus),
            "days_scored": int(np.isfinite(z).sum()), "high_days": int((d["flag"] == "high").sum()),
            "low_days": int((d["flag"] == "low").sum()), "episodes": len(sl),
            "episode_spans": "; ".join(f"{d.index[a].date()}..{d.index[b].date()}" for a, b in sl)})
        for i, (a, b) in enumerate(pos):
            win = z[max(0, a - EVENT_PAD): b + EVENT_PAD + 1]
            own = z[a: b + 1]
            long_rows.append({
                "event_id": f"E{i + 1}", "variant": name, "family": family, "in_consensus": bool(consensus),
                "testable": bool(np.isfinite(own).any() and d["pool_n"].iloc[a: b + 1].max() >= MIN_POOL_TESTABLE),
                "detected": bool(_overlaps(sl, a, b)),
                "peak_z": float(np.nanmax(win)) if np.isfinite(win).any() else float("nan")})
    return pd.DataFrame(rows), pd.DataFrame(long_rows), frames


# --------------------------------------------------------------------------- 3: random data loss
def thinning_experiments(cal, df, events, reps, seed=11):
    """Re-score after losing data at random. The episode days themselves (+- 5 days) are never removed, so this tests
    the effect of missing data on the BASELINE and on false alarms, not the trivial loss of the event itself."""
    rng = np.random.default_rng(seed)
    pos = _event_positions(df, events)
    n = len(df)
    protect = _protect_mask(n, pos, 5)
    base_valid = df["valid"].to_numpy()
    years = df["year"].to_numpy()
    scenarios = {"random 10% of days lost": "iid10", "random 25% of days lost": "iid25",
                 "three random 15-day outages per year": "blocks"}
    out = {}
    for label, kind in scenarios.items():
        det = np.zeros((reps, len(pos)), bool)
        new_eps = np.zeros(reps)
        for r in range(reps):
            drop = np.zeros(n, bool)
            if kind in ("iid10", "iid25"):
                drop = rng.random(n) < (0.10 if kind == "iid10" else 0.25)
            else:
                for y in np.unique(years):
                    idx = np.flatnonzero(years == y)
                    for st in rng.integers(0, max(len(idx) - 15, 1), 3):
                        drop[idx[st]: idx[st] + 15] = True
            drop &= ~protect
            d = A.score(cal, valid=base_valid & ~drop)
            sl = A.episode_slices(d["z"].to_numpy())
            for i, (a, b) in enumerate(pos):
                det[r, i] = _overlaps(sl, a, b)
            new_eps[r] = sum(not any(s <= b + EVENT_PAD and e >= a - EVENT_PAD for a, b in pos) for s, e in sl)
        out[label] = {"replicates": int(reps), "detection_rate_per_event": [float(x) for x in det.mean(axis=0)],
                      "new_episodes_per_record_mean": float(new_eps.mean()),
                      "records_with_any_new_episode": float((new_eps > 0).mean())}
    return out


# --------------------------------------------------------------------------- 1: seasonality
def seasonality_checks(df, naive):
    z = df["z"]
    ok = z.notna()
    month = df.index.month
    groups = [z[ok & (month == m)].to_numpy() for m in range(1, 13)]
    groups = [g for g in groups if len(g) > 5]
    med = {int(m): float(np.median(z[ok & (month == m)])) for m in range(1, 13) if (ok & (month == m)).sum() > 5}
    spread = {int(m): float(stats.median_abs_deviation(z[ok & (month == m)], scale="normal"))
              for m in range(1, 13) if (ok & (month == m)).sum() > 5}
    clim = df["value"].groupby(month).median()
    peak_months = list(clim.sort_values(ascending=False).index[:3])

    def share(d):
        h = d.index[d["flag"] == "high"]
        return None if len(h) == 0 else float(np.isin(h.month, peak_months).mean())

    return {
        "monthly_median_z": med, "monthly_robust_sd_z": spread,
        "max_abs_monthly_median_z": float(max(abs(v) for v in med.values())),
        "kruskal_wallis_p_location": float(stats.kruskal(*groups).pvalue) if len(groups) > 2 else None,
        "brown_forsythe_p_scale": float(stats.levene(*groups, center="median").pvalue) if len(groups) > 2 else None,
        "reading": ("monthly median z close to 0 and robust SD close to 1 mean the seasonal cycle is removed; with ~4000 "
                    "days the tests detect tiny departures, so judge the effect sizes (max_abs_monthly_median_z)."),
        "highest_climatology_months": [int(m) for m in peak_months],
        "high_flags_in_peak_months": {"seasonal_baseline": share(df), "no_seasonal_baseline_control": share(naive)},
        "high_days_by_month": {
            "seasonal_baseline": {int(m): int(((df["flag"] == "high") & (month == m)).sum()) for m in range(1, 13)},
            "no_seasonal_baseline_control": {int(m): int(((naive["flag"] == "high") & (naive.index.month == m)).sum()) for m in range(1, 13)}},
        "episodes_without_seasonal_baseline": int(len(A.episode_slices(naive["z"].to_numpy()))),
    }


# --------------------------------------------------------------------------- 2: extreme values
def baseline_distortion(df):
    x = np.log1p(df["value"].to_numpy(float))
    ok = df["valid"].to_numpy() & np.isfinite(x)
    args = (x, df["doy"].to_numpy(), df["year"].to_numpy(), ok)
    c1, s1, _ = A.baseline(*args, method="median_mad")
    c2, s2, _ = A.baseline(*args, method="mean_sd")
    f = np.isfinite(c1) & np.isfinite(c2)
    ratio = (s2 / s1)[f]
    shift = ((c2 - c1) / s1)[f]
    return {
        "sd_over_robust_sigma": {"median": float(np.median(ratio)), "p90": float(np.percentile(ratio, 90)),
                                 "max": float(ratio.max())},
        "mean_minus_median_in_robust_sigma": {"median": float(np.median(shift)), "p10": float(np.percentile(shift, 10)),
                                               "p90": float(np.percentile(shift, 90))},
        "reading": ("a ratio above 1 means the classical SD of the same-season pool is inflated by extreme values, which "
                    "makes a classical z-score LESS sensitive; a mean far from the median means the extremes also move the "
                    "centre. The production z uses median / MAD, so it is not affected."),
    }


# --------------------------------------------------------------------------- 3: missing observations
def missing_observation_checks(cal, df, frames):
    status = cal["status"]
    unobs = status.isin(("viirs_not_observed", "partial_day", "absent", "observed_obscured"))
    near_gap = unobs.shift(1, fill_value=False) | unobs.shift(-1, fill_value=False)
    low = df["flag"] == "low"
    out = {
        "days_by_status": {k: int(v) for k, v in status.value_counts().items()},
        "scored_days_that_are_not_valid": int(((~df["valid"]) & df["z"].notna()).sum()),
        "low_flags": int(low.sum()),
        "low_flags_next_to_an_unobserved_day": int((low & near_gap).sum()),
        "low_flags_on_genuine_zero_days": int((low & (status == "measured_zero")).sum()),
    }
    name = "outage-suspect days re-admitted"
    if name in frames:
        d2 = frames[name]
        l2 = d2["flag"] == "low"
        on_susp = l2 & (status == "measured_low_coverage")
        out.update({
            "low_flags_if_outage_suspect_days_were_kept": int(l2.sum()),
            "of_which_on_outage_suspect_days": int(on_susp.sum()),
            "share_of_low_flags_explained_by_outage_suspect_days": float(on_susp.sum() / max(int(l2.sum()), 1)),
            "high_flags_if_outage_suspect_days_were_kept": int((d2["flag"] == "high").sum()),
            "high_flags_production": int((df["flag"] == "high").sum())})
    out["reading"] = ("missing days are NaN (never zero) and cannot be scored; a day that lost overpasses is biased low and "
                      "would be flagged as a 'low' anomaly, so such days are excluded from the production analysis. "
                      "Missing observations cannot create a high anomaly; they can only make the baseline slightly noisier "
                      "(see thinning_experiments).")
    return out


# --------------------------------------------------------------------------- 5: distribution-free peak position
def pool_position(df, events, window=None):
    """Where the peak day of each episode sits among the same-season values of the other years (no model, no scale)."""
    window = ANOM_WINDOW_DAYS if window is None else window
    x = np.log1p(df["value"].to_numpy(float))
    ok = df["valid"].to_numpy() & np.isfinite(x)
    rows = []
    for _, e in events.iterrows():
        d = pd.Timestamp(e["peak_date"])
        i = df.index.get_loc(d)
        dist = np.abs(df["doy"].to_numpy() - df["doy"].iloc[i])
        dist = np.minimum(dist, 365 - dist)
        pool = x[ok & (dist <= window) & (df["year"].to_numpy() != df["year"].iloc[i])]
        if len(pool) == 0:
            rows.append((np.nan, 0, False))
            continue
        rows.append(((np.sum(pool < x[i]) + 0.5 * np.sum(pool == x[i])) / len(pool), int(len(pool)), bool(x[i] > pool.max())))
    return rows


# --------------------------------------------------------------------------- 6: null model
def null_calibration(df, events, n_sims, seed=5):
    """What ordinary variability alone would produce.

    The null is estimated from the NON-event days (event days +-3 removed), so the episodes do not define their own null:
      - empirical null: centre and scale of the central part of z; expected number of days above each threshold under a
        Gaussian and under a heavy-tailed Student-t fitted to the non-event days; observed / expected = estimated false
        discovery proportion of a z threshold (Efron).
      - Monte-Carlo records with the real validity mask and the lag-1 autocorrelation of the non-event days, run through
        the SAME episode rule; the family-wise p of an episode is the share of null records that contain an episode at
        least as long and at least as high.
    Both nulls are crude (stationary, one lag of memory); the t null is the one to read. Neither models changes of
    sensor, algorithm or land cover."""
    z = df["z"].to_numpy(float)
    fin = np.isfinite(z)
    pos = _event_positions(df, events)
    ev = _protect_mask(len(z), pos, 3)
    zm = np.where(fin & ~ev, z, np.nan)
    zn = zm[np.isfinite(zm)]
    mu0 = float(np.median(zn))
    s0 = float(stats.iqr(zn) / 1.349)
    nu, loc, scale = (float(v) for v in stats.t.fit(zn))
    a, b = zm[:-1], zm[1:]
    okp = np.isfinite(a) & np.isfinite(b)
    phi = float(np.clip(np.corrcoef(a[okp], b[okp])[0, 1], 0.0, 0.95))
    n_fin = int(fin.sum())
    tails = []
    for t in NULL_THRESHOLDS:
        obs = int((z[fin] >= t).sum())
        eg = float(n_fin * stats.norm.sf((t - mu0) / s0))
        et = float(n_fin * stats.t.sf((t - loc) / scale, nu))
        tails.append({"z_threshold": t, "observed_days": obs, "expected_gaussian_null": eg, "expected_t_null": et,
                      "false_discovery_proportion_gaussian": min(1.0, eg / obs) if obs else None,
                      "false_discovery_proportion_t": min(1.0, et / obs) if obs else None})
    rng = np.random.default_rng(seed)
    n = len(z)
    spans = {"gaussian": [], "t": []}
    peaks = {"gaussian": [], "t": []}
    counts = {"gaussian": np.zeros(n_sims), "t": np.zeros(n_sims)}
    done = 0
    while done < n_sims:
        k = min(500, n_sims - done)
        g = signal.lfilter([np.sqrt(1 - phi ** 2)], [1, -phi], rng.standard_normal((k, n)), axis=1)
        sims = {"gaussian": mu0 + s0 * g, "t": loc + scale * stats.t.ppf(stats.norm.cdf(g), nu)}
        for kind, zz in sims.items():
            zz[:, ~fin] = np.nan
            for r in range(k):
                sl = A.episode_slices(zz[r])
                counts[kind][done + r] = len(sl)
                spans[kind].append(np.array([e - s + 1 for s, e in sl], int))
                peaks[kind].append(np.array([np.nanmax(zz[r][s: e + 1]) for s, e in sl], float))
        done += k
    per_event = []
    for i, (a_, b_) in enumerate(pos):
        span, pk = b_ - a_ + 1, float(np.nanmax(z[a_: b_ + 1]))
        rec = {"event_id": f"E{i + 1}", "span_days": int(span), "peak_z": pk}
        for kind in ("gaussian", "t"):
            hit = sum(bool(((sp >= span) & (p >= pk)).any()) for sp, p in zip(spans[kind], peaks[kind]))
            rec[f"p_family_{kind}"] = float((hit + 1) / (n_sims + 1))
        per_event.append(rec)
    n_obs = len(pos)
    return {
        "null_parameters": {"centre": mu0, "scale": s0, "student_t": {"df": nu, "loc": loc, "scale": scale},
                            "lag1_autocorrelation": phi, "non_event_days": int(len(zn)), "simulated_records": int(n_sims)},
        "tail_counts": tails,
        "episodes": {"observed": n_obs,
                     "expected_per_record_gaussian": float(counts["gaussian"].mean()),
                     "expected_per_record_t": float(counts["t"].mean()),
                     "p_at_least_observed_count_gaussian": float(((counts["gaussian"] >= n_obs).sum() + 1) / (n_sims + 1)),
                     "p_at_least_observed_count_t": float(((counts["t"] >= n_obs).sum() + 1) / (n_sims + 1))},
        "per_event": per_event,
        "reading": ("p_family = chance that a record of this length and gap pattern contains an episode at least this long "
                    "and this high by ordinary variability alone. It is an order-of-magnitude screen under a stationary "
                    "model, not a significance claim about fire regimes."),
    }


def annual_level_trend(df):
    v = np.log1p(df["value"])
    g = v.groupby(df["year"]).agg(["median", "count"])
    g = g[g["count"] >= 300]
    if len(g) < 5:
        return {"note": "fewer than 5 complete years"}
    tau, p = stats.kendalltau(g.index.to_numpy(), g["median"].to_numpy())
    slope = stats.theilslopes(g["median"].to_numpy(), g.index.to_numpy())[0]
    return {"years": [int(y) for y in g.index], "kendall_tau": float(tau), "p": float(p),
            "theil_sen_log_change_per_year": float(slope), "relative_change_per_year": float(np.expm1(slope)),
            "reading": ("a significant trend means the leave-one-year-out baseline mixes different activity levels; the "
                        "'annual level removed' variant checks that this does not create or hide episodes.")}


# --------------------------------------------------------------------------- orchestration
def label(frac):
    return "ROBUST" if frac >= ROBUST_FRAC else ("MODERATE" if frac >= MODERATE_FRAC else "FRAGILE")


def validate(cal, df, events):
    """Run every check; returns a dict with the augmented episode table (`events`), the tables and the JSON report."""
    ev = events.copy()
    summary, long, frames = run_variants(cal, df, ev)
    naive = frames["NO seasonal baseline (whole year pooled)"]
    thin = thinning_experiments(cal, df, ev, ANOM_REPLICATES) if len(ev) else {}
    null = null_calibration(df, ev, ANOM_NULL_SIMS) if len(ev) else None
    pp = pool_position(df, ev) if len(ev) else []

    ev.insert(0, "event_id", [f"E{i + 1}" for i in range(len(ev))])
    cols = {k: [] for k in ("variants_testable", "variants_detected", "detected_fraction", "thinning_detection_rate",
                            "robustness", "detected_by_classical_z", "peak_z_min_over_variants",
                            "peak_z_median_over_variants", "peak_pool_percentile", "peak_pool_n",
                            "peak_exceeds_all_pool", "null_p_family_t", "null_p_family_gaussian", "assessment")}
    for i, eid in enumerate(ev["event_id"]):
        L = long[long["event_id"] == eid]
        cons = L[L["in_consensus"] & L["testable"]]
        n_t, n_d = int(len(cons)), int(cons["detected"].sum())
        frac = n_d / n_t if n_t else float("nan")
        rates = [thin[s]["detection_rate_per_event"][i] for s in thin] if thin else []
        thin_rate = float(np.mean(rates)) if rates else float("nan")
        worst = np.nanmin([frac, thin_rate]) if (np.isfinite(frac) or np.isfinite(thin_rate)) else float("nan")
        classical = L[L["variant"] == "classical mean / SD z-score"]
        cl = bool(classical["detected"].iloc[0]) if len(classical) else None
        pk = L[L["variant"] != "NO seasonal baseline (whole year pooled)"]["peak_z"].dropna()
        pct, npool, above = pp[i]
        nf = null["per_event"][i] if null else {}
        rob = label(worst) if np.isfinite(worst) else "NOT_TESTABLE"
        for k, v in (("variants_testable", n_t), ("variants_detected", n_d), ("detected_fraction", frac),
                     ("thinning_detection_rate", thin_rate), ("robustness", rob), ("detected_by_classical_z", cl),
                     ("peak_z_min_over_variants", float(pk.min()) if len(pk) else np.nan),
                     ("peak_z_median_over_variants", float(pk.median()) if len(pk) else np.nan),
                     ("peak_pool_percentile", pct), ("peak_pool_n", npool), ("peak_exceeds_all_pool", above),
                     ("null_p_family_t", nf.get("p_family_t")), ("null_p_family_gaussian", nf.get("p_family_gaussian"))):
            cols[k].append(v)
        cols["assessment"].append(
            f"{rob}: detected in {n_d}/{n_t} consensus variants and {thin_rate:.0%} of data-loss replicates; "
            f"{'also' if cl else 'NOT'} found by the classical z; peak day above {pct:.1%} of {npool} same-season values "
            f"of other years{' (a new maximum)' if above else ''}; family-wise p under the heavy-tailed null "
            f"{nf.get('p_family_t', float('nan')):.3f}. Statistical candidate only.")
    for k, v in cols.items():
        ev[k] = v

    seas = seasonality_checks(df, naive)
    dist = baseline_distortion(df)
    miss = missing_observation_checks(cal, df, frames)
    counts = ev["robustness"].value_counts().to_dict() if len(ev) else {}
    text = (f"{len(ev)} episode(s): " + ", ".join(f"{v} {k}" for k, v in counts.items()) if len(ev) else "no episodes") + \
           f"; {miss['low_flags']} low-flag days ({miss['low_flags_next_to_an_unobserved_day']} next to an unobserved day); " \
           f"without a seasonal baseline the same analysis would find {seas['episodes_without_seasonal_baseline']} episode(s)"
    report = {
        "summary": text,
        "definitions": {"event_overlap_padding_days": EVENT_PAD, "robust_fraction": ROBUST_FRAC, "moderate_fraction": MODERATE_FRAC,
                        "consensus_variants": [r["variant"] for r in summary.to_dict("records") if r["in_consensus"]],
                        "min_baseline_values_for_a_variant_to_count": MIN_POOL_TESTABLE,
                        "robustness_rule": "min(share of consensus variants that still detect the episode, share of random "
                                           "data-loss replicates that still detect it); >= 0.8 ROBUST, >= 0.5 MODERATE, else FRAGILE"},
        "seasonality": seas, "extreme_values": dist, "missing_observations": miss,
        "thinning_experiments": thin, "annual_level_trend": annual_level_trend(df), "null_model": null,
        "episodes": ev.to_dict("records"),
    }
    return {"events": ev, "variants": summary, "event_variants": long, "report": report, "summary_text": text,
            "naive": naive, "frames": {k: v for k, v in frames.items() if k in ("production analysis",
                                                                                "classical mean / SD z-score")}}


# --------------------------------------------------------------------------- outputs
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (pd.Timestamp, np.datetime64)) or hasattr(o, "isoformat"):
        return str(o)
    return o


def plot_validation(v, df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#8a8a85", "#33322f"
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    L = v["event_variants"]
    ax = axes[0]
    ids = list(v["events"]["event_id"])
    for i, eid in enumerate(ids):
        g = L[(L["event_id"] == eid) & (L["variant"] != "NO seasonal baseline (whole year pooled)")]
        ax.scatter(g["peak_z"], [i] * len(g), s=22, color=[BLUE if c else GREY for c in g["in_consensus"]], alpha=0.8)
    ax.axvline(ANOM_Z, color=ORANGE, ls="--", lw=1, label=f"episode threshold z = {ANOM_Z:g}")
    ax.set_yticks(range(len(ids)), [f"{e}\n{v['events']['start'].iloc[i]}" for i, e in enumerate(ids)], fontsize=8)
    ax.set_xlabel("peak z over the episode window"); ax.set_title("Peak z of each episode under every variant\n(blue = consensus, grey = informational)")
    ax.legend(fontsize=8); ax.grid(True, axis="x", alpha=0.3)
    ax = axes[1]
    s = v["report"]["seasonality"]["high_days_by_month"]
    m = np.arange(1, 13)
    ax.bar(m - 0.2, [s["seasonal_baseline"][k] for k in m], 0.4, color=BLUE, label="seasonal baseline (production)")
    ax.bar(m + 0.2, [s["no_seasonal_baseline_control"][k] for k in m], 0.4, color=GREY, label="no seasonal baseline (control)")
    ax.set_xticks(m); ax.set_xlabel("month"); ax.set_ylabel("days flagged high")
    ax.set_title("Days flagged high, by month\n(seasonal baseline vs no-season negative control)"); ax.legend(fontsize=8)
    ax = axes[2]
    nm = v["report"]["null_model"]
    if nm:
        z = df["z"].dropna().to_numpy()
        zz = np.linspace(max(z.min(), -6), z.max() + 0.5, 300)
        ax.hist(z, bins=60, density=True, color=BLUE, alpha=0.35, label="all scored days")
        p = nm["null_parameters"]
        ax.plot(zz, stats.norm.pdf(zz, p["centre"], p["scale"]), color=INK, lw=1.2, label="Gaussian null")
        t = p["student_t"]
        ax.plot(zz, stats.t.pdf((zz - t["loc"]) / t["scale"], t["df"]) / t["scale"], color=ORANGE, lw=1.5, label=f"Student-t null (df {t['df']:.1f})")
        ax.set_yscale("log"); ax.set_ylim(1e-5, 2)
        ax.set_xlabel("robust z"); ax.set_title("Distribution of z (all scored days) and the nulls\nfitted on the non-event days only")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(v, df):
    (PROCESSED_DIR / "anomaly_validation.json").write_text(json.dumps(_jsonable(v["report"]), indent=2))
    v["variants"].to_csv(PROCESSED_DIR / "anomaly_variants.csv", index=False)
    v["event_variants"].to_csv(PROCESSED_DIR / "anomaly_event_variants.csv", index=False)
    plot_validation(v, df, PROCESSED_DIR / "anomaly_validation.png")


def main():
    cal = A.load_series()
    df = A.score(cal)
    ev = A.find_events(df)
    v = validate(cal, df, ev)
    write_outputs(v, df)
    print(v["summary_text"])


if __name__ == "__main__":
    main()
