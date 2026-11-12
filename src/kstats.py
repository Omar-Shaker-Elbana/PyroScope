"""Pure statistical helpers for the MODIS -> VIIRS scale k (no I/O, no pipeline state).

Everything here is deterministic given its inputs (random numbers come from a generator the caller passes in), so
it can be unit-tested against a known answer in `python -m src.selftest`.

Contents
 - cell_id / cell_area_km2 / edge_split_probability : the lat/lon aggregation grid and what it costs
 - bootstrap_ratio                                  : cluster bootstrap of a ratio of sums (vectorised)
 - heterogeneity                                    : random-effects view of per-year log k (Cochran Q, I^2, tau
                                                      with a Q-profile interval, prediction interval for a new year)
 - infer_missing_days                               : which UTC days are absent from a raw file (not "no fire")
"""
import numpy as np
from scipy import optimize, stats

R_KM = 6371.0088
KM_PER_DEG = R_KM * np.pi / 180.0


# --------------------------------------------------------------------------- grid
def cell_id(lat, lon, deg: float) -> np.ndarray:
    """Integer id of the `deg` x `deg` lat/lon cell containing each point (row-major: lat band * 1e6 + lon band)."""
    ci = np.floor((np.asarray(lat, float) + 90.0) / deg).astype("int64")
    cj = np.floor((np.asarray(lon, float) + 180.0) / deg).astype("int64")
    return ci * 1_000_000 + cj


def cell_area_km2(lat_centre, deg: float):
    """Exact area (km^2) of a `deg` x `deg` lat/lon cell centred at `lat_centre`.
    A = R^2 * d_lambda * (sin(phi_north) - sin(phi_south)); it shrinks with cos(latitude), so a lat/lon grid is NOT
    equal-area: totals per cell are only comparable between cells at similar latitudes."""
    lat = np.asarray(lat_centre, float)
    s = np.sin(np.radians(np.clip(lat + deg / 2.0, -90.0, 90.0))) - np.sin(np.radians(np.clip(lat - deg / 2.0, -90.0, 90.0)))
    return R_KM ** 2 * np.radians(deg) * s


def edge_split_probability(distance_km: float, deg: float, lat: float) -> float:
    """Probability that two detections `distance_km` apart (random orientation, random position) fall into DIFFERENT
    cells of a `deg` grid at latitude `lat`: Buffon's needle on a square lattice, 4 d / (pi s) for d << s, with
    s the geometric-mean cell side. It is the 'same fire, two cells' error of the grid."""
    s = deg * KM_PER_DEG * np.sqrt(max(np.cos(np.radians(lat)), 1e-6))
    return float(min(1.0, 4.0 * distance_km / (np.pi * s)))


# --------------------------------------------------------------------------- bootstrap
def bootstrap_ratio(arr, rng, n_boot: int, chunk: int = 250) -> np.ndarray:
    """Cluster bootstrap of sum(numerator) / sum(denominator).

    arr : (n_clusters, 2) array, column 0 = denominator sum per cluster, column 1 = numerator sum per cluster.
    Whole clusters are resampled with replacement (neighbouring pixels of one fire are not independent).
    Returns `n_boot` resampled ratios."""
    arr = np.asarray(arr, float)
    n = len(arr)
    out = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(b, n))
        out[s:s + b] = arr[idx, 1].sum(axis=1) / arr[idx, 0].sum(axis=1)
    return out


def ratio_ci(arr, rng, n_boot: int):
    """(k, lo, hi, se_log) of a ratio of sums with a cluster-bootstrap 95 % percentile interval."""
    arr = np.asarray(arr, float)
    k = float(arr[:, 1].sum() / arr[:, 0].sum())
    b = bootstrap_ratio(arr, rng, n_boot)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return k, float(lo), float(hi), float(np.std(np.log(b), ddof=1))


# --------------------------------------------------------------------------- between-year heterogeneity
def _qgen(y, se, tau2):
    w = 1.0 / (se ** 2 + tau2)
    mu = float((w * y).sum() / w.sum())
    return float((w * (y - mu) ** 2).sum()), mu, w


def heterogeneity(y, se):
    """Is the year-to-year scatter of a log-scale factor bigger than its sampling error?

    y  : per-year log k.      se : per-year standard error of log k (e.g. cluster-bootstrap SD).
    Returns None for fewer than 3 years, else a dict with
      Q, df, p_Q       Cochran's Q and its chi-square p-value (H0: one common k, scatter = sampling error only)
      I2               share of the scatter beyond sampling error (Higgins' I^2)
      tau, tau_ci95    DerSimonian-Laird between-year SD of log k and its Q-profile 95 % interval
                       (Viechtbauer 2007). tau = 0.10 means a typical year differs from the mean by ~10 %.
      mu_random, se_mu_random   random-effects mean (log) and its SE
    (prediction_interval() turns tau into the interval to carry into any value that uses k.)
    The max-min spread of per-year k is NOT a valid stability test: it grows with the number of years and with
    sampling noise, and is dominated by the smallest year."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    n = len(y)
    if n < 3 or not np.all(np.isfinite(y)) or not np.all(se > 0):
        return None
    df = n - 1
    q0, mu_f, w = _qgen(y, se, 0.0)
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q0 - df) / c)

    def solve(target):  # Q_gen(tau2) decreases in tau2; find where it hits the chi-square quantile
        if q0 <= target:
            return 0.0
        hi = 1.0
        while _qgen(y, se, hi)[0] > target and hi < 1e8:
            hi *= 4.0
        return float(optimize.brentq(lambda t: _qgen(y, se, t)[0] - target, 0.0, hi))

    lo2 = solve(stats.chi2.ppf(0.975, df))
    hi2 = solve(stats.chi2.ppf(0.025, df))
    _, mu_r, w_r = _qgen(y, se, tau2)
    se_mu = float(np.sqrt(1.0 / w_r.sum()))
    return {
        "n_years": int(n), "Q": float(q0), "df": int(df), "p_Q": float(stats.chi2.sf(q0, df)),
        "I2": float(max(0.0, (q0 - df) / q0)) if q0 > 0 else 0.0,
        "tau": float(np.sqrt(tau2)), "tau_ci95": [float(np.sqrt(lo2)), float(np.sqrt(hi2))],
        "mu_random": mu_r, "se_mu_random": se_mu,
    }


def prediction_interval(k, se_log, tau, n_years):
    """Approximate 95 % prediction interval for k in a year NOT in the sample (Higgins et al. 2009), centred on the
    k that is actually applied: k * exp(+-t * sqrt(tau^2 + se_log^2)), t = Student-t quantile with n_years-2 df.
    It combines the sampling error of the pooled k with the between-year SD, so it is wider than the plain
    confidence interval exactly when k is not stable between years. With no between-year information
    (tau = None) it degrades to the confidence-interval half-width."""
    t = stats.t.ppf(0.975, max(int(n_years) - 2, 1)) if tau is not None else 1.96
    half = float(t * np.sqrt((tau or 0.0) ** 2 + se_log ** 2))
    return [float(k * np.exp(-half)), float(k * np.exp(half))]


# --------------------------------------------------------------------------- missing days
def infer_missing_days(counts, alpha: float = 1e-4, window: int = 30) -> np.ndarray:
    """Which days of a daily raw-row-count series are MISSING (outage / not delivered) rather than 'no detections'?

    counts : 1-D array of rows per day over every day of the file's span (0 where the day is absent).
    Every maximal run of L empty days is judged against the days around it (+-`window`, run excluded): their mean mu
    and variance give a negative-binomial model of a normal day, P0 = P(0 rows | normal day). The run is a gap if
    P0**L < alpha, i.e. if that many empty days in a row is implausible *for that file at that time*.
      - dense regional / global files (tens of detections per day): P0 ~ 0, so even ONE empty day is a gap;
      - sparse files (a few detections per day): P0 is large, so only long runs are gaps;
      - a file that is empty all around (P0 = 1) never reveals a gap.
    This replaces a rule that used a Laplace-smoothed share of empty days and therefore treated a single empty
    day of a dense file (e.g. 2024-09-26 of S-NPP over Egypt, ~90 rows on every normal day) as 'observed, zero fire'."""
    cnt = np.asarray(counts, float)
    missing = np.zeros(len(cnt), dtype=bool)
    i = 0
    while i < len(cnt):
        if cnt[i] > 0:
            i += 1
            continue
        j = i
        while j + 1 < len(cnt) and cnt[j + 1] == 0:
            j += 1
        run = j - i + 1
        around = np.r_[cnt[max(0, i - window):i], cnt[j + 1:j + 1 + window]]
        if len(around) == 0:
            gap = run >= 7
        else:
            mu = float(around.mean())
            var = float(around.var(ddof=1)) if len(around) > 2 else mu
            if mu <= 0:
                p0 = 1.0
            elif var <= mu * 1.0001:           # no over-dispersion: Poisson
                p0 = float(np.exp(-mu))
            else:                              # negative binomial with mean mu, variance var
                r = mu * mu / (var - mu)
                p0 = float((r / (r + mu)) ** r)
            gap = bool(p0 ** run < alpha)
        if gap:
            missing[i:j + 1] = True
        i = j + 1
    return missing
