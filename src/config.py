import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
# Sub-folder of data/ for every output. Change it to run a sensitivity analysis next to the main run
# (e.g. PYRO_PROCESSED_SUBDIR=processed_nonstatic) without overwriting data/processed/.
PROCESSED_DIR = ROOT / "data" / (os.getenv("PYRO_PROCESSED_SUBDIR") or "processed")

# Optional time filter (inclusive, YYYY-MM-DD). None keeps everything in the files.
START_DATE = os.getenv("PYRO_START") or None
END_DATE = os.getenv("PYRO_END") or None

# Optional region filter as "west,south,east,north". None = global (default).
# Example: PYRO_BBOX="24.7,21.7,36.9,31.7" restricts to an Egypt-like box.
# If west > east the box is treated as crossing the antimeridian.
_bbox = os.getenv("PYRO_BBOX")
BBOX = tuple(float(v) for v in _bbox.split(",")) if _bbox else None

# Optional country mask: a GeoJSON Polygon/MultiPolygon in lon/lat. Applied AFTER PYRO_BBOX, so the
# rectangle is only a coarse pre-filter. Example: PYRO_AOI_FILE=data/aoi/egypt.geojson
_aoi = os.getenv("PYRO_AOI_FILE")
AOI_FILE = (Path(_aoi) if Path(_aoi).is_absolute() else ROOT / _aoi) if _aoi else None

# PYRO_EXCLUDE_STATIC=1 keeps only FIRMS type 0 (presumed vegetation fire) and drops static land sources
# (type 2: flares, industry), offshore (type 3) and volcanoes (type 1). Rows without a type (NRT files) are kept.
EXCLUDE_STATIC = os.getenv("PYRO_EXCLUDE_STATIC", "0") == "1"

# Global VIIRS 375 m data is large: files are processed in chunks of this many rows.
CHUNK_ROWS = int(os.getenv("PYRO_CHUNK", 500_000))

# Phase B (harmonize.py)
CELL_DEG = float(os.getenv("PYRO_CELL_DEG", 0.5))        # calendar grid size (degrees)
MATCH_MAX_MIN = float(os.getenv("PYRO_MATCH_MIN", 180))  # max |time gap| MODIS<->VIIRS (minutes)
DAY_TOL_MIN = float(os.getenv("PYRO_DAY_TOL_MIN", 30))   # slack when testing a UTC day is fully covered
MIN_PAIRS_FIT = int(os.getenv("PYRO_MIN_PAIRS", 50))     # fewer matched pixels than this -> no FRP fit

 

# --- Multi-year runs / trustworthy k (preprocess.py, harmonize.py) ---
# k is fitted only on matched pixels whose median |time gap| is at most this many minutes.
K_FIT_MAX_DT_MIN = float(os.getenv("PYRO_K_MAX_DT_MIN", 30))
# A k fitted on fewer close-in-time matched pixels than this can never be graded HIGH.
K_TRUST_MIN_PAIRS = int(os.getenv("PYRO_K_TRUST_PAIRS", 500))
# Large archives automatically switch to the disk-backed, day-batched harmonizer.
# PYRO_OUT_OF_CORE=1 forces streaming; PYRO_OUT_OF_CORE=0 selects the legacy in-memory path.
# The row threshold is a dispatch heuristic based on combined standardized input rows, not a hard cap.
OUT_OF_CORE = os.getenv("PYRO_OUT_OF_CORE", "auto").strip().lower()
OUT_OF_CORE_ROWS = int(os.getenv("PYRO_OUT_OF_CORE_ROWS", 3_000_000))
# Set PYRO_DEDUPE=0 to skip the cross-file duplicate check when your files are known not to overlap.
DEDUPE = os.getenv("PYRO_DEDUPE", "1") != "0"
# VIIRS satellite used as the calendar reference. Default: the one observed on the most days.
REF_SAT = os.getenv("PYRO_REF_SAT") or None

# --- Harmonization hardening (harmonize.py, harmonize_validation.py) ---
# A MODIS satellite is only given a scale k when it has its OWN close-in-time matches with VIIRS. By default a
# satellite without them (e.g. Terra vs S-NPP: ~150 min apart) gets NO VIIRS-equivalent FRP (NaN), instead of
# silently inheriting another satellite's k. PYRO_APPLY_UNVALIDATED_K=1 applies an own low-trust fit, or the k of
# the best-validated satellite, and labels every such value (scale_basis) as unvalidated.
APPLY_UNVALIDATED_K = os.getenv("PYRO_APPLY_UNVALIDATED_K", "0") == "1"
# Cluster-bootstrap replicates for every confidence interval on k (10 000 keeps the Monte-Carlo error of a 95 %
# interval width well below 1 percentage point, so a trust grade cannot flip because of the random seed).
N_BOOT = int(os.getenv("PYRO_N_BOOT", 10_000))
# k_trust can only be HIGH if the 95 % upper limit of the between-year SD of log k (tau) is at most this value.
K_TAU_MAX = float(os.getenv("PYRO_K_TAU_MAX", 0.15))
# Conditional scale (kmodel.py): k as a function of fire size / day-night / time gap instead of one constant.
# The covariates are chosen by cross-validation with whole cells held out; if none beats the constant k, the constant
# stays. PYRO_K_MODEL=0 always applies the constant k (the model is then not even fitted).
K_MODEL = os.getenv("PYRO_K_MODEL", "1") != "0"
K_MODEL_BOOT = int(os.getenv("PYRO_K_MODEL_BOOT", 2000))          # cluster-bootstrap replicates for its coefficient CIs
K_MODEL_FOLDS = int(os.getenv("PYRO_K_MODEL_FOLDS", 10))          # spatial cross-validation folds (whole cells)
# MODIS FRP is clipped to the central (1 - 2 q) of the fitted pixels before the model is evaluated: beyond that range the
# ratio is held at its boundary value instead of extrapolating a log-linear trend from a handful of very large fires.
K_MODEL_CLIP_Q = float(os.getenv("PYRO_K_MODEL_CLIP_Q", 0.001))
# Grid sizes (degrees) of the spatial-aggregation sensitivity analysis (harmonize_validation.grid_sensitivity).
GRID_SENS_DEG = tuple(float(x) for x in (os.getenv("PYRO_GRID_SENS") or "0.05,0.1,0.25,0.5,1,2").split(","))
# Partial-outage detector for the reference VIIRS series (see harmonize.flag_low_coverage). A day is flagged when
#   (a) it has <= 1 overpass cluster AND fewer than COV_MIN_FRAC x the local median number of detections, or
#   (b) it has fewer than COV_EXTREME_FRAC x the local median number of detections (a level reached by < 1 % of days),
# but only where at least COV_MIN_EXPECTED detections per day are normal (a sparse region cannot reveal an outage this way).
COV_MIN_FRAC = float(os.getenv("PYRO_COV_MIN_FRAC", 0.5))
COV_EXTREME_FRAC = float(os.getenv("PYRO_COV_EXTREME_FRAC", 0.2))
COV_MIN_EXPECTED = float(os.getenv("PYRO_COV_MIN_EXPECTED", 20))
COV_CLUSTER_GAP_MIN = float(os.getenv("PYRO_COV_GAP_MIN", 25))   # minutes between detections that start a new overpass

# --- Cloud obscuration (cloud_mask.py, harmonize.py): the third observation status "observed, obscured" ---
# PYRO_CLOUD_FILE = CSV with one row per UTC day (and optionally per satellite): date, clear_fraction (0..1) - see
# cloud_mask.py for the accepted columns. Unset (default) = no cloud information: every output is identical to a run
# without this feature. A day that is not in the file has unknown cloud cover and is never treated as clear or cloudy.
_cloud = os.getenv("PYRO_CLOUD_FILE")
CLOUD_FILE = (Path(_cloud) if Path(_cloud).is_absolute() else ROOT / _cloud) if _cloud else None
# An observed day with 0 detections whose clear-sky share is BELOW this value is `observed_obscured` (harmonized value
# NaN), not a measured zero: with that much cloud, "no detection" is not evidence of "no fire". A convention, not an estimate.
CLOUD_MIN_CLEAR = float(os.getenv("PYRO_CLOUD_MIN_CLEAR", 0.5))
if not 0.0 <= CLOUD_MIN_CLEAR <= 1.0:
    raise SystemExit(f"PYRO_CLOUD_MIN_CLEAR must be a share between 0 and 1, got {CLOUD_MIN_CLEAR:g} (0.5 = half of the area clear).")
# PYRO_CLOUD_OBSCURE_NONZERO=1 also marks cloud-limited days that DID have detections as observed_obscured (their FRP is
# then a lower bound and is left out of the anomaly series). Default 0: detections are real, the day keeps its status
# and is only flagged in the calendar column viirs_cloud_limited.
CLOUD_OBSCURE_NONZERO = os.getenv("PYRO_CLOUD_OBSCURE_NONZERO", "0") == "1"

# --- Anomalies and critical burning windows (anomaly.py) ---
ANOM_MIN_YEARS = int(os.getenv("PYRO_ANOM_MIN_YEARS", 5))   # fewer valid years -> flagged UNRELIABLE
ANOM_Z = float(os.getenv("PYRO_ANOM_Z", 3.0))               # robust z for an anomalous day
ANOM_WINDOW_DAYS = int(os.getenv("PYRO_ANOM_WINDOW", 7))    # +- days of season used for the baseline
CRIT_PCT = float(os.getenv("PYRO_CRIT_PCT", 75))            # percentile of the climatology = "critical"
# Statistical validation of the anomaly episodes (anomaly_validation.py). Set PYRO_ANOM_VALIDATE=0 to skip it.
ANOM_VALIDATE = os.getenv("PYRO_ANOM_VALIDATE", "1") != "0"
ANOM_INCLUDE_LOW_COVERAGE = os.getenv("PYRO_ANOM_INCLUDE_LOW_COVERAGE", "0") == "1"  # keep outage-suspect days in the analysis
# PYRO_ANOM_DETREND=1 removes each year's median level before scoring. Off by default (keeps the baseline definition
# of earlier runs), but anomaly_validation.json reports the annual-level trend and the effect of removing it: a
# significant trend makes the leave-one-year-out baseline too low for early years and too high for late years.
ANOM_DETREND = os.getenv("PYRO_ANOM_DETREND", "0") == "1"
ANOM_REPLICATES = int(os.getenv("PYRO_ANOM_REPLICATES", 30))   # random-thinning replicates per scenario
ANOM_NULL_SIMS = int(os.getenv("PYRO_ANOM_NULL_SIMS", 2000))   # Monte-Carlo records for the false-alarm null

# --- HotSync-style active-cell anomaly score (runs beside the robust FRP detector) ---
HOTSYNC_ANOM_Z = float(os.getenv("PYRO_HOTSYNC_ANOM_Z", 2.5))
HOTSYNC_ANOM_WINDOW_DAYS = int(os.getenv("PYRO_HOTSYNC_ANOM_WINDOW", 7))
HOTSYNC_ANOM_MIN_POOL = int(os.getenv("PYRO_HOTSYNC_ANOM_MIN_POOL", 10))
HOTSYNC_ANOM_MIN_ACTIVITY = float(os.getenv("PYRO_HOTSYNC_ANOM_MIN_ACTIVITY", 5))
HOTSYNC_ANOM_MIN_OVERLAP_DAYS = int(os.getenv("PYRO_HOTSYNC_ANOM_MIN_OVERLAP_DAYS", 30))

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
