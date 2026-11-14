# PyroScope: MODIS–VIIRS global experiment

PyroScope harmonizes NASA FIRMS MODIS and VIIRS 375 m active-fire detections **worldwide** (NASA Space Apps challenge "Harmonization of MODIS and VIIRS hot spots"). Every step runs globally by default; `PYRO_BBOX` / `PYRO_AOI_FILE` only restrict the region. Egypt was the author's regional development test case, not a limit of the method.

## Scientific scope

**Phase A (ingestion and exploration, implemented):** MODIS Collection 6.1 active-fire CSVs and VIIRS 375 m active-fire CSVs (S-NPP, NOAA-20 and/or NOAA-21). This phase validates ingestion, cleaning and exploratory analysis (`inspect_data`, `preprocess`, `plot_overview`); the harmonization itself is Phase B.

**Phase B (implemented, `src/harmonize.py`):** footprint-aware matching of MODIS and VIIRS 375 m detections, an aggregate FRP scale `k` **per MODIS satellite** (bootstrap CI, between-year heterogeneity analysis, prediction interval, trust grade) and a unified burning-activity calendar on the VIIRS reference scale with explicit observation status. The time-gap window used to fit `k` is 30 minutes and is widened automatically only when no MODIS satellite has enough matches inside it (e.g. Aqua against NOAA-20 / J1), which caps the trust grade. It needs years of data to be trusted (see `k_trust` in `frp_mapping.json`). Optionally also obtain the VIIRS 750 m product (`VNP14`, `VJ114`), which NASA describes as designed for continuity with MODIS. Do not label the 375 m CSV as 750 m.

**Phase C (implemented, `src/anomaly.py`, `src/anomaly_validation.py`):** anomaly detection against a leave-one-year-out seasonal baseline, anomalous episodes with a statistical validation of each one, and critical burning windows, plus a second HotSync-style active-cell activity score (`src/hotsync_anomaly.py`, section 8). It needs multi-year history (>= 5 valid years recommended).

Methodology, the evidence behind each design decision and the known limits are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Data in this repository

* `sample_data/synthetic_multiyear/` holds **synthetic** 2016-2022 FIRMS-format files (not real fires). They are only for running the pipeline end to end (`sample_data/README.txt`) and give no scientific result.
* **Real FIRMS data is not committed** (`data/raw/` is git-ignored; the files are large). The pipeline itself is global: with no `PYRO_BBOX` / `PYRO_AOI_FILE` every step reads whatever is in `data/raw/` for the whole planet.
* The figures labelled "Egypt 2015-2025" in this README, `CHANGELOG.md` and `docs/METHODOLOGY.md` (k values, the ~27 % per 10x FRP slope, VIIRS-to-Aqua total ratios, anomaly counts) come from the author's **regional development run** (Egypt bounding box `--bbox 24.7,21.7,36.9,31.7`). They show what the outputs look like; they are not global results, the scale `k` and its trust grade must be re-estimated for every region and period, and they cannot be checked from this repository alone. To approximate that run, download MODIS_SP and VIIRS_SNPP_SP for 2015-2025 with that `--bbox` (section 7), optionally set `PYRO_AOI_FILE=data/aoi/egypt.geojson` (the only country mask that ships), and run the pipeline; expect close, not identical, numbers because FIRMS archives are reprocessed over time.
* Quick global check without an account: put the FIRMS global 24h files `MODIS_C6_1_Global_24h.csv` and `J1_VIIRS_C2_Global_24h.csv` (section 1) in `data/raw/` and run `preprocess` and `harmonize`. This only demonstrates the pipeline (see "Single-day demo").

## 1. Download data

NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/download/ (archive) or https://firms.modaps.eosdis.nasa.gov/active_fire/ (global 24h/48h/7d NRT files, handy for development).

Put every file in `data/raw/`. Filenames only need to contain `modis` or `viirs` (case-insensitive); multiple files per sensor are read and combined automatically, so yearly or per-satellite files work as-is.

Suggested for the real experiment: MODIS C6.1 (Terra + Aqua) and VIIRS 375 m (S-NPP, plus NOAA-20 if wanted), global, standard-quality (SP) archive. S-NPP and Aqua share a ~13:30 overpass until Aqua's orbit drifted (early 2020s), so that pair is the best match candidate for 2012-2022; for later years expect few close-in-time matches and a lower `k_trust` (in a 2026-11 global 24h test the median Aqua-NOAA-20 gap was about 2 hours).

For a quick global demo, download the FIRMS global 24h files (PowerShell):

```powershell
Invoke-WebRequest https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv -OutFile data\raw\MODIS_C6_1_Global_24h.csv
Invoke-WebRequest https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_24h.csv -OutFile data\raw\J1_VIIRS_C2_Global_24h.csv
```

**Size warning:** this is a global product. VIIRS 375 m produces tens of millions of rows per year, so the code reads files in chunks (`PYRO_CHUNK`, default 500,000 rows) and never loads a whole file at once. Start with a month or a single year.

## 2. Install

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Inspect raw files

```bash
python -m core.src.inspect_data
```

## 4. Standardize

```bash
python -m core.src.preprocess
```

Default is global with no date filter and no region mask (leave `PYRO_BBOX` and `PYRO_AOI_FILE` unset for a world run; the Egypt values below are only examples). Optional environment variables:

| Variable | Meaning | Example |
|---|---|---|
| `PYRO_START`, `PYRO_END` | Inclusive UTC date range | `2024-01-01`, `2024-12-31` |
| `PYRO_BBOX` | `west,south,east,north`; west > east crosses the antimeridian | `24.7,21.7,36.9,31.7` (Egypt, example) |
| `PYRO_CHUNK` | Raw FIRMS rows per ingestion chunk | `200000` |
| `PYRO_OUT_OF_CORE` | `auto` (default), `1` force disk-backed harmonization, `0` force legacy in-memory mode | `1` |
| `PYRO_OUT_OF_CORE_ROWS` | Estimated combined MODIS + VIIRS rows at which `auto` switches to disk-backed harmonization | `3000000` |
| `PYRO_STREAM_CHUNK_ROWS` | Standardized CSV rows staged per chunk (defaults to `PYRO_CHUNK`) | `200000` |
| `PYRO_STREAM_SAMPLE_ROWS` | Maximum matched-pixel sample for conditional-k and ancillary diagnostics in streaming mode | `100000` |
| `PYRO_STREAM_DIAGNOSTIC_ROWS` | Maximum raw-pixel sample per sensor for grid diagnostics | `150000` |
| `PYRO_AOI_FILE` | GeoJSON polygon used as a country mask after the bbox | `data/aoi/egypt.geojson` (the only mask that ships; supply your own polygon for other countries) |
| `PYRO_EXCLUDE_STATIC` | `1` keeps only FIRMS `type` 0 (drops flares, industry, offshore, volcanoes) | `1` |
| `PYRO_PROCESSED_SUBDIR` | Output folder under `data/` (for side-by-side sensitivity runs) | `processed_nonstatic` |

macOS/Linux: `PYRO_BBOX="24.7,21.7,36.9,31.7" python -m core.src.preprocess`
Windows (PowerShell): `$env:PYRO_BBOX="24.7,21.7,36.9,31.7"; python -m core.src.preprocess`

Outputs in `data/processed/`:
- `modis_standardized.csv`, `viirs_standardized.csv` (sorted by time within each chunk, not globally)
- `daily_detection_counts.csv` (per sensor, satellite, day)

## 5. Plot a first comparison

```bash
python -m core.src.plot_overview
```

Outputs: `detection_counts.png` (daily, or monthly if the data span > 90 days), `frp_distributions.png`, `detection_density_map.png` (1° global grid).

## 6. Harmonize and build the calendar

```bash
python -m core.src.harmonize
```

Clamps both sensors to their common time window, assigns each VIIRS 375 m pixel to one MODIS footprint, estimates an aggregate FRP scale `k` (VIIRS-equivalent FRP = `k` x MODIS FRP) **separately for each MODIS satellite**, and writes a calendar on the VIIRS reference scale. Optional variables: `PYRO_K_MAX_DT_MIN` (default `auto`, see "Time-gap window of k"), `PYRO_K_WIDEN_STEPS` (default `45,60,75,90`), `PYRO_SINGLE_DAY_DEMO` (default `0`, see "Single-day demo"), `PYRO_CELL_DEG` (default 0.5), `PYRO_MATCH_MIN` (default 180 minutes), `PYRO_MIN_PAIRS` (default 50), `PYRO_DAY_TOL_MIN` (default 30), `PYRO_N_BOOT` (default 10000 bootstrap replicates), `PYRO_GRID_SENS` (grid sizes of the sensitivity sweep, default `0.05,0.1,0.25,0.5,1,2`), `PYRO_K_MODEL` (`0` always applies the constant k), `PYRO_K_MODEL_BOOT` (default 2000), `PYRO_K_MODEL_FOLDS` (default 10), `PYRO_K_MODEL_CLIP_Q` (default 0.001).

Outputs in `data/processed/`: `calendar_daily.csv`, `calendar_cells.csv` (cell-day table with `cell_area_km2`; filter it to any area of interest), `matched_pairs.csv`, `frp_mapping.json`, `frp_matchup.png`, `k_by_year.png`, `grid_sensitivity.csv`, `grid_sensitivity.png`.

**The scale k, per satellite.** `frp_mapping.json` holds one entry per MODIS satellite under `modis_satellites`, with `k`, its CI, `k_prediction_interval95_new_year`, `k_by_year`, `k_year_heterogeneity` (Cochran Q, I2, between-year SD tau), `k_sensitivity` (how k moves with time gap, day/night, source type, fire size, trimming), `k_ci_by_cluster_unit`, `k_trust` (HIGH / MEDIUM / LOW / NOT_VALIDATED) and `k_trust_problems` (exactly what is missing). `k` is fitted only on matched pixels within the k time-gap window: 30 minutes by default (`PYRO_K_MAX_DT_MIN=auto`), widened to the first of 45 / 60 / 75 / 90 minutes that gives enough matches only when 30 does not (see "Time-gap window of k"); a number fixes the window. A satellite with fewer than `PYRO_MIN_PAIRS` such pixels (Terra against S-NPP: ~150 minutes apart) gets **no k and no VIIRS-equivalent FRP** (`NaN`); its k is never borrowed from another satellite unless you set `PYRO_APPLY_UNVALIDATED_K=1`, and then every value is labelled in `scale_basis`. HIGH needs a window that did not have to be widened, >= 500 pixels (`PYRO_K_TRUST_PAIRS`), >= 30 days, >= 3 years with enough matches, a bootstrap CI <= 20 % wide, no significant between-year heterogeneity and a 95 % upper limit of tau <= `PYRO_K_TAU_MAX` (default 0.15), and a leave-one-year-out check within sampling error. The max-min spread of per-year k is reported but is not a stability test (it is dominated by sampling noise).

**k as a function of fire size (`src/kmodel.py`).** One constant k assumes every fire size and overpass gap shares one ratio. `harmonize` therefore also fits `VIIRS FRP = MODIS FRP x k(MODIS FRP, night, time gap)` (quasi-Poisson regression with log link and the MODIS FRP as an offset, so the constant k is its special case and fitted totals equal observed totals). Which covariates stay is decided by cross-validation with whole 0.5-degree cells held out (all 8 subsets, simplest within one standard error of the best); if none beats the constant k, the constant stays. On the author's Egypt 2015-2025 run (data not in this repository, see "Data in this repository") only fire size survives: k falls ~27 % per 10x MODIS FRP (95 % CI -40 % to -15 %), so the Aqua scale is ~1.07 for 5 MW pixels, ~0.89 for 20 MW and ~0.71 for 100 MW pixels. Which terms survive depends on the data: a 2-day global 24h test with a widened 90-minute window selected fire size and time gap and found k falling ~75 % per 10x MODIS FRP (850 matched pixels, trust MEDIUM, much of the ratio is fire evolution between the overpasses), so treat these numbers as properties of a dataset, not constants. The model is applied at `dt = 0` (simultaneous overpass), FRP is clipped to the central 99.8 % of the fit (no extrapolation), and `scale_basis` becomes `own_conditional` (`scale_form` = `conditional`). `k_applied_to_calendar` is the FRP-weighted scale actually applied. Details and the full CV table are in `frp_mapping.json` (`modis_satellites.<sat>.k_model`), `k_model.png` and `docs/METHODOLOGY.md` section 1.5. The headline `k`, its CI and `k_trust` are still the constant ratio of sums; the model changes only `modis_frp_viirs_equiv_*` and `frp_viirs_equiv_mw` (the FRP anomaly series is VIIRS-only and does not move). A conditional model is never transferred to another satellite.

**Observation status in the calendar.** A day with no detections is not a day without observations. Every sensor-level number is `NaN` on a day that sensor did not deliver data. `harmonized_status` is one of `measured`, `measured_zero`, `measured_low_coverage` (a day that lost overpasses: a lower bound), `viirs_not_observed` and `partial_day` (plus `observed_obscured` when a cloud file is supplied, see the next paragraph, and `demo_partial_window` only with `PYRO_SINGLE_DAY_DEMO=1`); `harmonized_frp_mw_viirs_equiv` is the VIIRS reference value on `measured`, `measured_zero` and `measured_low_coverage` and `NaN` otherwise (`demo_partial_window` carries a lower-bound VIIRS total but is never `harmonized_is_measured`), and `harmonized_source` is never MODIS. MODIS is not substituted for a missing VIIRS day (in the author's Egypt run VIIRS totals are ~9x the k-scaled Aqua totals and correlate weakly with them; check `daily_total_relationship` for your own data). Days with no VIIRS data but Aqua detections are marked `modis_only_evidence`. Per-source flags: `viirs_observed`, `aqua_observed`, `terra_observed`, `modis_observed` (`day_observed` = VIIRS and MODIS both present). `*_observed` is inferred from row counts per satellite (a day is missing when an empty day is implausible for that file; one empty day is a gap in a dense file, only long runs in a sparse one). Outage-suspect days are controlled by `PYRO_COV_MIN_FRAC` (0.5), `PYRO_COV_EXTREME_FRAC` (0.2), `PYRO_COV_MIN_EXPECTED` (20). The calendar uses ONE VIIRS reference satellite for the whole period (`PYRO_REF_SAT`, default: the one observed on most days). `modis_frp_viirs_equiv_mw` (+ `_lo_mw` / `_hi_mw` from the prediction interval of k) covers only satellites that have a k.

**Cloud-obscured days (optional, `PYRO_CLOUD_FILE`).** A satellite can pass over and see nothing because cloud hides the ground; FIRMS files carry no cloud information, so an observed day with 0 detections used to be read as "no fire". Give `harmonize` a CSV with one row per UTC day (columns `date` and `clear_fraction`, the share of your area of interest that was clear, 0-1; `cloud_fraction`, `clear_percent` and `cloud_percent` also work; an optional `satellite` column restricts a row to one satellite) and an observed day with **0 detections and a clear share below `PYRO_CLOUD_MIN_CLEAR` (default 0.5)** becomes `harmonized_status = observed_obscured`: harmonized value `NaN`, never scored by `anomaly`. A day that is not in the file has unknown cloud cover and keeps its old status; it is never treated as clear. Days with detections keep their status and are only flagged (`viirs_cloud_limited`); `PYRO_CLOUD_OBSCURE_NONZERO=1` also marks them `observed_obscured`. The calendar gains `viirs_clear_fraction` and `viirs_cloud_limited`, and `frp_mapping.json` gains a `cloud_obscuration` block (days covered, days marked, warnings). Without `PYRO_CLOUD_FILE` nothing changes. Build the file by averaging, per UTC day, the clear share of a cloud-mask product (e.g. MOD35 / MYD35, or a VIIRS cloud mask) clipped to the same area as your fire data (`cloud_mask.clear_share_from_mod35` decodes the MOD35 `Cloud_Mask` byte); check it first with `python -m core.src.cloud_mask your_file.csv`. Details: `docs/METHODOLOGY.md` section 2.5.

**Why the harmonized series is VIIRS-referenced, and where MODIS enters.** The harmonized record is expressed on the VIIRS 375 m scale on purpose. A merged MODIS+VIIRS series needs one sensor's value to stand in for the other on days one is missing, and the data do not support that here: in the author's Egypt run VIIRS totals are ~9x the k-scaled Aqua totals and correlate weakly with them (`daily_total_relationship`), because k is a ratio conditional on a MODIS detection and cannot recover the many small fires only VIIRS sees. MODIS therefore contributes four things, none of them a silent substitution: (1) the footprint-matched pixel table (`matched_pairs.csv`); (2) the per-satellite scale k with its CI, prediction interval and trust grade, i.e. the quantitative bridge between the two records; (3) MODIS expressed on the VIIRS scale with uncertainty (`modis_frp_viirs_equiv_mw`, `_lo_mw`, `_hi_mw`), so MODIS can be read on the same axis (the calendar covers only the window in which both sensors have data; the pipeline does not extend it to MODIS-only years before the VIIRS era, although `k` could be applied there by hand); (4) `modis_only_evidence` on days VIIRS did not observe. Terra has no k against an afternoon VIIRS reference (~150 min apart) and so adds only detection counts; `PYRO_APPLY_UNVALIDATED_K=1` applies an unvalidated k and labels every value.

**Time-gap window of k (J1 / NOAA-20).** `PYRO_K_MAX_DT_MIN` defaults to `auto`: matches within 30 minutes are used, and only if no MODIS satellite has enough of them the window is widened to 45, 60, 75 or 90 minutes (`PYRO_K_WIDEN_STEPS`). Aqua and NOAA-20 were ~50 minutes apart at their nominal times, but Aqua has since drifted: in a 2026-11 global 24h test the median gap was ~2 hours and the window widened to 90 minutes (`k window: ... widened to 90 min`, Aqua k from 850 matched pixels, trust MEDIUM). The widened window applies to every MODIS satellite in that run. A widened k is capped below HIGH trust and recorded in `frp_mapping.json` (`scale_policy.k_window`); part of its ratio is fire evolution between the overpasses. Set a number to fix the window.

**Single-day demo (FIRMS 24h files).** A FIRMS 24h NRT file can start and end part-way through a UTC day; by default such a partial day is `partial_day` with no harmonized value. (The 2026-11 global J1 and MODIS 24h files covered two UTC days almost completely, came out `measured`, and the demo flag changed nothing.) To label partial days for a demo, run

```bash
PYRO_SINGLE_DAY_DEMO=1 python -m core.src.harmonize
```

The partial day becomes `demo_partial_window` with the VIIRS total (a lower bound) and a printed summary of VIIRS vs MODIS-on-the-VIIRS-scale totals and k. It is a pipeline demonstration, never a validated record, and `anomaly` ignores it.

**Spatial grid.** Matching, k and every daily FRP total use native pixels and do not depend on the 0.5 degree grid; the grid is used for the bootstrap clusters, the active-cell counts, the cell-level correlations and `calendar_cells.csv`. `grid_sensitivity.csv` re-computes those quantities for several cell sizes; see `docs/METHODOLOGY.md`, section 4, for why 0.5 degrees, what it loses, and how to read the cell metrics (they are properties of the grid; quote them with `cell_deg`).

Limits: 24h NRT files only demonstrate the pipeline. A usable calendar needs years of archive data.

## 7. Multi-year archive

1. Get a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Download standard-processing (`_SP`) archive files into `data/raw/`:

```powershell
$env:FIRMS_MAP_KEY="your_key"
python -m core.src.download_firms --start 2015-01-01 --end 2025-12-31 --bbox 24.7,21.7,36.9,31.7
```

   The `--bbox` above is only the Egypt example: use your own region, or drop it for the whole globe (see the size warning). Default sources are `MODIS_SP` (Terra + Aqua) and `VIIRS_SNPP_SP`. VIIRS S-NPP 375 m starts 2012-01-20, NOAA-20 2018-04-01, NOAA-21 2023-03-01. `_SP` lags real time by about 5 months (use `_NRT` sources for the latest months). Without `--bbox` the request is global: tens of millions of VIIRS rows per year, so start with a region. Alternatively request files from the FIRMS archive page (https://firms.modaps.eosdis.nasa.gov/download/; delivered by e-mail) and save them in `data/raw/` with `modis` or `viirs` in the name.
3. Run the pipeline with the same region settings (`PYRO_BBOX`; none for a global run) for every step: `preprocess`, `plot_overview`, `harmonize`, `anomaly`.
4. For a trustworthy `k`, include years when Aqua and S-NPP crossed the same place within minutes (S-NPP and Aqua both ~13:30 until Aqua's orbit was allowed to drift in the early 2020s; check the printed local solar times). Check `k_trust` of each satellite under `modis_satellites` in `frp_mapping.json`.
5. Large-run processing: `preprocess` reads raw files in chunks and keeps cross-file deduplication keys in a temporary SQLite database. `harmonize` automatically switches to disk-backed processing when the estimated combined standardized row count reaches `PYRO_OUT_OF_CORE_ROWS` (default 3 million); use `PYRO_OUT_OF_CORE=1` to force it. See section 10 for memory and disk requirements.

## 8. Anomalies and critical burning windows

```bash
python -m core.src.selftest     # must print SELFTEST PASSED
python -m core.src.anomaly
```

Needs `calendar_daily.csv` from `harmonize` covering several years (>= 2 to run, `PYRO_ANOM_MIN_YEARS`, default 5, to be flagged reliable). The analysed series is the VIIRS reference value on days with `harmonized_status` `measured` or `measured_zero`; absent, partial (including `demo_partial_window`), outage-suspect and cloud-obscured days are missing, never zero (`PYRO_ANOM_INCLUDE_LOW_COVERAGE=1` keeps the outage-suspect days; cloud-obscured days are always excluded). Each day is compared with the same season (+- `PYRO_ANOM_WINDOW` days, default 7) in all OTHER years (median / MAD of log1p FRP, robust z). `PYRO_ANOM_DETREND=1` removes each year's median level first (the series has a significant trend in the author's Egypt run; see `anomaly_validation.json`).

Outputs in `data/processed/`: `anomalies_daily.csv` (with `status`), `anomaly_events.csv` (episodes: >= 3 elevated days, peak z >= `PYRO_ANOM_Z`, default 3, plus the validation columns below), `critical_windows.csv` (days at/above the `PYRO_CRIT_PCT` percentile, default 75, of the multi-year median, with their share of annual FRP), `season_by_year.csv` (10 % / peak / 90 % of FRP per season), `anomaly_report.json`, `anomaly_overview.png`.

**HotSync-style activity score (`src/hotsync_anomaly.py`, runs inside `python -m core.src.anomaly`).** A companion to the FRP detector. It scores the daily number of active 0.5-degree cells (the `calendar_cells` grid, not HotSync's ~2 km grid) with a mean/SD z-score against the same season (+- `PYRO_HOTSYNC_ANOM_WINDOW` days, default 7) in all other years. A day is flagged when z >= `PYRO_HOTSYNC_ANOM_Z` (default 2.5), the baseline has at least `PYRO_HOTSYNC_ANOM_MIN_POOL` values (10) and the activity is at least `PYRO_HOTSYNC_ANOM_MIN_ACTIVITY` (5). The series is anchored to valid VIIRS days: where both sensors were observed it uses the union of MODIS and VIIRS active cells; on VIIRS-observed days without MODIS coverage it uses the VIIRS cells times a VIIRS-to-union scale learned on overlap days (needs `PYRO_HOTSYNC_ANOM_MIN_OVERLAP_DAYS`, default 30). A MODIS-only day never fills a missing VIIRS day. Outputs: `hotsync_activity_*` columns in `anomalies_daily.csv`, `hotsync_activity_anomalies.csv`, `hotsync_activity_overview.png` and a `hotsync_activity` block in `anomaly_report.json`. It does not replace or change the FRP detector, its validation or its episodes.

**Validation (on by default, ~25 s; `PYRO_ANOM_VALIDATE=0` skips it).** A z-score above 3 does not prove that an event is exceptional, so every episode is re-tested: other baseline windows, an IQR instead of the MAD, earlier / later halves of the years, neighbouring years skipped, a causal past-only baseline, the annual level removed, random data loss (`PYRO_ANOM_REPLICATES`, default 30), the classical mean/SD z-score, and a Monte-Carlo null (`PYRO_ANOM_NULL_SIMS`, default 2000) estimated from the non-event days. It also checks residual seasonality, extreme-value distortion of the baseline, low flags next to gaps and on outage-suspect days, and the annual-level trend. `anomaly_events.csv` gains `robustness` (ROBUST / MODERATE / FRAGILE), `variants_detected` / `variants_testable`, `thinning_detection_rate`, `detected_by_classical_z`, `peak_pool_percentile`, `null_p_family_t` / `null_p_family_gaussian` and a text `assessment`. Full results: `anomaly_validation.json`, `anomaly_variants.csv`, `anomaly_event_variants.csv`, `anomaly_validation.png`. `robustness` is a statement about the statistics, not proof that fire activity was exceptional. 'low' flags can still be outages rather than less fire.

## 9. Optional external validation against burned area (Edit 7)

After `harmonize` and `anomaly`, you can compare each anomaly episode with NASA/LP DAAC MODIS MCD64A1 Collection 6.1 monthly burned-area granules. Download the HDF granules for the region and years you want to validate (`MCD64A1/download_mcd64a1.py` is a helper that downloads from NASA LAADS, which requires an Earthdata login; it reads `MCD64A1/required_tiles.json`, which in this repository lists the Egypt tiles h20-h21 / v05-v06 and the Egypt events, and nothing in the repository generates it, so for another region write your own file with the same keys: `tiles`, `events`, `collection`, `cell_deg`), then run:

```bash
python -m core.src.external_validation --mcd64-dir /path/to/MCD64A1
```

The defaults use `data/processed/anomaly_events.csv` and `data/processed/calendar_cells.csv`; outputs are `external_validation.csv` and `external_validation.json` in the configured processed directory. You can override inputs/outputs with `--events-csv`, `--cells-csv`, `--out-csv` and `--out-json`. Use `--collection 061` for Collection 6.1 (the default), `--cell-deg` to match `PYRO_CELL_DEG`, and `--min-coverage` to change the minimum valid land-pixel coverage required before a zero-burn result can be called confirmed.

The event footprint is the set of VIIRS cells in `calendar_cells.csv` observed during that event. The tool reads the MCD64A1 `Burn Date`, `QA`, `First Day` and `Last Day` subdatasets by name; checks the exact event day-of-year interval across every month touched by the event; clips analysis to the event cells; calculates area from the source raster pixel transform; and compares the same calendar window in other archive years (leave-one-year-out seasonal reference). Spatial agreement is the share of event VIIRS cells containing at least one MCD64A1 burned pixel. Reports include a complete `burned_area_km2` only when coverage is adequate, a separate `burned_area_observed_km2` subtotal when coverage is partial, seasonal median / 90th percentile, event-to-baseline ratio, empirical baseline percentile, QA coverage, missing tile counts and spatial overlap.

**Optional raster dependencies:** install `rasterio` and `pyproj` with `pip install rasterio pyproj`; the Rasterio/GDAL build must support HDF4. These are not added to `requirements.txt` and are not needed for the core pipeline or `python -m core.src.selftest`. The self-test exercises synthetic QA/date/coverage cases without real satellite files.

Interpretation matters: `BURNED_AREA_DETECTED` is supporting evidence; `NO_BURN_IN_VALID_COVERAGE` means no burn was mapped in adequately covered event cells and dates; `MCD64_FILES_MISSING` and `INCONCLUSIVE_LOW_COVERAGE` are not negative evidence. A shortened mapping period, unmapped pixels and unburned special-condition QA codes are not silently counted as observed zero. MCD64A1 is a separate burned-area product, but it uses MODIS observations (including active-fire information) and is not a fully independent sensor family from MODIS inputs. Burned area and active-fire FRP are different quantities; do not expect one-to-one event matches. This optional check never changes anomaly scores or existing outputs.

## Important scientific notes

- FIRMS acquisition times are UTC. The code preserves UTC.
- Satellite codes are mapped to names (T = Terra, A = Aqua, N = S-NPP, N20 = NOAA-20, N21 = NOAA-21). Check unfamiliar codes in your files.
- Do not treat MODIS and VIIRS confidence as interchangeable: MODIS is numeric (0–100), VIIRS is categorical (`low`/`nominal`/`high`, or `l`/`n`/`h`).
- Keep all detections initially. Apply confidence filters only as a documented sensitivity analysis.
- Global data includes non-vegetation heat sources (gas flares, industry, volcanoes). The `type` column is kept when the source file has it (NRT files do not); use it to flag static sources.
- NRT (`*NRT`) and standard-processing versions can differ slightly; the `version` column is kept so you can check for mixing.
- FRP is in MW in the FIRMS tables, but always inspect the product metadata.
- A nearby MODIS and VIIRS detection is only a *candidate match*, not proof that both sensors saw the same fire. Overpass times differ, as do pixel footprints, view geometry and sensitivity. Do not assume the nominal 10:30 / 13:30 schedules: in the 2015-2025 standard-processing files for Egypt used by the author (not included in this repository) the median local solar time of daytime detections is about 10.7 h (Terra), 13.3 h (Aqua) and 13.2 h (S-NPP). Aqua drifts from about 13.2 h (2015-2022) to about 14.4 h in 2025 and Terra from about 10.9 h to about 9.7 h, while S-NPP stays near 13.1-13.2 h, so close-in-time Aqua-S-NPP matches become rarer after 2022 and Terra has almost no usable matches with S-NPP. In the 2026-11 global 24h files the median local time of Aqua daytime detections was already about 15.7 h and of Terra about 8.7 h. `python -m core.src.harmonize` prints the real time gaps for your files.
- Do not train a correction model by randomly splitting nearby detections. Split by date or fire-event groups to avoid leakage.
- A day with zero detections is not a day without observations: check `harmonized_status` (and the `*_observed` flags) before treating a zero as 'no fire'. Never fill a missing VIIRS day with MODIS.
- A MODIS satellite is only harmonized if it has close-in-time overpasses with the VIIRS reference. In the author's Egypt files Aqua (13:30) does, Terra (10:30) does not; do not describe the Aqua k as a universal MODIS-to-VIIRS mapping.
- The scale k is a ratio of totals conditional on a MODIS detection; it does not recover the many small fires only VIIRS detects (VIIRS sees ~56 times as many detections as Aqua over Egypt; the factor changes with region and year).
- Cell-level metrics (active-cell ratios, cell correlations) depend on the grid size (`grid_sensitivity.csv`); a lat/lon cell is not equal-area (`cell_area_km2`).


## 10. Out-of-core processing for large archives (Edit 8)

The normal commands are unchanged. By default, `python -m core.src.harmonize` estimates the combined row count of the two standardized CSVs from a small byte sample. At or above `PYRO_OUT_OF_CORE_ROWS` (default `3000000` combined rows), it runs the disk-backed implementation in `src/harmonize_streaming.py`; below that threshold it keeps using the established in-memory implementation. You can force either path:

```bash
# Force disk-backed processing
PYRO_OUT_OF_CORE=1 python -m core.src.harmonize

# Force the legacy in-memory implementation for comparison (only when RAM is sufficient)
PYRO_OUT_OF_CORE=0 python -m core.src.harmonize
```

The large-run path requires no Dask or other new mandatory package. It uses Python's built-in SQLite support to stage standardized rows in bounded CSV chunks, match one MODIS/VIIRS satellite pair at a time for each UTC day, spill matched pairs and year-by-cell sufficient statistics to disk, and persist cell/day aggregates in SQLite. The daily calendar is assembled from SQL summaries; `calendar_cells.csv` and `matched_pairs.csv` are exported without loading the complete output files into pandas at once. Preprocessing also moves cross-chunk deduplication state from a growing in-memory hash array into a temporary SQLite table. Edit 7's event-cell lookup reads `calendar_cells.csv` in chunks, too.

The headline constant `k`, its spatial-cell bootstrap interval, and per-year cell summaries use all qualifying matched pixels. The conditional `k` model and pixel-level ancillary diagnostics use a deterministic sample capped by `PYRO_STREAM_SAMPLE_ROWS` (default 100,000); cluster-unit diagnostic intervals use at most 10,000 sampled rows and 200 bootstrap replicates. Grid diagnostics use bounded samples (`PYRO_STREAM_DIAGNOSTIC_ROWS`, default 150,000 per sensor). These sample-based diagnostics are labelled in `frp_mapping.json`; they must not be described as exact full-archive estimates. `calendar_daily.csv` is built from full SQL aggregates, not from those samples.

**Resource limits still apply.** Disk-backed processing trades RAM for temporary disk: allow substantial free space for SQLite indexes, matched-pair spill files and outputs (the exact amount depends on input size and overlap). Temporary staging files are removed after a successful run. A run killed midway can leave `pyroscope_dedupe_*` / `pyroscope_stream_*` folders in the output directory; the next `preprocess` or `harmonize` run deletes such folders when they are more than an hour old. Matching still holds one UTC day's rows for one MODIS satellite and the corresponding VIIRS time window in memory; an unusually dense single day can therefore require more RAM than an ordinary day. The default 0.5-degree analysis grid keeps the exact bootstrap's cell-level sufficient statistics bounded by the active grid cells, but very fine global grids increase that cost. `PYRO_OUT_OF_CORE=0` deliberately disables this protection.

`python -m core.src.plot_overview` already scans standardized CSVs in chunks. `calendar_daily.csv` is a daily time series and remains small compared with the raw detections; downstream anomaly analysis continues to read that daily calendar.
