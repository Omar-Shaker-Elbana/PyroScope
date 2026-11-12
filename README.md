# PyroScope: MODIS–VIIRS global experiment

Starter project for preparing NASA FIRMS active-fire observations (global scope) for MODIS–VIIRS harmonization.

## Scientific scope

**Phase A (this starter):** MODIS Collection 6.1 active-fire CSVs and VIIRS 375 m active-fire CSVs (S-NPP, NOAA-20 and/or NOAA-21). This phase validates data ingestion, cleaning and exploratory analysis. It is *not* the final MODIS-continuity harmonization experiment.

**Phase B (implemented, `src/harmonize.py`):** footprint-aware matching of MODIS and VIIRS 375 m detections, an aggregate FRP scale `k` **per MODIS satellite** (bootstrap CI, between-year heterogeneity analysis, prediction interval, trust grade) and a unified burning-activity calendar on the VIIRS reference scale with explicit observation status. It needs years of data to be trusted (see `k_trust` in `frp_mapping.json`). Optionally also obtain the VIIRS 750 m product (`VNP14`, `VJ114`), which NASA describes as designed for continuity with MODIS. Do not label the 375 m CSV as 750 m.

**Phase C (implemented, `src/anomaly.py`, `src/anomaly_validation.py`):** anomaly detection against a leave-one-year-out seasonal baseline, anomalous episodes with a statistical validation of each one, and critical burning windows. It needs multi-year history (>= 5 valid years recommended).

Methodology, the evidence behind each design decision and the known limits are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## 1. Download data

NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/download/ (archive) or https://firms.modaps.eosdis.nasa.gov/active_fire/ (global 24h/48h/7d NRT files, handy for development).

Put every file in `data/raw/`. Filenames only need to contain `modis` or `viirs` (case-insensitive); multiple files per sensor are read and combined automatically, so yearly or per-satellite files work as-is.

Suggested for the real experiment: MODIS C6.1 (Terra + Aqua) and VIIRS 375 m (S-NPP, plus NOAA-20 if wanted), global, standard-quality (SP) archive. S-NPP and Aqua have similar afternoon overpasses, so that pair is the best match candidate.

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
python -m src.inspect_data
```

## 4. Standardize

```bash
python -m src.preprocess
```

Default is global with no date filter. Optional environment variables:

| Variable | Meaning | Example |
|---|---|---|
| `PYRO_START`, `PYRO_END` | Inclusive UTC date range | `2024-01-01`, `2024-12-31` |
| `PYRO_BBOX` | `west,south,east,north`; west > east crosses the antimeridian | `24.7,21.7,36.9,31.7` |
| `PYRO_CHUNK` | Rows per chunk | `200000` |
| `PYRO_AOI_FILE` | GeoJSON polygon used as a country mask after the bbox | `data/aoi/egypt.geojson` |
| `PYRO_EXCLUDE_STATIC` | `1` keeps only FIRMS `type` 0 (drops flares, industry, offshore, volcanoes) | `1` |
| `PYRO_PROCESSED_SUBDIR` | Output folder under `data/` (for side-by-side sensitivity runs) | `processed_nonstatic` |

macOS/Linux: `PYRO_BBOX="24.7,21.7,36.9,31.7" python -m src.preprocess`
Windows (PowerShell): `$env:PYRO_BBOX="24.7,21.7,36.9,31.7"; python -m src.preprocess`

Outputs in `data/processed/`:
- `modis_standardized.csv`, `viirs_standardized.csv` (sorted by time within each chunk, not globally)
- `daily_detection_counts.csv` (per sensor, satellite, day)

## 5. Plot a first comparison

```bash
python -m src.plot_overview
```

Outputs: `detection_counts.png` (daily, or monthly if the data span > 90 days), `frp_distributions.png`, `detection_density_map.png` (1° global grid).

## 6. Harmonize and build the calendar

```bash
python -m src.harmonize
```

Clamps both sensors to their common time window, assigns each VIIRS 375 m pixel to one MODIS footprint, estimates an aggregate FRP scale `k` (VIIRS-equivalent FRP = `k` x MODIS FRP) **separately for each MODIS satellite**, and writes a calendar on the VIIRS reference scale. Optional variables: `PYRO_CELL_DEG` (default 0.5), `PYRO_MATCH_MIN` (default 180 minutes), `PYRO_MIN_PAIRS` (default 50), `PYRO_DAY_TOL_MIN` (default 30), `PYRO_N_BOOT` (default 10000 bootstrap replicates), `PYRO_GRID_SENS` (grid sizes of the sensitivity sweep, default `0.05,0.1,0.25,0.5,1,2`).

Outputs in `data/processed/`: `calendar_daily.csv`, `calendar_cells.csv` (cell-day table with `cell_area_km2`; filter it to any area of interest), `matched_pairs.csv`, `frp_mapping.json`, `frp_matchup.png`, `k_by_year.png`, `grid_sensitivity.csv`, `grid_sensitivity.png`.

**The scale k, per satellite.** `frp_mapping.json` holds one entry per MODIS satellite under `modis_satellites`, with `k`, its CI, `k_prediction_interval95_new_year`, `k_by_year`, `k_year_heterogeneity` (Cochran Q, I2, between-year SD tau), `k_sensitivity` (how k moves with time gap, day/night, source type, fire size, trimming), `k_ci_by_cluster_unit`, `k_trust` (HIGH / MEDIUM / LOW / NOT_VALIDATED) and `k_trust_problems` (exactly what is missing). `k` is fitted only on matched pixels within `PYRO_K_MAX_DT_MIN` minutes (default 30). A satellite with fewer than `PYRO_MIN_PAIRS` such pixels (Terra against S-NPP: ~150 minutes apart) gets **no k and no VIIRS-equivalent FRP** (`NaN`); its k is never borrowed from another satellite unless you set `PYRO_APPLY_UNVALIDATED_K=1`, and then every value is labelled in `scale_basis`. HIGH needs >= 500 pixels (`PYRO_K_TRUST_PAIRS`), >= 30 days, >= 3 years with enough matches, a bootstrap CI <= 20 % wide, no significant between-year heterogeneity and a 95 % upper limit of tau <= `PYRO_K_TAU_MAX` (default 0.15), and a leave-one-year-out check within sampling error. The max-min spread of per-year k is reported but is not a stability test (it is dominated by sampling noise).

**Observation status in the calendar.** A day with no detections is not a day without observations. Every sensor-level number is `NaN` on a day that sensor did not deliver data. `harmonized_status` is one of `measured`, `measured_zero`, `measured_low_coverage` (a day that lost overpasses: a lower bound), `viirs_not_observed` and `partial_day`; `harmonized_frp_mw_viirs_equiv` is the VIIRS reference value on the first three and `NaN` otherwise, and `harmonized_source` is never MODIS. MODIS is not substituted for a missing VIIRS day (on this data VIIRS totals are ~9x the k-scaled Aqua totals and correlate weakly with them; `daily_total_relationship`). Days with no VIIRS data but Aqua detections are marked `modis_only_evidence`. Per-source flags: `viirs_observed`, `aqua_observed`, `terra_observed`, `modis_observed` (`day_observed` = VIIRS and MODIS both present). `*_observed` is inferred from row counts per satellite (a day is missing when an empty day is implausible for that file; one empty day is a gap in a dense file, only long runs in a sparse one). Outage-suspect days are controlled by `PYRO_COV_MIN_FRAC` (0.5), `PYRO_COV_EXTREME_FRAC` (0.2), `PYRO_COV_MIN_EXPECTED` (20). The calendar uses ONE VIIRS reference satellite for the whole period (`PYRO_REF_SAT`, default: the one observed on most days). `modis_frp_viirs_equiv_mw` (+ `_lo_mw` / `_hi_mw` from the prediction interval of k) covers only satellites that have a k.

**Spatial grid.** Matching, k and every daily FRP total use native pixels and do not depend on the 0.5 degree grid; the grid is used for the bootstrap clusters, the active-cell counts, the cell-level correlations and `calendar_cells.csv`. `grid_sensitivity.csv` re-computes those quantities for several cell sizes; see `docs/METHODOLOGY.md`, section 4, for why 0.5 degrees, what it loses, and how to read the cell metrics (they are properties of the grid; quote them with `cell_deg`).

Limits: 24h NRT files only demonstrate the pipeline. A usable calendar needs years of archive data.

## 7. Multi-year archive

1. Get a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
2. Download standard-processing (`_SP`) archive files into `data/raw/`:

```powershell
$env:FIRMS_MAP_KEY="your_key"
python -m src.download_firms --start 2015-01-01 --end 2025-12-31 --bbox 24.7,21.7,36.9,31.7
```

   Default sources are `MODIS_SP` (Terra + Aqua) and `VIIRS_SNPP_SP`. VIIRS S-NPP 375 m starts 2012-01-20, NOAA-20 2018-04-01, NOAA-21 2023-03-01. `_SP` lags real time by about 5 months (use `_NRT` sources for the latest months). Without `--bbox` the request is global: tens of millions of VIIRS rows per year, so start with a region. Alternatively request files from the FIRMS archive page (https://firms.modaps.eosdis.nasa.gov/download/; delivered by e-mail) and save them in `data/raw/` with `modis` or `viirs` in the name.
3. Run the pipeline with the same `PYRO_BBOX` for every step: `preprocess`, `plot_overview`, `harmonize`, `anomaly`.
4. For a trustworthy `k`, include years when Aqua and S-NPP crossed the same place within minutes (S-NPP and Aqua both ~13:30 until Aqua's orbit was allowed to drift in the early 2020s; check the printed local solar times). Check `k_trust` of each satellite under `modis_satellites` in `frp_mapping.json`.
5. Memory: `harmonize` keeps both standardized files in memory and stops with a message above `PYRO_MAX_ROWS` (15 million rows per file). For the whole planet, run region by region or year by year.

## 8. Anomalies and critical burning windows

```bash
python -m src.selftest     # must print SELFTEST PASSED
python -m src.anomaly
```

Needs `calendar_daily.csv` from `harmonize` covering several years (>= 2 to run, `PYRO_ANOM_MIN_YEARS`, default 5, to be flagged reliable). The analysed series is the VIIRS reference value on days with `harmonized_status` `measured` or `measured_zero`; absent, partial and outage-suspect days are missing, never zero (`PYRO_ANOM_INCLUDE_LOW_COVERAGE=1` keeps the outage-suspect days). Each day is compared with the same season (+- `PYRO_ANOM_WINDOW` days, default 7) in all OTHER years (median / MAD of log1p FRP, robust z). `PYRO_ANOM_DETREND=1` removes each year's median level first (the series has a significant trend on the shipped data; see `anomaly_validation.json`).

Outputs in `data/processed/`: `anomalies_daily.csv` (with `status`), `anomaly_events.csv` (episodes: >= 3 elevated days, peak z >= `PYRO_ANOM_Z`, default 3, plus the validation columns below), `critical_windows.csv` (days at/above the `PYRO_CRIT_PCT` percentile, default 75, of the multi-year median, with their share of annual FRP), `season_by_year.csv` (10 % / peak / 90 % of FRP per season), `anomaly_report.json`, `anomaly_overview.png`.

**Validation (on by default, ~25 s; `PYRO_ANOM_VALIDATE=0` skips it).** A z-score above 3 does not prove that an event is exceptional, so every episode is re-tested: other baseline windows, an IQR instead of the MAD, earlier / later halves of the years, neighbouring years skipped, a causal past-only baseline, the annual level removed, random data loss (`PYRO_ANOM_REPLICATES`, default 30), the classical mean/SD z-score, and a Monte-Carlo null (`PYRO_ANOM_NULL_SIMS`, default 2000) estimated from the non-event days. It also checks residual seasonality, extreme-value distortion of the baseline, low flags next to gaps and on outage-suspect days, and the annual-level trend. `anomaly_events.csv` gains `robustness` (ROBUST / MODERATE / FRAGILE), `variants_detected` / `variants_testable`, `thinning_detection_rate`, `detected_by_classical_z`, `peak_pool_percentile`, `null_p_family_t` / `null_p_family_gaussian` and a text `assessment`. Full results: `anomaly_validation.json`, `anomaly_variants.csv`, `anomaly_event_variants.csv`, `anomaly_validation.png`. `robustness` is a statement about the statistics, not proof that fire activity was exceptional. 'low' flags can still be outages rather than less fire.

## Important scientific notes

- FIRMS acquisition times are UTC. The code preserves UTC.
- Satellite codes are mapped to names (T = Terra, A = Aqua, N = S-NPP, N20 = NOAA-20, N21 = NOAA-21). Check unfamiliar codes in your files.
- Do not treat MODIS and VIIRS confidence as interchangeable: MODIS is numeric (0–100), VIIRS is categorical (`low`/`nominal`/`high`, or `l`/`n`/`h`).
- Keep all detections initially. Apply confidence filters only as a documented sensitivity analysis.
- Global data includes non-vegetation heat sources (gas flares, industry, volcanoes). The `type` column is kept when the source file has it (NRT files do not); use it to flag static sources.
- NRT (`*NRT`) and standard-processing versions can differ slightly; the `version` column is kept so you can check for mixing.
- FRP is in MW in the FIRMS tables, but always inspect the product metadata.
- A nearby MODIS and VIIRS detection is only a *candidate match*, not proof that both sensors saw the same fire. Overpass times differ, as do pixel footprints, view geometry and sensitivity. Do not assume the nominal 10:30 / 13:30 schedules: in the 2015-2025 standard-processing files for Egypt supplied with this project the median local solar time of daytime detections is about 10.7 h (Terra), 13.3 h (Aqua) and 13.2 h (S-NPP). Aqua drifts from about 13.2 h (2015-2022) to about 14.4 h in 2025 and Terra from about 10.9 h to about 9.7 h, while S-NPP stays near 13.1-13.2 h, so close-in-time Aqua-S-NPP matches become rarer after 2022 and Terra has almost no usable matches with S-NPP. `python -m src.harmonize` prints the real time gaps for your files.
- Do not train a correction model by randomly splitting nearby detections. Split by date or fire-event groups to avoid leakage.
- A day with zero detections is not a day without observations: check `harmonized_status` (and the `*_observed` flags) before treating a zero as 'no fire'. Never fill a missing VIIRS day with MODIS.
- A MODIS satellite is only harmonized if it has close-in-time overpasses with the VIIRS reference. In the Egypt files Aqua (13:30) does, Terra (10:30) does not; do not describe the Aqua k as a universal MODIS-to-VIIRS mapping.
- The scale k is a ratio of totals conditional on a MODIS detection; it does not recover the many small fires only VIIRS detects (VIIRS sees ~56 times as many detections as Aqua over Egypt).
- Cell-level metrics (active-cell ratios, cell correlations) depend on the grid size (`grid_sensitivity.csv`); a lat/lon cell is not equal-area (`cell_area_km2`).
