# Changelog: backend hardening after the methodological review

Backend only (no frontend work). Evidence and reasoning: `docs/METHODOLOGY.md`. A unified diff against
`pyroscope_modis_viirs_not_final.zip` is in `docs/CHANGES_vs_not_final.patch`.

## Follow-up: external burned-area validation (review point 7)

* New `src/external_validation.py`: optional event-level comparison with MODIS MCD64A1 Collection 6.1. It uses the
  actual Phase C event columns (`start`, `end`) and derives each event footprint from VIIRS `cell_id` values in
  `calendar_cells.csv`; it does not require invented event bounding-box columns.
* Reads the named `Burn Date`, `QA`, `First Day` and `Last Day` HDF subdatasets, handles every month crossed by an event,
  matches the Burn Date day-of-year to the event window, clips to event cells, and calculates burned area from the
  source raster transform/CRS units.
* Adds same-season leave-one-year-out burned-area baselines, empirical percentiles, cell-level overlap, QA coverage and
  explicit statuses for partial coverage / missing tiles. A complete zero is emitted only when coverage is adequate;
  partial observed area is kept in a separate field. Missing data is never converted to confirmed zero.
* `README.md` and `docs/METHODOLOGY.md`: usage, optional HDF4-capable Rasterio/PyProj dependencies, QA semantics and
  scientific interpretation. `src/selftest.py`: synthetic QA/date/coverage tests; no satellite files are needed for these.
* No changes to anomaly scoring, harmonization, mandatory dependencies or existing output files. The external check is
  run manually and does not download MCD64A1 granules.

## Follow-up: cloud-obscured days as a third observation status (review point 6)

* New `src/cloud_mask.py`: reads and validates an optional daily clear-sky table (`PYRO_CLOUD_FILE`: `date`,
  `clear_fraction`; `cloud_fraction`, `clear_percent`, `cloud_percent` and a `satellite` column also accepted), selects
  the rows of the reference VIIRS satellite, and decodes a MOD35 `Cloud_Mask` byte (`clear_share_from_mod35`).
  `python -m src.cloud_mask file.csv` checks a file. A bad file stops the run before the slow steps; a day that is
  not in the file has unknown cloud cover, never "clear".
* `harmonize.py`: `build_calendar(..., clear=None)`. An observed, complete day with 0 detections and a clear share below
  `PYRO_CLOUD_MIN_CLEAR` (0.5) is `harmonized_status = observed_obscured` (harmonized value NaN, `harmonized_source`
  none), not a measured zero; the cloud explanation takes precedence over `measured_low_coverage`. Days with detections
  keep their status and get `viirs_cloud_limited` (`PYRO_CLOUD_OBSCURE_NONZERO=1` marks them obscured too). New calendar
  columns `viirs_clear_fraction`, `viirs_cloud_limited` and a `cloud_obscuration` block in `frp_mapping.json`, only when a
  cloud file is given.
* `anomaly.py` / `anomaly_validation.py`: `observed_obscured` days are missing (never zero, never scored, not re-admitted
  by the outage-suspect variant); counted in `anomaly_report.json` (`cloud_obscured_days_excluded`, only when present) and
  in the "not usable" reason; treated as unobserved neighbours in the low-flag check.
* New settings: `PYRO_CLOUD_FILE`, `PYRO_CLOUD_MIN_CLEAR`, `PYRO_CLOUD_OBSCURE_NONZERO`. **Without `PYRO_CLOUD_FILE` every
  output is identical to before.**
* `selftest.py`: `check_cloud` (decoder with planted bytes, file validation, satellite selection, every status case
  including precedence, the threshold edge and unknown days, equivalence with a fully clear sky, and the real
  `anomaly.load_series` path).
* Not included: downloading or clipping MOD35 / VIIRS cloud-mask granules (see `docs/METHODOLOGY.md` section 2.5).

## Follow-up: k as a model of fire size (review point 1, first half)

* New `src/kmodel.py`: quasi-Poisson regression of VIIRS footprint FRP on MODIS FRP, day/night and time gap (offset
  `log(MODIS FRP)`, so it nests the constant k and preserves totals), covariates chosen by 10-fold cross-validation with
  whole cells held out and a one-standard-error rule, coefficient CIs by cell-cluster bootstrap.
* `harmonize.py`: `fit_scale` adds `k_model` to the satellite's info (the constant k, its CI and `k_trust` are
  unchanged); `decide_scales` applies it when adopted (`scale_basis = own_conditional`); `apply_scale` evaluates it at
  `dt = 0` with FRP clipped to the fitted range; new `applied_scale` -> `k_applied_to_calendar`, `scale_form`.
  A conditional model is never transferred to another satellite; Terra is still NaN.
* New outputs: `k_model.png`; `frp_matchup.png` shows the conditional curve; `k_model` block in `frp_mapping.json`.
* New settings: `PYRO_K_MODEL` (`0` = constant k, outputs identical to before), `PYRO_K_MODEL_BOOT`,
  `PYRO_K_MODEL_FOLDS`, `PYRO_K_MODEL_CLIP_Q`.
* Egypt 2015-2025: only fire size is selected; k falls 27 % per 10x MODIS FRP (CI -40 % to -15 %); held-out deviance
  3.1 % lower; 11-year `modis_frp_viirs_equiv_mw` sum -1.0 %. See `docs/METHODOLOGY.md` section 1.5.
* `selftest.py`: planted-truth checks for the model (nesting, exact totals, placebo, recovery, clipping, application,
  switch-off).

## What changed, by review point

### A. The harmonization factor
* **One k per MODIS satellite** (`fit_scale` is called per satellite). Aqua: k = 0.821. Terra: **no k** (3 matched pixels
  within 30 minutes of S-NPP; the 1,288 matches at a median 149 minutes are reported only as `k_diagnostic_all_gaps`).
* **No silent transfer.** A satellite without a k gets `NaN` VIIRS-equivalent FRP. `PYRO_APPLY_UNVALIDATED_K=1` is an
  explicit opt-in; every value is labelled in `scale_basis` (`own`, `own_low_trust`, `transferred:<sat>`, `none`).
* **Stability judged against sampling error.** New `kstats.heterogeneity`: Cochran Q, I^2, between-year SD tau with a
  Q-profile interval. Result: the 45 % max-min spread is compatible with sampling noise (Q p = 0.13, I^2 = 35 %), but tau
  cannot be bounded below 25 %, so `k_trust` stays MEDIUM. The raw spread is still reported, no longer used as a criterion.
* **Uncertainty carried forward:** `k_prediction_interval95_new_year` (0.65-1.03) and `modis_frp_viirs_equiv_lo_mw` / `_hi_mw`.
* **Bootstrap:** 10,000 replicates (was 1,000; at 1,000 the CI width straddled the 20 % grading threshold across seeds),
  vectorised. Leave-one-year-out check now judged against sampling error with a Bonferroni correction.
* **New diagnostics:** `k_sensitivity` (time gap, day/night, source type, confidence, fire size, trimming, estimator),
  `k_ci_by_cluster_unit`, `n_sites_0p01deg`, `k_by_year.png`.

### B. Missing observations
* Sensor-level numbers are `NaN` (not 0) on days a sensor did not deliver data; counts are nullable integers.
* New `harmonized_status` (`measured`, `measured_zero`, `measured_low_coverage`, `viirs_not_observed`, `partial_day`),
  `harmonized_is_measured`, `modis_only_evidence`, per-source `*_observed` flags.
* **MODIS is never substituted for a missing VIIRS day** (VIIRS totals are ~9.2x the k-scaled Aqua totals, R^2 = 0.12).
* **Gap inference rewritten** (`kstats.infer_missing_days`, per satellite, negative-binomial): finds all 36 VIIRS-absent days
  including single-day gaps the old rule missed (2024-09-26 was scored as "observed, zero fire" and filled from MODIS,
  producing a spurious 'low' anomaly, z = -3.41). Requires per-satellite counts: re-run `preprocess`.
* **Partial-outage detector** (`viirs_low_coverage`, `viirs_overpass_clusters`, `viirs_expected_detections`): 35 days.

### C. Anomaly validation (`src/anomaly_validation.py`, on by default)
* Re-tests every episode under other windows, scale estimator, baseline periods, annual-level removal, random data loss
  and outages, the classical z, and a Monte-Carlo null fitted on the non-event days; checks seasonality, extreme-value
  distortion, missing-observation artefacts and the annual-level trend (significant: -5.6 %/year, p = 0.0003).
* `anomaly_events.csv` gains `robustness` and supporting columns; new `anomaly_validation.json`, `anomaly_variants.csv`,
  `anomaly_event_variants.csv`, `anomaly_validation.png`.
* The scoring itself is unchanged by default (median / MAD, +-7 days, leave-one-year-out). New options:
  `PYRO_ANOM_DETREND`, `PYRO_ANOM_INCLUDE_LOW_COVERAGE`. Validity now follows `harmonized_status`.

### D. Spatial resolution
* `grid_sensitivity.csv/.png` and `grid_sensitivity` in `frp_mapping.json`: what a 0.05-2 degree grid changes (cell metrics,
  cell correlations, CI of k, grid-edge error) and what it does not (k, daily FRP totals: verified identical).
* `cell_area_km2` in `calendar_cells.csv`; `cell_day_corr_log` next to the (aggregation-inflated) `cell_total_corr_log`.
* Justification, what is lost and how to read the cell metrics: `docs/METHODOLOGY.md` section 4.

## Results on the shipped Egypt 2015-2025 data

| | before | after |
|---|---|---|
| k | 0.818 pooled (1,864 Aqua + 3 Terra pixels), applied to Terra too | Aqua 0.821; Terra none |
| valid days in the anomaly series | 3,963 | 3,944 (-33 outage-suspect days, -1 MODIS-filled VIIRS gap day, +15 days previously lost to MODIS-side gaps while VIIRS was fine) |
| 'low' anomaly days | 34 | 16 |
| episodes | 4 | the same 4: E2, E3 ROBUST, E4 MODERATE, E1 FRAGILE |
| calendar value on unobserved days | 0.0 (11 days) / MODIS-scaled (25 days) | NaN |

## Schema changes that may affect readers of the outputs

* `frp_mapping.json`: the top-level `k`, `k_ci95`, `k_trust`, `k_by_year`, ... moved to `modis_satellites.<Satellite>`;
  new top-level `k_by_modis_satellite`, `k_applied_to_calendar`, `scale_basis`, `scale_policy`, `observation_inference`,
  `harmonized_status_counts`, `daily_total_relationship`, `grid_sensitivity`. `detection_sensitivity` keeps the all-MODIS
  numbers and adds `like_for_like` (Aqua x S-NPP).
* `calendar_daily.csv`: `harmonized_frp_mw_viirs_equiv` is `NaN` unless the status is measured; `harmonized_source` is
  `VIIRS S-NPP` or `none` (the labels `MODIS scaled` and `no detections` no longer occur); `modis_frp_viirs_equiv_mw` is
  Aqua only; new columns listed above plus `aqua_*` / `terra_*` detections and FRP.
* `calendar_cells.csv`: new `cell_area_km2`, `scale_basis`; `frp_viirs_equiv_mw` is `NaN` for satellites without a k.
* `*_coverage.json`: new `raw_rows_per_day_by_satellite` (re-run `preprocess`).
* `anomalies_daily.csv`: new `status` column.

## New settings

`PYRO_APPLY_UNVALIDATED_K`, `PYRO_N_BOOT`, `PYRO_K_TAU_MAX`, `PYRO_GRID_SENS`, `PYRO_COV_MIN_FRAC`, `PYRO_COV_EXTREME_FRAC`,
`PYRO_COV_MIN_EXPECTED`, `PYRO_COV_GAP_MIN`, `PYRO_ANOM_VALIDATE`, `PYRO_ANOM_INCLUDE_LOW_COVERAGE`, `PYRO_ANOM_DETREND`,
`PYRO_ANOM_REPLICATES`, `PYRO_ANOM_NULL_SIMS`.

## Verification

`python -m src.selftest` (synthetic data with known answers) passes; it now also tests per-satellite k, Terra receiving no
k, detection of real between-year variation, bootstrap coverage (93 % over 150 replicates), gap / zero / outage / partial-day
semantics, grid-independence of k, a planted burst (ROBUST) and a placebo (no ROBUST episode), outage days excluded,
and the effect of a trend. Selftest and the full pipeline give the same results (to 1e-7) under pandas 2.2 / numpy 1.26 /
scipy 1.11 and pandas 3.0 / numpy 2.5 / scipy 1.18. The whole chain (`preprocess`, `plot_overview`, `harmonize`, `anomaly`)
was re-run from the raw files; outputs in `data/processed/` are current. `output.txt` (a console capture from before these
changes) was not regenerated.


## Follow-up: disk-backed out-of-core processing (review point 8)

* `src/config.py` / `src/harmonize.py`: remove the fixed `PYRO_MAX_ROWS` stop. The normal `python -m src.harmonize` command now dispatches automatically to disk-backed processing above `PYRO_OUT_OF_CORE_ROWS` combined estimated rows (`PYRO_OUT_OF_CORE=1` forces streaming; `0` forces the legacy in-memory path).
* New `src/harmonize_streaming.py`: chunk-stages both standardized CSVs into temporary SQLite tables; matches one UTC day and one satellite pair at a time; spills full matched pairs to disk; computes full-stream constant k and its cell bootstrap from disk-backed year/cell sufficient statistics; stores cell/day aggregates in SQLite; computes the complete daily calendar from SQL summaries; exports `calendar_cells.csv` in chunks. Conditional-k fitting and pixel-level diagnostics use deterministic bounded samples and are labelled in `frp_mapping.json`.
* `src/preprocess.py`: keeps the existing raw chunking and outputs, but moves the cross-chunk duplicate-key history from an in-memory NumPy array to temporary SQLite storage.
* `src/external_validation.py`: reads `calendar_cells.csv` in chunks when collecting VIIRS event cells, avoiding a full-calendar DataFrame per anomaly event.
* Settings: `PYRO_OUT_OF_CORE`, `PYRO_OUT_OF_CORE_ROWS`, `PYRO_STREAM_CHUNK_ROWS`, `PYRO_STREAM_SAMPLE_ROWS` and `PYRO_STREAM_DIAGNOSTIC_ROWS`. No new mandatory dependency.
* Tests: deterministic chunk-independent sample selection; streamed calendar and cell-output parity against the in-memory builder; exact disk-backed k/year-cell summaries; optional validation's chunked event-cell lookup; and a synthetic end-to-end out-of-core harmonization run.
* Limits: disk space is traded for RAM; one instrument/day matching window still resides in memory, so unusually dense single days may require a regional split. The auto threshold is a heuristic, not a hard resource guarantee.
