# Changelog: backend hardening after the methodological review

Backend only (no frontend work). Evidence and reasoning: `docs/METHODOLOGY.md`. A unified diff against
`pyroscope_modis_viirs_not_final.zip` is in `docs/CHANGES_vs_not_final.patch`.

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
