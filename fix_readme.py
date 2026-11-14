"""Apply the README corrections (global scope + logic changes). Run from the repo root:  python fix_readme.py
Each edit is an exact find/replace that must match exactly once. The script is ALL-OR-NOTHING: it prints OK or MISSING
for every edit and writes README.md (after saving README.md.bak) only if every edit matched, so a half-applied README
is impossible. If the corrections are already in, it says so and changes nothing."""
import re, sys, shutil
from pathlib import Path

P = Path("README.md")
raw = P.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
text = raw
EN = "\u2013"

def fx(old, new):  # plain find/replace, must occur exactly once
    return ("fx", old, new)
def rx(pattern, fn):  # regex on one line, must match exactly once
    return ("rx", pattern, fn)

PARA_HOTSYNC = (
    "**HotSync-style activity score (`src/hotsync_anomaly.py`, runs inside `python -m src.anomaly`).** A companion to the FRP "
    "detector. It scores the daily number of active 0.5-degree cells (the `calendar_cells` grid, not HotSync's ~2 km grid) "
    "with a mean/SD z-score against the same season (+- `PYRO_HOTSYNC_ANOM_WINDOW` days, default 7) in all other years. A day "
    "is flagged when z >= `PYRO_HOTSYNC_ANOM_Z` (default 2.5), the baseline has at least `PYRO_HOTSYNC_ANOM_MIN_POOL` values "
    "(10) and the activity is at least `PYRO_HOTSYNC_ANOM_MIN_ACTIVITY` (5). The series is anchored to valid VIIRS days: where "
    "both sensors were observed it uses the union of MODIS and VIIRS active cells; on VIIRS-observed days without MODIS "
    "coverage it uses the VIIRS cells times a VIIRS-to-union scale learned on overlap days (needs "
    "`PYRO_HOTSYNC_ANOM_MIN_OVERLAP_DAYS`, default 30). A MODIS-only day never fills a missing VIIRS day. Outputs: "
    "`hotsync_activity_*` columns in `anomalies_daily.csv`, `hotsync_activity_anomalies.csv`, `hotsync_activity_overview.png` "
    "and a `hotsync_activity` block in `anomaly_report.json`. It does not replace or change the FRP detector, its validation "
    "or its episodes.")

DEMO_FILES = (
    "For a quick global demo, download the FIRMS global 24h files (PowerShell):" + nl + nl + "```powershell" + nl +
    "Invoke-WebRequest https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv -OutFile data\\raw\\MODIS_C6_1_Global_24h.csv" + nl +
    "Invoke-WebRequest https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_24h.csv -OutFile data\\raw\\J1_VIIRS_C2_Global_24h.csv" + nl + "```")

edits = [
 # ---- scope: worldwide, not Egypt
 ("intro", fx("Starter project for preparing NASA FIRMS active-fire observations (global scope) for MODIS" + EN + "VIIRS harmonization.",
   "PyroScope harmonizes NASA FIRMS MODIS and VIIRS 375 m active-fire detections **worldwide** (NASA Space Apps challenge \"Harmonization of MODIS and VIIRS hot spots\"). Every step runs globally by default; `PYRO_BBOX` / `PYRO_AOI_FILE` only restrict the region. Egypt was the author's regional development test case, not a limit of the method.")),
 ("phase A title", fx("**Phase A (this starter):**", "**Phase A (ingestion and exploration, implemented):**")),
 ("phase A text", fx("This phase validates data ingestion, cleaning and exploratory analysis. It is *not* the final MODIS-continuity harmonization experiment.",
   "This phase validates ingestion, cleaning and exploratory analysis (`inspect_data`, `preprocess`, `plot_overview`); the harmonization itself is Phase B.")),
 ("phase B", fx("with explicit observation status. It needs years of data",
   "with explicit observation status. The time-gap window used to fit `k` is 30 minutes and is widened automatically only when no MODIS satellite has enough matches inside it (e.g. Aqua against NOAA-20 / J1), which caps the trust grade. It needs years of data")),
 ("phase C", fx("and critical burning windows. It needs multi-year history (>= 5 valid years recommended).",
   "and critical burning windows, plus a second HotSync-style active-cell activity score (`src/hotsync_anomaly.py`, section 8). It needs multi-year history (>= 5 valid years recommended).")),
 ("data bullet", rx(r"^\* \*\*Real FIRMS data is not committed\*\*[^\r\n]*", lambda m:
   "* **Real FIRMS data is not committed** (`data/raw/` is git-ignored; the files are large). The pipeline itself is global: with no `PYRO_BBOX` / `PYRO_AOI_FILE` every step reads whatever is in `data/raw/` for the whole planet." + nl +
   "* The figures labelled \"Egypt 2015-2025\" in this README, `CHANGELOG.md` and `docs/METHODOLOGY.md` (k values, the ~27 % per 10x FRP slope, VIIRS-to-Aqua total ratios, anomaly counts) come from the author's **regional development run** (Egypt bounding box `--bbox 24.7,21.7,36.9,31.7`). They show what the outputs look like; they are not global results, the scale `k` and its trust grade must be re-estimated for every region and period, and they cannot be checked from this repository alone. To approximate that run, download MODIS_SP and VIIRS_SNPP_SP for 2015-2025 with that `--bbox` (section 7), optionally set `PYRO_AOI_FILE=data/aoi/egypt.geojson` (the only country mask that ships), and run the pipeline; expect close, not identical, numbers because FIRMS archives are reprocessed over time." + nl +
   "* Quick global check without an account: put the FIRMS global 24h files `MODIS_C6_1_Global_24h.csv` and `J1_VIIRS_C2_Global_24h.csv` (section 1) in `data/raw/` and run `preprocess` and `harmonize`. This only demonstrates the pipeline (see \"Single-day demo\").")),
 # ---- section 1 / 4
 ("sec1 suggested", rx(r"^Suggested for the real experiment:[^\r\n]*", lambda m:
   "Suggested for the real experiment: MODIS C6.1 (Terra + Aqua) and VIIRS 375 m (S-NPP, plus NOAA-20 if wanted), global, standard-quality (SP) archive. S-NPP and Aqua share a ~13:30 overpass until Aqua's orbit drifted (early 2020s), so that pair is the best match candidate for 2012-2022; for later years expect few close-in-time matches and a lower `k_trust` (in a 2026-11 global 24h test the median Aqua-NOAA-20 gap was about 2 hours)." + nl + nl + DEMO_FILES)),
 ("sec4 intro", fx("Default is global with no date filter. Optional environment variables:",
   "Default is global with no date filter and no region mask (leave `PYRO_BBOX` and `PYRO_AOI_FILE` unset for a world run; the Egypt values below are only examples). Optional environment variables:")),
 ("bbox example", fx("| `24.7,21.7,36.9,31.7` |", "| `24.7,21.7,36.9,31.7` (Egypt, example) |")),
 ("aoi example", fx("| `data/aoi/egypt.geojson` |", "| `data/aoi/egypt.geojson` (the only mask that ships; supply your own polygon for other countries) |")),
 # ---- section 6 logic changes
 ("sec6 variables", fx("Optional variables: `PYRO_CELL_DEG` (default 0.5),",
   "Optional variables: `PYRO_K_MAX_DT_MIN` (default `auto`, see \"Time-gap window of k\"), `PYRO_K_WIDEN_STEPS` (default `45,60,75,90`), `PYRO_SINGLE_DAY_DEMO` (default `0`, see \"Single-day demo\"), `PYRO_CELL_DEG` (default 0.5),")),
 ("k window", fx("`k` is fitted only on matched pixels within `PYRO_K_MAX_DT_MIN` minutes (default 30).",
   "`k` is fitted only on matched pixels within the k time-gap window: 30 minutes by default (`PYRO_K_MAX_DT_MIN=auto`), widened to the first of 45 / 60 / 75 / 90 minutes that gives enough matches only when 30 does not (see \"Time-gap window of k\"); a number fixes the window.")),
 ("k trust", fx("HIGH needs >= 500 pixels (`PYRO_K_TRUST_PAIRS`),", "HIGH needs a window that did not have to be widened, >= 500 pixels (`PYRO_K_TRUST_PAIRS`),")),
 ("k model note", fx("~0.71 for 100 MW pixels.",
   "~0.71 for 100 MW pixels. Which terms survive depends on the data: a 2-day global 24h test with a widened 90-minute window selected fire size and time gap and found k falling ~75 % per 10x MODIS FRP (850 matched pixels, trust MEDIUM, much of the ratio is fire evolution between the overpasses), so treat these numbers as properties of a dataset, not constants.")),
 ("anomaly series", fx("(the anomaly series is VIIRS-only and does not move)", "(the FRP anomaly series is VIIRS-only and does not move)")),
 ("status list", fx("and `partial_day` (plus `observed_obscured` when a cloud file is supplied, see the next paragraph); `harmonized_frp_mw_viirs_equiv` is the VIIRS reference value on the first three and `NaN` otherwise,",
   "and `partial_day` (plus `observed_obscured` when a cloud file is supplied, see the next paragraph, and `demo_partial_window` only with `PYRO_SINGLE_DAY_DEMO=1`); `harmonized_frp_mw_viirs_equiv` is the VIIRS reference value on `measured`, `measured_zero` and `measured_low_coverage` and `NaN` otherwise (`demo_partial_window` carries a lower-bound VIIRS total but is never `harmonized_is_measured`),")),
 ("9x claim 1", fx("(on this data VIIRS totals are ~9x the k-scaled Aqua totals and correlate weakly with them; `daily_total_relationship`)",
   "(in the author's Egypt run VIIRS totals are ~9x the k-scaled Aqua totals and correlate weakly with them; check `daily_total_relationship` for your own data)")),
 ("9x claim 2", fx("and the data do not support that here: on the shipped Egypt data VIIRS totals are ~9x the k-scaled Aqua totals",
   "and the data do not support that here: in the author's Egypt run VIIRS totals are ~9x the k-scaled Aqua totals")),
 ("pre-VIIRS claim", fx("so the pre-VIIRS / MODIS-only record can be read on the same axis;",
   "so MODIS can be read on the same axis (the calendar covers only the window in which both sensors have data; the pipeline does not extend it to MODIS-only years before the VIIRS era, although `k` could be applied there by hand);")),
 ("j1 window", fx("Aqua vs NOAA-20 is ~50 minutes apart, so with the J1 file you will see `k window: ... widened to 60 min`.",
   "Aqua and NOAA-20 were ~50 minutes apart at their nominal times, but Aqua has since drifted: in a 2026-11 global 24h test the median gap was ~2 hours and the window widened to 90 minutes (`k window: ... widened to 90 min`, Aqua k from 850 matched pixels, trust MEDIUM). The widened window applies to every MODIS satellite in that run.")),
 ("demo claim", fx("The global 24h NRT files never cover a full UTC day, so by default their one day is `partial_day` with no harmonized value. For a demo run",
   "A FIRMS 24h NRT file can start and end part-way through a UTC day; by default such a partial day is `partial_day` with no harmonized value. (The 2026-11 global J1 and MODIS 24h files covered two UTC days almost completely, came out `measured`, and the demo flag changed nothing.) To label partial days for a demo, run")),
 # ---- section 7-10
 ("sec7 bbox", fx("Default sources are `MODIS_SP` (Terra + Aqua) and `VIIRS_SNPP_SP`.",
   "The `--bbox` above is only the Egypt example: use your own region, or drop it for the whole globe (see the size warning). Default sources are `MODIS_SP` (Terra + Aqua) and `VIIRS_SNPP_SP`.")),
 ("sec7 step3", fx("Run the pipeline with the same `PYRO_BBOX` for every step:", "Run the pipeline with the same region settings (`PYRO_BBOX`; none for a global run) for every step:")),
 ("sec8 series", fx("absent, partial, outage-suspect and cloud-obscured days are missing, never zero",
   "absent, partial (including `demo_partial_window`), outage-suspect and cloud-obscured days are missing, never zero")),
 ("sec8 hotsync", rx(r"^Outputs in `data/processed/`: `anomalies_daily\.csv`[^\r\n]*", lambda m: m.group(0) + nl + nl + PARA_HOTSYNC)),
 ("sec9 download", fx("Download the HDF granules for the region and years you want to validate, then run:",
   "Download the HDF granules for the region and years you want to validate (`MCD64A1/download_mcd64a1.py` is a helper that downloads from NASA LAADS, which requires an Earthdata login; it reads `MCD64A1/required_tiles.json`, which in this repository lists the Egypt tiles h20-h21 / v05-v06 and the Egypt events, and nothing in the repository generates it, so for another region write your own file with the same keys: `tiles`, `events`, `collection`, `cell_deg`), then run:")),
 ("notes drift", fx("Terra has almost no usable matches with S-NPP.",
   "Terra has almost no usable matches with S-NPP. In the 2026-11 global 24h files the median local time of Aqua daytime detections was already about 15.7 h and of Terra about 8.7 h.")),
 ("notes 56x", fx("(VIIRS sees ~56 times as many detections as Aqua over Egypt).", "(VIIRS sees ~56 times as many detections as Aqua over Egypt; the factor changes with region and year).")),
 ("sec10 cleanup", fx("Temporary staging files are removed after a successful run.",
   "Temporary staging files are removed after a successful run. A run killed midway can leave `pyroscope_dedupe_*` / `pyroscope_stream_*` folders in the output directory; the next `preprocess` or `harmonize` run deletes such folders when they are more than an hour old.")),
]

SENTINEL = "harmonizes NASA FIRMS MODIS and VIIRS 375 m active-fire detections **worldwide**"
if SENTINEL in text:
    print("README.md already contains these corrections; nothing changed."); sys.exit(0)

failed = []
for name, (kind, a, b) in edits:
    if kind == "fx":
        n = text.count(a)
        if n == 1:
            text = text.replace(a, b); print(f"OK       {name}")
        else:
            print(f"MISSING  {name}  (found {n} times)"); failed.append(name)
    else:
        ms = list(re.finditer(a, text, flags=re.M))
        if len(ms) == 1:
            m = ms[0]; text = text[:m.start()] + b(m) + text[m.end():]; print(f"OK       {name}")
        else:
            print(f"MISSING  {name}  (matched {len(ms)} lines)"); failed.append(name)

if failed:
    print(f"\nNOTHING WRITTEN: {len(failed)} edit(s) did not match exactly once: {', '.join(failed)}")
    print("Send me the MISSING lines (or the README text around them) and I will adjust those edits.")
    sys.exit(1)
shutil.copyfile(P, "README.md.bak")
P.write_bytes(text.encode("utf-8"))
print(f"\nAll {len(edits)} edits applied. Backup: README.md.bak")
