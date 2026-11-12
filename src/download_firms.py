"""Download multi-year FIRMS archive data (Area API) into data/raw/, one file per source and year.

You need a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
Pass it with --key or the FIRMS_MAP_KEY environment variable (it is never printed or written to disk).

Examples (PowerShell, from the project folder):
    $env:FIRMS_MAP_KEY="your_key"
    python -m src.download_firms --start 2018-04-01 --end 2023-12-31 --bbox 24.7,21.7,36.9,31.7
    python -m src.download_firms --start 2019-01-01 --end 2019-12-31 --sources MODIS_SP VIIRS_SNPP_SP

Notes
 * Default sources: MODIS_SP (Terra+Aqua) and VIIRS_SNPP_SP (the reference VIIRS satellite).
   Add VIIRS_NOAA20_SP / VIIRS_NOAA21_SP if wanted. *_SP is standard (science-quality) processing, which
   lags real time by about 5 months; use the *_NRT names for the most recent months.
 * Without --bbox the request is GLOBAL. A global VIIRS 375 m year is tens of millions of rows (many GB);
   start with a bounding box (or one year) unless you really need the whole planet.
 * The API returns at most 5 days per request, so a year is ~73 requests. Finished files are skipped on
   re-runs (use --force to refetch); an interrupted file is written as *.part and restarted.
 * If the API refuses a 5-day request, that request is retried day by day and only the refused days are
   skipped and listed at the end (missing days are handled as gaps by the later steps).
 * File names contain 'modis' or 'viirs' exactly as src.preprocess expects.
"""
import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

from src.config import RAW_DIR

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# First date with archive data per source (FIRMS archive page).
FIRST_DATE = {
    "MODIS_SP": date(2000, 11, 1), "MODIS_NRT": date(2000, 11, 1),
    "VIIRS_SNPP_SP": date(2012, 1, 20), "VIIRS_SNPP_NRT": date(2012, 1, 20),
    "VIIRS_NOAA20_SP": date(2018, 4, 1), "VIIRS_NOAA20_NRT": date(2018, 4, 1),
    "VIIRS_NOAA21_SP": date(2023, 3, 1), "VIIRS_NOAA21_NRT": date(2023, 3, 1),
}
STEP_DAYS = 5  # API maximum for DAY_RANGE


class ApiError(Exception):
    def __init__(self, code, body):
        super().__init__(f"HTTP {code}: {body}" if code else body)
        self.code = code


RATE_WAIT_S = 60     # pause between retries when FIRMS says we are over the request limit
RATE_TRIES = 15      # ~15 minutes in total (the limit is counted per 10 minutes)
_key_worked = False  # becomes True after the first successful request in this run


def fetch(url: str, retries: int = 4) -> str:
    """GET with retries. FIRMS answers 'HTTP 400: Invalid MAP_KEY' when a VALID key is over its request
    limit (5,000 transactions per 10 minutes), so once the key has worked we WAIT and retry instead of
    giving up. Other 400/401/403/404 replies are final."""
    global _key_worked
    last, waits, attempt = "unknown error", 0, 0
    while True:
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                text = r.read().decode("utf-8", errors="replace")
            _key_worked = True
            return text
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace").strip()[:300]
            limited = e.code == 429 or (e.code == 400 and "MAP_KEY" in body and _key_worked)
            if limited:
                waits += 1
                if waits > RATE_TRIES:
                    raise ApiError(e.code, f"still refused after {RATE_TRIES} min of waiting: {body}")
                print(f"  rate limit (HTTP {e.code} {body!r}); waiting {RATE_WAIT_S}s "
                      f"({waits}/{RATE_TRIES}) ...", flush=True)
                time.sleep(RATE_WAIT_S)
                continue
            if e.code in (400, 401, 403, 404):
                raise ApiError(e.code, body or e.reason)
            last = f"HTTP {e.code}: {body or e.reason}"
        except (urllib.error.URLError, TimeoutError) as e:
            last = f"{type(e).__name__} {e}"
        attempt += 1
        if attempt >= retries:
            raise ApiError(None, f"failed after {retries} tries: {last}")
        time.sleep(2 ** attempt * 3)


def download_range(key, source, area, d0, d1, out_path, base_url):
    """Fetch d0..d1 in 5-day requests. If a request is refused, retry it day by day and SKIP only the
    days the API refuses (they are returned so you can see them). Returns (rows, skipped_days)."""
    part = out_path.with_suffix(".csv.part")
    part.unlink(missing_ok=True)
    state = {"header": None, "rows": 0}
    skipped = []

    def get(d, n, f):
        text = fetch(f"{base_url}/{key}/{source}/{area}/{n}/{d.isoformat()}")
        lines = text.strip().splitlines()
        if not lines or not lines[0].lower().startswith("latitude"):
            raise ApiError(None, f"unexpected reply: {text[:200]!r}")
        if state["header"] is None:
            state["header"] = lines[0]
            f.write(lines[0] + "\n")
        elif lines[0] != state["header"]:
            raise SystemExit(f"Column layout changed inside {source} at {d}; aborting.")
        f.write("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""))
        state["rows"] += len(lines) - 1

    with open(part, "w", encoding="utf-8", newline="") as f:
        d = d0
        while d <= d1:
            n = min(STEP_DAYS, (d1 - d).days + 1)
            try:
                get(d, n, f)
            except ApiError as e5:
                if "MAP_KEY" in str(e5) or e5.code in (401, 403):
                    raise SystemExit(f"{source}: the API refused the key/request: {str(e5).replace(key, '***')}. "
                                     "If this key worked earlier you are probably over the request limit: wait 10 minutes "
                                     "and run the same command again (finished files are skipped).")
                for k in range(n):  # retry day by day
                    day = d + timedelta(days=k)
                    try:
                        get(day, 1, f)
                    except ApiError as e1:
                        skipped.append((day, str(e1).replace(key, "***")))
                        print(f"  skipped {day}: {str(e1).replace(key, '***')}", flush=True)
            d += timedelta(days=n)
    part.replace(out_path)
    return state["rows"], skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default=os.getenv("FIRMS_MAP_KEY"), help="FIRMS MAP_KEY (or env FIRMS_MAP_KEY)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--bbox", default=None, help="west,south,east,north (default: global)")
    ap.add_argument("--sources", nargs="+", default=["MODIS_SP", "VIIRS_SNPP_SP"], choices=sorted(FIRST_DATE))
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.key:
        sys.exit("No MAP_KEY. Get one at https://firms.modaps.eosdis.nasa.gov/api/map_key/ "
                 "and pass --key or set FIRMS_MAP_KEY.")
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    if start > end:
        sys.exit("--start is after --end")
    area = a.bbox.replace(" ", "") if a.bbox else "world"
    if a.bbox and len(area.split(",")) != 4:
        sys.exit("--bbox must be west,south,east,north")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_skipped = []
    for source in a.sources:
        s0 = max(start, FIRST_DATE[source])
        if s0 > end:
            print(f"{source}: no data before {FIRST_DATE[source]}; skipped")
            continue
        if s0 > start:
            print(f"{source}: starts {FIRST_DATE[source]}; start moved from {start}")
        for year in range(s0.year, end.year + 1):
            d0, d1 = max(s0, date(year, 1, 1)), min(end, date(year, 12, 31))
            tag = "" if area == "world" else "_bbox"
            name = f"{source.lower()}{tag}_{d0.isoformat()}_{d1.isoformat()}.csv"
            out = RAW_DIR / name
            if out.exists() and not a.force:
                print(f"{name}: already there, skipped")
                continue
            print(f"{name}: downloading ...", flush=True)
            n, skipped = download_range(a.key, source, area, d0, d1, out, a.base_url)
            print(f"{name}: {n:,} rows" + (f", {len(skipped)} day(s) skipped" if skipped else ""))
            all_skipped += [(source, d_, m_) for d_, m_ in skipped]
    if all_skipped:
        print(f"\nWARNING: {len(all_skipped)} day(s) were refused by the API and are MISSING from the files:")
        for s_, d_, m_ in all_skipped[:40]:
            print(f"  {s_} {d_}: {m_}")
        print("Check https://firms.modaps.eosdis.nasa.gov/api/data_availability/ or retry later; send this list if unsure.")
    print("Done. Next: python -m src.preprocess")


if __name__ == "__main__":
    main()