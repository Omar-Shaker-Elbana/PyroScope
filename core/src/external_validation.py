"""Optional external validation of PyroScope anomaly events with MODIS MCD64A1.

The normal PyroScope pipeline does not import this module and its anomaly scores are
never modified. Run it after ``python -m src.anomaly`` and download the matching
MCD64A1 Collection 6.1 monthly HDF granules separately.

MCD64A1 ``Burn Date`` is day-of-year (1..366; 0 means unburned, negative values
are unmapped/water). QA bit 0 is land, bit 1 is the valid-data flag, bit 2 marks
a shortened mapping period, and bits 5..7 encode special conditions for some
unburned pixels. See the MCD64A1 Collection 6.1 User's Guide:
https://lpdaac.usgs.gov/documents/1006/MCD64_User_Guide_V61.pdf

Raster reading is deliberately optional. Install a Rasterio build with HDF4
support (``pip install rasterio pyproj``; some platforms require GDAL/HDF4 system
libraries) only when running this module. No new dependency is required by the
core PyroScope pipeline or its self-test.
"""
from __future__ import annotations

import argparse
import calendar
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.src.config import CELL_DEG, PROCESSED_DIR


_PRODUCT_RE = re.compile(
    r"MCD64A1\.A(?P<year>\d{4})(?P<doy>\d{3})\.h(?P<h>\d{2})v(?P<v>\d{2})\.(?P<collection>\d{3})(?:\.|$)",
    re.IGNORECASE,
)
# MODIS global sinusoidal tile grid (meters; 36 x 18 tiles).
_TILE_SIZE_M = 1_111_950.5196666666
_HALF_WORLD_X_M = 20_015_109.354
_HALF_WORLD_Y_M = 10_007_554.677
_SINUSOIDAL_RADIUS_M = 6_371_007.181


# --------------------------------------------------------------------------- input conventions

def parse_mcd64_filename(path: str | Path) -> dict | None:
    """Parse the MCD64A1 YYYY/DOY start date, tile and collection from a filename."""
    name = Path(path).name
    match = _PRODUCT_RE.search(name)
    if not match:
        return None
    year, doy = int(match.group("year")), int(match.group("doy"))
    try:
        first_day = date(year, 1, 1) + timedelta(days=doy - 1)
    except ValueError:
        return None
    # MCD64A1 names encode the first day of the calendar month, not an
    # approximation such as day-of-year // 30.
    if first_day.day != 1:
        return None
    return {
        "year": year,
        "month": first_day.month,
        "first_day": first_day,
        "tile": f"h{int(match.group('h')):02d}v{int(match.group('v')):02d}",
        "collection": match.group("collection"),
    }


def _event_date_columns(events: pd.DataFrame) -> tuple[str, str]:
    start_col = next((c for c in ("start", "start_date", "onset_date") if c in events.columns), None)
    end_col = next((c for c in ("end", "end_date") if c in events.columns), None)
    if start_col is None or end_col is None:
        raise ValueError(
            "Events CSV must contain the PyroScope columns 'start' and 'end' "
            "(aliases 'start_date'/'end_date' and 'onset_date'/'end_date' are also accepted)."
        )
    return start_col, end_col


def _load_event_cells(cells_csv: str | Path, start: pd.Timestamp, end: pd.Timestamp,
                      cell_deg: float, chunksize: int = 500_000) -> tuple[set[int], dict[int, tuple[float, float]]]:
    """Return event VIIRS cells with bounded-memory, chunked calendar reads.

    Cell centres are invariant for a given ``cell_id`` on a fixed project grid, so
    retaining the first encountered centre is equivalent to the legacy median.
    """
    header = pd.read_csv(cells_csv, nrows=0).columns
    required = {"date", "cell_id", "sensor_family"}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{cells_csv} is missing required calendar_cells columns: {', '.join(missing)}")
    has_centres = {"cell_lat", "cell_lon"}.issubset(header)
    usecols = ["date", "cell_id", "sensor_family"] + (["cell_lat", "cell_lon"] if has_centres else [])
    start_day, end_day = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    centres: dict[int, tuple[float, float]] = {}
    for chunk in pd.read_csv(cells_csv, usecols=usecols, chunksize=max(1, int(chunksize)), low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        viirs = chunk["sensor_family"].astype(str).str.strip().str.upper().eq("VIIRS")
        during = chunk["date"].between(start_day, end_day, inclusive="both")
        selected = chunk.loc[viirs & during].copy()
        if selected.empty:
            continue
        selected["cell_id"] = pd.to_numeric(selected["cell_id"], errors="coerce")
        selected = selected.dropna(subset=["cell_id"])
        if selected.empty:
            continue
        if has_centres:
            selected["cell_lat"] = pd.to_numeric(selected["cell_lat"], errors="coerce")
            selected["cell_lon"] = pd.to_numeric(selected["cell_lon"], errors="coerce")
            selected = selected.dropna(subset=["cell_lat", "cell_lon"])
        for row in selected.drop_duplicates("cell_id").itertuples(index=False):
            cid = int(row.cell_id)
            if has_centres:
                centres.setdefault(cid, (float(row.cell_lon), float(row.cell_lat)))
            else:
                i, j = divmod(cid, 1_000_000)
                centres.setdefault(cid, (-180.0 + (j + 0.5) * cell_deg, -90.0 + (i + 0.5) * cell_deg))
    return set(centres), centres


def _cell_bounds(cell_ids: Iterable[int], cell_deg: float) -> dict[int, tuple[float, float, float, float]]:
    """Decode PyroScope's cell_id convention into (west, south, east, north)."""
    out = {}
    for raw in cell_ids:
        i, j = divmod(int(raw), 1_000_000)
        west = -180.0 + j * cell_deg
        south = -90.0 + i * cell_deg
        out[int(raw)] = (west, south, west + cell_deg, south + cell_deg)
    return out


def _sinusoidal_tile(lon: float, lat: float) -> str:
    """Approximate MODIS h/v tile for an interior lon/lat point."""
    phi = math.radians(float(lat))
    lam = math.radians(float(lon))
    x = _SINUSOIDAL_RADIUS_M * lam * math.cos(phi)
    y = _SINUSOIDAL_RADIUS_M * phi
    h = int(math.floor((x + _HALF_WORLD_X_M) / _TILE_SIZE_M))
    v = int(math.floor((_HALF_WORLD_Y_M - y) / _TILE_SIZE_M))
    h = min(35, max(0, h))
    v = min(17, max(0, v))
    return f"h{h:02d}v{v:02d}"


def _tiles_for_cells(cell_ids: Iterable[int], cell_deg: float) -> dict[str, set[int]]:
    """Map each active cell to every MODIS tile sampled within its footprint.

    A 5 x 5 interior sample prevents cells straddling a MODIS tile boundary from
    being assigned to only one side of that boundary.
    """
    tile_cells: dict[str, set[int]] = defaultdict(set)
    fractions = np.linspace(0.01, 0.99, 5)
    bounds = _cell_bounds(cell_ids, cell_deg)
    for cid, (west, south, east, north) in bounds.items():
        for fy in fractions:
            lat = south + (north - south) * float(fy)
            for fx in fractions:
                lon = west + (east - west) * float(fx)
                tile_cells[_sinusoidal_tile(lon, lat)].add(cid)
    return dict(tile_cells)


def _month_segments(start: date, end: date):
    """Yield inclusive (month_start, segment_start, segment_end) date triples."""
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        last = date(cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1])
        yield cursor, max(start, cursor), min(end, last)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def _shift_year(value: date, target_year: int) -> date:
    """Shift a calendar date by year, clipping 29 February to 28 February."""
    day = min(value.day, calendar.monthrange(target_year, value.month)[1])
    return date(target_year, value.month, day)


# --------------------------------------------------------------------------- MCD64A1 quality / array calculations

def _summarize_arrays(burn_date: np.ndarray, qa: np.ndarray, pixel_cell_ids: np.ndarray,
                      start_doy: int, end_doy: int, pixel_area_km2: float,
                      first_day: np.ndarray | None = None,
                      last_day: np.ndarray | None = None) -> dict:
    """Summarize a cropped MCD64A1 raster window for selected event cells.

    ``pixel_cell_ids`` is the PyroScope cell id for each raster pixel, or -1
    outside the event cells. Positive burns count as evidence when QA bits 0/1
    are set. A no-burn observation is considered fully usable only when QA bit 2
    does not indicate a shortened mapping period (or First/Last Day encompass
    the complete requested date segment) and the unburned special-condition
    code in bits 5..7 is zero.
    """
    burn = np.asarray(burn_date)
    quality = np.asarray(qa).astype(np.uint8, copy=False)
    cell_map = np.asarray(pixel_cell_ids)
    if burn.shape != quality.shape or burn.shape != cell_map.shape:
        raise ValueError("Burn Date, QA and pixel-cell arrays must have identical shapes")
    if not (1 <= int(start_doy) <= 366 and 1 <= int(end_doy) <= 366 and start_doy <= end_doy):
        raise ValueError("Invalid day-of-year interval for an MCD64A1 monthly segment")
    # 255 is the byte fill value in common MCD64A1 QA encodings; never decode it as land/valid.
    not_fill = quality != 255
    land = ((quality & 1) != 0) & not_fill
    valid_flag = ((quality & 2) != 0) & not_fill
    valid_date = (burn >= 0) & (burn <= 366)
    valid_land = land & valid_flag & valid_date & (cell_map >= 0)
    shortened = (quality & 4) != 0
    if first_day is not None and last_day is not None:
        first = np.asarray(first_day)
        last = np.asarray(last_day)
        if first.shape != burn.shape or last.shape != burn.shape:
            raise ValueError("First Day and Last Day arrays must match Burn Date dimensions")
        period_covered = (~shortened) | ((first > 0) & (last > 0) & (first <= start_doy) & (last >= end_doy))
    else:
        period_covered = ~shortened
    special_condition = (quality >> 5) & 7
    reliable_unburned = valid_land & period_covered & (special_condition == 0)
    is_burned_in_window = valid_land & (burn >= 1) & (burn >= start_doy) & (burn <= end_doy)
    in_cells = cell_map >= 0
    land_in_cells = land & in_cells
    burned_ids = np.unique(cell_map[is_burned_in_window]).astype(np.int64)
    covered_ids = np.unique(cell_map[reliable_unburned]).astype(np.int64)
    land_count = int(np.count_nonzero(land_in_cells))
    known_count = int(np.count_nonzero(reliable_unburned))
    burned_count = int(np.count_nonzero(is_burned_in_window))
    return {
        "land_pixel_observations": land_count,
        "valid_land_pixel_observations": known_count,
        "valid_burn_date_pixels": int(np.count_nonzero(valid_land)),
        "burned_pixels": burned_count,
        "burned_area_km2": float(burned_count * pixel_area_km2),
        "shortened_mapping_pixels": int(np.count_nonzero(land_in_cells & shortened)),
        "special_condition_pixels": int(np.count_nonzero(land_in_cells & (special_condition > 0))),
        "burned_cell_ids": set(int(x) for x in burned_ids if x >= 0),
        "covered_cell_ids": set(int(x) for x in covered_ids if x >= 0),
    }


def _coverage_fraction(stats: dict) -> float | None:
    denominator = int(stats.get("land_pixel_observations", 0))
    if denominator <= 0:
        return None
    return float(stats.get("valid_land_pixel_observations", 0)) / denominator


def _period_status(stats: dict, min_coverage: float) -> str:
    if stats.get("burned_pixels", 0) > 0:
        return "BURNED_AREA_DETECTED" if stats.get("complete_coverage") else "BURNED_AREA_DETECTED_PARTIAL_COVERAGE"
    if stats.get("missing_tile_count", 0) > 0:
        return "MCD64_FILES_MISSING"
    if stats.get("complete_coverage", False):
        return "NO_BURN_IN_VALID_COVERAGE"
    if stats.get("files_read", 0) == 0:
        return "MCD64_FILES_MISSING"
    return "INCONCLUSIVE_LOW_COVERAGE"


# --------------------------------------------------------------------------- raster I/O (optional)

def _optional_raster_modules():
    try:
        import rasterio
        from rasterio.windows import Window, from_bounds
        from rasterio.warp import transform_bounds
        from pyproj import Transformer
    except ImportError as exc:
        raise RuntimeError(
            "Reading MCD64A1 HDF files requires Rasterio and PyProj. Install them with "
            "`pip install rasterio pyproj` and ensure your Rasterio/GDAL build supports HDF4. "
            "These are optional and are not needed for the normal PyroScope pipeline."
        ) from exc
    return rasterio, Window, from_bounds, transform_bounds, Transformer


def _subdataset_paths(hdf_path: Path, rasterio) -> dict[str, str]:
    """Find MCD64A1 layers by name, never by a brittle fixed subdataset index."""
    try:
        with rasterio.open(hdf_path) as root:
            subdatasets = list(root.subdatasets)
    except Exception as exc:
        raise RuntimeError(f"Cannot open HDF container {hdf_path}: {exc}") from exc
    if not subdatasets:
        raise RuntimeError(f"{hdf_path} contains no readable HDF subdatasets (HDF4 support may be missing)")
    found: dict[str, str] = {}
    for subdataset in subdatasets:
        label = re.sub(r"[^a-z0-9]+", " ", subdataset.lower()).strip()
        if "burn date" in label and "uncertainty" not in label:
            found["burn"] = subdataset
        elif re.search(r"(?:^|\s)qa(?:\s|$)", label):
            found["qa"] = subdataset
        elif "first day" in label:
            found["first"] = subdataset
        elif "last day" in label:
            found["last"] = subdataset
    if "burn" not in found or "qa" not in found:
        raise RuntimeError(
            f"{hdf_path} does not expose the required MCD64A1 'Burn Date' and 'QA' layers; "
            f"found subdatasets: {subdatasets}"
        )
    return found


def _pixel_area_km2(dataset) -> float:
    transform = dataset.transform
    area_native = abs(transform.a * transform.e - transform.b * transform.d)
    if not np.isfinite(area_native) or area_native <= 0:
        raise RuntimeError("MCD64A1 raster has an invalid pixel transform")
    units = getattr(dataset.crs, "linear_units_factor", None)
    if units is not None:
        factor = float(units[1])
    else:
        unit_name = str(getattr(dataset.crs, "linear_units", "")).lower()
        if unit_name in ("metre", "meter", "metres", "meters"):
            factor = 1.0
        elif unit_name in ("kilometre", "kilometer", "kilometres", "kilometers"):
            factor = 1000.0
        else:
            raise RuntimeError(f"Cannot convert raster CRS linear units to km²: {dataset.crs}")
    return float(area_native * factor * factor / 1_000_000.0)


def _read_month_tile(hdf_path: Path, segment_start: date, segment_end: date,
                     cell_ids: set[int], bounds: dict[int, tuple[float, float, float, float]],
                     cell_deg: float, raster_modules) -> dict:
    rasterio, Window, from_bounds, transform_bounds, Transformer = raster_modules
    layers = _subdataset_paths(hdf_path, rasterio)
    with rasterio.open(layers["burn"]) as burn_ds, rasterio.open(layers["qa"]) as qa_ds:
        if burn_ds.crs is None:
            raise RuntimeError(f"{hdf_path}: Burn Date layer has no CRS")
        if burn_ds.width != qa_ds.width or burn_ds.height != qa_ds.height or burn_ds.transform != qa_ds.transform:
            raise RuntimeError(f"{hdf_path}: Burn Date and QA grids are not aligned")
        west = min(bounds[cid][0] for cid in cell_ids)
        south = min(bounds[cid][1] for cid in cell_ids)
        east = max(bounds[cid][2] for cid in cell_ids)
        north = max(bounds[cid][3] for cid in cell_ids)
        raster_bounds = transform_bounds("EPSG:4326", burn_ds.crs, west, south, east, north, densify_pts=21)
        try:
            window = from_bounds(*raster_bounds, transform=burn_ds.transform)
            window = window.round_offsets().round_lengths().intersection(
                Window(0, 0, burn_ds.width, burn_ds.height)
            )
        except Exception:
            return {"land_pixel_observations": 0, "valid_land_pixel_observations": 0,
                    "valid_burn_date_pixels": 0, "burned_pixels": 0, "burned_area_km2": 0.0,
                    "shortened_mapping_pixels": 0, "special_condition_pixels": 0,
                    "burned_cell_ids": set(), "covered_cell_ids": set(), "files_read": 0}
        if window.width <= 0 or window.height <= 0:
            return {"land_pixel_observations": 0, "valid_land_pixel_observations": 0,
                    "valid_burn_date_pixels": 0, "burned_pixels": 0, "burned_area_km2": 0.0,
                    "shortened_mapping_pixels": 0, "special_condition_pixels": 0,
                    "burned_cell_ids": set(), "covered_cell_ids": set(), "files_read": 0}
        burn = burn_ds.read(1, window=window)
        qa = qa_ds.read(1, window=window)
        first = last = None
        if "first" in layers and "last" in layers:
            with rasterio.open(layers["first"]) as first_ds, rasterio.open(layers["last"]) as last_ds:
                for label, layer_ds in (("First Day", first_ds), ("Last Day", last_ds)):
                    if (layer_ds.width != burn_ds.width or layer_ds.height != burn_ds.height
                            or layer_ds.transform != burn_ds.transform or layer_ds.crs != burn_ds.crs):
                        raise RuntimeError(f"{hdf_path}: {label} grid does not align with Burn Date")
                first = first_ds.read(1, window=window)
                last = last_ds.read(1, window=window)
        win_transform = burn_ds.window_transform(window)
        cols = np.arange(int(window.width), dtype=np.float64) + 0.5
        rows = np.arange(int(window.height), dtype=np.float64) + 0.5
        col_grid, row_grid = np.meshgrid(cols, rows)
        xx = (win_transform.c + col_grid * win_transform.a + row_grid * win_transform.b)
        yy = (win_transform.f + col_grid * win_transform.d + row_grid * win_transform.e)
        transformer = Transformer.from_crs(burn_ds.crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(xx, yy)
        ii = np.floor((np.asarray(lat) + 90.0) / cell_deg).astype(np.int64)
        jj = np.floor((np.asarray(lon) + 180.0) / cell_deg).astype(np.int64)
        mapped = ii * 1_000_000 + jj
        inside = np.isin(mapped, np.fromiter(cell_ids, dtype=np.int64))
        pixel_cell_ids = np.where(inside, mapped, -1)
        area = _pixel_area_km2(burn_ds)
        stats = _summarize_arrays(
            burn, qa, pixel_cell_ids, segment_start.timetuple().tm_yday,
            segment_end.timetuple().tm_yday, area, first, last,
        )
        stats["files_read"] = 1
        return stats


# --------------------------------------------------------------------------- period aggregation / reporting

def _index_product_files(directory: str | Path, collection: str = "061") -> dict[tuple[int, int], dict[str, Path]]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"MCD64A1 directory does not exist: {root}")
    index: dict[tuple[int, int], dict[str, Path]] = defaultdict(dict)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".hdf", ".h4"):
            continue
        parsed = parse_mcd64_filename(path)
        if parsed is None or (collection and parsed["collection"] != str(collection)):
            continue
        key = (parsed["year"], parsed["month"])
        tile = parsed["tile"]
        # Multiple production versions of one tile/month can exist in copied
        # archives. Keep one deterministic (lexicographically newest) version.
        previous = index[key].get(tile)
        if previous is None or path.name > previous.name:
            index[key][tile] = path
    return dict(index)


def _sum_stats(total: dict, part: dict) -> None:
    for key in ("land_pixel_observations", "valid_land_pixel_observations", "valid_burn_date_pixels",
                "burned_pixels", "shortened_mapping_pixels", "special_condition_pixels"):
        total[key] += int(part.get(key, 0))
    total["burned_area_km2"] += float(part.get("burned_area_km2", 0.0))
    total["burned_cell_ids"].update(part.get("burned_cell_ids", set()))
    total["covered_cell_ids"].update(part.get("covered_cell_ids", set()))
    total["files_read"] += int(part.get("files_read", 0))


def _period_stats(start: date, end: date, tile_cells: dict[str, set[int]],
                  cell_bounds: dict[int, tuple[float, float, float, float]],
                  file_index: dict[tuple[int, int], dict[str, Path]], cell_deg: float,
                  raster_modules, min_coverage: float = 0.80) -> dict:
    total = {
        "land_pixel_observations": 0, "valid_land_pixel_observations": 0,
        "valid_burn_date_pixels": 0, "burned_pixels": 0, "burned_area_km2": 0.0,
        "shortened_mapping_pixels": 0, "special_condition_pixels": 0,
        "burned_cell_ids": set(), "covered_cell_ids": set(), "files_read": 0,
        "files_expected": 0, "missing_tile_count": 0, "missing_tile_ids": set(), "missing_tile_months": set(),
    }
    for month_start, seg_start, seg_end in _month_segments(start, end):
        available = file_index.get((month_start.year, month_start.month), {})
        total["files_expected"] += len(tile_cells)
        missing = set(tile_cells) - set(available)
        total["missing_tile_count"] += len(missing)
        total["missing_tile_ids"].update(missing)
        total["missing_tile_months"].update(f"{month_start.isoformat()}:{tile}" for tile in missing)
        for tile, ids in tile_cells.items():
            path = available.get(tile)
            if path is None:
                continue
            part = _read_month_tile(path, seg_start, seg_end, ids, cell_bounds, cell_deg, raster_modules)
            _sum_stats(total, part)
    coverage = _coverage_fraction(total)
    total["coverage_fraction"] = coverage
    total["complete_coverage"] = bool(
        total["files_expected"] > 0 and total["missing_tile_count"] == 0
        and total["files_read"] > 0 and coverage is not None and coverage >= min_coverage
    )
    return total


def _empirical_percentile(value: float, reference: list[float]) -> float | None:
    if not reference:
        return None
    values = np.asarray(reference, dtype=float)
    less = int(np.count_nonzero(values < value))
    equal = int(np.count_nonzero(values == value))
    return float(100.0 * (less + 0.5 * equal) / len(values))


def validate_events(events_df: pd.DataFrame, mcd64_dir: str | Path,
                    cells_csv: str | Path | None = None,
                    output_csv: str | Path | None = None,
                    output_json: str | Path | None = None,
                    cell_deg: float = CELL_DEG, min_coverage: float = 0.80,
                    collection: str = "061") -> tuple[pd.DataFrame, dict]:
    """Validate PyroScope events against MCD64A1 monthly burned-area data.

    Event spatial footprints are the unique VIIRS ``cell_id`` values in
    ``calendar_cells.csv`` during each event. Burned-area totals are computed
    only in those event cells, using the actual source raster pixel area and
    exact Burn Date day-of-year interval. The seasonal reference repeats the
    same calendar window in every other year available in the MCD64A1 archive.

    ``cells_csv`` defaults to PyroScope's ``calendar_cells.csv``. The default
    outputs are ``external_validation.csv`` and ``external_validation.json``
    in the configured processed directory.
    """
    if not (0 < float(cell_deg) <= 10):
        raise ValueError("cell_deg must be greater than 0 and no larger than 10 degrees")
    if not (0 < float(min_coverage) <= 1):
        raise ValueError("min_coverage must be in the interval (0, 1]")
    if cells_csv is None:
        cells_csv = PROCESSED_DIR / "calendar_cells.csv"
    if output_csv is None:
        output_csv = PROCESSED_DIR / "external_validation.csv"
    if output_json is None:
        output_json = PROCESSED_DIR / "external_validation.json"
    start_col, end_col = _event_date_columns(events_df)
    file_index = _index_product_files(mcd64_dir, collection=collection)
    if not file_index:
        raise FileNotFoundError(
            f"No MCD64A1 HDF files for collection {collection} found under {mcd64_dir}. "
            "Expected names such as MCD64A1.A2020032.hXXvYY.061.*.hdf."
        )
    raster_modules = _optional_raster_modules()
    available_years = sorted({year for year, _ in file_index})
    results = []
    for row_number, (_, event) in enumerate(events_df.iterrows(), start=1):
        start_ts = pd.to_datetime(event[start_col], errors="coerce")
        end_ts = pd.to_datetime(event[end_col], errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts) or end_ts < start_ts:
            raise ValueError(f"Invalid start/end dates in event row {row_number}: {event[start_col]!r}, {event[end_col]!r}")
        start_date, end_date = start_ts.date(), end_ts.date()
        raw_event_id = event.get("event_id", None)
        event_id = str(raw_event_id) if pd.notna(raw_event_id) and str(raw_event_id).strip() else f"E{row_number:03d}"
        active_cells, _ = _load_event_cells(cells_csv, start_ts.normalize(), end_ts.normalize(), cell_deg)
        if not active_cells:
            results.append({
                "event_id": event_id, "start": start_date.isoformat(), "end": end_date.isoformat(),
                "event_active_cells": 0, "burned_area_km2": None, "baseline_median_km2": None,
                "baseline_q90_km2": None, "event_to_baseline_ratio": None, "baseline_percentile": None,
                "baseline_n_years": 0, "coverage_fraction": None, "missing_tile_count": None,
                "burned_pixels": None, "burned_active_cells": None, "share_event_cells_with_burn": None,
                "status": "NO_VIIRS_EVENT_CELLS",
            })
            continue
        cell_bounds = _cell_bounds(active_cells, cell_deg)
        tile_cells = _tiles_for_cells(active_cells, cell_deg)
        observed = _period_stats(start_date, end_date, tile_cells, cell_bounds,
                                 file_index, cell_deg, raster_modules, min_coverage)
        status = _period_status(observed, min_coverage)
        # Same month/day window in all other archive years: this is a
        # leave-one-year-out seasonal comparison, consistent with Phase C.
        baseline_values: list[float] = []
        baseline_years: list[int] = []
        for target_year in available_years:
            if target_year == start_date.year:
                continue
            shifted_start = _shift_year(start_date, target_year)
            shifted_end = _shift_year(end_date, end_date.year + (target_year - start_date.year))
            if shifted_end < shifted_start:
                continue
            baseline = _period_stats(shifted_start, shifted_end, tile_cells, cell_bounds,
                                     file_index, cell_deg, raster_modules, min_coverage)
            if baseline["complete_coverage"]:
                baseline_values.append(float(baseline["burned_area_km2"]))
                baseline_years.append(target_year)
        baseline_median = float(np.median(baseline_values)) if baseline_values else None
        baseline_q90 = float(np.percentile(baseline_values, 90)) if baseline_values else None
        event_area_complete = bool(observed["complete_coverage"])
        ratio = (float(observed["burned_area_km2"]) / baseline_median
                 if event_area_complete and baseline_median is not None and baseline_median > 0 else None)
        percentile = (_empirical_percentile(float(observed["burned_area_km2"]), baseline_values)
                      if event_area_complete else None)
        observed_area = (float(observed["burned_area_km2"])
                         if observed["files_read"] > 0 and observed["land_pixel_observations"] > 0 else None)
        complete_area = float(observed["burned_area_km2"]) if event_area_complete else None
        burned_cells = set(observed["burned_cell_ids"])
        out = {
            "event_id": event_id,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "event_active_cells": len(active_cells),
            "burned_area_km2": complete_area,
            "burned_area_observed_km2": observed_area,
            "baseline_median_km2": baseline_median,
            "baseline_q90_km2": baseline_q90,
            "event_to_baseline_ratio": ratio,
            "baseline_percentile": percentile,
            "baseline_n_years": len(baseline_values),
            "baseline_years": ";".join(str(y) for y in baseline_years),
            "coverage_fraction": observed["coverage_fraction"],
            "coverage_complete": event_area_complete,
            "land_pixel_observations": observed["land_pixel_observations"],
            "valid_land_pixel_observations": observed["valid_land_pixel_observations"],
            "missing_tile_count": observed["missing_tile_count"],
            "missing_tile_ids": ";".join(sorted(observed["missing_tile_ids"])),
            "missing_tile_months": ";".join(sorted(observed["missing_tile_months"])),
            "files_read": observed["files_read"],
            "burned_pixels": observed["burned_pixels"],
            "burned_active_cells": len(burned_cells),
            "share_event_cells_with_burn": float(len(burned_cells) / len(active_cells)),
            "shortened_mapping_pixels": observed["shortened_mapping_pixels"],
            "special_condition_pixels": observed["special_condition_pixels"],
            "status": status,
        }
        results.append(out)
    out_df = pd.DataFrame(results)
    payload = {
        "product": "MODIS MCD64A1 Collection 6.1",
        "collection": str(collection),
        "cell_deg": float(cell_deg),
        "minimum_coverage_fraction_for_confirmed_no_burn": float(min_coverage),
        "events": int(len(out_df)),
        "status_counts": {str(k): int(v) for k, v in out_df.get("status", pd.Series(dtype=str)).value_counts().items()},
        "results": out_df.replace({np.nan: None}).to_dict(orient="records"),
        "interpretation": (
            "MCD64A1 measures burned area from MODIS surface-reflectance time series, not active-fire FRP. "
            "A positive burned-area match is supporting evidence, not a one-to-one equivalence. "
            "A complete burned_area_km2 total (including a true zero) is reported only when valid spatial/temporal coverage is adequate; "
            "burned_area_observed_km2 is a partial observed subtotal when coverage is incomplete. "
            "missing granules, invalid QA, shortened mapping periods and unburned special-condition pixels "
            "must not be interpreted as confirmed zero fire. Spatial agreement is measured on PyroScope's "
            "configured cell grid and is therefore grid-resolution dependent."
        ),
    }
    for path in (output_csv, output_json):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    Path(output_json).write_text(json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8")
    return out_df, payload


# --------------------------------------------------------------------------- fast synthetic checks (no raster files / optional packages needed)

def self_test() -> None:
    """Small deterministic checks included in ``python -m src.selftest``."""
    feb_2020 = parse_mcd64_filename("MCD64A1.A2020032.h18v05.061.2021032.hdf")
    assert feb_2020 and feb_2020["first_day"] == date(2020, 2, 1) and feb_2020["month"] == 2
    dec_2020 = parse_mcd64_filename("MCD64A1.A2020336.h18v05.061.2021032.hdf")
    assert dec_2020 and dec_2020["first_day"] == date(2020, 12, 1)
    assert parse_mcd64_filename("MCD64A1.A2020002.h18v05.061.bad.hdf") is None
    segs = list(_month_segments(date(2020, 12, 30), date(2021, 1, 2)))
    assert [(s.isoformat(), a.isoformat(), b.isoformat()) for s, a, b in segs] == [
        ("2020-12-01", "2020-12-30", "2020-12-31"),
        ("2021-01-01", "2021-01-01", "2021-01-02"),
    ]
    assert _shift_year(date(2020, 2, 29), 2021) == date(2021, 2, 28)

    # QA=3: land + valid. QA=1: land but insufficient data. QA=7: land + valid,
    # but shortened mapping period. An exact burn remains evidence; a shortened
    # unburned pixel is not counted as a confirmed no-burn observation.
    burn = np.array([[150, 0, 0, -1, 152]], dtype=np.int16)
    qa = np.array([[3, 3, 1, 3, 7]], dtype=np.uint8)
    cell_ids = np.array([[101, 101, 101, 101, 102]], dtype=np.int64)
    first = np.array([[1, 1, 1, 1, 140]], dtype=np.int16)
    last = np.array([[366, 366, 366, 366, 160]], dtype=np.int16)
    stats = _summarize_arrays(burn, qa, cell_ids, 149, 153, 0.21, first, last)
    assert stats["land_pixel_observations"] == 5
    assert stats["valid_land_pixel_observations"] == 3  # QA=1 and BurnDate=-1 are not usable
    assert stats["burned_pixels"] == 2
    assert stats["burned_area_km2"] == 0.42
    assert stats["burned_cell_ids"] == {101, 102}
    assert _coverage_fraction(stats) == 3 / 5
    assert _period_status({**stats, "complete_coverage": False, "missing_tile_count": 0, "files_read": 1}, .8) == "BURNED_AREA_DETECTED_PARTIAL_COVERAGE"
    zero = _summarize_arrays(np.array([[0, 0]], dtype=np.int16), np.array([[3, 3]], dtype=np.uint8),
                             np.array([[101, 101]], dtype=np.int64), 149, 153, .21)
    zero.update({"complete_coverage": True, "missing_tile_count": 0, "files_read": 1})
    assert _period_status(zero, .8) == "NO_BURN_IN_VALID_COVERAGE"
    missing = {"burned_pixels": 0, "missing_tile_count": 1, "files_read": 0,
               "land_pixel_observations": 0, "valid_land_pixel_observations": 0}
    assert _period_status(missing, .8) == "MCD64_FILES_MISSING"
    assert _period_status({"burned_pixels": 0, "missing_tile_count": 0, "files_read": 1,
                           "land_pixel_observations": 100, "valid_land_pixel_observations": 50,
                           "complete_coverage": False}, .8) == "INCONCLUSIVE_LOW_COVERAGE"
    assert _empirical_percentile(2.0, [0.0, 1.0, 2.0, 3.0]) == 62.5

    # The calendar can be much larger than the event file; exercise the chunked cell lookup.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="pyroscope_event_cells_") as td:
        cells_path = Path(td) / "calendar_cells.csv"
        pd.DataFrame([
            {"date": "2020-01-01", "sensor_family": "VIIRS", "cell_id": 1000001,
             "cell_lat": 0.25, "cell_lon": 0.75},
            {"date": "2020-01-02", "sensor_family": "MODIS", "cell_id": 1000002,
             "cell_lat": 0.25, "cell_lon": 1.25},
            {"date": "2020-01-03", "sensor_family": "VIIRS", "cell_id": 1000003,
             "cell_lat": 0.25, "cell_lon": 1.75},
            {"date": "2020-01-04", "sensor_family": "VIIRS", "cell_id": 1000001,
             "cell_lat": 0.25, "cell_lon": 0.75},
        ]).to_csv(cells_path, index=False)
        found, centres = _load_event_cells(cells_path, pd.Timestamp("2020-01-03"),
                                           pd.Timestamp("2020-01-04"), 0.5, chunksize=1)
        assert found == {1000001, 1000003} and centres[1000003] == (1.75, 0.25)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate PyroScope anomaly events against MODIS MCD64A1 burned-area data"
    )
    parser.add_argument("--mcd64-dir", required=True, help="Root directory containing MCD64A1 HDF files (searched recursively)")
    parser.add_argument("--events-csv", default=str(PROCESSED_DIR / "anomaly_events.csv"),
                        help="PyroScope anomaly_events.csv (default: data/processed/anomaly_events.csv)")
    parser.add_argument("--cells-csv", default=str(PROCESSED_DIR / "calendar_cells.csv"),
                        help="PyroScope calendar_cells.csv (default: data/processed/calendar_cells.csv)")
    parser.add_argument("--out-csv", default=str(PROCESSED_DIR / "external_validation.csv"))
    parser.add_argument("--out-json", default=str(PROCESSED_DIR / "external_validation.json"))
    parser.add_argument("--cell-deg", type=float, default=CELL_DEG, help=f"PyroScope calendar cell size in degrees (default: {CELL_DEG:g})")
    parser.add_argument("--min-coverage", type=float, default=0.80,
                        help="Minimum valid land-pixel coverage required to call an observed no-burn result confirmed (default: 0.80)")
    parser.add_argument("--collection", default="061", help="MCD64A1 collection code to use (default: 061)")
    args = parser.parse_args(argv)
    events_path = Path(args.events_csv)
    if not events_path.exists():
        parser.error(f"Events file not found: {events_path}. Run `python -m src.anomaly` first.")
    cells_path = Path(args.cells_csv)
    if not cells_path.exists():
        parser.error(f"Cell calendar not found: {cells_path}. Run `python -m src.harmonize` first.")
    try:
        events = pd.read_csv(events_path)
        result, report = validate_events(
            events, args.mcd64_dir, cells_path, args.out_csv, args.out_json,
            args.cell_deg, args.min_coverage, args.collection,
        )
    except Exception as exc:
        print(f"External validation failed: {exc}", file=sys.stderr)
        return 2
    print(f"External validation wrote {args.out_csv} and {args.out_json}")
    if len(result):
        print(result[["event_id", "burned_area_km2", "baseline_median_km2", "baseline_percentile", "coverage_fraction", "status"]].to_string(index=False))
    print(json.dumps({"events": report["events"], "status_counts": report["status_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
