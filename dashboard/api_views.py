
import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from core.src.config import PROCESSED_DIR, OUTPUTS_DIR


# ---------------------------------------------------------
# DATA ACCESS
# ---------------------------------------------------------

@lru_cache(maxsize=12)
def _cached_csv(path, modified_ns, file_size):
    """Cache CSV files until their modification time or size changes."""
    return pd.read_csv(path)


def read_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path.name)

    stat = path.stat()

    return _cached_csv(
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
    ).copy()


def read_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path.name)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def error_response(message, status=400, **extra):
    return JsonResponse(
        {
            "success": False,
            "error": message,
            **extra,
        },
        status=status,
    )


def records_to_json(frame):
    """Convert a DataFrame to JSON-safe dictionaries.
    
    Pandas converts NaN and NaT to JSON null.
    """
    if frame.empty:
        return []

    return json.loads(
        frame.to_json(
            orient="records",
            date_format="iso",
        )
    )


def number_or_none(value):
    if pd.isna(value):
        return None

    return float(value)


# ---------------------------------------------------------
# COMMON DATASET LOADING
# ---------------------------------------------------------

CELL_COLUMNS = {
    "date",
    "sensor_family",
    "source_satellite",
    "cell_id",
    "cell_lat",
    "cell_lon",
    "detections",
    "frp_mw",
    "frp_viirs_equiv_mw",
}


def load_cell_data():
    cells = read_csv(PROCESSED_DIR / "calendar_cells.csv")
    mapping = read_json(PROCESSED_DIR / "frp_mapping.json")

    missing = CELL_COLUMNS.difference(cells.columns)

    if missing:
        raise ValueError(
            f"calendar_cells.csv is missing columns: {sorted(missing)}"
        )

    cells["date"] = pd.to_datetime(
        cells["date"],
        errors="coerce",
    )

    cells["sensor_family"] = (
        cells["sensor_family"].astype(str).str.upper()
    )

    cells["source_satellite"] = (
        cells["source_satellite"].astype(str)
    )

    for column in [
        "cell_lat",
        "cell_lon",
        "detections",
        "frp_mw",
        "frp_viirs_equiv_mw",
    ]:
        cells[column] = pd.to_numeric(
            cells[column],
            errors="coerce",
        )

    cells = cells.dropna(
        subset=["date", "cell_lat", "cell_lon", "cell_id"]
    )

    reference_satellite = mapping.get(
        "reference_viirs_satellite"
    )

    if not reference_satellite:
        raise ValueError(
            "frp_mapping.json does not specify "
            "reference_viirs_satellite."
        )

    return cells, mapping, str(reference_satellite)


def apply_date_filter(frame, request):
    start = request.GET.get("start")
    end = request.GET.get("end")

    try:
        if start:
            start = pd.Timestamp(start)
            if start.tzinfo is not None:
                raise ValueError("Use UTC dates without timezone offsets.")

        if end:
            end = pd.Timestamp(end)
            if end.tzinfo is not None:
                raise ValueError("Use UTC dates without timezone offsets.")

    except (ValueError, TypeError):
        raise ValueError(
            "Invalid date. Use YYYY-MM-DD for start and end."
        )

    if start is not None and end is not None and start > end:
        raise ValueError("start cannot be later than end.")

    if start is not None:
        frame = frame[frame["date"] >= start]

    if end is not None:
        frame = frame[frame["date"] <= end]

    return frame


def get_modis_trust(mapping):
    satellites = mapping.get("modis_satellites", {})

    result = {}

    for satellite, info in satellites.items():
        result[satellite] = {
            "k": info.get("k"),
            "k_trust": info.get("k_trust"),
            "k_ci95": info.get("k_ci95"),
            "scale_basis": info.get("scale_basis"),
            "k_trust_problems": info.get("k_trust_problems", []),
        }

    return result


# ---------------------------------------------------------
# 1. API STATUS
# ---------------------------------------------------------

@require_GET
def api_status(request):
    artifacts = {
        "calendar_cells": PROCESSED_DIR / "calendar_cells.csv",
        "calendar_daily": PROCESSED_DIR / "calendar_daily.csv",
        "frp_mapping": PROCESSED_DIR / "frp_mapping.json",
        "anomalies_daily": PROCESSED_DIR / "anomalies_daily.csv",
        "anomaly_events": PROCESSED_DIR / "anomaly_events.csv",
        "critical_windows": PROCESSED_DIR / "critical_windows.csv",
        "anomaly_report": PROCESSED_DIR / "anomaly_report.json",
        "forecast": OUTPUTS_DIR / "forecast_by_country.csv",
        "recent_observations": OUTPUTS_DIR / "recent_by_country.csv",
    }

    available = {
        name: {
            "available": path.exists(),
            "filename": path.name,
        }
        for name, path in artifacts.items()
    }

    return JsonResponse({
        "success": True,
        "status": "ready" if (
            available["calendar_cells"]["available"]
            and available["frp_mapping"]["available"]
        ) else "incomplete",
        "artifacts": available,
    })


# ---------------------------------------------------------
# 2. GLOBAL MAP API
# ---------------------------------------------------------

@require_GET
def api_map(request):
    try:
        cells, mapping, reference_satellite = load_cell_data()

        grid_deg = float(request.GET.get("grid_deg", 5))

        if not 0.5 <= grid_deg <= 20:
            return error_response(
                "grid_deg must be between 0.5 and 20."
            )

        cells = cells[
            (cells["sensor_family"] == "VIIRS")
            & (
                cells["source_satellite"]
                == reference_satellite
            )
        ].copy()

        cells = apply_date_filter(cells, request)

        if cells.empty:
            return JsonResponse({
                "success": True,
                "reference_satellite": reference_satellite,
                "grid_deg": grid_deg,
                "period": None,
                "summary": {
                    "detections": 0,
                    "active_cell_days": 0,
                    "frp_mw": 0,
                },
                "cells": [],
                "note": "No detection records exist for this period.",
            })

        cells["lat_bin"] = (
            (cells["cell_lat"] + 90) / grid_deg
        ).apply(math.floor)

        cells["lon_bin"] = (
            (cells["cell_lon"] + 180) / grid_deg
        ).apply(math.floor)

        grouped = (
            cells.groupby(
                ["lat_bin", "lon_bin"],
                as_index=False,
            )
            .agg(
                detections=("detections", "sum"),
                active_cell_days=("cell_id", "size"),
                active_grid_cells=("cell_id", "nunique"),
                frp_mw=("frp_mw", "sum"),
            )
        )

        grouped["lat"] = (
            -90 + (grouped["lat_bin"] + 0.5) * grid_deg
        )

        grouped["lon"] = (
            -180 + (grouped["lon_bin"] + 0.5) * grid_deg
        )

        grouped = grouped.sort_values(
            "active_cell_days",
            ascending=False,
        )

        return JsonResponse({
            "success": True,
            "reference_satellite": reference_satellite,
            "grid_deg": grid_deg,

            "period": {
                "start": cells["date"].min().strftime("%Y-%m-%d"),
                "end": cells["date"].max().strftime("%Y-%m-%d"),
            },

            "summary": {
                "detections": int(cells["detections"].sum()),
                "active_cell_days": int(len(cells)),
                "frp_mw": number_or_none(
                cells["frp_mw"].sum(min_count=1)
            ),
            },

            "cells": records_to_json(
                grouped[
                    [
                        "lat",
                        "lon",
                        "detections",
                        "active_cell_days",
                        "active_grid_cells",
                        "frp_mw",
                    ]
                ]
            ),
        })

    except FileNotFoundError as exc:
        return error_response(
            f"Required pipeline output is missing: {exc}",
            status=503,
        )

    except ValueError as exc:
        return error_response(str(exc))

    except Exception:
        return error_response(
            "An unexpected error occurred while generating the map.",
            status=500,
        )


# ---------------------------------------------------------
# 3. REGIONAL ANALYSIS API
# ---------------------------------------------------------

def parse_bounds(request):
    required = ["west", "south", "east", "north"]

    try:
        bounds = {
            key: float(request.GET[key])
            for key in required
        }
    except (KeyError, ValueError, TypeError):
        raise ValueError(
            "Provide west, south, east and north coordinates."
        )

    west = bounds["west"]
    south = bounds["south"]
    east = bounds["east"]
    north = bounds["north"]

    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Longitude must be between -180 and 180.")

    if not (-90 <= south < north <= 90):
        raise ValueError("Invalid latitude boundaries.")

    return bounds


def filter_bounds(frame, bounds):
    west = bounds["west"]
    south = bounds["south"]
    east = bounds["east"]
    north = bounds["north"]

    lat_mask = (
        (frame["cell_lat"] >= south)
        & (frame["cell_lat"] <= north)
    )

    # Support areas crossing the international date line.
    if west <= east:
        lon_mask = (
            (frame["cell_lon"] >= west)
            & (frame["cell_lon"] <= east)
        )
    else:
        lon_mask = (
            (frame["cell_lon"] >= west)
            | (frame["cell_lon"] <= east)
        )

    return frame[lat_mask & lon_mask].copy()


def daily_sensor_summary(frame, include_calibrated=False):
    if frame.empty:
        return pd.DataFrame()

    grouped = frame.groupby("date")

    result = grouped.agg(
    detections=("detections", "sum"),
    active_cells=("cell_id", "nunique"),
    )

    result["frp_mw"] = grouped["frp_mw"].sum(min_count=1)

    if include_calibrated:
        result["frp_viirs_equiv_mw"] = (
            grouped["frp_viirs_equiv_mw"].sum(min_count=1)
        )

    return result


@require_GET
def api_region(request):
    try:
        bounds = parse_bounds(request)

        cells, mapping, reference_satellite = load_cell_data()

        cells = apply_date_filter(cells, request)
        region = filter_bounds(cells, bounds)

        viirs = region[
            (region["sensor_family"] == "VIIRS")
            & (
                region["source_satellite"]
                == reference_satellite
            )
        ].copy()

        modis = region[
            region["sensor_family"] == "MODIS"
        ].copy()

        viirs_daily = daily_sensor_summary(viirs)

        modis_daily = daily_sensor_summary(
            modis,
            include_calibrated=True,
        )

        # Count distinct active spatial cells per day across
        # the available MODIS and reference-VIIRS records.
        active = pd.concat(
    [
        modis[["date", "cell_id"]],
        viirs[["date", "cell_id"]],
    ],
    ignore_index=True,
).drop_duplicates()

        if active.empty:
            union_daily = pd.DataFrame()
        else:
            union_daily = (
                active.groupby("date")
                .agg(
                    unique_active_grid_cells=(
                        "cell_id",
                        "nunique",
                    )
                )
            )

        daily = pd.DataFrame()

        if not viirs_daily.empty:
            daily = viirs_daily.rename(columns={
                "detections": "viirs_detections",
                "active_cells": "viirs_active_cells",
                "frp_mw": "viirs_frp_mw",
            })

        if not modis_daily.empty:
            modis_daily = modis_daily.rename(columns={
                "detections": "modis_detections",
                "active_cells": "modis_active_cells",
                "frp_mw": "modis_frp_mw",
                "frp_viirs_equiv_mw": (
                    "modis_frp_viirs_equiv_mw"
                ),
            })

            daily = daily.join(
                modis_daily,
                how="outer",
            )

        if not union_daily.empty:
            daily = daily.join(
                union_daily,
                how="outer",
            )

        daily = daily.sort_index().reset_index()

        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

        calibrated_frp = pd.to_numeric(
            modis.get(
                "frp_viirs_equiv_mw",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )

        summary = {
            "recorded_modis_detections": int(
                modis["detections"].sum()
            ) if not modis.empty else 0,

            "recorded_viirs_detections": int(
                viirs["detections"].sum()
            ) if not viirs.empty else 0,

            "modis_active_cell_days": int(
                len(modis[["date", "cell_id"]].drop_duplicates())
            ),

            "viirs_active_cell_days": int(
                len(viirs[["date", "cell_id"]].drop_duplicates())
            ),

            "unique_active_cell_days": int(len(active)),

            "viirs_frp_mw": number_or_none(
                viirs["frp_mw"].sum()
            ) if not viirs.empty else None,

            "modis_frp_viirs_equiv_mw": (
                number_or_none(calibrated_frp.sum(min_count=1))
                if calibrated_frp.notna().any()
                else None
            ),

            "days_with_detection_records": int(len(daily)),
        }

        return JsonResponse({
            "success": True,

            "region": bounds,

            "reference_satellite": reference_satellite,

            "summary": summary,

            "calibration": get_modis_trust(mapping),

            "daily": records_to_json(daily),

            "notes": [
                "VIIRS is reported on its reference scale.",
                "MODIS calibrated FRP is included only where a valid scale exists.",
                "Missing dates are not interpreted as zero-fire days.",
                "Cell-centre filtering is an approximation of the requested geographical boundary.",
                "Unique active cell-days count the spatial union of available sensor records.",
            ],
        })

    except FileNotFoundError as exc:
        return error_response(
            f"Required pipeline output is missing: {exc}",
            status=503,
        )

    except ValueError as exc:
        return error_response(str(exc))

    except Exception:
        return error_response(
            "An unexpected error occurred during regional analysis.",
            status=500,
        )


# ---------------------------------------------------------
# 4. COUNTRY FORECAST API
# ---------------------------------------------------------

@require_GET
def api_forecast(request):
    try:
        forecasts = read_csv(
            OUTPUTS_DIR / "forecast_by_country.csv"
        )

        required = {
            "date",
            "horizon_days",
            "predicted",
            "low_p10",
            "high_p90",
        }

        missing = required.difference(forecasts.columns)

        if missing:
            raise ValueError(
                f"Forecast file is missing columns: {sorted(missing)}"
            )

        forecasts["date"] = pd.to_datetime(
            forecasts["date"],
            errors="coerce",
        )

        forecasts = forecasts.dropna(subset=["date"])

        country = request.GET.get("country", "").strip()

        if country:
            candidates = [
                col for col in ["country", "name", "iso3", "code"]
                if col in forecasts.columns
            ]

            if not candidates:
                raise ValueError(
                    "The forecast file has no country identifier column."
                )

            mask = pd.Series(False, index=forecasts.index)

            for column in candidates:
                mask |= (
                    forecasts[column]
                    .astype(str)
                    .str.casefold()
                    == country.casefold()
                )

            forecasts = forecasts[mask]

        limit = int(request.GET.get("limit", 5000))

        if not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10000.")

        forecasts = forecasts.sort_values(
            ["date", "horizon_days"]
        ).head(limit)

        forecasts["date"] = (
            forecasts["date"].dt.strftime("%Y-%m-%d")
        )

        # Recent observations are optional.
        recent_path = OUTPUTS_DIR / "recent_by_country.csv"

        if recent_path.exists():
            recent = read_csv(recent_path)

            if country:
                for column in ["country", "name", "iso3", "code"]:
                    if column in recent.columns:
                        recent = recent[
                            recent[column]
                            .astype(str)
                            .str.casefold()
                            == country.casefold()
                        ]
                        break

            if "date" in recent.columns:
                recent["date"] = pd.to_datetime(
                    recent["date"],
                    errors="coerce",
                ).dt.strftime("%Y-%m-%d")

            recent_records = records_to_json(
                recent.tail(2000)
            )
        else:
            recent_records = []

        return JsonResponse({
            "success": True,

            "country_filter": country or None,

            "count": len(forecasts),

            "forecasts": records_to_json(forecasts),

            "recent_observations": recent_records,
        })

    except FileNotFoundError as exc:
        return error_response(
            f"Required forecast output is missing: {exc}",
            status=503,
        )

    except ValueError as exc:
        return error_response(str(exc))

    except Exception:
        return error_response(
            "An unexpected error occurred while loading forecasts.",
            status=500,
        )
