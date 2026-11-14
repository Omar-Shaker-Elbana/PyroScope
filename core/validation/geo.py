"""core.validation.geo
=======================
A small cell_id -> country lookup used by core.forecast.harmonized_country_data
(training, step 4). core.validation.confidence already has its own copy of this
same join for backtesting (_cell_country_map / _country_borders) -- it is
intentionally NOT refactored to use this module here, since that would mean
editing confidence.py's exact text sight-unseen. Both copies do the same thing;
consolidate them yourself later if you want, once you can diff them directly.
"""
from __future__ import annotations

import warnings

import pandas as pd

_CELL_COUNTRY: dict[int, str] = {}


def cell_country_map(cells: pd.DataFrame, countries) -> pd.Series:
    """cell_id -> country code (countries["country"]) for every row in `cells`
    (needs cell_id, cell_lat, cell_lon columns). Resolved cell_ids are cached
    for the life of the process, so repeated calls over a growing calendar
    stay cheap. Cells just offshore are snapped to the nearest coast (<= 0.3 deg).
    `countries` is a GeoDataFrame with ("country", "geometry") columns, i.e.
    whatever core.forecast.legacy_hotsync_data.load_countries() returns.
    """
    unseen = cells.loc[~cells["cell_id"].isin(_CELL_COUNTRY), ["cell_id", "cell_lat", "cell_lon"]]
    unseen = unseen.drop_duplicates("cell_id")
    if len(unseen):
        import geopandas as gpd

        pts = gpd.GeoDataFrame(
            {"cell_id": unseen["cell_id"].to_numpy()},
            geometry=gpd.points_from_xy(unseen["cell_lon"], unseen["cell_lat"]),
            crs="EPSG:4326",
        )
        ref = countries[["country", "geometry"]]
        j = gpd.sjoin(pts, ref, how="left", predicate="within").drop_duplicates("cell_id")
        miss = j["country"].isna()
        if miss.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                near = gpd.sjoin_nearest(
                    pts[pts["cell_id"].isin(j.loc[miss, "cell_id"])], ref, how="left", max_distance=0.3
                ).drop_duplicates("cell_id")
            j = j.set_index("cell_id")
            j.loc[near["cell_id"].to_numpy(), "country"] = near["country"].to_numpy()
            j = j.reset_index()
        for cid, country in zip(j["cell_id"], j["country"]):
            if pd.notna(country):
                _CELL_COUNTRY[int(cid)] = country

    return cells["cell_id"].map(_CELL_COUNTRY)
