"""Regional data for the mechanism tests.

- Temperature: NOAA GHCN-CAMS monthly 2m air temperature, 0.5 degree land grid,
  1948-present, averaged over each region's polygon.
- Boundaries: NUTS 2024 (NUTS2 from the repo, NUTS3 from Eurostat GISCO).
- Nordic statistics offices through their PxWeb APIs (Statistics Sweden, Statistics
  Norway, Statistics Finland, Luke), cached as JSON.
"""

import hashlib
import json
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.windows import Window

from strategy.data import CACHE
from strategy.signals import DATA

TEMP_URL = "https://downloads.psl.noaa.gov/Datasets/ghcncams/air.mon.mean.nc"
NUTS3_URL = ("https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
             "NUTS_RG_10M_2024_4326_LEVL_3.geojson")

# Europe window of the 0.5 degree grid: 25W-45E, 34N-72N.
WEST, EAST, SOUTH, NORTH, RES = -25.0, 45.0, 34.0, 72.0, 0.5
EUROPE = Affine(RES, 0, WEST, 0, -RES, NORTH)

# Statistics Sweden county codes -> NUTS3 (Kalmar and Gotland are reported together).
SE_COUNTIES = {
    "01": ["SE110"], "03": ["SE121"], "04": ["SE122"], "05": ["SE123"], "06": ["SE211"],
    "07": ["SE212"], "08+09": ["SE213", "SE214"], "08": ["SE213"], "09": ["SE214"],
    "10": ["SE221"], "12": ["SE224"], "13": ["SE231"], "14": ["SE232"], "17": ["SE311"],
    "18": ["SE124"], "19": ["SE125"], "20": ["SE312"], "21": ["SE313"], "22": ["SE321"],
    "23": ["SE322"], "24": ["SE331"], "25": ["SE332"],
}
# Statistics Finland / Luke regions (maakunta) -> NUTS3.
FI_REGIONS = {
    "MK01": ["FI1B1"], "MK02": ["FI1C1"], "MK04": ["FI196"], "MK05": ["FI1C2"],
    "MK06": ["FI19B"], "MK07": ["FI1C6"], "MK08": ["FI1C7"], "MK09": ["FI1C5"],
    "MK10": ["FI1DA"], "MK11": ["FI1DB"], "MK12": ["FI1DC"], "MK13": ["FI198"],
    "MK14": ["FI199"], "MK15": ["FI19A"], "MK16": ["FI1D5"], "MK17": ["FI1D9"],
    "MK18": ["FI1D8"], "MK19": ["FI1D7"],
}
# Statistics Norway house price regions -> NUTS3. City regions use their county.
NO_REGIONS = {
    "001": ["NO081"], "002": ["NO0A1"], "003": ["NO0A2"], "004": ["NO060"],
    "005": ["NO084"], "006": ["NO083", "NO085", "NO093", "NO094"], "007": ["NO020"],
    "008": ["NO092", "NO0A1"], "009": ["NO0A3", "NO0A2"], "010": ["NO060"],
    "011": ["NO071", "NO072", "NO073"],
}


# --- Boundaries -------------------------------------------------------------------

def nuts_boundaries(level):
    """NUTS 2024 polygons in lon/lat, indexed by NUTS_ID."""
    if level == 2:
        g = gpd.read_file(DATA.parent / "NUTS_RG_01M_2024_3035_LEVL_2.geojson").to_crs(4326)
    else:
        path = CACHE / "nuts3_2024_10m.geojson"
        if not path.exists():
            gpd.read_file(NUTS3_URL).to_file(path, driver="GeoJSON")
        g = gpd.read_file(path)
    return g.set_index("NUTS_ID")


def merged(boundaries, mapping):
    """One polygon per statistics region, from a region -> [NUTS ids] mapping."""
    return gpd.GeoSeries({r: boundaries.loc[ids].union_all() for r, ids in mapping.items()},
                         crs=boundaries.crs)


# --- Temperature ------------------------------------------------------------------

def _europe_grid():
    """Monthly temperature in deg C for the Europe window: (months, rows, cols), dates."""
    path = CACHE / "ghcn_cams_air_mon_mean.nc"
    if not path.exists():
        urllib.request.urlretrieve(TEMP_URL, path)
    with rasterio.open(f"netcdf:{path}:air") as src:
        hours = json.loads(src.tags()["NETCDF_DIM_time_VALUES"].replace("{", "[").replace("}", "]"))
        rows = slice(int((90 - NORTH) / RES), int((90 - SOUTH) / RES))
        # The file runs 0-360E, so Europe is split across the date line at 0E.
        west = src.read(window=Window.from_slices(rows, (int((360 + WEST) / RES), 720)), masked=True)
        east = src.read(window=Window.from_slices(rows, (0, int(EAST / RES))), masked=True)
    grid = np.ma.concatenate([west, east], axis=2).astype("float64") - 273.15
    dates = pd.Timestamp("1800-01-01") + pd.to_timedelta(hours, unit="h")
    return grid.filled(np.nan), dates + pd.offsets.MonthEnd(0)


def regional_temperature(regions):
    """Monthly mean temperature (deg C) per region, from a GeoSeries of lon/lat polygons.

    Each grid cell whose centre falls inside a region counts towards it; regions too
    small to contain a cell centre use the cell at their centroid.
    """
    grid, dates = _europe_grid()
    n_months, n_rows, n_cols = grid.shape
    values = grid.reshape(n_months, -1)
    cols = {}
    for name, geom in regions.items():
        mask = rasterize([(geom, 1)], out_shape=(n_rows, n_cols), transform=EUROPE, fill=0).ravel() == 1
        if not mask.any():  # smaller than a grid cell: use the cell at its centroid
            c = geom.centroid
            row, col = int((NORTH - c.y) / RES), int((c.x - WEST) / RES)
            if not (0 <= row < n_rows and 0 <= col < n_cols):  # outside Europe (e.g. Canaries)
                cols[name] = np.full(n_months, np.nan)
                continue
            mask[row * n_cols + col] = True
        cols[name] = np.nanmean(values[:, mask], axis=1)
    return pd.DataFrame(cols, index=dates)


def annual_temperature(monthly, months=None):
    """Calendar-year mean of monthly temperatures, optionally only some months (e.g. 6-8)."""
    m = monthly if months is None else monthly[monthly.index.month.isin(months)]
    counts = m.groupby(m.index.year).count()
    need = 12 if months is None else len(months)
    return m.groupby(m.index.year).mean().where(counts == need)


# --- PxWeb (Nordic statistics offices) ----------------------------------------------

def _jsonstat(d):
    cats = []
    for dim in d["id"]:
        idx = d["dimension"][dim]["category"]["index"]
        cats.append(sorted(idx, key=idx.get) if isinstance(idx, dict) else list(idx))
    values = d["value"]
    if isinstance(values, dict):  # sparse encoding
        dense = [None] * int(np.prod(d["size"]))
        for k, v in values.items():
            dense[int(k)] = v
        values = dense
    index = pd.MultiIndex.from_product(cats, names=d["id"])
    return pd.Series(pd.to_numeric(pd.Series(values), errors="coerce").values, index=index)


def pxweb(url, selection, refresh=False):
    """Query a PxWeb table; `selection` maps variable code -> list of values, or "*" for all.

    Returns a Series with one index level per variable.
    """
    key = hashlib.md5((url + json.dumps(selection, sort_keys=True)).encode()).hexdigest()[:12]
    path = CACHE / f"pxweb_{key}.json"
    if not path.exists() or refresh:
        query = {"query": [{"code": c, "selection": {"filter": "all", "values": ["*"]} if v == "*"
                            else {"filter": "item", "values": v}}
                           for c, v in selection.items()],
                 "response": {"format": "json-stat2"}}
        req = urllib.request.Request(url, json.dumps(query).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    return _jsonstat(json.loads(path.read_text(encoding="utf-8-sig")))
