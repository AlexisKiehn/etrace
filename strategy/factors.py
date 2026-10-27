"""Candidate country factors to blend with the climate score.

Each factor returns either a static Series (country -> value) or a point-in-time
DataFrame (rows = date the value became public, columns = countries). Annual
Eurostat data for year Y is treated as public from 1 January of Y+2, which is
conservative for national accounts and regional population.

Every factor has a sign fixed before testing (FACTORS below), so that "higher z-score
= go long" without choosing the direction from backtest results.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from strategy.signals import COLD, DATA, HOT_ARID

ROOT = DATA.parent
CACHE = Path(__file__).resolve().parent / "cache"
PUBLICATION_LAG = 2  # years between end of data year and use in the backtest

# Land borders with countries outside the NUTS file (Russia, Belarus, Ukraine).
# They would otherwise be counted as coastline. Source: CIA World Factbook, km.
MISSING_LAND_BORDERS = {"FI": 1309, "NO": 191, "PL": 209 + 375 + 498}


def _read_estat(name):
    df = pd.read_csv(DATA / name, sep="\t", dtype=str)
    key = df.columns[0]
    idx = df[key].str.split(",", expand=True)
    idx.columns = key.split("\\")[0].split(",")
    vals = df.drop(columns=key)
    vals.columns = [int(c.strip()) for c in vals.columns]
    vals = vals.apply(lambda s: pd.to_numeric(s.str.extract(r"([-\d.]+)")[0], errors="coerce"))
    return pd.concat([idx, vals], axis=1)


def _annual(name, countries, **dims):
    """Country x year table from a Eurostat TSV, filtered on the given dimensions."""
    df = _read_estat(name)
    for k, v in dims.items():
        df = df[df[k] == v]
    return df[df.geo.isin(countries)].set_index("geo").select_dtypes("number").T.sort_index()


def _point_in_time(annual):
    """Re-index a year x country table by the date each year becomes public."""
    out = annual.ffill().copy()
    out.index = pd.to_datetime([f"{y + PUBLICATION_LAG}-01-01" for y in out.index])
    return out


def _population(countries):
    return _annual("estat_demo_r_d2jan.tsv", countries, sex="T", age="TOTAL")


def gdp_level(countries):
    """GDP per head in purchasing power standards (log)."""
    pps = _annual("estat_nama_10r_2gdp.tsv", countries, unit="MIO_PPS_EU27_2020") * 1e6
    return _point_in_time(np.log(pps / _population(countries)))


def gdp_growth(countries, years=3):
    """Annualised growth of nominal GDP per head in national currency."""
    nac = _annual("estat_nama_10r_2gdp.tsv", countries, unit="MIO_NAC")
    per_head = np.log(nac / _population(countries)).ffill()
    return _point_in_time(per_head.diff(years) / years)


def pop_growth(countries, years=3):
    """Annualised population growth, a proxy for housing demand."""
    return _point_in_time(np.log(_population(countries)).diff(years) / years)


def coastline_km(countries):
    """Km of coastline per country: boundary not shared with another country.

    Slow (geometry ops on the 1:1M file), so cached in strategy/cache.
    """
    path = CACHE / "coastline_km.csv"
    if path.exists():
        cached = pd.read_csv(path, index_col=0).iloc[:, 0]
        if set(countries) <= set(cached.index):
            return cached[countries]
    g = gpd.read_file(ROOT / "NUTS_RG_01M_2024_3035_LEVL_2.geojson")
    shapes = g.dissolve("CNTR_CODE").geometry
    coast = {}
    for c in countries:
        others = shapes.drop(c).union_all().buffer(100)  # 100 m tolerance for shared borders
        boundary = shapes[c].boundary
        km = (boundary.length - boundary.intersection(others).length) / 1000
        km -= MISSING_LAND_BORDERS.get(c, 0)
        coast[c] = km if km > 50 else 0.0  # landlocked: leftover is digitising noise
    coast = pd.Series(coast, name="coastline_km")
    CACHE.mkdir(exist_ok=True)
    coast.to_csv(path)
    return coast


def coastline_per_capita(countries):
    """log(1 + km of coastline per million inhabitants); 0 for landlocked countries."""
    pop = _population(countries).loc[2020]
    return np.log1p(coastline_km(countries) / pop * 1e6).rename("coastline_per_capita")


def coastal_pop_share(countries):
    """Share of population living in NUTS2 regions classified as coastal."""
    df = pd.read_csv(DATA / "cleaned_final_interpolated_dataset.csv")
    df = df[df.year == df.year.max()].drop_duplicates("geo")
    df["country"] = df.geo.str[:2]
    df = df[df.country.isin(countries)]
    return df.groupby("country").apply(
        lambda g: np.average(g.is_coastal, weights=g["pop"]), include_groups=False
    ).rename("coastal_pop_share")


def climate_shift_100y(countries):
    """Observed Koppen shift 1901-1930 -> 1991-2020: cold-climate loss minus hot/arid gain.

    Same construction as signals.koppen_shift, but using observed history instead of
    a projection. Uses the anchor years of the smoothed history file, which equal the
    raw 30-year Koppen-Geiger maps.
    """
    hist = pd.read_csv(DATA / "NUTS2_Climate_History_Smooth.csv")
    then = hist[hist.YEAR == 1915].set_index("NUTS_ID")
    now = hist[hist.YEAR == 2005].set_index("NUTS_ID")

    def share(df, cols):
        return df.reindex(columns=cols).fillna(0).sum(axis=1)

    score = (share(then, COLD) - share(now, COLD)) - (share(now, HOT_ARID) - share(then, HOT_ARID))
    pop = pd.read_csv(DATA / "cleaned_final_interpolated_dataset.csv")
    pop = pop.dropna(subset=["pop"]).sort_values("year").drop_duplicates("geo", keep="last")
    df = pd.DataFrame({"score": score.dropna(), "pop": pop.set_index("geo")["pop"]}).dropna()
    df["country"] = df.index.str[:2]
    df = df[df.country.isin(countries)]
    return df.groupby("country").apply(
        lambda g: np.average(g.score, weights=g["pop"]), include_groups=False
    ).rename("climate_shift_100y") * 100


# name -> (function, sign, hypothesis). Signs were fixed before running any test.
FACTORS = {
    "gdp_level": (gdp_level, +1, "richer countries adapt better and sustain property demand"),
    "gdp_growth": (gdp_growth, +1, "faster income growth supports rents and valuations"),
    "pop_growth": (pop_growth, +1, "population growth drives housing demand"),
    "coastline_per_capita": (coastline_per_capita, -1, "more coast per head = more sea-level and storm exposure"),
    "coastal_pop_share": (coastal_pop_share, -1, "more people on the coast = more flood exposure"),
    "climate_shift_100y": (climate_shift_100y, +1, "countries whose climate already improved over the last century keep benefiting"),
}


def zscore(x, cap=2.0):
    """Cross-sectional z-score capped at +/-cap, so one outlier (e.g. Finland) can't
    dominate; for a DataFrame, row by row."""
    if isinstance(x, pd.DataFrame):
        z = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1, ddof=0), axis=0)
    else:
        z = (x - x.mean()) / x.std(ddof=0)
    return z.clip(-cap, cap)


def factor_score(name, countries):
    fn, sign, _ = FACTORS[name]
    return sign * zscore(fn(countries))


def blend(parts):
    """Equal-weight average of z-scored parts. Static parts are broadcast over dates."""
    dynamic = [p for p in parts if isinstance(p, pd.DataFrame)]
    if not dynamic:
        return sum(parts) / len(parts)
    index = dynamic[0].index
    for d in dynamic[1:]:
        index = index.union(d.index)
    frames = [
        p.reindex(index).ffill() if isinstance(p, pd.DataFrame)
        else pd.DataFrame([p.values] * len(index), index=index, columns=p.index)
        for p in parts
    ]
    # Average whatever parts are available for each country and date.
    return sum(f.fillna(0) for f in frames) / sum(f.notna() for f in frames)
