"""Time-varying country signals: momentum, dividend yield vs bonds, tourism surprise.

Each returns a DataFrame indexed by the date the value is known (month end, or the
assumed publication date for annual data), with one column per country. Higher =
more attractive; signs are fixed here, before testing.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from strategy.data import load_bond_yields, load_dividends
from strategy.factors import _read_estat
from strategy.signals import DATA
from strategy.universe import REAL_ESTATE


def _by_country(per_ticker, countries):
    """Median across each country's instruments."""
    return pd.DataFrame({
        c: per_ticker.reindex(columns=REAL_ESTATE[c]).median(axis=1) for c in countries
    })


def country_momentum(prices, countries):
    """12-1 month return of each country's listed real estate."""
    monthly = prices.ffill().resample("ME").last()
    mom = monthly.shift(1) / monthly.shift(12) - 1
    return _by_country(mom, countries)


def yield_spread(countries, start="2010-01-01"):
    """Trailing 12-month dividend yield minus the 10-year government bond yield.

    Bond yields are monthly averages, so the value for month m is used from the end of
    month m+1.
    """
    tickers = [t for c in countries for t in REAL_ESTATE[c]]
    close, divs = load_dividends(tickers, start)
    ttm = divs.rolling("365D").sum()
    # Cap at 15%: one-off special distributions (or bad data, e.g. Citycon in 2026)
    # would otherwise swamp the cross-section.
    dy = (ttm / close.ffill()).resample("ME").last().clip(upper=0.15)
    # Need a full year of dividend history before the yield means anything.
    dy = dy[dy.index >= dy.index[0] + pd.DateOffset(years=1)]
    bonds = load_bond_yields(countries).shift(1) / 100
    # Latest months may not be published yet: carry the last known yield forward.
    bonds = bonds.reindex(bonds.index.union(dy.index)).ffill().reindex(dy.index)
    return _by_country(dy, countries) - bonds


# --- Tourism surprise -------------------------------------------------------------

# XGBoost settings from etrace's tourism model (XGB_v2.ipynb).
XGB_PARAMS = {
    "objective": "reg:squarederror", "tree_method": "hist", "learning_rate": 0.1,
    "max_depth": 11, "min_child_weight": 16, "subsample": 0.77, "colsample_bytree": 0.96,
    "lambda": 0.07, "alpha": 8.48, "random_state": 42, "verbosity": 0,
}
N_ROUNDS = 300
FIRST_YEAR = 2009  # first year predicted; trained on 2000-2008


def _nuts2(name, **dims):
    df = _read_estat(name)
    for k, v in dims.items():
        df = df[df[k] == v]
    df = df[df.geo.str.len() == 4].set_index("geo").select_dtypes("number")
    return df.stack().rename_axis(["geo", "year"])


def tourism_panel():
    """NUTS2 x year panel with etrace's features and actual nights spent, 2000-2024.

    GDP per head is lagged one year, since regional GDP comes out well after the
    tourism data. Climate shares come from the smoothed Koppen history and are held
    at their 2020 values afterwards.
    """
    panel = pd.DataFrame({
        "nights": _nuts2("estat_tour_occ_nin2.tsv", c_resid="TOTAL", unit="NR", nace_r2="I551-I553"),
        "pop": _nuts2("estat_demo_r_d2jan.tsv", sex="T", age="TOTAL"),
        "employment_rate": _nuts2("estat_lfst_r_lfe2emprt.tsv", sex="T", age="Y20-64"),
        "gdp_capita": _nuts2("estat_nama_10r_2gdp.tsv", unit="EUR_HAB")
        .groupby(level="geo").shift(1),
    }).reset_index()
    panel = panel[panel.year >= 2000]

    area = pd.read_csv(DATA / "nuts2_areas.csv").set_index("NUTS_ID").area_km2
    panel["area_km2"] = panel.geo.map(area)
    panel["pop_dens"] = panel["pop"] / panel.area_km2

    hist = pd.read_csv(DATA / "NUTS2_Climate_History_Smooth.csv").rename(
        columns={"NUTS_ID": "geo", "YEAR": "year"})
    panel["climate_year"] = panel.year.clip(upper=hist.year.max())
    panel = panel.merge(hist.rename(columns={"year": "climate_year"}), on=["geo", "climate_year"])
    return panel.drop(columns="climate_year").dropna(subset=["nights", "area_km2"])


def tourism_predictions(panel):
    """Walk-forward: each year is predicted by a model trained only on earlier years."""
    features = [c for c in panel.columns if c not in ("geo", "year", "nights")]
    out = []
    for year in range(FIRST_YEAR, panel.year.max() + 1):
        train, test = panel[panel.year < year], panel[panel.year == year]
        model = xgb.train(XGB_PARAMS, xgb.DMatrix(train[features], np.log1p(train.nights)), N_ROUNDS)
        pred = np.expm1(model.predict(xgb.DMatrix(test[features])))
        out.append(test[["geo", "year", "nights"]].assign(predicted=pred))
    return pd.concat(out)


def tourism_surprise(countries, publication_month=7):
    """Change in how much a country's actual tourist nights beat etrace's prediction.

    surprise(Y) = gap(Y) - gap(Y-1), with gap = log(actual / predicted), summed over
    regions reported in both years. Positive = tourism demand improving faster than the
    model's fundamentals (climate, population, income) explain. Year Y's annual data is
    used from July of Y+1.
    """
    preds = tourism_predictions(tourism_panel())
    preds["country"] = preds.geo.str[:2]
    preds = preds[preds.country.isin(countries)]

    rows = {}
    for year in range(FIRST_YEAR + 1, preds.year.max() + 1):
        cur = preds[preds.year == year].set_index("geo")
        prev = preds[preds.year == year - 1].set_index("geo")
        both = cur.index.intersection(prev.index)
        cur, prev = cur.loc[both], prev.loc[both]

        def gap(df):
            sums = df.groupby("country")[["nights", "predicted"]].sum()
            return np.log(sums.nights / sums.predicted)

        rows[pd.Timestamp(f"{year + 1}-{publication_month:02d}-01")] = gap(cur) - gap(prev)
    return pd.DataFrame(rows).T.reindex(columns=countries)


SIGNALS = {
    "momentum": "12-1 month return of the country's listed real estate (+)",
    "yield_spread": "dividend yield minus 10y government bond yield (+)",
    "tourism_surprise": "change in actual vs etrace-predicted tourist nights (+)",
}
