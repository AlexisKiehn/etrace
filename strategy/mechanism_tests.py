"""Does warming actually make cold regions more valuable? Regional tests of the mechanism.

    python -m strategy.mechanism_tests

All comparisons are between regions of the same country (country fixed effects), so
national interest rates, currencies and business cycles cancel out.

1. Tourism, ~280 NUTS2 regions in Europe, 2000-2024 (no 2020-21):
   a. Weather panel: log tourist nights on this and last summer's temperature, with
      region and country-year fixed effects. Thesis: a warmer summer brings more
      tourists to cold regions (and fewer to hot ones).
   b. Long difference: change in log nights 2000-03 -> 2016-19 on summer warming over
      the same period, controlling for population and income growth.
2. House prices, 49 regions in Sweden, Norway and Finland, 2005 -> 2024: price growth
   on baseline temperature and warming, controlling for population growth. Thesis:
   colder regions, and those that warmed more, gained.
3. Forests: Swedish forest-property prices by county (2000-02 -> 2023-25) and Finnish
   stumpage prices by region (2016-17 -> 2024-25), on baseline temperature. Thesis:
   northern forests gain. (Forest growth by region could not be fetched: the Swedish
   forest inventory API was down.)

Temperatures: NOAA GHCN-CAMS, averaged over each region (strategy/regional_data.py).
"""

import numpy as np
import pandas as pd
from scipy import stats

from strategy.data import load_eurostat
from strategy.dynamic_signals import _nuts2
from strategy.regional_data import (FI_REGIONS, NO_REGIONS, SE_COUNTIES, annual_temperature, merged,
                                    nuts_boundaries, pxweb, regional_temperature)
from strategy.run_backtest import OUT

SCB = "https://api.scb.se/OV0104/v1/doris/en/ssd/BO/BO0501/"
SSB = "https://data.ssb.no/api/v0/en/table/"
STATFIN = "https://pxdata.stat.fi/PxWeb/api/v1/en/StatFin/"
LUKE = "https://statdb.luke.fi/PxWeb/api/v1/en/LUKE/"

SUMMER_CENTRE = 18.0  # deg C; interactions are centred here
ANNUAL_CENTRE = 6.0
BASELINE = (1991, 2020)
WARMING = ((1991, 2000), (2015, 2024))


# --- Regression helpers -------------------------------------------------------------

class Fit:
    """OLS with robust (HC1) or cluster-robust standard errors."""

    def __init__(self, y, X, cluster=None):
        Xv, yv = X.to_numpy(float), np.asarray(y, float)
        n, k = Xv.shape
        bread = np.linalg.pinv(Xv.T @ Xv)
        self.beta = bread @ Xv.T @ yv
        e = yv - Xv @ self.beta
        if cluster is None:
            u = Xv * e[:, None]
            meat, self.dof = u.T @ u * n / (n - k), n - k
        else:
            g = pd.factorize(np.asarray(cluster))[0]
            G = g.max() + 1
            u = np.zeros((G, k))
            np.add.at(u, g, Xv * e[:, None])
            meat, self.dof = u.T @ u * G / (G - 1) * (n - 1) / (n - k), G - 1
        self.V = pd.DataFrame(bread @ meat @ bread, X.columns, X.columns)
        self.beta = pd.Series(self.beta, X.columns)
        self.n = n

    def effect(self, weights):
        """Linear combination of coefficients, e.g. {"T": 1, "T x base": -5}."""
        w = pd.Series(weights).reindex(self.beta.index, fill_value=0)
        est, se = w @ self.beta, np.sqrt(w @ self.V @ w)
        p = 2 * stats.t.sf(abs(est / se), self.dof)
        return pd.Series({"estimate": est, "se": se, "t": est / se, "p": p})

    def table(self, names):
        return pd.DataFrame({n: self.effect({n: 1}) for n in names}).T


def demean(df, cols, groups, tol=1e-10, max_iter=500):
    """Remove several sets of fixed effects by alternating projections."""
    out = df[cols].astype(float)
    for _ in range(max_iter):
        before = out
        for g in groups:
            out = out - out.groupby(df[g]).transform("mean")
        if (out - before).abs().to_numpy().max() < tol:
            break
    return out


def country_dummies(country):
    return pd.get_dummies(country, prefix="c", dtype=float)


def print_effects(title, rows):
    print(f"\n{title}")
    print(pd.DataFrame(rows).T[["estimate", "se", "t", "p"]].round(3).to_string())


# --- 1. Tourism -----------------------------------------------------------------------

def tourism_data():
    temps = regional_temperature(nuts_boundaries(2).geometry)
    summer = annual_temperature(temps, [6, 7, 8])
    base = summer.loc[BASELINE[0]:BASELINE[1]].mean()
    panel = pd.DataFrame({
        "nights": _nuts2("estat_tour_occ_nin2.tsv", c_resid="TOTAL", unit="NR", nace_r2="I551-I553"),
        "pop": _nuts2("estat_demo_r_d2jan.tsv", sex="T", age="TOTAL"),
        "gdp": _nuts2("estat_nama_10r_2gdp.tsv", unit="EUR_HAB"),
    }).reset_index()
    summer_long = summer.stack().rename("T").rename_axis(["year", "geo"]).reset_index()
    panel = panel.merge(summer_long, on=["geo", "year"])
    panel["T_lag"] = panel.merge(summer_long.assign(year=summer_long.year + 1),
                                 on=["geo", "year"], how="left", suffixes=("", "_prev"))["T_prev"]
    panel["base"] = panel.geo.map(base)
    panel["country"] = panel.geo.str[:2]
    return panel[(panel.nights > 0) & panel.base.notna()]


def tourism_weather_panel(panel):
    df = panel[panel.year.between(2000, 2024) & ~panel.year.isin([2020, 2021])].dropna(
        subset=["T", "T_lag"]).copy()
    df["ln_nights"] = np.log(df.nights)
    df["T x base"] = df["T"] * (df.base - SUMMER_CENTRE)
    df["T_lag x base"] = df["T_lag"] * (df.base - SUMMER_CENTRE)
    df["country_year"] = df.country + df.year.astype(str)
    cols = ["ln_nights", "T", "T x base", "T_lag", "T_lag x base"]
    dm = demean(df, cols, ["geo", "country_year"])
    fit = Fit(dm.ln_nights, dm[cols[1:]], cluster=df.geo)
    rows = {}
    for label, b in [("cold region (summer 13C, e.g. north Sweden)", 13),
                     ("mild region (summer 17C, e.g. Denmark)", 17),
                     ("warm region (summer 21C, e.g. Bavaria)", 21),
                     ("hot region (summer 25C, e.g. Andalusia)", 25)]:
        d = b - SUMMER_CENTRE
        rows[label] = fit.effect({"T": 1, "T x base": d, "T_lag": 1, "T_lag x base": d}) * [100, 100, 1, 1]
    rows["difference, cold minus hot region"] = fit.effect(
        {"T x base": -12, "T_lag x base": -12}) * [100, 100, 1, 1]
    print_effects(f"1a. Tourist nights after a summer 1C warmer than usual, % "
                  f"(this + next year; {fit.n} region-years, {df.geo.nunique()} regions)", rows)
    return fit


def tourism_long_difference(panel):
    def window(lo, hi):
        w = panel[panel.year.between(lo, hi)]
        g = w.groupby("geo")
        out = pd.DataFrame({"ln_nights": g.nights.apply(lambda s: np.log(s).mean()),
                            "T": g["T"].mean(), "ln_pop": g["pop"].apply(lambda s: np.log(s).mean()),
                            "ln_gdp": g.gdp.apply(lambda s: np.log(s).mean()), "n": g.size()})
        return out[out.n >= 3]

    a, b = window(2000, 2003), window(2016, 2019)
    d = (b - a).replace([np.inf, -np.inf], np.nan).dropna().drop(columns="n")
    d["base"] = panel.groupby("geo").base.first().reindex(d.index)
    d["dT x base"] = d["T"] * (d.base - SUMMER_CENTRE)
    d["country"] = d.index.str[:2]
    X = pd.concat([d[["T", "dT x base", "ln_pop", "ln_gdp"]], country_dummies(d.country)], axis=1)
    fit = Fit(d.ln_nights, X)
    rows = {}
    for label, b in [("cold region (summer 13C)", 13), ("warm region (summer 21C)", 21),
                     ("hot region (summer 25C)", 25)]:
        rows[label] = fit.effect({"T": 1, "dT x base": b - SUMMER_CENTRE}) * [100, 100, 1, 1]
    rows["difference, cold minus hot region"] = fit.effect({"dT x base": -12}) * [100, 100, 1, 1]
    rows["population growth (elasticity)"] = fit.effect({"ln_pop": 1})
    print_effects(f"1b. Growth in tourist nights 2000-03 -> 2016-19 per 1C of summer warming, % "
                  f"({fit.n} regions; mean summer warming {d['T'].mean():.2f}C)", rows)
    return fit


# --- 2. House prices --------------------------------------------------------------------

def region_climate(mapping):
    temps = regional_temperature(merged(nuts_boundaries(3), mapping))
    annual = annual_temperature(temps)
    (a0, a1), (b0, b1) = WARMING
    return pd.DataFrame({"base": annual.loc[BASELINE[0]:BASELINE[1]].mean(),
                         "warming": annual.loc[b0:b1].mean() - annual.loc[a0:a1].mean()})


def population_growth(mapping, start=2005, end=2024):
    """Annualised log population growth per region, from the first year >= start."""
    geos = sorted({g for ids in mapping.values() for g in ids})
    pop = load_eurostat("demo_r_pjanaggr3", "A.NR.T.TOTAL." + "+".join(geos))
    pop.index = pop.index.year
    out = {}
    for region, ids in mapping.items():
        total = pop[ids].loc[start:end].dropna().sum(axis=1)
        total = total[total > 0]
        out[region] = np.log(total.iloc[-1] / total.iloc[0]) / (total.index[-1] - total.index[0])
    return pd.Series(out)


def house_prices(start=2005, end=2024):
    se_map = {k: v for k, v in SE_COUNTIES.items() if k not in ("08", "09")}
    se = pxweb(SCB + "BO0501A/FastpiPSLanAr",
               {"Lan": list(se_map), "ContentsCode": ["BO0501R5"], "Tid": "*"})
    se = se.droplevel("ContentsCode").unstack("Lan")
    no = pxweb(SSB + "07221", {"Region": list(NO_REGIONS), "Boligtype": ["00"],
                               "ContentsCode": ["Boligindeks"], "Tid": "*"})
    no = no.droplevel(["Boligtype", "ContentsCode"]).unstack("Region").replace(0, np.nan)
    no = no.groupby(no.index.str[:4]).mean()  # quarters -> years
    fi = pxweb(STATFIN + "ashi/13mz.px",
               {"alue_43_20220407": list(FI_REGIONS), "talotyyppi_5_20111209": ["0"],
                "huoneluku_1_20111212": ["00"], "contentscode": ["ind00"], "timeperiod_y": "*"})
    fi = fi.unstack("alue_43_20220407").droplevel([1, 2, 3])

    frames = []
    for country, prices, mapping in [("SE", se, se_map), ("NO", no, NO_REGIONS), ("FI", fi, FI_REGIONS)]:
        prices.index = prices.index.astype(int)
        df = region_climate(mapping)
        df["growth"] = np.log(prices.loc[end] / prices.loc[start]) / (end - start)
        df["pop_growth"] = population_growth(mapping, start, end)
        df["country"] = country
        frames.append(df.rename(index=lambda r: f"{country}-{r}"))
    return pd.concat(frames)


def house_price_test(df):
    df = df.dropna().copy()
    df["base_c"] = df.base - ANNUAL_CENTRE
    X = pd.concat([df[["base_c", "warming", "pop_growth"]], country_dummies(df.country)], axis=1)
    fit = Fit(df.growth, X)
    rows = {
        "1C colder baseline": fit.effect({"base_c": -1}) * [100, 100, 1, 1],
        "1C more warming since the 1990s": fit.effect({"warming": 1}) * [100, 100, 1, 1],
        "1 pp faster population growth": fit.effect({"pop_growth": 0.01}) * [100, 100, 1, 1],
    }
    print_effects(f"2. Nordic house prices 2005-2024: extra annual price growth, pp/yr "
                  f"({fit.n} regions; within-country)", rows)
    north = {"SE": ["SE-21", "SE-22", "SE-23", "SE-24", "SE-25"], "NO": ["NO-011"],
             "FI": ["FI-MK17", "FI-MK18", "FI-MK19"]}
    rel = {}
    for c, ids in north.items():
        sub = df[df.country == c]
        rel[c] = {"north": sub.loc[ids, "growth"].mean() * 100,
                  "rest": sub.drop(ids).growth.mean() * 100,
                  "north pop": sub.loc[ids, "pop_growth"].mean() * 100,
                  "rest pop": sub.drop(ids).pop_growth.mean() * 100}
    print("   north = Norrland / Nord-Norge / North Ostrobothnia, Kainuu, Lapland (%/yr):")
    print(pd.DataFrame(rel).T.round(2).to_string())
    return fit


# --- 3. Forests ------------------------------------------------------------------------

def forest_prices():
    mapping = {k: v for k, v in SE_COUNTIES.items() if k != "08+09"}
    raw = pxweb(SCB + "BO0501B/FastprisLBRegSkogAr",
                {"Region": list(mapping), "Skogsbruksvarde": ["67-99", "100"],
                 "ContentsCode": ["BO0501E2", "BO0501E3", "BO0501E5"], "Tid": "*"})
    t = raw.unstack("ContentsCode")
    t.columns = ["number", "price", "coef"]
    t["value"] = t.number * t.price
    t["assessed"] = t.value / t.coef
    by = t.groupby(level=["Region", "Tid"])[["number", "value", "assessed"]].sum()
    by["avg_price"] = by.value / by.number
    by["coef"] = by.value / by.assessed

    def window(col, lo, hi):
        w = by[col].unstack("Region")
        w.index = w.index.astype(int)
        return np.log(w.loc[lo:hi]).mean()

    se = region_climate(mapping)
    se["growth_price"] = (window("avg_price", 2023, 2025) - window("avg_price", 2000, 2002)) / 23
    se["growth_coef"] = (window("coef", 2023, 2025) - window("coef", 2000, 2002)) / 23
    se["sales"] = by.number.groupby(level="Region").mean()

    fi_raw = pxweb(LUKE + "met/teokau/v/0500_teokau.px",
                   {"INFO": ["E_M3"], "A": "*", "MK": list(FI_REGIONS), "KAUP": ["PKAUP"],
                    "PTL": ["TUK_MA", "TUK_KU", "KUI_MA", "KUI_KU"]})
    fi = fi_raw.droplevel(["INFO", "KAUP"]).unstack("A")
    fi.columns = fi.columns.astype(int)
    growth = (np.log(fi[[2024, 2025]]).mean(axis=1) - np.log(fi[[2016, 2017]]).mean(axis=1)) / 8.5
    fi_reg = region_climate(FI_REGIONS)
    fi_reg["growth"] = growth.groupby(level="MK").mean()  # average over the four assortments
    return se, fi_reg


def forest_test(se, fi):
    rows = {}
    for label, col in [("Sweden forest property, average price", "growth_price"),
                       ("Sweden forest property, price / assessed value", "growth_coef")]:
        d = se.dropna(subset=[col])
        X = pd.DataFrame({"const": 1.0, "base_c": d.base - ANNUAL_CENTRE}, index=d.index)
        rows[f"{label}: 1C colder county"] = Fit(d[col], X).effect({"base_c": -1}) * [100, 100, 1, 1]
    d = fi.dropna()
    X = pd.DataFrame({"const": 1.0, "base_c": d.base - ANNUAL_CENTRE}, index=d.index)
    rows["Finland stumpage price: 1C colder region"] = Fit(d.growth, X).effect({"base_c": -1}) * [100, 100, 1, 1]
    print_effects("3. Forest prices: extra annual growth for a 1C colder region, pp/yr", rows)
    north_se = ["21", "22", "23", "24", "25"]
    north_fi = ["MK17", "MK18", "MK19"]
    print(f"   Sweden forest property, Norrland vs rest (avg price, %/yr): "
          f"{se.loc[north_se, 'growth_price'].mean():.1%} vs {se.drop(north_se).growth_price.mean():.1%}")
    print(f"   Finland stumpage, north vs rest (%/yr): "
          f"{fi.loc[north_fi, 'growth'].mean():.1%} vs {fi.drop(north_fi).growth.mean():.1%}")


def main():
    pd.set_option("display.width", 200)
    panel = tourism_data()
    tourism_weather_panel(panel)
    tourism_long_difference(panel)
    houses = house_prices()
    house_price_test(houses)
    se_forest, fi_forest = forest_prices()
    forest_test(se_forest, fi_forest)
    OUT.mkdir(exist_ok=True)
    houses.to_csv(OUT / "mechanism_house_prices.csv")
    se_forest.to_csv(OUT / "mechanism_forest_sweden.csv")
    fi_forest.to_csv(OUT / "mechanism_forest_finland.csv")


if __name__ == "__main__":
    main()
