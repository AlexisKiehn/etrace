"""150-year test of the investor's country traits, 1870-2020.

    python -m strategy.history_tests

The traits come from the investor's own views (education, family, honest and capable
government, sustainable debt, sensible defence spending); the values come from data,
so the test is free of hindsight. Signs were fixed before testing:

    trait            measure                                          sign  source
    debt             government debt / GDP                             -    Macrohistory (JST)
    population       5-year population growth                          +    JST
    schooling        average years of schooling, ages 15-64            +    Lee-Lee via OWID
    transparency     V-Dem political corruption index                  -    V-Dem via OWID
    rule of law      V-Dem rule of law index                           +    V-Dem via OWID
    defence          2-5% of GDP good, more so if rising from <2%;    rule  SIPRI / GMSD via OWID
                     above 5% bad
    -- secondary, shorter history --
    fertility        children per woman (mostly from 1950)             +    Gapminder/UN via OWID
    inequality       top-10% income share (mostly from 1980)           -    WID via OWID
    debt trend       5-year change in debt / GDP                       -    JST

The primary test is one composite: the average z-score of the first six. Each year t,
countries are ranked on traits measured in t-1, and compared on real stock returns
(local currency) over the next 3 and 5 years, relative to the average country.
"""

import numpy as np
import pandas as pd
from scipy import stats

from strategy.data import load_jst, load_owid
from strategy.factors import zscore
from strategy.run_backtest import OUT
from strategy.signal_tests import newey_west_t

HORIZONS = (3, 5)
PERIODS = {"1870-2015": (1870, 2015), "1870-1945": (1870, 1945), "1950-2015": (1950, 2015)}
PRIMARY = ["debt", "population", "schooling", "transparency", "rule_of_law", "defence"]
MIN_COUNTRIES = 8


def returns(jst, column="eq_tr"):
    """Annual log real returns, year x country. Clipped at -90%/+900% (hyperinflations)."""
    wide = jst.pivot(index="year", columns="iso", values=column)
    infl = jst.pivot(index="year", columns="iso", values="cpi").pct_change(fill_method=None)
    real = np.log1p(wide) - np.log1p(infl)
    return real.clip(np.log(0.1), np.log(10))


def defence_score(share):
    """Investor's rule: 2-5% of GDP is good (+1), +1 more if it was below 2% five years
    earlier (rearming from a low base), above 5% is bad (-1), below 2% neutral (0)."""
    rising = share.shift(5) < 2
    score = pd.DataFrame(0.0, index=share.index, columns=share.columns)
    score[(share >= 2) & (share <= 5)] = 1.0
    score[(share >= 2) & (share <= 5) & rising] = 2.0
    score[share > 5] = -1.0
    return score.where(share.notna())


def traits(jst):
    """Year x country tables, oriented so that higher = better (signs as in the docstring)."""
    countries = sorted(jst.iso.unique())
    years = pd.RangeIndex(1860, 2021, name="year")

    def owid(slug):
        return load_owid(slug).reindex(index=years, columns=countries)

    debt = jst.pivot(index="year", columns="iso", values="debtgdp").reindex(years)
    pop = jst.pivot(index="year", columns="iso", values="pop").reindex(years)
    # Spending: SIPRI where it exists (closest to NATO's 2% definition), long-run GMSD before.
    military = owid("military-expenditure-share-gdp").combine_first(
        owid("military-spending-as-a-share-of-gdp-gmsd"))
    return {
        "debt": -debt,
        "population": np.log(pop / pop.shift(5)) / 5,
        # Five-yearly observations: carry the last one forward (no look-ahead interpolation).
        "schooling": owid("mean-years-of-schooling-long-run").ffill(limit=4),
        "transparency": -owid("political-corruption-index"),
        "rule_of_law": owid("rule-of-law-index"),
        "defence": defence_score(military),
        "fertility": owid("children-born-per-woman"),
        "inequality": -owid("income-share-top-10-before-tax-wid"),
        "debt_trend": -(debt - debt.shift(5)),
    }


def composite(t, names=PRIMARY, min_parts=4, weights=None):
    """Weighted average of the traits' cross-country z-scores (default: equal weights)."""
    weights = weights or {}
    stacked = pd.concat([zscore(t[n].dropna(how="all")) for n in names], keys=names)
    w = pd.Series([weights.get(n, 1.0) for n in stacked.index.get_level_values(0)], index=stacked.index)
    total = stacked.mul(w, axis=0).groupby(level=1).sum()
    weight = stacked.notna().mul(w, axis=0).groupby(level=1).sum()
    count = stacked.notna().groupby(level=1).sum()
    return (total / weight).where(count >= min_parts)


def forward(ret, h):
    """Log return over years t+1..t+h, relative to the cross-country average."""
    fwd = ret[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)
    return fwd.sub(fwd.mean(axis=1), axis=0)


def ic_series(signal, ret, h):
    """Each year: rank correlation of the signal (as of t-1) with the next h years' returns."""
    sig, fwd = signal.shift(1), forward(ret, h)
    ic = {}
    for year in fwd.index:
        if year not in sig.index:
            continue
        both = pd.concat([sig.loc[year], fwd.loc[year]], axis=1).dropna()
        if len(both) >= MIN_COUNTRIES:
            ic[year] = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic
    return pd.Series(ic, dtype=float)


def long_short(signal, ret, h):
    """Top third minus bottom third, each year's picks held for h years (overlapping).
    Returns the annual log return of the long/short book."""
    sig = signal.shift(1).reindex(ret.index)
    legs = []
    for year in ret.index:
        s = sig.loc[year].dropna()
        s = s[ret.columns.intersection(s.index)]
        if len(s) < MIN_COUNTRIES:
            continue
        top, bottom = s[s >= s.quantile(2 / 3)].index, s[s <= s.quantile(1 / 3)].index
        for k in range(1, h + 1):
            if year + k in ret.index:
                r = ret.loc[year + k]
                legs.append((year + k, r[top].mean() - r[bottom].mean()))
    return pd.DataFrame(legs, columns=["year", "ls"]).groupby("year").ls.mean()


def evaluate(signal, ret, h):
    ic = ic_series(signal, ret, h)
    ls = long_short(signal, ret, h)
    row = {}
    for label, (lo, hi) in PERIODS.items():
        i = ic[(ic.index >= lo) & (ic.index <= hi)]
        row[f"IC {label}"] = i.mean()
        row[f"t {label}"] = newey_west_t(i, lags=h) if len(i) > h + 2 else np.nan
    full = ls[ls.index >= 1871]
    row["L/S %/yr"] = full.mean() * 100
    row["L/S t"] = newey_west_t(full, lags=h)
    row["L/S %/yr 1950+"] = ls[ls.index >= 1951].mean() * 100
    row["years"] = len(ic)
    return pd.Series(row)


def placebo(signal, ret, h, n=1000, seed=0):
    """Share of country-shuffled signals with a lower full-period mean IC."""
    actual = ic_series(signal, ret, h).mean()
    rng = np.random.default_rng(seed)
    cols = signal.columns
    fake = [ic_series(signal.set_axis(cols[rng.permutation(len(cols))], axis=1), ret, h).mean()
            for _ in range(n)]
    return (np.array(fake) < actual).mean()


def residualise(signal, control):
    """Remove the part of the signal explained by a control, country by country each year."""
    out = {}
    for year in signal.index.intersection(control.index):
        both = pd.concat([signal.loc[year], control.loc[year]], axis=1).dropna()
        if len(both) >= MIN_COUNTRIES:
            slope, intercept = np.polyfit(both.iloc[:, 1], both.iloc[:, 0], 1)
            out[year] = both.iloc[:, 0] - (intercept + slope * both.iloc[:, 1])
    return pd.DataFrame(out).T


def robustness(t, jst, stocks, h=5):
    """Is the composite just hindsight in expert-coded data, just rich countries, or FX?"""
    gdp = np.log(jst.pivot(index="year", columns="iso", values="rgdpmad"))
    usd = jst.pivot(index="year", columns="iso", values="xrusd")
    stocks_usd = (np.log1p(jst.pivot(index="year", columns="iso", values="eq_tr"))
                  - np.log(usd / usd.shift(1))).clip(np.log(0.1), np.log(10))
    comp = t["COMPOSITE (primary)"]
    checks = {
        "composite": (comp, stocks),
        "hard data only (debt, population, schooling, defence)":
            (composite(t, ["debt", "population", "schooling", "defence"], min_parts=3), stocks),
        "composite, after removing GDP per head": (residualise(comp, gdp), stocks),
        "GDP per head alone": (gdp, stocks),
        "composite, returns in US dollars": (comp, stocks_usd),
        "composite, defence at half weight (live model)":
            (composite(t, weights={"defence": 0.5}), stocks),
    }
    return pd.DataFrame({k: evaluate(s, r, h) for k, (s, r) in checks.items()}).T


def main():
    pd.set_option("display.width", 220)
    jst = load_jst()
    t = traits(jst)
    t["COMPOSITE (primary)"] = composite(t)
    stocks = returns(jst, "eq_tr")
    housing = returns(jst, "housing_tr")

    tables = {}
    for h in HORIZONS:
        table = pd.DataFrame({name: evaluate(sig, stocks, h) for name, sig in t.items()}).T
        print(f"\nStocks, next {h} years vs other countries (real, local currency); "
              f"IC = rank correlation, t = Newey-West")
        print(table.round(3).to_string())
        tables[h] = table

    comp = t["COMPOSITE (primary)"]
    print("\nComposite, placebo (1000 country shuffles) and housing:")
    for h in HORIZONS:
        pct = placebo(comp, stocks, h)
        hs = evaluate(comp, housing, h)
        print(f"  {h}y: stocks beat {pct:.0%} of shuffles | housing IC {hs['IC 1870-2015']:.3f} "
              f"(t {hs['t 1870-2015']:.2f}), housing L/S {hs['L/S %/yr']:.1f}%/yr")

    print("\nRobustness, 5-year horizon:")
    print(robustness(t, jst, stocks).round(3).to_string())

    OUT.mkdir(exist_ok=True)
    pd.concat(tables, names=["horizon"]).to_csv(OUT / "history_tests.csv")


if __name__ == "__main__":
    main()
