"""Country model: the investor's traits (70%) plus the investor's own views (30%).

    python -m strategy.country_model              # today's scores and positions
    python -m strategy.country_model --backtest   # data part on country ETFs since 2001

Data part: the six traits tested over 150 years in history_tests.py (debt, population
growth, schooling, corruption, rule of law, defence at half weight), measured today for
a global universe. Views part: expert_views.py.

Two long/short books: developed and emerging markets. Countries are scored only against
others in their book, so governance gaps don't put every emerging market at the bottom.
Each book is reviewed every 6 months: long its top third, short its bottom third, each
held 3-5 years. The books are weighted 80/20, one weight per investable country.

The backtest can only check the data part (the views were recorded in 2026). It uses
US-listed iShares country ETFs in US dollars and is the first test that includes
emerging markets (China, India, Korea, Poland).
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from strategy import backtest
from strategy.data import load_imf, load_jst, load_owid, load_prices
from strategy.expert_views import RECORDED, expert_score
from strategy.factors import zscore
from strategy.history_tests import composite, defence_score
from strategy.run_backtest import OUT
from strategy.signal_tests import newey_west_t

DATA_WEIGHT, VIEW_WEIGHT = 0.7, 0.3
WEIGHTS = {"defence": 0.5}
TRACKING = Path(__file__).resolve().parent / "tracking"

# ISO3 -> US-listed ETF (USD). Russia is in the scores but not investable (sanctions).
ETFS = {
    "USA": "SPY", "CHN": "FXI", "IND": "INDA", "JPN": "EWJ", "KOR": "EWY", "CAN": "EWC",
    "GBR": "EWU", "DEU": "EWG", "FRA": "EWQ", "ITA": "EWI", "ESP": "EWP", "NLD": "EWN",
    "BEL": "EWK", "AUT": "EWO", "CHE": "EWL", "SWE": "EWD", "NOR": "NORW", "FIN": "EFNL",
    "DNK": "EDEN", "POL": "EPOL",
}
NOT_INVESTABLE = {"RUS"}
COUNTRIES = sorted(set(ETFS) | NOT_INVESTABLE)
EMERGING = {"CHN", "IND", "KOR", "POL", "RUS"}  # MSCI emerging markets (Russia until 2022)
BOOKS = {"developed": sorted(set(COUNTRIES) - EMERGING), "emerging": sorted(EMERGING)}
BOOK_WEIGHTS = {"developed": 0.8, "emerging": 0.2}  # 16 and 4 investable countries
BENCHMARKS = {"ACWI": "MSCI All Country World", "VGK": "FTSE Developed Europe"}


def trait_panel():
    """Year x country tables for the six traits, higher = better, as in history_tests."""
    years = pd.RangeIndex(1990, date.today().year, name="year")

    def owid(slug):
        return load_owid(slug).reindex(index=years, columns=COUNTRIES)

    jst = load_jst()
    jst_debt = jst.pivot(index="year", columns="iso", values="debtgdp") * 100
    debt = load_imf().reindex(index=years, columns=COUNTRIES).combine_first(
        jst_debt.reindex(index=years, columns=COUNTRIES))
    pop = owid("population")
    military = owid("military-expenditure-share-gdp").combine_first(
        owid("military-spending-as-a-share-of-gdp-gmsd"))
    return {
        "debt": -debt,
        "population": np.log(pop / pop.shift(5)) / 5,
        "schooling": owid("mean-years-of-schooling-long-run").ffill(limit=5),
        "transparency": -owid("political-corruption-index"),
        "rule_of_law": owid("rule-of-law-index"),
        "defence": defence_score(military),
    }


def book_signal(panel, members):
    """Composite of the traits, z-scored only across the book's members."""
    return composite({k: v[members] for k, v in panel.items()}, weights=WEIGHTS)


def raw_values():
    """Latest reported value of each trait per country, in natural units, for display."""
    t = trait_panel()
    latest = {k: v.ffill().iloc[-1] for k, v in t.items()}
    return pd.DataFrame({
        "debt % GDP": -latest["debt"], "pop growth %/yr": latest["population"] * 100,
        "schooling yrs": latest["schooling"], "corruption (0-1)": -latest["transparency"],
        "rule of law (0-1)": latest["rule_of_law"], "defence score": latest["defence"],
    })


def scores():
    """Today's data score, view score and final score per country, within its book."""
    t = {k: v.ffill() for k, v in trait_panel().items()}  # latest report per country
    views = pd.Series(expert_score())
    tables = []
    for book, members in BOOKS.items():
        out = pd.DataFrame({"book": book, "data score": zscore(book_signal(t, members).iloc[-1]),
                            "your view": views.reindex(members)})
        out["view score"] = zscore(out["your view"])
        out["final"] = DATA_WEIGHT * out["data score"] + VIEW_WEIGHT * out["view score"].fillna(0)
        tables.append(out.sort_values("final", ascending=False))
    return pd.concat(tables)


def positions(table):
    """In each book, long the top third and short the bottom third of the investable countries."""
    pos = pd.Series("", index=table.index)
    for _, book in table.groupby("book", sort=False):
        inv = book[~book.index.isin(NOT_INVESTABLE)].sort_values("final", ascending=False)
        n = len(inv) // 3
        if n:
            pos[inv.index[:n]] = "LONG"
            pos[inv.index[-n:]] = "SHORT"
    pos[list(NOT_INVESTABLE & set(table.index))] = "(not investable)"
    return pos


# --- Backtest of the data part on country ETFs ----------------------------------------

def _picks(signal, country, reviews):
    """At each review: (top third, bottom third, all live countries) of one book."""
    picks = {}
    for d in reviews:
        # Annual data for year y is used from the following year (published with a lag).
        s = signal.loc[: d.year - 1].ffill().iloc[-1].reindex(country.columns)
        live = country.loc[:d].iloc[-1].notna() & country.loc[:d].count().ge(12)
        s = s[live].dropna().sort_values(ascending=False)
        n = len(s) // 3
        if n:  # a book needs at least 3 live countries
            picks[d] = (list(s.index[:n]), list(s.index[-n:]), list(s.index))
    return picks


def _book_returns(country, picks, h, cost_bps):
    """Monthly returns of one book, every review's picks held for h months (overlapping)."""
    rows = []
    for m in country.index[country.index > min(picks)]:
        active = [d for d in picks if d < m <= d + pd.DateOffset(months=h)]
        if not active:
            continue
        r = country.loc[m]
        ls = np.mean([r[picks[d][0]].mean() - r[picks[d][1]].mean() for d in active])
        top = np.mean([r[picks[d][0]].mean() for d in active])
        ew = np.mean([r[picks[d][2]].mean() for d in active])
        new = sum(d == m - pd.offsets.MonthEnd(1) for d in active)  # tranches formed last month
        cost = new / len(active) * 2 * cost_bps / 1e4
        rows.append({"date": m, "long/short": ls - cost, "top third": top - cost / 2,
                     "equal weight": ew})
    return pd.DataFrame(rows).set_index("date")


def etf_backtest(hold_months=(36, 60), cost_bps=10, start="2001-06-30"):
    """{holding period: {book or "combined": monthly returns}}. Until the emerging book
    has 3 live ETFs (2011) the combined book is the developed book alone."""
    prices = load_prices(list(ETFS.values()) + list(BENCHMARKS), start="1996-01-01", name="country_etfs")
    monthly = prices.resample("ME").last().pct_change(fill_method=None)
    country = monthly[list(ETFS.values())].set_axis(list(ETFS), axis=1)
    panel = trait_panel()
    reviews = [d for d in monthly.index if d.month in (6, 12) and d >= pd.Timestamp(start)]
    picks = {}
    for book, members in BOOKS.items():
        investable = [c for c in members if c in ETFS]
        picks[book] = _picks(book_signal(panel, members), country[investable], reviews)

    results = {}
    for h in hold_months:
        books = {b: _book_returns(country[list(ETFS)], p, h, cost_bps) for b, p in picks.items()}
        combined = {}
        for col in books["developed"]:
            parts = pd.concat({b: df[col] for b, df in books.items()}, axis=1)
            w = parts.notna() * pd.Series(BOOK_WEIGHTS)
            combined[col] = (parts.fillna(0) * w).sum(axis=1) / w.sum(axis=1)
        books = {"combined": pd.DataFrame(combined), **books}
        results[h] = {b: df.join(monthly[list(BENCHMARKS)]) for b, df in books.items()}
    return results, picks


def summarise(df, h):
    rows = {}
    for col in df:
        r = df[col].dropna()
        st = backtest.stats(r)
        rows[col] = {"CAGR": st.CAGR, "Vol": st.Vol, "Sharpe": st.Sharpe, "MaxDD": st.MaxDD,
                     "t": newey_west_t(r, lags=h // 6), "from": f"{r.index[0]:%Y-%m}"}
    ex = (df["top third"] - df["equal weight"]).dropna()
    rows["top third minus equal weight"] = {"CAGR": ex.mean() * 12, "t": newey_west_t(ex, lags=h // 6)}
    ex = (df["top third"] - df["ACWI"]).dropna()
    rows["top third minus ACWI"] = {"CAGR": ex.mean() * 12, "t": newey_west_t(ex, lags=h // 6),
                                    "from": f"{ex.index[0]:%Y-%m}"}
    return pd.DataFrame(rows).T


def by_period(books):
    """Average annual long/short return of each book, by decade and over its whole history."""
    periods = {"2001-2010": ("2001", "2010"), "2011-2020": ("2011", "2020"), "2021-": ("2021", None),
               "all": (None, None)}
    return pd.DataFrame({p: {b: df["long/short"].loc[lo:hi].mean() * 12 for b, df in books.items()}
                         for p, (lo, hi) in periods.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backtest", action="store_true")
    a = p.parse_args()
    pd.set_option("display.width", 220)

    table = scores()
    table["position"] = positions(table)
    table["ETF"] = pd.Series(ETFS)
    print(f"Country scores ({DATA_WEIGHT:.0%} data traits, {VIEW_WEIGHT:.0%} your views of {RECORDED}), "
          f"scored within each book:")
    print(table.join(raw_values()).round(2).to_string())

    TRACKING.mkdir(exist_ok=True)
    path = TRACKING / f"{date.today():%Y-%m-%d}.csv"
    table.to_csv(path)
    print(f"\nSaved for forward tracking: {path}")

    if a.backtest:
        results, _ = etf_backtest()
        for h, books in results.items():
            for book, df in books.items():
                weights = "80/20 developed/emerging" if book == "combined" else "scored within the book"
                print(f"\n{book.capitalize()} ({weights}), data part only, country ETFs in USD, "
                      f"reviewed every 6 months, held {h // 12} years:")
                print(summarise(df, h).round(3).to_string())
            print(f"\nLong/short %/yr by decade, held {h // 12} years:")
            print((by_period(books) * 100).round(1).to_string())
        OUT.mkdir(exist_ok=True)
        pd.concat({(h, b): df for h, books in results.items() for b, df in books.items()},
                  names=["hold_months", "book"]).to_csv(OUT / "country_model_backtest.csv")


if __name__ == "__main__":
    main()
