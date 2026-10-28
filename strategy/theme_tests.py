"""Theme checks: Nordic capital offices, northern Sweden real estate, northern forests.

    python -m strategy.theme_tests

These are small baskets (2-6 names), so the output is descriptive: how each theme
did by period, how sensitive it is to bond yields, and whether simply holding it
for 6 or 12 months after a positive 12-month return (time-series momentum) helped.
None of this is a statistically strong test.
"""

import numpy as np
import pandas as pd

from strategy import backtest
from strategy.data import load_bond_yields, load_prices
from strategy.signal_tests import newey_west_t

THEMES = {
    "Nordic capital offices vs continental offices": (
        # Stockholm CBD / inner-city and Oslo office landlords
        ["HUFV-A.ST", "FABG.ST", "ATRLJ-B.ST", "ENTRA.OL"],
        # Paris, Madrid and Barcelona office landlords
        ["GFC.PA", "COV.PA", "COL.MC", "MRL.MC"],
        ["SE", "NO"], ["FR", "ES"],
    ),
    "Northern Sweden real estate vs Swedish property": (
        ["DIOS.ST"],  # Dios: Lulea, Umea, Skelleftea, Sundsvall, Ostersund
        ["CAST.ST", "FABG.ST", "WALL-B.ST", "SAGA-B.ST", "ATRLJ-B.ST", "HUFV-A.ST"],
        ["SE"], ["SE"],
    ),
    "Northern forest owners vs pulp and paper": (
        ["SCA-B.ST", "HOLM-B.ST"],             # large forest holdings in northern Sweden
        ["STERV.HE", "UPM.HE", "METSB.HE"],    # mainly pulp, paper and packaging
        ["SE"], ["FI"],
    ),
    "Northern forest owners vs US timber REITs": (
        ["SCA-B.ST", "HOLM-B.ST"],
        ["WY", "RYN", "PCH"],
        ["SE"], ["US"],
    ),
}
PERIODS = {
    "2011-2018": ("2011", "2018"),
    "2019-2021": ("2019", "2021"),
    "2022-2023 (rate shock)": ("2022", "2023"),
    "2024-2026": ("2024", "2026"),
}


def basket(prices, tickers):
    monthly = prices[tickers].ffill().resample("ME").last().pct_change(fill_method=None)
    return monthly.mean(axis=1)["2011":]


def yield_beta(r, yields):
    """Return per +1 percentage point move in the 10y yield (monthly regression)."""
    dy = yields.mean(axis=1).diff().reindex(r.index)
    both = pd.concat([r, dy], axis=1).dropna()
    return np.polyfit(both.iloc[:, 1], both.iloc[:, 0], 1)[0]


def trend_hold(r, horizon):
    """Hold the spread for `horizon` months in tranches; each tranche is long if the
    spread's 12-1 month return was positive when it was formed, else short."""
    cum = np.log1p(r)
    signal = np.sign(cum.rolling(11).sum().shift(1))  # months t-11..t-1, skipping t
    tranches = pd.concat([signal.shift(k) for k in range(1, horizon + 1)], axis=1).mean(axis=1)
    return (tranches * r).dropna()


def main():
    tickers = sorted({t for a, b, _, _ in THEMES.values() for t in a + b})
    prices = load_prices(tickers, name="theme_prices")
    yields = load_bond_yields(["SE", "NO", "FI", "FR", "ES"])

    for theme, (longs, shorts, long_ccy, short_ccy) in THEMES.items():
        a, b = basket(prices, longs), basket(prices, shorts)
        spread = (a - b).dropna()
        print(f"\n=== {theme}")
        print(f"long:  {', '.join(longs)}\nshort: {', '.join(shorts)}")

        rows = {}
        for label, (s, e) in PERIODS.items():
            rows[label] = {
                "long ann.": (1 + a[s:e]).prod() ** (12 / len(a[s:e])) - 1,
                "short ann.": (1 + b[s:e]).prod() ** (12 / len(b[s:e])) - 1,
                "spread ann.": spread[s:e].mean() * 12,
            }
        st = backtest.stats(spread)
        rows["full period"] = {"long ann.": backtest.stats(a).CAGR, "short ann.": backtest.stats(b).CAGR,
                               "spread ann.": spread.mean() * 12}
        print(pd.DataFrame(rows).T.round(3).to_string())
        print(f"spread: Sharpe {st.Sharpe:.2f}, t {newey_west_t(spread):.2f}, max DD {st.MaxDD:.0%}")

        ly = yields.reindex(columns=[c for c in long_ccy if c in yields])
        if len(ly.columns):
            print(f"long basket: {yield_beta(a, ly):+.1%} per +1pp in its 10y yield")
        for h in (6, 12):
            tr = trend_hold(spread, h)
            ts = backtest.stats(tr)
            print(f"trend-following the spread, {h}m hold: Sharpe {ts.Sharpe:.2f} "
                  f"(t {newey_west_t(tr, lags=h):.2f}), after 2018: {backtest.stats(tr['2019':]).Sharpe:.2f}")


if __name__ == "__main__":
    main()
