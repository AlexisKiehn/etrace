"""Buy once and hold indefinitely: long-only climate winners plus the timber sleeve.

    python -m strategy.buy_and_hold

No trend filter, no vol targeting, no rebalancing and no shorts (a short held forever
pays borrow fees and grows as it goes against you). Every entry month from 2011 to
2023 is tested, held to today, both FX-hedged (local currency) and unhedged in euros.

Entry months overlap and share the same end date, so they are not independent. The
placebo instead shuffles the climate scores across countries 200 times and buys and
holds those random picks, to see whether choosing the climate winners mattered.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from strategy import backtest
from strategy.backtest import Config
from strategy.data import load_prices
from strategy.signals import country_scores
from strategy.universe import COMMODITIES, COUNTRY_NAMES, REAL_ESTATE, all_tickers

INDEX = "EXSA.DE"  # iShares STOXX Europe 600, in EUR: the "just buy the market" option
SUFFIX_CCY = {".ST": "SEK", ".OL": "NOK", ".SW": "CHF", ".CO": "DKK", ".WA": "PLN"}
EURO_SUFFIXES = (".HE", ".DE", ".AS", ".BR", ".PA", ".MC", ".VI")
ENTRIES = pd.date_range("2011-01-31", "2023-12-31", freq="ME")

BUY_ONCE = dict(long_short=False, trend_filter=False, staggered=False, hold_months=0,
                vol_target=1.0, max_leverage=1.0)  # fully invested, unlevered


def currency(ticker):
    for suffix, ccy in SUFFIX_CCY.items():
        if ticker.endswith(suffix):
            return ccy
    return "EUR" if ticker.endswith(EURO_SUFFIXES) else "USD"  # US-listed ETFs


def in_euros(prices):
    ccys = sorted({currency(t) for t in prices} - {"EUR"})
    fx = load_prices([f"EUR{c}=X" for c in ccys], name="fx").reindex(prices.index).ffill()
    out = prices.copy()
    for t in prices:
        if currency(t) != "EUR":
            out[t] = prices[t] / fx[f"EUR{currency(t)}=X"]
    return out


def portfolios(countries):
    climate = country_scores(countries).score
    return {
        "climate winners + timber": (climate, Config(**BUY_ONCE)),
        "climate winners only": (climate, Config(**BUY_ONCE, commodity_weight=0.0)),
        "all 14 countries": (pd.Series(1.0, index=countries), Config(**BUY_ONCE, commodity_weight=0.0)),
        "timber only": (pd.Series(0.0, index=countries), Config(**BUY_ONCE, commodity_weight=1.0)),
    }


def hold_from(entry, scores, prices, cfg):
    """Monthly net returns of a portfolio bought at the `entry` month end and never traded."""
    start = (entry - pd.DateOffset(months=12)).replace(day=1)
    return backtest.run(scores, prices.loc[start:], cfg)[0].net_ret


def index_from(entry, index_px):
    return index_px.resample("ME").last().pct_change().loc[entry:].iloc[1:]


def summarise(paths, bench):
    rows = {}
    for entry, r in paths.items():
        equity = (1 + r).cumprod()
        years = len(r) / 12
        rows[entry] = {
            "CAGR": equity.iloc[-1] ** (1 / years) - 1,
            "MaxDD": (equity / equity.cummax() - 1).min(),
            "beats all-countries": equity.iloc[-1] > (1 + bench[entry]).prod(),
        }
    df = pd.DataFrame(rows).T.astype(float)
    return pd.Series({
        "median CAGR": df.CAGR.median(), "worst CAGR": df.CAGR.min(),
        "entries in profit": (df.CAGR > 0).mean(), "median MaxDD": df.MaxDD.median(),
        "worst MaxDD": df.MaxDD.min(), "beats all 14 countries": df["beats all-countries"].mean(),
        "CAGR if bought 2011-01": df.CAGR.iloc[0],
    })


def placebo(scores, cfg, prices, n=200, seed=0):
    """Median CAGR over yearly entries for the real scores and for shuffled ones."""
    entries = ENTRIES[ENTRIES.month == 1]

    def median_cagr(s):
        return np.median([backtest.stats(hold_from(e, s, prices, cfg)).CAGR for e in entries])

    rng = np.random.default_rng(seed)
    shuffles = [pd.Series(rng.permutation(scores.values), index=scores.index) for _ in range(n)]
    fake = Parallel(n_jobs=-2)(delayed(median_cagr)(s) for s in shuffles)
    actual = median_cagr(scores)
    return actual, np.median(fake), (np.array(fake) < actual).mean()


def main():
    countries = list(REAL_ESTATE)
    local = load_prices(all_tickers())
    index_px = load_prices([INDEX], name="index")[INDEX]
    ports = portfolios(countries)

    for label, prices in [("FX-hedged (local currency)", local), ("unhedged, in euros", in_euros(local))]:
        paths = {}
        for name, (scores, cfg) in ports.items():
            res = Parallel(n_jobs=-2)(delayed(hold_from)(e, scores, prices, cfg) for e in ENTRIES)
            paths[name] = dict(zip(ENTRIES, res))
        paths[f"European equities ({INDEX})"] = {e: index_from(e, index_px) for e in ENTRIES}
        bench = paths["all 14 countries"]
        table = pd.DataFrame({name: summarise(p, bench) for name, p in paths.items()}).T
        print(f"\nBought once, held to today, {label}: {len(ENTRIES)} entry months 2011-2023")
        print(table.round(2).to_string())

    actual, fake, pct = placebo(*ports["climate winners only"], local)
    print(f"\nPlacebo (climate winners only, hedged, January entries): median CAGR {actual:.1%} "
          f"vs {fake:.1%} for shuffled countries; beats {pct:.0%} of 200 shuffles")

    scores, cfg = ports["climate winners + timber"]
    end = local.index[-1]
    w = backtest.target_weights(scores, local.pct_change(fill_method=None), local.resample("ME").last(),
                                end, cfg)
    w = w[w > 0].sort_values(ascending=False)
    names = {t: COUNTRY_NAMES[c] for c, ts in REAL_ESTATE.items() for t in ts}
    names.update({t: "timber" for t in COMMODITIES})
    print(f"\nWhat it would buy today ({end:%Y-%m-%d}):")
    print(pd.DataFrame({"weight": w.round(3), "country": w.index.map(names)}).to_string())


if __name__ == "__main__":
    main()
