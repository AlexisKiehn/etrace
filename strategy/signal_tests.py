"""Test the time-varying signals: momentum, yield spread, tourism surprise.

    python -m strategy.signal_tests

For each signal (and the static climate score, for comparison):
  - IC: monthly rank correlation between the signal and next month's country returns,
    with a Newey-West t-stat, before and after 2018.
  - Pure tilt: long/short book on the signal alone, no trend filter, no commodities.
  - Full strategy: the signal used in place of the climate score in the base config.
  - Placebo: pure tilt with country labels shuffled (200 times).

Rule, fixed before running: a signal is used if its placebo beats >= 95% of shuffles
and its IC after 2018 is positive. Used signals are blended with the climate score.
"""

import numpy as np
import pandas as pd
from scipy import stats

from strategy import backtest
from strategy.backtest import Config
from strategy.data import load_prices
from strategy.dynamic_signals import SIGNALS, country_momentum, tourism_surprise, yield_spread
from strategy.factor_selection import IN_SAMPLE_END, evaluate, placebo
from strategy.factors import blend, zscore
from strategy.run_backtest import OUT
from strategy.signals import country_scores
from strategy.universe import REAL_ESTATE, all_tickers

PURE_TILT = Config(trend_filter=False, commodity_weight=0.0)


def country_returns(prices, countries):
    monthly = prices.ffill().resample("ME").last().pct_change(fill_method=None)
    return pd.DataFrame({c: monthly.reindex(columns=REAL_ESTATE[c]).mean(axis=1) for c in countries})


def as_of(signal, dates):
    """Signal value known at each date (static scores are broadcast)."""
    if isinstance(signal, pd.Series):
        return pd.DataFrame([signal.values] * len(dates), index=dates, columns=signal.index)
    return signal.reindex(signal.index.union(dates)).ffill().reindex(dates)


def newey_west_t(x, lags=6):
    x = x.dropna()
    e = x - x.mean()
    n = len(x)
    lrv = e @ e / n + 2 * sum((1 - l / (lags + 1)) * (e.values[l:] @ e.values[:-l]) / n
                              for l in range(1, lags + 1))
    return x.mean() / np.sqrt(lrv / n)


def information_coefficient(signal, rets):
    sig = as_of(signal, rets.index).shift(1)  # signal at month end t vs return in t+1
    ic = {}
    for d in rets.index:
        both = pd.concat([sig.loc[d], rets.loc[d]], axis=1).dropna()
        if len(both) >= 5:
            ic[d] = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic
    ic = pd.Series(ic)["2011-02":]
    is_, oos = ic[:IN_SAMPLE_END], ic[IN_SAMPLE_END:]
    return pd.Series({
        "IC IS": is_.mean(), "IC t IS": newey_west_t(is_),
        "IC OOS": oos.mean(), "IC t OOS": newey_west_t(oos),
        "IC full": ic.mean(), "IC t full": newey_west_t(ic),
    })


def main():
    countries = list(REAL_ESTATE)
    prices = load_prices(all_tickers())
    rets = country_returns(prices, countries)

    signals = {
        "climate (static)": zscore(country_scores(countries).score),
        "momentum": zscore(country_momentum(prices, countries)),
        "yield_spread": zscore(yield_spread(countries)),
        "tourism_surprise": zscore(tourism_surprise(countries)),
    }

    rows = {}
    for name, s in signals.items():
        print(f"testing {name} ...", flush=True)
        ic = information_coefficient(s, rets)
        tilt = evaluate(s, prices, PURE_TILT)
        full = evaluate(s, prices)
        actual, pct = placebo(s, prices, cfg=PURE_TILT)
        rows[name] = pd.concat([
            ic,
            tilt[["Sharpe IS", "Sharpe OOS", "Sharpe full"]].add_prefix("tilt "),
            full[["Sharpe IS", "Sharpe OOS", "Sharpe full"]].add_prefix("strat "),
            pd.Series({"placebo pct": pct}),
        ])
    table = pd.DataFrame(rows).T
    pd.set_option("display.width", 200)
    print("\nInformation coefficient (rank corr. with next-month country return)")
    print(table.filter(like="IC").round(3).to_string())
    print("\nSharpe: pure tilt (no trend filter, no commodities) and full strategy")
    print(table.filter(regex="tilt|strat|placebo").round(2).to_string())

    used = [n for n in SIGNALS if table.loc[n, "placebo pct"] >= 0.95 and table.loc[n, "IC OOS"] > 0]
    print(f"\nSignals passing the rule (placebo >= 95% and positive IC after 2018): {used or 'none'}")
    if used:
        combo = blend([signals["climate (static)"]] + [signals[n] for n in used])
        out = pd.DataFrame({
            "climate only": evaluate(signals["climate (static)"], prices),
            "climate + " + " + ".join(used): evaluate(combo, prices),
        }).T
        print("\nFull strategy with the passing signals blended in")
        print(out.round(2).to_string())
        actual, pct = placebo(combo, prices, cfg=PURE_TILT)
        print(f"Blend placebo (pure tilt): Sharpe {actual:.2f}, beats {pct:.0%} of shuffles")

    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "signal_tests.csv")


if __name__ == "__main__":
    main()
