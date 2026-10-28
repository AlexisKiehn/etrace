"""Test all country signals for 6- and 12-month holding periods.

    python -m strategy.horizon_tests

For each signal and horizon H (6, 12 months):
  - IC: rank correlation between the signal and the next H months' country returns
    (monthly, overlapping; Newey-West t-stat with H lags), before and after 2018.
  - Pure tilt: long/short on the signal alone, no trend filter, no commodities, held H
    months in overlapping tranches.
  - Placebo: the same with country labels shuffled, 200 times.

Rule, fixed before running: a signal is used at horizon H if its placebo beats >= 95%
of shuffles and its IC after 2018 is positive. Used signals are blended equally with
the climate score and compared, in the full strategy, with what we have now.
"""

import numpy as np
import pandas as pd
from scipy import stats

from strategy import more_signals
from strategy.backtest import Config
from strategy.data import load_prices
from strategy.dynamic_signals import country_momentum, tourism_surprise, yield_spread
from strategy.factor_selection import IN_SAMPLE_END, evaluate, placebo
from strategy.factors import blend, zscore
from strategy.run_backtest import OUT
from strategy.signal_tests import as_of, country_returns, newey_west_t
from strategy.signals import country_scores
from strategy.universe import REAL_ESTATE, all_tickers

HORIZONS = (6, 12)


def forward_ic(signal, rets, horizon):
    fwd = np.log1p(rets).rolling(horizon).sum().shift(-horizon)
    sig = as_of(signal, rets.index)
    ic = {}
    for d in rets.index:
        both = pd.concat([sig.loc[d], fwd.loc[d]], axis=1).dropna()
        if len(both) >= 5:
            ic[d] = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic
    ic = pd.Series(ic)["2011-01":]
    is_, oos = ic[:IN_SAMPLE_END], ic[IN_SAMPLE_END:]
    return pd.Series({
        "IC IS": is_.mean(), "IC OOS": oos.mean(), "IC full": ic.mean(),
        "IC t full": newey_west_t(ic, lags=horizon),
    })


def load_signals(prices, countries):
    signals = {
        "climate (static)": ("climate", country_scores(countries).score),
        "momentum": ("market", country_momentum(prices, countries)),
        "yield_spread": ("market", yield_spread(countries)),
        "tourism_surprise": ("climate/etrace", tourism_surprise(countries)),
    }
    for name, (group, fn) in more_signals.SIGNALS.items():
        signals[name] = (group, fn(countries))
    return {name: (group, zscore(s)) for name, (group, s) in signals.items()}


def main():
    countries = list(REAL_ESTATE)
    prices = load_prices(all_tickers())
    rets = country_returns(prices, countries)
    signals = load_signals(prices, countries)

    rows = []
    for h in HORIZONS:
        tilt = Config(trend_filter=False, commodity_weight=0.0, hold_months=h)
        for name, (group, s) in signals.items():
            print(f"H={h:2d}  {name} ...", flush=True)
            ic = forward_ic(s, rets, h)
            perf = evaluate(s, prices, tilt)[["Sharpe IS", "Sharpe OOS", "Sharpe full"]]
            _, pct = placebo(s, prices, cfg=tilt)
            rows.append({"horizon": h, "signal": name, "group": group, **ic,
                         **perf.add_prefix("tilt "), "placebo pct": pct})
    table = pd.DataFrame(rows).set_index(["horizon", "signal"])
    pd.set_option("display.width", 220)
    print("\n" + table.round(2).to_string())
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "horizon_tests.csv")

    climate = signals["climate (static)"][1]
    momentum = signals["momentum"][1]
    for h in HORIZONS:
        t = table.loc[h]
        used = [n for n in t.index if n != "climate (static)"
                and t.loc[n, "placebo pct"] >= 0.95 and t.loc[n, "IC OOS"] > 0]
        print(f"\nH={h}: signals passing the rule: {used or 'none'}")
        variants = {
            "current default (climate + momentum, monthly)": (blend([climate, momentum]), Config()),
            f"climate only, {h}m hold": (climate, Config(hold_months=h)),
            f"climate + momentum, {h}m hold": (blend([climate, momentum]), Config(hold_months=h)),
        }
        if used and used != ["momentum"]:
            parts = [climate] + [signals[n][1] for n in used]
            variants[f"climate + {' + '.join(used)}, {h}m hold"] = (blend(parts), Config(hold_months=h))
        out = pd.DataFrame({k: evaluate(s, prices, cfg) for k, (s, cfg) in variants.items()}).T
        print(out.round(2).to_string())


if __name__ == "__main__":
    main()
