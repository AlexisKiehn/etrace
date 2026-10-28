"""Robustness checks: scenario / config sweep and a placebo test with shuffled scores.

    python -m strategy.robustness
"""

import numpy as np
import pandas as pd

from strategy import backtest
from strategy.data import load_prices
from strategy.signals import country_scores
from strategy.universe import REAL_ESTATE, all_tickers

SCENARIOS = ["ssp126_2041_2070", "ssp245_2041_2070", "ssp585_2041_2070", "ssp585_2071_2099"]
VARIANTS = {
    "base": backtest.Config(),
    "long-only": backtest.Config(long_short=False),
    "no trend filter": backtest.Config(trend_filter=False),
    "no commodities": backtest.Config(commodity_weight=0.0),
}


def sharpe(scores, prices, cfg):
    result, _ = backtest.run(scores, prices, cfg)
    return backtest.stats(result.net_ret)


def main(n_placebo=200, seed=0):
    prices = load_prices(all_tickers())
    countries = list(REAL_ESTATE)

    rows = []
    for scen in SCENARIOS:
        scores = country_scores(countries, scen).score
        for name, cfg in VARIANTS.items():
            s = sharpe(scores, prices, cfg)
            rows.append({"scenario": scen, "variant": name, **s[["CAGR", "Vol", "Sharpe", "MaxDD"]]})
    print(pd.DataFrame(rows).set_index(["scenario", "variant"]).round(3).to_string())

    # Placebo: same construction, country scores randomly reassigned across countries.
    real = country_scores(countries).score
    cfg = backtest.Config(commodity_weight=0.0)
    actual = sharpe(real, prices, cfg).Sharpe
    rng = np.random.default_rng(seed)
    placebo = [
        sharpe(pd.Series(rng.permutation(real.values), index=real.index), prices, cfg).Sharpe
        for _ in range(n_placebo)
    ]
    pct = (np.array(placebo) < actual).mean()
    print(f"\nPlacebo (real estate only, {n_placebo} shuffles): actual Sharpe {actual:.2f}, "
          f"placebo median {np.median(placebo):.2f}, 95th pct {np.percentile(placebo, 95):.2f}, "
          f"actual beats {pct:.0%} of shuffles")


if __name__ == "__main__":
    main()
