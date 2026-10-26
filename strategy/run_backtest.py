"""Backtest the European climate macro strategy.

    python -m strategy.run_backtest
    python -m strategy.run_backtest --scenario ssp585_2041_2070 --long-only
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from strategy import backtest
from strategy.data import load_prices
from strategy.signals import country_scores
from strategy.universe import COUNTRY_NAMES, REAL_ESTATE, all_tickers

OUT = Path(__file__).resolve().parent / "output"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="ssp245_2041_2070",
                   help="Koppen projection, e.g. ssp126_2041_2070 or ssp585_2071_2099")
    p.add_argument("--koppen-weight", type=float, default=0.5)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--long-only", action="store_true")
    p.add_argument("--no-trend", action="store_true")
    p.add_argument("--commodity-weight", type=float, default=0.2)
    p.add_argument("--vol-target", type=float, default=0.10)
    p.add_argument("--cost-bps", type=float, default=15.0)
    p.add_argument("--refresh", action="store_true", help="re-download prices")
    a = p.parse_args()

    scores = country_scores(list(REAL_ESTATE), a.scenario, a.koppen_weight)
    print(f"\nClimate scores ({a.scenario}):")
    print(scores.rename(index=COUNTRY_NAMES).round(2).to_string())

    prices = load_prices(all_tickers(), a.start, a.refresh)
    cfg = backtest.Config(
        long_short=not a.long_only, trend_filter=not a.no_trend,
        commodity_weight=a.commodity_weight, vol_target=a.vol_target, cost_bps=a.cost_bps,
    )
    result, weights = backtest.run(scores.score, prices, cfg)
    bench = backtest.benchmark(prices, result.index[0]).reindex(result.index)

    table = pd.DataFrame({
        "Strategy": backtest.stats(result.net_ret),
        "EU real estate EW": backtest.stats(bench),
    })
    table.loc["Corr to bench", "Strategy"] = result.net_ret.corr(bench)
    table.loc["Avg turnover/mo", "Strategy"] = result.turnover.mean()
    table.loc["Avg gross exp", "Strategy"] = result.gross_exposure.mean()
    print(f"\nBacktest {result.index[0]:%Y-%m} to {result.index[-1]:%Y-%m}, net of costs:")
    print(table.round(3).to_string())

    latest = weights.iloc[-1]
    print(f"\nCurrent positions ({weights.index[-1]:%Y-%m-%d}):")
    print(latest[latest != 0].sort_values(ascending=False).round(3).to_string())

    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / "returns.csv")
    weights.to_csv(OUT / "weights.csv")
    scores.to_csv(OUT / "country_scores.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1], sharex=True)
    result.equity.plot(ax=ax1, label="Climate macro (net)")
    (1 + bench).cumprod().plot(ax=ax1, label="EU real estate EW", alpha=0.7)
    ax1.set_yscale("log")
    ax1.set_title(f"European climate macro strategy - {a.scenario}")
    ax1.legend()
    (result.equity / result.equity.cummax() - 1).plot(ax=ax2, color="firebrick")
    ax2.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve.png", dpi=120)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
