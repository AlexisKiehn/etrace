"""Test candidate factors one at a time and keep the ones that improve the strategy.

    python -m strategy.factor_selection

Protocol, chosen to limit overfitting:
  1. Each factor's sign is fixed in advance (factors.FACTORS).
  2. Selection uses in-sample data only (trades up to 2018). 2019 onwards is held out.
  3. Factors are added one at a time, in the listed order, to the climate score with
     equal weights. A factor is kept if it raises the in-sample Sharpe by at least
     MIN_GAIN.
  4. The chosen blend is then judged on the held-out period and with a placebo test
     (blend scores shuffled across countries), against the climate score alone.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from strategy import backtest
from strategy.backtest import Config
from strategy.data import load_prices
from strategy.factors import FACTORS, blend, factor_score, zscore
from strategy.robustness import sharpe
from strategy.run_backtest import OUT
from strategy.signals import country_scores
from strategy.universe import REAL_ESTATE, all_tickers

IN_SAMPLE_END = "2018-12-31"
MIN_GAIN = 0.05
N_PLACEBO = 200


def evaluate(scores, prices, cfg=Config()):
    r = backtest.run(scores, prices, cfg)[0].net_ret
    return pd.Series({
        "Sharpe IS": backtest.stats(r[:IN_SAMPLE_END]).Sharpe,
        "Sharpe OOS": backtest.stats(r[IN_SAMPLE_END:].iloc[1:]).Sharpe,
        "Sharpe full": backtest.stats(r).Sharpe,
        "CAGR full": backtest.stats(r).CAGR,
        "MaxDD full": backtest.stats(r).MaxDD,
    })


def _shuffle(scores, rng):
    perm = rng.permutation(len(scores.columns if isinstance(scores, pd.DataFrame) else scores))
    if isinstance(scores, pd.DataFrame):
        return scores.set_axis(scores.columns[perm], axis=1)
    return pd.Series(scores.values[perm], index=scores.index)


def placebo(scores, prices, n=N_PLACEBO, seed=0, cfg=Config(commodity_weight=0.0)):
    """Share of country-shuffled scores the real scores beat (real estate book only)."""
    actual = sharpe(scores, prices, cfg).Sharpe
    rng = np.random.default_rng(seed)
    shuffles = [_shuffle(scores, rng) for _ in range(n)]
    fake = Parallel(n_jobs=-2)(delayed(lambda s: sharpe(s, prices, cfg).Sharpe)(s) for s in shuffles)
    return actual, (np.array(fake) < actual).mean()


def main():
    prices = load_prices(all_tickers())
    countries = list(REAL_ESTATE)
    climate = zscore(country_scores(countries).score)
    candidates = {name: factor_score(name, countries) for name in FACTORS}

    print("Step 1: each factor on its own (full strategy, own sign)")
    solo = pd.DataFrame({"climate (current)": evaluate(climate, prices)})
    for name, s in candidates.items():
        solo[name] = evaluate(s, prices)
    print(solo.T.round(2).to_string())

    print(f"\nStep 2: add factors one by one, keep if in-sample Sharpe rises >= {MIN_GAIN}")
    kept, parts = [], [climate]
    best = evaluate(climate, prices)
    log = [{"step": "climate only", **best, "kept": True}]
    for name, s in candidates.items():
        trial = evaluate(blend(parts + [s]), prices)
        keep = trial["Sharpe IS"] - best["Sharpe IS"] >= MIN_GAIN
        log.append({"step": f"+ {name}", **trial, "kept": keep})
        if keep:
            kept.append(name)
            parts.append(s)
            best = trial
    log = pd.DataFrame(log).set_index("step")
    print(log.round(2).to_string())

    final = blend(parts)
    print(f"\nSelected factors: climate + {kept or 'nothing'}")
    print("\nStep 3: placebo test, real estate book only "
          f"({N_PLACEBO} country shuffles; >= 95% would be significant)")
    for label, s in [("climate only", climate), ("selected blend", final)]:
        actual, pct = placebo(s, prices)
        print(f"  {label:15s} Sharpe {actual:.2f}, beats {pct:.0%} of shuffles")

    OUT.mkdir(exist_ok=True)
    solo.T.to_csv(OUT / "factor_solo.csv")
    log.to_csv(OUT / "factor_selection.csv")


if __name__ == "__main__":
    main()
