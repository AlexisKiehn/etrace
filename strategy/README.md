# European climate macro strategy

A long/short macro strategy built on etrace's climate projections. It goes long
listed real estate in European countries that become more attractive as the
climate warms (the Nordics) and short the ones that lose out (Spain, Austria,
Germany). A small sleeve holds Nordic forestry and timber.

## Signal (`signals.py`)

The country climate score is an equal blend of two z-scored components:

| Component | Source | What it measures |
|---|---|---|
| Tourism climate elasticity | etrace stacking model predictions, 2041-2070, 7 SSPs | % change in predicted tourism nights per °C of scenario warming. Population and GDP inputs are identical across scenarios, so this isolates the effect of climate. |
| Köppen shift | Köppen-Geiger history (2020) vs SSP projection | Population-weighted loss of subarctic/tundra area minus gain of hot or arid area |

## Portfolio (`backtest.py`)

The book rebalances monthly:

1. Country weights are proportional to the demeaned score. Positive scores go long, negative scores go short, and each side gets equal gross.
2. Within a country, weights are inverse-volatility.
3. A trend filter drops a long whose 12-1 month momentum is negative, and a short whose momentum is positive.
4. Commodities get 20% of gross. Each name is capped at 15%.
5. The whole book is scaled to 10% annualised volatility, with leverage capped at 2x. Costs are 15 bps on turnover.
6. Returns are in local currency, so the book is assumed to be FX-hedged.

## Run

```bash
pip install -r requirements.txt
python -m strategy.run_backtest                      # base case, SSP2-4.5
python -m strategy.run_backtest --scenario ssp585_2071_2099 --long-only
python -m strategy.robustness                        # scenario sweep + placebo test
```

Outputs (returns, weights, scores, equity chart) are written to `strategy/output/`.

## Results (Feb 2011 to Oct 2026, net of costs)

| Variant (SSP2-4.5) | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Base long/short | 5.7% | 9.6% | 0.62 | -19% |
| Long-only | 8.7% | 10.2% | 0.87 | -19% |
| No trend filter | 2.0% | 8.0% | 0.29 | -16% |
| No commodity sleeve | 3.6% | 9.8% | 0.41 | -25% |
| EU real estate, equal weight | 8.7% | 17.3% | 0.57 | -33% |

The results barely change across SSP scenarios (Sharpe 0.61 to 0.64). In a placebo test, 200 random reassignments of the country scores had a median Sharpe of 0.27 and a 95th percentile of 0.61. The real climate scores reached 0.41, beating 76% of the shuffles. That is not statistically significant. Most of the return comes from the trend filter and the timber sleeve, not from the climate tilt.

## Caveats

- **The signal is static.** The scores come from 2041-2070 projections, so over the backtest the tilt never changes. Only the trend filter and vol scaling move positions. The backtest shows how such a tilt would have behaved, not that the signal predicted returns.
- **The tourism model is cross-sectional.** It learned that warm Mediterranean regions get a lot of tourists, so warming raises predicted tourism in Greece and Italy too. The Köppen component offsets this. Use `--koppen-weight` to change the blend.
- **The universe is thin.** Finland, Norway and Austria have one listed name each, and Italy, Greece, Poland and Denmark are proxied by country ETFs. Data comes from Yahoo Finance and has survivorship bias: delisted names such as Kojamo and Cofinimmo are excluded.
