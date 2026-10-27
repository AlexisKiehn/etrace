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
python -m strategy.run_backtest                      # climate + momentum, SSP2-4.5
python -m strategy.run_backtest --signals            # climate score alone (original)
python -m strategy.run_backtest --scenario ssp585_2071_2099 --long-only
python -m strategy.robustness                        # scenario sweep + placebo test
python -m strategy.signal_tests                      # momentum / yield spread / tourism surprise
```

Outputs (returns, weights, scores, equity chart) are written to `strategy/output/`.

## Results (Feb 2011 to Oct 2026, net of costs)

The default now blends the climate score with country momentum (see below): CAGR 7.7%, vol 10.0%, Sharpe 0.80, max drawdown −20%, t-stat 3.2. The table below is the original climate-only strategy.

| Variant (SSP2-4.5, climate only) | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Base long/short | 5.7% | 9.6% | 0.62 | -19% |
| Long-only | 8.7% | 10.2% | 0.87 | -19% |
| No trend filter | 2.0% | 8.0% | 0.29 | -16% |
| No commodity sleeve | 3.6% | 9.8% | 0.41 | -25% |
| EU real estate, equal weight | 8.7% | 17.3% | 0.57 | -33% |

The results barely change across SSP scenarios (Sharpe 0.61 to 0.64). In a placebo test, 200 random reassignments of the country scores had a median Sharpe of 0.27 and a 95th percentile of 0.61. The real climate scores reached 0.41, beating 76% of the shuffles. That is not statistically significant. Most of the return comes from the trend filter and the timber sleeve, not from the climate tilt.

## Extra factors (`factors.py`, `factor_selection.py`)

```bash
python -m strategy.factor_selection
```

Six candidate factors were tested against the climate score. Each factor's sign was fixed before testing. GDP and population data are lagged two years to avoid look-ahead.

| Factor | Sign | Source |
|---|---|---|
| GDP per head (PPS, log) | + | Eurostat nama_10r_2gdp |
| GDP per head growth, 3y | + | Eurostat nama_10r_2gdp |
| Population growth, 3y | + | Eurostat demo_r_d2jan |
| Coastline per head | − | NUTS 2024 1:1M geometries |
| Coastal population share | − | NUTS2 coastal flag |
| Observed Köppen shift, 1901-1930 to 1991-2020 | + | Köppen-Geiger history |

Factors were added one at a time and kept if they raised the Sharpe up to 2018 by at least 0.05. 2019 onwards was held out.

| | Sharpe to 2018 | Sharpe 2019+ | Placebo |
|---|---|---|---|
| Climate score only | 0.84 | 0.43 | beats 76% |
| + GDP level + GDP growth (selected) | 1.33 | −0.03 | beats 64% |

The selected factors fail on the held-out years, so they are **not** used. Every factor looked good before 2018 (in-sample Sharpe 1.0 to 1.3) and none held up afterwards. Population growth on its own came closest: Sharpe 0.35 after 2018, and it beats 91% of placebo shuffles. That is still not significant, and it was only singled out after seeing the held-out results.

## Time-varying signals (`dynamic_signals.py`, `signal_tests.py`)

| Signal | Definition | Data |
|---|---|---|
| Momentum | 12-1 month return of the country's listed real estate | Prices |
| Yield spread | Trailing 12m dividend yield minus 10y government bond yield (bond yield lagged one month) | Yahoo dividends, FRED/OECD yields |
| Tourism surprise | Change in log(actual / predicted) tourist nights. Predictions come from etrace's XGBoost setup, refitted walk-forward so each year is predicted only from earlier years. Year Y is used from July of Y+1. | Eurostat tour_occ_nin2 plus etrace features |

Rule, fixed before running: use a signal if its pure tilt (no trend filter, no commodities) beats ≥95% of country-shuffled placebos **and** its IC after 2018 is positive.

| Signal | IC (t), full | IC after 2018 | Pure tilt Sharpe | Placebo | Used |
|---|---|---|---|---|---|
| Climate score (static) | −0.01 (−0.4) | 0.03 | −0.12 | 46% | (baseline) |
| Momentum | 0.06 (2.0) | 0.06 | 0.62 | 100% | **yes** |
| Yield spread | −0.01 (−0.1) | −0.04 | −0.09 | 36% | no |
| Tourism surprise | 0.05 (2.0) | 0.06 | 0.25 | 82% | no |

| Full strategy | Sharpe to 2018 | Sharpe 2019+ | Sharpe full |
|---|---|---|---|
| Climate only | 0.84 | 0.43 | 0.63 |
| Climate + momentum | 1.10 | 0.51 | 0.80 |

The improvement comes from momentum. Momentum used alone scores about the same (Sharpe 0.83), and the climate tilt on its own loses money (pure tilt Sharpe −0.12). Tourism surprise is the one worth watching: its IC is positive in both halves and significant after 2018 (t = 2.2), but the placebo (82%) misses the bar.

## Caveats

- **Many things were tested.** Nine factors and signals in total. Momentum's placebo result (better than all 200 shuffles, p < 0.005) survives a Bonferroni correction for that. The blend's 97% does not.
- **Momentum overlaps the trend filter.** Both use 12-1 month returns, so the default book leans heavily on one well-known effect.
- **The climate score is static.** The scores come from 2041-2070 projections, so over the backtest the tilt never changes. Only the trend filter and vol scaling move positions. The backtest shows how such a tilt would have behaved, not that the signal predicted returns.
- **The tourism model is cross-sectional.** It learned that warm Mediterranean regions get a lot of tourists, so warming raises predicted tourism in Greece and Italy too. The Köppen component offsets this. Use `--koppen-weight` to change the blend.
- **The universe is thin.** Finland, Norway and Austria have one listed name each, and Italy, Greece, Poland and Denmark are proxied by country ETFs. Data comes from Yahoo Finance and has survivorship bias: delisted names such as Kojamo and Cofinimmo are excluded.
