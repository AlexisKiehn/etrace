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

The book rebalances monthly. By default each month's new positions are held for 12 months, so the book is an average of 12 overlapping slices (`--hold-months`; use 1 for the old fully-monthly rebalance). Each slice is built like this:

1. Country weights are proportional to the demeaned score. Positive scores go long, negative scores go short, and each side gets equal gross.
2. Within a country, weights are inverse-volatility.
3. A trend filter drops a long whose 12-1 month momentum is negative, and a short whose momentum is positive.
4. Commodities get 20% of gross. Each name is capped at 15%.
5. The whole book is scaled to 10% annualised volatility, with leverage capped at 2x. Costs are 15 bps on turnover.
6. Returns are in local currency, so the book is assumed to be FX-hedged.

## Run

```bash
pip install -r requirements.txt
python -m strategy.run_backtest                      # climate + momentum, 12-month hold, SSP2-4.5
python -m strategy.run_backtest --hold-months 6      # 6-month hold
python -m strategy.run_backtest --signals            # climate score alone (original)
python -m strategy.run_backtest --scenario ssp585_2071_2099 --long-only
python -m strategy.robustness                        # scenario sweep + placebo test
python -m strategy.signal_tests                      # momentum / yield spread / tourism surprise
python -m strategy.horizon_tests                     # all signals at 6 and 12 months (~30 min)
python -m strategy.theme_tests                       # Nordic offices, northern Sweden, forests
python -m strategy.buy_and_hold                      # buy once, never trade (~2 min)
python -m strategy.mechanism_tests                   # does warming lift cold regions? (~2 min)
```

Outputs (returns, weights, scores, equity chart) are written to `strategy/output/`.

## Results (Feb 2011 to Oct 2026, net of costs)

The default blends the climate score with country momentum and holds positions for 12 months: CAGR 7.2%, vol 9.0%, Sharpe 0.82, max drawdown −13%, turnover 13% a month. With monthly rebalancing the same blend has Sharpe 0.80 but a −20% drawdown and four times the turnover. The table below is the original climate-only strategy, rebalanced monthly.

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

## Holding period and more signals (`more_signals.py`, `horizon_tests.py`)

| Holding period | Climate only | Climate + momentum | Sharpe 2019+ (blend) | Turnover/month |
|---|---|---|---|---|
| 1 month | 0.63 | 0.80 | 0.51 | ~50% |
| 6 months | 0.77 | 0.84 | 0.68 | ~17% |
| 12 months | 0.80 | 0.82 | 0.78 | ~12% |
| 2 years | 0.74 | 0.76 | 0.71 | ~8% |
| 3 years | 0.79 | 0.85 | 0.79 | ~7% |
| 5 years | 0.69 | 0.74 | 0.59 | ~6% |

Holding 6-12 months cuts costs and stops the trend filter from flipping every month. Results are much steadier before and after 2018. Longer than a year adds nothing: the climate score never changes, so a longer hold just lets the trend filter and vol scaling go stale. With 16 years of data, a 3-5 year horizon also leaves only 3-5 independent periods.

Each signal was tested against 6- and 12-month forward returns. The rule is the same as before: placebo ≥ 95% and positive IC after 2018.

| Signal | Group | IC t, 12m | IC after 2018, 12m | Placebo 6m / 12m | Used |
|---|---|---|---|---|---|
| Momentum | market | 2.1 | 0.07 | 98% / 96% | yes |
| Tourism surprise | climate/etrace | 2.0 | 0.11 | 88% / 78% | no (closest) |
| House price momentum | economic | 1.9 | 0.03 | 81% / 86% | no |
| Net migration | economic | 1.2 | −0.08 | 74% / 72% | no |
| Political stability (World Bank) | political | 0.1 | −0.12 | 55% / 54% | no |
| Political stability, 3y change | political | −0.1 | −0.03 | 69% / 74% | no |
| 10y yield change (rates / fiscal stress) | political/rates | 0.7 | 0.09 | 69% / 78% | no |
| Restaurants & hotels share of spending | cultural | −0.3 | 0.12 | 55% / 50% | no |
| Dividend yield minus 10y yield | market | −0.6 | −0.09 | 39% / 44% | no |
| FDI inflows, 3y average | economic | −2.6 | −0.14 | 6% / 8% | no |
| Climate score (static) | climate | −0.5 | 0.08 | 46% / 51% | baseline |

- Political and cultural signals show nothing.
- House prices and net migration worked before 2018 and not after.
- FDI inflows are significantly *negative*: high-inflow countries underperformed. That was not the hypothesis. Holding-company flows (Netherlands, Switzerland, Belgium) make the series noisy, so it was not flipped after the fact.
- Economic policy uncertainty indices were skipped: FRED only has them for Germany and Italy.

## Themes (`theme_tests.py`)

Small baskets in local currency, so these are descriptive only:

| Theme | 2011-18 | 2019-21 | 2022-23 | 2024-26 | Spread Sharpe |
|---|---|---|---|---|---|
| Nordic capital offices (Hufvudstaden, Fabege, Atrium Ljungberg, Entra) minus Paris/Madrid offices | +6.6% | +7.0% | −2.0% | −12.8% | 0.11 |
| Northern Sweden (Diös) minus Swedish property | −2.7% | +9.9% | −7.0% | +0.3% | −0.02 |
| Northern forest owners (SCA, Holmen) minus pulp & paper (Stora Enso, UPM, Metsä) | +6.3% | +10.4% | +2.2% | +4.3% | 0.31 |
| Northern forest owners minus US timber REITs (Weyerhaeuser, Rayonier) | +13.7% | +8.0% | +3.4% | +6.4% | 0.38 |

- **Nordic offices** outperformed while rates were low and then lost it all. The basket falls about 13% per 1-point rise in Swedish/Norwegian 10y yields. The premium was a leveraged low-rate trade, not a capital-influx or startup effect. Free venture-capital data per country wasn't available to test the startup angle directly.
- **Northern Sweden** real estate shows no premium from the green-industry build-out (Northvolt, Stegra). Trend-following it lost money.
- **Northern forest owners** beat both comparison groups in every period, and are much less rate-sensitive (−2.5% per 1-point rise). Trend-timing the spread at 6 or 12 months did not help. It is two stocks, and the finding was not used to change the sleeve.

## Buy once and hold (`buy_and_hold.py`)

Long-only and unlevered, bought at one month end and never traded. There are no shorts, because a short held forever pays borrow fees and grows as it goes against you. There is also no trend filter or vol scaling. All 156 entry months from 2011 to 2023 were tested, each held to October 2026:

| Portfolio (FX-hedged) | Median CAGR | Worst CAGR | Entries in profit | Median / worst drawdown | Bought Jan 2011 |
|---|---|---|---|---|---|
| Climate winners + timber (80/20) | 8% | 3% | 100% | −31% / −40% | 10% |
| Climate winners only | 8% | 5% | 100% | −42% / −55% | 8% |
| All 14 countries | 7% | 5% | 100% | −34% / −40% | 6% |
| Timber only (SCA, Holmen, WOOD) | 7% | −8% | 76% | −37% / −42% | 14% |
| European equities (STOXX 600, EXSA.DE) | 9% | 7% | 100% | −23% / −23% | 8% |

Unhedged in euros, each property portfolio loses about a point a year (SEK and NOK weakened).

- **Every property portfolio made money from every entry month,** because European property rose over 2011-2026. That says more about the period, with falling rates until 2021, than about the strategy.
- **Picking the climate winners did not clearly help.** Shuffling the climate scores across countries and buying those random picks gave a median CAGR of 6.6%, against 7.8% for the real picks. The real picks beat 72% of 200 shuffles, which is not significant.
- **Simply buying European equities did better,** with higher returns, the best worst case and half the drawdown.
- **Buy-and-hold throws away the parts that tested significant.** Momentum and the trend filter both need trading.
- **Today's buy list is concentrated.** Citycon, the only Finnish name, would be 18% of the portfolio.

## Mechanism tests (`regional_data.py`, `mechanism_tests.py`)

Stock returns can't separate the climate thesis from interest rates, so these tests look at the mechanism directly, comparing regions within the same country. Regional temperatures are from NOAA GHCN-CAMS (0.5° monthly grid) averaged over NUTS polygons. House and forest prices come from Statistics Sweden, Statistics Norway, Statistics Finland and Luke (PxWeb APIs).

| Test | Thesis predicts | Result | p |
|---|---|---|---|
| Tourist nights after a 1°C warmer summer, 286 regions, 2000-2024 | up in cold regions, down in hot | cold +0.4%, hot −2.3%; gap 2.7 pp | 0.18 |
| Tourist nights 2000-03 → 2016-19 vs summer warming, 152 regions | same | cold +1.5%, hot −6.4% per °C; gap 7.9 pp | 0.47 |
| Nordic house prices 2005-2024, 49 regions: 1°C colder baseline | faster growth | +0.07 pp/yr | 0.03 |
| Same: 1°C more warming since the 1990s | faster growth | **−1.1 pp/yr** | 0.006 |
| Swedish forest-property prices by county, 1°C colder | faster growth | −0.04 pp/yr (Norrland 6.3% vs rest 6.8% a year) | 0.64 |
| Finnish stumpage prices 2016-2025, 1°C colder region | faster growth | +0.12 pp/yr (north 6.4% vs rest 6.0% a year) | 0.035 |

House price regressions control for population growth and use country fixed effects. The tourism panel uses region and country-year fixed effects with standard errors clustered by region.

- **Tourism moves in the predicted direction but is far too small and noisy to confirm.** Regional warming so far (about 0.6°C in summer since 2000) is small.
- **Northern house prices held up better than population trends predict**, especially in Sweden (4.9% vs 4.5% a year with much weaker population growth). But regions that warmed *more* saw *slower* growth, the opposite of the mechanism. Within these countries the south warmed more than the north (about 1.2-1.4°C vs 0.6-0.9°C in Sweden). The northern strength more likely reflects the mining and green-industry investment in Norrbotten and Västerbotten than climate.
- **No northern premium in Swedish forest land.** The small Finnish stumpage effect fits a demand shock (Metsä Fibre's Kemi mill in Lapland opened in 2023) as well as climate.
- Regional forest growth could not be tested: the Swedish forest inventory (SLU) API was down.

Overall: the mechanism is not visible yet. Warming so far is a fraction of what is projected, so this is "not yet detectable", not "false".

## Caveats

- **Many things were tested.** Nine factors and signals in total. Momentum's placebo result (better than all 200 shuffles, p < 0.005) survives a Bonferroni correction for that. The blend's 97% does not.
- **Momentum overlaps the trend filter.** Both use 12-1 month returns, so the default book leans heavily on one well-known effect.
- **The climate score is static.** The scores come from 2041-2070 projections, so over the backtest the tilt never changes. Only the trend filter and vol scaling move positions. The backtest shows how such a tilt would have behaved, not that the signal predicted returns.
- **The tourism model is cross-sectional.** It learned that warm Mediterranean regions get a lot of tourists, so warming raises predicted tourism in Greece and Italy too. The Köppen component offsets this. Use `--koppen-weight` to change the blend.
- **The universe is thin.** Finland, Norway and Austria have one listed name each, and Italy, Greece, Poland and Denmark are proxied by country ETFs. Data comes from Yahoo Finance and has survivorship bias: delisted names such as Kojamo and Cofinimmo are excluded.
