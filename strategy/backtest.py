"""Monthly-rebalanced climate macro backtest.

Portfolio construction at each month end:
  1. Country weight proportional to its climate score (long winners, short losers
     in long/short mode; only positive scores in long-only mode).
  2. Within a country, inverse-volatility weights across its instruments.
  3. Trend filter: a position is dropped when the instrument's 12-1 month momentum
     disagrees with its direction (no longs in a downtrend, no shorts in an uptrend).
  4. Real estate book and commodity sleeve are scaled to their risk budgets, then the
     whole portfolio is scaled to a target volatility with a leverage cap.
With hold_months = H > 1 the book is H overlapping tranches: each month a new tranche
is formed from that month's signal and kept for H months, and the portfolio is their
average. This gives an H-month holding period while still producing monthly returns.
With staggered = False the whole book is instead rebalanced once every H months and
left to drift in between. Costs are charged on turnover.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy.universe import COMMODITIES, REAL_ESTATE


@dataclass
class Config:
    long_short: bool = True
    trend_filter: bool = True
    commodity_weight: float = 0.2   # share of gross exposure in the commodity sleeve
    vol_target: float = 0.10        # annualised
    max_leverage: float = 2.0
    cost_bps: float = 15.0          # per unit of turnover
    vol_lookback: int = 63          # trading days
    cov_lookback: int = 126
    max_position: float = 0.15      # cap per instrument, as a share of gross before vol scaling
    hold_months: int = 1            # each month's new positions are held this long (overlapping)
    staggered: bool = True          # False: trade the whole book once every hold_months instead
                                    # (hold_months=0: buy once and never trade again)
    offset: int = 0                 # with staggered=False, which month of the cycle to trade in


def _momentum(monthly, date):
    """12-1 month total return as of `date`."""
    hist = monthly.loc[:date]
    if len(hist) < 13:
        return pd.Series(np.nan, index=monthly.columns)
    return hist.iloc[-2] / hist.iloc[-13] - 1


def _inv_vol(rets, tickers):
    vol = rets[tickers].std()
    w = (1 / vol).replace([np.inf], np.nan).dropna()
    return w / w.sum() if len(w) else w


def target_weights(scores, daily_rets, monthly_px, date, cfg):
    window = daily_rets.loc[:date].iloc[-cfg.vol_lookback:]
    live = window.columns[window.notna().sum() > cfg.vol_lookback * 0.8]
    mom = _momentum(monthly_px, date)

    country = scores if cfg.long_short else scores.clip(lower=0)
    if cfg.long_short:
        country = country - country.mean()

    re_w = pd.Series(dtype=float)
    for c, s in country.items():
        tickers = [t for t in REAL_ESTATE.get(c, []) if t in live]
        if not tickers or s == 0:
            continue
        re_w = pd.concat([re_w, s * _inv_vol(window, tickers)])

    com_w = pd.Series({t: d for t, d in COMMODITIES.items() if t in live}, dtype=float)
    if len(com_w):
        com_w = com_w * _inv_vol(window, list(com_w.index))

    if cfg.trend_filter:
        re_w = re_w[np.sign(mom.reindex(re_w.index)) == np.sign(re_w)]
        com_w = com_w[np.sign(mom.reindex(com_w.index)) == np.sign(com_w)]

    # Each long/short side of the real estate book gets equal gross; empty side stays cash.
    re_budget = 1 - cfg.commodity_weight
    parts = []
    for side in (re_w[re_w > 0], re_w[re_w < 0]):
        if len(side):
            parts.append(side / side.abs().sum() * re_budget / (2 if cfg.long_short else 1))
    if len(com_w):
        parts.append(com_w / com_w.abs().sum() * cfg.commodity_weight)
    if not parts:
        return pd.Series(dtype=float)
    w = pd.concat(parts).groupby(level=0).sum()
    w = w.clip(-cfg.max_position, cfg.max_position)

    cov_win = daily_rets.loc[:date, w.index].iloc[-cfg.cov_lookback:].fillna(0)
    port_vol = np.sqrt(w @ cov_win.cov() @ w * 252)
    if port_vol > 0:
        w *= min(cfg.vol_target / port_vol, cfg.max_leverage / w.abs().sum())
    return w


def run(scores, prices, cfg=Config()):
    # Forward-fill across exchange holidays so returns aren't lost on the following day.
    prices = prices.ffill()
    daily_rets = prices.pct_change(fill_method=None)
    monthly_px = prices.resample("ME").last()
    monthly_rets = monthly_px.pct_change(fill_method=None)
    dates = monthly_px.index

    weights = {}
    prev = pd.Series(dtype=float)
    tranches = deque(maxlen=cfg.hold_months)
    rows = []
    for i in range(12, len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        # Time-varying scores (rows = publication date) are used as of the rebalance date.
        if isinstance(scores, pd.DataFrame):
            known = scores.loc[:d]
            s = known.iloc[-1].dropna() if len(known) else pd.Series(dtype=float)
        else:
            s = scores
        if cfg.staggered:
            tranches.append(target_weights(s, daily_rets, monthly_px, d, cfg))
            w = pd.concat(list(tranches), axis=1).fillna(0).sum(axis=1) / len(tranches)
        elif i == 12 or (cfg.hold_months and (i - 12 - cfg.offset) % cfg.hold_months == 0):
            w = target_weights(s, daily_rets, monthly_px, d, cfg)
        else:
            w = prev  # hold: positions drift with prices, no trading
        turnover = w.subtract(prev, fill_value=0).abs().sum()
        r = (w * monthly_rets.loc[nxt, w.index].fillna(0)).sum()
        rows.append({"date": nxt, "gross_ret": r, "turnover": turnover,
                     "net_ret": r - turnover * cfg.cost_bps / 1e4,
                     "gross_exposure": w.abs().sum(), "net_exposure": w.sum()})
        weights[d] = w
        # Drift weights through the month so next turnover is measured correctly.
        grown = w * (1 + monthly_rets.loc[nxt, w.index].fillna(0))
        prev = grown / (1 + r) if len(w) else w

    result = pd.DataFrame(rows).set_index("date")
    result["equity"] = (1 + result.net_ret).cumprod()
    return result, pd.DataFrame(weights).T.fillna(0)


def benchmark(prices, start):
    """Equal-weight long-only basket of all European listed real estate in the universe."""
    tickers = [t for ts in REAL_ESTATE.values() for t in ts]
    monthly = prices[tickers].ffill().resample("ME").last().pct_change(fill_method=None)
    r = monthly.loc[start:].mean(axis=1)
    return r


def stats(r, periods=12):
    r = r.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / periods
    cagr = equity.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(periods)
    dd = (equity / equity.cummax() - 1).min()
    return pd.Series({
        "CAGR": cagr, "Vol": vol, "Sharpe": r.mean() / r.std() * np.sqrt(periods),
        "MaxDD": dd, "Calmar": cagr / abs(dd) if dd else np.nan,
        "HitRate": (r > 0).mean(), "Years": years,
    })
