"""Market data, cached to disk: prices and dividends (Yahoo Finance), bond yields (FRED)."""

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent / "cache"

# OECD 10-year government bond yields on FRED (monthly average, %).
FRED_10Y = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01{}M156N"
FRED_CODES = {"EL": "GR"}  # FRED uses ISO codes; NUTS uses EL for Greece


def load_prices(tickers, start="2010-01-01", refresh=False):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"prices_{start}.csv"
    if path.exists() and not refresh:
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(tickers) <= set(prices.columns):
            return prices[tickers]

    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    prices = raw.reindex(columns=tickers)
    missing = prices.columns[prices.isna().all()].tolist()
    if missing:
        print(f"No price data for: {missing}")
    prices.to_csv(path)
    return prices


def load_dividends(tickers, start="2010-01-01", refresh=False):
    """Daily unadjusted closes and cash dividends per share, for dividend yields."""
    CACHE.mkdir(exist_ok=True)
    paths = CACHE / f"close_{start}.csv", CACHE / f"dividends_{start}.csv"
    if all(p.exists() for p in paths) and not refresh:
        close, divs = (pd.read_csv(p, index_col=0, parse_dates=True) for p in paths)
        if set(tickers) <= set(close.columns):
            return close[tickers], divs[tickers]

    raw = yf.download(tickers, start=start, auto_adjust=False, actions=True, progress=False)
    close = raw["Close"].reindex(columns=tickers)
    divs = raw["Dividends"].reindex(columns=tickers).fillna(0)
    close.to_csv(paths[0])
    divs.to_csv(paths[1])
    return close, divs


def load_bond_yields(countries, refresh=False):
    """Monthly 10-year government bond yields in %, indexed by month end."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "bond_yields_10y.csv"
    if path.exists() and not refresh:
        yields = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(countries) <= set(yields.columns):
            return yields[countries]

    series = {}
    for c in countries:
        df = pd.read_csv(FRED_10Y.format(FRED_CODES.get(c, c)), index_col=0, parse_dates=True)
        series[c] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    yields = pd.DataFrame(series)
    yields.index = yields.index + pd.offsets.MonthEnd(0)
    yields.to_csv(path)
    return yields
