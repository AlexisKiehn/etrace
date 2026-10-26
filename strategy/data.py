"""Daily adjusted close prices from Yahoo Finance, cached to disk."""

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent / "cache"


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
