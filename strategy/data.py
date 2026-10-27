"""Market data, cached to disk: prices and dividends (Yahoo Finance), bond yields (FRED)."""

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent / "cache"

# OECD 10-year government bond yields on FRED (monthly average, %).
FRED_10Y = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01{}M156N"
FRED_CODES = {"EL": "GR"}  # FRED uses ISO codes; NUTS uses EL for Greece


def load_prices(tickers, start="2010-01-01", refresh=False, name="prices"):
    """Daily total-return prices. `name` picks the cache file, so ad-hoc ticker sets
    don't overwrite the main universe's cached prices."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}_{start}.csv"
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


EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{}/{}?format=TSV&compressed=false"
WORLD_BANK_API = ("https://api.worldbank.org/v2/country/{}/indicator/{}"
                  "?format=json&date=1995:2030&per_page=20000{}")
ISO2 = {"EL": "GR"}  # NUTS uses EL for Greece


def _period(label):
    """Eurostat period label ('2015', '2015-Q3', '2015-07') -> period end date."""
    label = label.strip()
    if "-Q" in label:
        y, q = label.split("-Q")
        return pd.Period(year=int(y), quarter=int(q), freq="Q").end_time.normalize()
    if "-" in label:
        return pd.Period(label, freq="M").end_time.normalize()
    return pd.Timestamp(f"{label}-12-31")


def load_eurostat(dataset, key, refresh=False):
    """Eurostat series via the SDMX API, e.g. load_eurostat('prc_hpi_q', 'Q.TOTAL.I15_Q.SE+DE').

    Returns a date x geo table (dates are period ends); flags such as 'p' are dropped.
    """
    CACHE.mkdir(exist_ok=True)
    name = key.replace("+", "_")
    if len(name) > 100:  # long geo lists: keep the file name short
        name = hashlib.md5(key.encode()).hexdigest()[:12]
    path = CACHE / f"estat_{dataset}_{name}.tsv"
    if not path.exists() or refresh:
        df = pd.read_csv(EUROSTAT_API.format(dataset, key), sep="\t", dtype=str)
        df.to_csv(path, sep="\t", index=False)
    df = pd.read_csv(path, sep="\t", dtype=str)
    first = df.columns[0]
    dims = df[first].str.split(",", expand=True)
    dims = dims.loc[:, dims.nunique() > 1]  # keep only the dimensions that vary
    df.index = dims.iloc[:, 0] if dims.shape[1] == 1 else pd.MultiIndex.from_frame(dims)
    vals = df.drop(columns=first)
    vals.columns = [_period(c) for c in vals.columns]
    vals = vals.apply(lambda s: pd.to_numeric(s.str.extract(r"([-\d.]+)")[0], errors="coerce"))
    return vals.T.sort_index()


def load_world_bank(indicator, countries, source=None, refresh=False):
    """Annual World Bank indicator, year-end dates x countries (NUTS codes)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"wb_{indicator}.csv"
    if path.exists() and not refresh:
        out = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(countries) <= set(out.columns):
            return out[countries]
    codes = ";".join(ISO2.get(c, c) for c in countries)
    url = WORLD_BANK_API.format(codes, indicator, f"&source={source}" if source else "")
    with urllib.request.urlopen(url, timeout=60) as resp:
        rows = json.load(resp)[1]
    back = {ISO2.get(c, c): c for c in countries}
    df = pd.DataFrame([
        {"date": pd.Timestamp(f"{r['date']}-12-31"), "country": back[r["country"]["id"]], "value": r["value"]}
        for r in rows if r["country"]["id"] in back
    ])
    out = df.pivot(index="date", columns="country", values="value").sort_index()
    out.to_csv(path)
    return out[countries]
