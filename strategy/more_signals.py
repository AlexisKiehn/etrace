"""Political, cultural and economic country signals, point-in-time.

Each returns a DataFrame indexed by the date the value is assumed public, one column
per country, oriented so that higher = more attractive (signs fixed here, before any
test). Annual data for year Y is used from 1 January Y+2; quarterly house prices one
quarter after the quarter ends; monthly bond yields one month later.

Not included: economic policy uncertainty indices (FRED only has Germany and Italy),
and a 10y spread to Germany, which is the same as the yield change below once
z-scored across countries.
"""

import numpy as np
import pandas as pd

from strategy.data import load_bond_yields, load_eurostat, load_world_bank


def _annual_available(df, lag_years=2):
    """Re-date annual data to when it is public. Countries that report late keep their
    previous value for up to two years, so one early reporter doesn't stand alone."""
    out = df.dropna(how="all").ffill(limit=2)
    out.index = pd.to_datetime([f"{d.year + lag_years}-01-01" for d in out.index])
    return out


# --- Political ----------------------------------------------------------------------

def political_stability(countries):
    """World Bank governance indicator: political stability and absence of violence."""
    return _annual_available(load_world_bank("GOV_WGI_PV.EST", countries, source=3))


def political_stability_change(countries, years=3):
    """Change in political stability over 3 years: improving politics."""
    wgi = load_world_bank("GOV_WGI_PV.EST", countries, source=3)
    return _annual_available(wgi.diff(years))


def yield_change(countries):
    """Minus the 12-month change in the 10y government bond yield.

    Captures falling rates and easing political/fiscal stress (across countries this
    is the same as the change in the spread to Germany).
    """
    y = load_bond_yields(countries)
    return -(y - y.shift(12)).shift(1)


# --- Cultural -----------------------------------------------------------------------

def hospitality_share(countries):
    """Restaurants and hotels as a share of household spending (food/hospitality culture).

    Includes spending by foreign tourists, so it partly measures tourism too.
    """
    raw = load_eurostat("nama_10_co3_p3", "A.CP_MEUR.CP11+TOTAL." + "+".join(countries))
    share = raw.xs("CP11", axis=1, level=0) / raw.xs("TOTAL", axis=1, level=0)
    return _annual_available(share.reindex(columns=countries))


# --- Economic -----------------------------------------------------------------------

def house_price_momentum(countries):
    """Year-on-year change in the Eurostat house price index (no data for Greece, Switzerland)."""
    have = [c for c in countries if c not in ("EL", "CH")]
    hpi = load_eurostat("prc_hpi_q", "Q.TOTAL.I15_Q." + "+".join(have))
    mom = np.log(hpi).diff(4)
    mom.index = mom.index + pd.offsets.QuarterEnd(1)  # published about a quarter later
    return mom.dropna(how="all").ffill(limit=2).reindex(columns=countries)


def net_migration(countries):
    """Crude net migration rate per 1,000 inhabitants: demand for housing and offices."""
    return _annual_available(load_eurostat("demo_gind", "A.CNMIGRATRT." + "+".join(countries))
                             .reindex(columns=countries))


def fdi_inflows(countries, years=3):
    """Foreign direct investment inflows, % of GDP, 3-year average (capital influx).

    Noisy for countries with large holding-company flows (Netherlands, Belgium).
    """
    fdi = load_world_bank("BX.KLT.DINV.WD.GD.ZS", countries)
    return _annual_available(fdi.rolling(years).mean())


SIGNALS = {
    "political_stability": ("political", political_stability),
    "political_stability_change": ("political", political_stability_change),
    "yield_change": ("political/rates", yield_change),
    "hospitality_share": ("cultural", hospitality_share),
    "house_price_momentum": ("economic", house_price_momentum),
    "net_migration": ("economic", net_migration),
    "fdi_inflows": ("economic", fdi_inflows),
}
