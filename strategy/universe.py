"""Tradeable universe: listed real estate per country plus a climate commodity sleeve.

Tickers are Yahoo Finance symbols. Returns are taken in each instrument's local
currency, i.e. the book is assumed to be FX-hedged.
"""

# Country (NUTS country code) -> listed real estate (REITs / property companies),
# with a country equity ETF as fallback where listed real estate is thin.
REAL_ESTATE = {
    "SE": ["CAST.ST", "FABG.ST", "BALD-B.ST", "WALL-B.ST", "SAGA-B.ST"],
    "NO": ["ENTRA.OL"],
    "FI": ["CTY1S.HE"],
    "DK": ["EDEN"],
    "DE": ["VNA.DE", "LEG.DE", "TEG.DE"],
    "NL": ["ECMPA.AS", "WHA.AS"],
    "BE": ["WDP.BR", "AED.BR"],
    "FR": ["GFC.PA", "COV.PA", "LI.PA"],
    "AT": ["CAI.VI"],
    "CH": ["PSPN.SW", "SPSN.SW"],
    "PL": ["EPOL"],
    "ES": ["MRL.MC", "COL.MC"],
    "IT": ["EWI"],
    "EL": ["GREK"],
}

# Climate commodity sleeve: ticker -> thesis direction (+1 long / -1 short).
# A longer, warmer growing season in the Nordics raises forest growth and yields.
COMMODITIES = {
    "SCA-B.ST": +1,   # Svenska Cellulosa: largest private forest owner in Europe
    "HOLM-B.ST": +1,  # Holmen: Swedish forestry
    "WOOD": +1,       # iShares global timber & forestry
}

COUNTRY_NAMES = {
    "SE": "Sweden", "NO": "Norway", "FI": "Finland", "DK": "Denmark",
    "DE": "Germany", "NL": "Netherlands", "BE": "Belgium", "FR": "France",
    "AT": "Austria", "CH": "Switzerland", "PL": "Poland", "ES": "Spain",
    "IT": "Italy", "EL": "Greece",
}


def all_tickers():
    tickers = [t for ts in REAL_ESTATE.values() for t in ts]
    return tickers + list(COMMODITIES)
