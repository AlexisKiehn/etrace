"""Country climate-attractiveness scores derived from etrace outputs.

Two components, each z-scored across countries:

1. Tourism climate elasticity: the etrace stacking model's predicted nights spent
   (2041-2070) under 7 SSP scenarios. Socio-economic inputs are identical across
   scenarios, so the slope of log(nights) on scenario warming isolates the effect
   of climate on demand for a country.
2. Koppen shift: change in area share between 1991-2020 and the chosen projection
   of cold climates (subarctic/tundra), which get milder, versus hot or arid
   climates, which get harsher. Weighted by regional population.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "raw_data"

# Approximate mid-century (2041-2060) global warming in degC vs 1850-1900.
# SSP1-1.9 to SSP5-8.5 follow IPCC AR6 WG1 Table SPM.1; SSP4-3.4 and SSP4-6.0
# are not in that table and are interpolated.
SCENARIO_WARMING = {
    "ssp119": 1.6, "ssp126": 1.7, "ssp434": 1.9, "ssp245": 2.0,
    "ssp460": 2.0, "ssp370": 2.1, "ssp585": 2.4,
}

COLD = ["pct_Dfc", "pct_Dfd", "pct_Dsc", "pct_Dwc", "pct_ET", "pct_EF"]
HOT_ARID = ["pct_BWh", "pct_BWk", "pct_BSh", "pct_BSk", "pct_Csa"]


def _zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def tourism_elasticity(countries):
    """% change in predicted tourism nights per degC of scenario warming."""
    df = pd.read_csv(DATA / "FINAL_DATAFRAME_PREDICTIONS_V1", index_col=0)
    df["country"] = df.NUTS_ID.str[:2]
    df = df[df.country.isin(countries)]
    nights = df.groupby(["country", "scenario"]).pred_stacking.mean().unstack()
    warming = pd.Series(SCENARIO_WARMING)[nights.columns]
    x = warming - warming.mean()
    slopes = np.log(nights).sub(np.log(nights).mean(axis=1), axis=0) @ x / (x @ x)
    return slopes.rename("tourism_elasticity") * 100


def koppen_shift(countries, scenario="ssp245_2041_2070"):
    """Population-weighted (cold-climate loss) minus (hot/arid-climate gain)."""
    hist = pd.read_csv(DATA / "NUTS2_Climate_History_Smooth.csv")
    hist = hist[hist.YEAR == hist.YEAR.max()].set_index("NUTS_ID")
    proj = pd.read_csv(DATA / "NUTS2_SSP_Baseline_Projections.csv")
    proj = proj[proj.scenario == scenario].set_index("NUTS_ID")

    def share(df, cols):
        return df.reindex(columns=cols).fillna(0).sum(axis=1)

    cold_relief = share(hist, COLD) - share(proj, COLD)
    heat_stress = share(proj, HOT_ARID) - share(hist, HOT_ARID)
    score = (cold_relief - heat_stress).dropna()

    pop = pd.read_csv(DATA / "cleaned_final_interpolated_dataset.csv")
    # Latest available population per region (the file repeats region-years).
    pop = pop.dropna(subset=["pop"]).sort_values("year").drop_duplicates("geo", keep="last")
    pop = pop.set_index("geo")["pop"]
    df = pd.DataFrame({"score": score, "pop": pop}).dropna()
    df["country"] = df.index.str[:2]
    df = df[df.country.isin(countries)]
    weighted = df.groupby("country").apply(
        lambda g: np.average(g.score, weights=g["pop"]), include_groups=False
    )
    return weighted.rename("koppen_shift") * 100


def country_scores(countries, scenario="ssp245_2041_2070", koppen_weight=0.5):
    """Combined climate score per country, z-scored (positive = climate winner)."""
    t = tourism_elasticity(countries)
    k = koppen_shift(countries, scenario)
    out = pd.concat([t, k], axis=1)
    out["score"] = (1 - koppen_weight) * _zscore(out.tourism_elasticity) + koppen_weight * _zscore(
        out.koppen_shift
    )
    return out.sort_values("score", ascending=False)
