"""The investor's own forward-looking views, recorded before any testing.

These are kept separate from the data-driven traits because they cannot be backtested
without hindsight. They are tracked forward from RECORDED: the country model gives
them 30% of the final score (data traits get 70%).

Scores run from -2 (much worse) to +2 (much better) over the next 3-5+ years.
"""

RECORDED = "2026-10-27"

# Overall outlook: track record of turning crises into reform, strategic position,
# fiscal health, and how prepared the country is for the US-China contest.
OUTLOOK = {
    "DEU": +2,  # strongest EU member, between the US and China; must rearm and gain data independence
    "USA": -1,  # +2 on history, but in decline: unbearable debt, individualism, unfair to poorer workers
    "GBR": 0,
    "SWE": +1, "NOR": +1, "FIN": +1, "DNK": +1,  # Nordics
    "FRA": -1,  # needs drastic reform of education and public finances
    "ITA": -1,
    "CHN": +2,  # lifted millions out of poverty; but over-centralised, capital misallocated
    "ESP": +1,
    "POL": +2,
    "RUS": -1,  # resources and military, but violence, brain drain, weak innovation
    "JPN": -1,  # debt
    "KOR": -2,  # debt
    "CAN": 0, "NLD": 0, "AUT": 0, "BEL": 0,
    "CHE": -1,
    "IND": +1,
}

# Social cohesion trend: a shared goal while keeping some individualism.
COHESION = {
    "USA": -2, "CHN": +1, "DEU": -1, "FRA": -2, "GBR": -1, "RUS": -1, "POL": +2,
    "SWE": 0, "NOR": 0, "FIN": 0, "DNK": 0,
}

# Themes not yet turned into factors (candidates for sector tilts):
THEMES = {
    "sectors": ["energy (cheap, clean, abundant power)", "defence", "data / sovereign tech",
                "governing / govtech"],
    "advantages": ["energy, military, data and education autarky", "cold climates with cheap power "
                   "(Nordics, Canada)", "minerals for electrification", "ports"],
    "risks": ["unsustainable public debt (US, Japan, Korea, France)", "tropics get worse",
              "countries unprepared for the US-China contest lose most"],
    "society": ["transparency and honesty", "openness to people who contribute",
                "education as the biggest lever", "children and families", "long-term thinking",
                "protecting young people from social media, gambling and violence"],
}


def expert_score():
    """Average of outlook and cohesion where both exist, per ISO3 country."""
    countries = sorted(set(OUTLOOK) | set(COHESION))
    return {c: sum(v[c] for v in (OUTLOOK, COHESION) if c in v)
            / sum(c in v for v in (OUTLOOK, COHESION)) for c in countries}
