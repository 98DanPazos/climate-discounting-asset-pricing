"""
calibrate.py — estimate the discounting parameters from the project's data.

The original paper ASSUMED these numbers. Here each one comes from data:

Newell-Pizer   persistence (phi) and volatility (sigma) of log long-term
               government bond yields, annual averages, US (main) and Canada.
               Following Newell & Pizer (2003), the *uncertainty structure*
               is estimated from nominal yields (positive, long history) and
               then applied to a chosen *real* starting rate r0.
Starting rate  r0 = average ex-post real long rate (10y yield - CPI inflation).
Gamma          (a) Weitzman's (2001) survey of 2,160 economists, mu=3.96%,
                   sigma=2.94% — literature values, reported for comparison;
               (b) mean and std of annual US ex-post real rates — data values.
Ramsey SDF     mean and std of world GDP-per-capita growth (population-
               weighted, from the Step 1 panel); se of the mean = std/sqrt(T).
               Ethical parameters (rho, eta) are value judgements, not data,
               so we report two benchmark sets:
                 Stern (2007):    rho = 0.1%, eta = 1
                 Nordhaus (DICE): rho = 1.5%, eta = 1.45
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import DIR_PROCESSED, DIR_TABLES

WEITZMAN_SURVEY = {"mu": 0.0396, "sigma": 0.0294}
ETHICS = {
    "Stern":    {"rho": 0.001, "eta": 1.0},
    "Nordhaus": {"rho": 0.015, "eta": 1.45},
}


# ---------------------------------------------------------------------------
# Interest rates
# ---------------------------------------------------------------------------
def load_annual_rates() -> pd.DataFrame:
    """Annual averages of the monthly FRED series built in R/05_get_rates.R."""
    m = pd.read_csv(DIR_PROCESSED / "rates_monthly.csv", parse_dates=["date"])
    m["year"] = m["date"].dt.year
    # Keep only complete years so a partial current year doesn't bias averages
    counts = m.groupby("year")["date"].count()
    full_years = counts[counts == 12].index
    return m[m["year"].isin(full_years)].groupby("year").mean(numeric_only=True)


def fit_log_ar1(rate_pct: pd.Series) -> dict:
    """
    OLS on x_t = a + phi * x_{t-1} + e_t with x = log(rate).
    Returns phi, its standard error, the residual std (sigma, used for the
    AR(1) model) and the std of first differences (used for the random walk).
    """
    x = np.log(rate_pct.dropna() / 100.0)
    y, xlag = x.values[1:], x.values[:-1]
    X = np.column_stack([np.ones_like(xlag), xlag])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {
        "phi": float(beta[1]), "phi_se": float(se[1]),
        "sigma_ar1": float(np.sqrt(s2)),
        "sigma_rw": float(np.std(np.diff(x.values), ddof=1)),
        "years": [int(x.index.min()), int(x.index.max())], "n": int(len(x)),
    }


# ---------------------------------------------------------------------------
# Consumption growth (proxied by GDP per capita)
# ---------------------------------------------------------------------------
def world_growth_moments() -> dict:
    """
    World GDP-per-capita growth from the Step 1 panel, weighting each
    country's growth by its share of world GDP in the PREVIOUS year.

    Why GDP weights (not population weights)?  The Ramsey rule needs the
    growth of the world's average consumption. A GDP-weighted average of
    country growth rates approximates growth of aggregate world output.
    Population weights instead give China and India (fast growth, large
    populations) a weight far above their share of output, which pushes
    the mean to ~3% and inflates every Ramsey discount rate.
    Lagged weights avoid the mechanical link between a country's growth this
    year and its GDP share this year.
    """
    p = pd.read_csv(DIR_PROCESSED / "panel_country_year.csv")
    p = p.sort_values(["iso3c", "year"])
    p["gdp"] = p["gdp_pc"] * p["population"]
    p["gdp_lag"] = p.groupby("iso3c")["gdp"].shift(1)
    # only use a lag that really is last year (panel may have gaps)
    p.loc[p.groupby("iso3c")["year"].diff() != 1, "gdp_lag"] = np.nan
    p = p.dropna(subset=["growth", "gdp_lag"])
    w = (p.assign(wg=p["growth"] * p["gdp_lag"])
           .groupby("year")[["wg", "gdp_lag"]].sum())
    g = (w["wg"] / w["gdp_lag"]) / 100.0                 # pp -> decimal
    return {"mu_g": float(g.mean()), "sigma_g": float(g.std(ddof=1)),
            "se_mu": float(g.std(ddof=1) / np.sqrt(len(g))),
            "years": [int(g.index.min()), int(g.index.max())], "n": int(len(g))}


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------
def calibrate(save: bool = True) -> dict:
    rates = load_annual_rates()
    real_us = rates["us_real_lt"].dropna() / 100.0

    cal = {
        "newell_pizer": {
            "US": fit_log_ar1(rates["us_lt_yield"]),
            "Canada": fit_log_ar1(rates["ca_lt_yield"]),
        },
        "r0": {
            "us_mean_real_rate": float(real_us.mean()),
            "us_real_rate_std": float(real_us.std(ddof=1)),
            "years": [int(real_us.index.min()), int(real_us.index.max())],
            "latest_tips10": (float(rates["us_tips10"].dropna().iloc[-1] / 100)
                              if "us_tips10" in rates else None),
        },
        "gamma": {
            "weitzman_survey": WEITZMAN_SURVEY,
            "us_historical": {"mu": float(real_us.mean()),
                              "sigma": float(real_us.std(ddof=1))},
        },
        "growth": world_growth_moments(),
        "ethics": ETHICS,
    }
    if save:
        with open(DIR_TABLES / "discounting_calibration.json", "w") as f:
            json.dump(cal, f, indent=2)
    return cal
