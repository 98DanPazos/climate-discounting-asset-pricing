"""
scc.py — the social cost of carbon (SCC): Module 1 damages x Module 2 discounting.

The SCC is the present value of all future damages caused by emitting ONE
extra tonne of CO2 today:

    SCC = sum_{t=1..H}  DF(t) * D(t)          [$ per tonne]

where D(t) is the extra global damage in year t and DF(t) a discount schedule.

Chain of the calculation
------------------------
1. Climate:   one tonne raises global temperature permanently by
              delta = TCRE = 0.45 °C per 1,000 GtCO2 = 0.45e-12 °C per tonne
              (IPCC AR6 best estimate; likely range 0.27-0.63). Temperature
              is roughly proportional to cumulative emissions and stays up
              after emissions stop, so the pulse is a permanent step.
2. Baseline:  each country warms along a scenario (default: +1.3 °C more by
              2100, roughly SSP2-4.5, then flat) and its GDP grows along a
              stylised path (per-capita growth easing from 2% to 1% by 2100;
              population growth fading to zero by 2080).
3. Damages:   Module 1 gives growth's response to temperature:
                  d growth / dT = b1 + 2 b2 T      (pp per °C)
              so the pulse changes country i's growth in year t by
                  m_i(t) = (b1 + 2 b2 T_i(t)) * delta / 100      (log points)
              and its log-GDP by
                  L_i(t) = lambda * L_i(t-1) + m_i(t)
              * lambda = 1  "growth effect": losses accumulate forever
                            (what the regression literally estimates)
              * lambda = 0  "level effect": each year's loss is one-off
                            (output bounces back the next year)
              Dollar damage: D(t) = - sum_i Y_i(t) * L_i(t)
              (minus sign so that harm is a POSITIVE cost).
4. Linearity: m is linear in (b1, b2), so D(t) = b1*A(t) + b2*B(t) and
                  SCC = b1 * a + b2 * b,   a = sum DF*A,  b = sum DF*B
              The SCC's standard error then follows EXACTLY from the
              regression's covariance matrix V:  Var(SCC) = [a b] V [a b]'.

Units: GDP in constant 2015 US$ (World Bank), so the SCC is in 2015 US$ per
tonne of CO2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TCRE_BEST = 0.45e-12          # °C per tonne CO2 (IPCC AR6)
TCRE_LIKELY = (0.27e-12, 0.63e-12)


@dataclass
class Scenario:
    """All the assumptions that are not estimated, in one place."""
    base_year: int = 2025
    end_year: int = 2300                  # damages truncated here (EPA uses 2300)
    tcre: float = TCRE_BEST               # °C per tonne
    warming_by_2100: float = 1.3          # extra global warming, base year -> 2100
    gpc_growth_now: float = 0.020         # GDP per capita growth today
    gpc_growth_2100: float = 0.010        # ... easing linearly to this by 2100
    pop_growth_now: float = 0.008
    pop_zero_year: int = 2080             # population growth fades to 0 by here
    persistence: float = 1.0              # lambda: 1 = growth effect, 0 = level

    @property
    def horizon(self) -> int:
        return self.end_year - self.base_year

    def years(self) -> np.ndarray:
        return np.arange(1, self.horizon + 1)          # t = 1..H

    def calendar(self) -> np.ndarray:
        return self.base_year + self.years()

    def gdp_index(self) -> np.ndarray:
        """GDP relative to the base year, common to all countries."""
        cal = self.calendar()
        frac_2100 = np.clip((cal - self.base_year) / (2100 - self.base_year), 0, 1)
        g_pc = self.gpc_growth_now + (self.gpc_growth_2100 - self.gpc_growth_now) * frac_2100
        frac_pop = np.clip((cal - self.base_year) / (self.pop_zero_year - self.base_year), 0, 1)
        g_pop = self.pop_growth_now * (1 - frac_pop)
        return np.exp(np.cumsum(g_pc + g_pop))

    def warming(self) -> np.ndarray:
        """Baseline global warming above today, by year (°C)."""
        frac = np.clip(self.years() / (2100 - self.base_year), 0, 1)
        return self.warming_by_2100 * frac


@dataclass
class Countries:
    """Base-year inputs, one entry per country."""
    iso3c: np.ndarray
    gdp: np.ndarray          # base-year GDP, constant 2015 US$
    temp: np.ndarray         # baseline annual mean temperature, °C
    poor: np.ndarray         # 1 = poor (Module 1 split), 0 = rich

    def subset(self, mask: np.ndarray) -> "Countries":
        return Countries(self.iso3c[mask], self.gdp[mask], self.temp[mask], self.poor[mask])


def countries_from_panel(panel: pd.DataFrame, temp_years: int = 10) -> tuple[Countries, int]:
    """
    Build base-year inputs from the Module 1 panel:
      * GDP = latest available GDP per capita x population (within 3 years of
        the latest panel year, so countries with stale data drop out);
      * baseline temperature = mean of the last `temp_years` years (smooths
        out one-off hot/cold years);
      * poor = the Module 1 rich/poor flag.
    Returns the inputs and the base year.
    """
    p = panel.dropna(subset=["gdp_pc", "population"]).copy()
    base_year = int(p["year"].max())
    latest = (p[p["year"] >= base_year - 3]
              .sort_values("year").groupby("iso3c").tail(1)
              .set_index("iso3c"))
    t = panel.dropna(subset=["temp"])
    temp = (t[t["year"] > t["year"].max() - temp_years]
            .groupby("iso3c")["temp"].mean())
    df = latest.join(temp.rename("temp_base"), how="inner").dropna(subset=["temp_base", "poor"])
    c = Countries(iso3c=df.index.to_numpy(),
                  gdp=(df["gdp_pc"] * df["population"]).to_numpy(float),
                  temp=df["temp_base"].to_numpy(float),
                  poor=df["poor"].to_numpy(int))
    return c, base_year


def damage_components(c: Countries, sc: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """
    Return A(t), B(t) such that global damage D(t) = b1*A(t) + b2*B(t),
    in $ per tonne, for t = 1..H. (Sign: positive = harm.)
    """
    H, lam = sc.horizon, sc.persistence
    delta = sc.tcre
    T = c.temp[:, None] + sc.warming()[None, :]          # countries x years
    Y = c.gdp[:, None] * sc.gdp_index()[None, :]

    # Per-unit-coefficient growth shocks (log points per year)
    mA = np.full_like(T, delta / 100.0)                  # from b1
    mB = 2.0 * T * delta / 100.0                         # from b2

    # Accumulate into log-GDP losses with persistence lambda
    LA, LB = np.empty_like(mA), np.empty_like(mB)
    LA[:, 0], LB[:, 0] = mA[:, 0], mB[:, 0]
    for t in range(1, H):
        LA[:, t] = lam * LA[:, t - 1] + mA[:, t]
        LB[:, t] = lam * LB[:, t - 1] + mB[:, t]

    # Dollar damage = -(change in GDP); for tiny shocks Y*(e^L - 1) ≈ Y*L
    A = -(Y * LA).sum(axis=0)
    B = -(Y * LB).sum(axis=0)
    return A, B


@dataclass
class SCCResult:
    scc: float
    se: float
    a: float = 0.0
    b: float = 0.0
    damages: np.ndarray = field(default=None, repr=False)   # D(t), $ per tonne

    @property
    def ci(self) -> tuple[float, float]:
        return self.scc - 1.96 * self.se, self.scc + 1.96 * self.se


def scc(df: np.ndarray, coef_sets: list[tuple[Countries, dict]], sc: Scenario) -> SCCResult:
    """
    SCC for one discount schedule.

    df        : discount factors for years 0..>=H (from Module 2)
    coef_sets : list of (countries, coefficients) pairs. One pair = the same
                damage function for everyone (pooled). Two pairs = rich and
                poor countries each with their own estimates (heterogeneous).
                Coefficients dict needs b_temp, b_temp_sq, vcov (2x2).
                Separate samples are independent, so variances add.
    """
    H = sc.horizon
    if len(df) < H + 1:
        raise ValueError(f"discount schedule too short: need {H + 1} years")
    d = np.asarray(df[1:H + 1])
    total, var, damages = 0.0, 0.0, np.zeros(H)
    a_tot = b_tot = 0.0
    for countries, coef in coef_sets:
        A, B = damage_components(countries, sc)
        a, b = float(d @ A), float(d @ B)
        b1, b2 = coef["b_temp"], coef["b_temp_sq"]
        V = np.asarray(coef["vcov"], dtype=float)
        g = np.array([a, b])
        total += b1 * a + b2 * b
        var += float(g @ V @ g)
        damages += b1 * A + b2 * B
        a_tot += a
        b_tot += b
    return SCCResult(scc=total, se=float(np.sqrt(var)), a=a_tot, b=b_tot, damages=damages)
