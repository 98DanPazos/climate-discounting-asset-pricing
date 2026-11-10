"""
Unit tests for the social-cost-of-carbon module.   pytest python/tests -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from climate_discount import constant                                   # noqa: E402
from climate_discount.scc import Countries, Scenario, damage_components, scc  # noqa: E402


def one_country(gdp=1e12, temp=20.0, poor=0):
    return Countries(np.array(["X"]), np.array([gdp]), np.array([temp]), np.array([poor]))


def flat_scenario(**kw):
    """No growth, no warming: isolates the arithmetic."""
    base = dict(base_year=2025, end_year=2125, warming_by_2100=0.0,
                gpc_growth_now=0.0, gpc_growth_2100=0.0, pop_growth_now=0.0)
    base.update(kw)
    return Scenario(**base)


def test_level_effect_closed_form():
    """lambda=0, b2=0, flat GDP: SCC = -b1 * delta/100 * Y * sum DF."""
    sc = flat_scenario(persistence=0.0)
    c = one_country(gdp=1e12)
    coef = {"b_temp": -1.0, "b_temp_sq": 0.0, "vcov": np.zeros((2, 2))}
    df = constant(0.03, horizon=400).df
    res = scc(df, [(c, coef)], sc)
    expected = 1.0 * sc.tcre / 100 * 1e12 * df[1:101].sum()
    assert np.isclose(res.scc, expected)


def test_growth_effect_accumulates():
    """With lambda=1 the year-t loss is t times the year-1 loss (flat world)."""
    sc = flat_scenario(persistence=1.0)
    A, _ = damage_components(one_country(), sc)
    assert np.allclose(A / A[0], np.arange(1, sc.horizon + 1))


def test_scc_is_linear_in_coefficients():
    sc = Scenario(base_year=2025, end_year=2200)
    c = one_country(temp=25.0)
    df = constant(0.02, horizon=400).df
    r1 = scc(df, [(c, {"b_temp": 0.3, "b_temp_sq": -0.02, "vcov": np.eye(2)})], sc)
    r2 = scc(df, [(c, {"b_temp": 0.6, "b_temp_sq": -0.04, "vcov": np.eye(2)})], sc)
    assert np.isclose(r2.scc, 2 * r1.scc)
    assert np.isclose(r1.scc, 0.3 * r1.a - 0.02 * r1.b)


def test_hot_country_harmed_cold_country_benefits():
    """With an inverted-U damage function, warming hurts hot and helps cold."""
    sc = Scenario(base_year=2025, end_year=2100)
    coef = {"b_temp": 0.3, "b_temp_sq": -0.02, "vcov": np.zeros((2, 2))}   # T* = 7.5°C
    df = constant(0.02, horizon=400).df
    assert scc(df, [(one_country(temp=25.0), coef)], sc).scc > 0
    assert scc(df, [(one_country(temp=-5.0), coef)], sc).scc < 0


def test_heterogeneous_with_equal_coefficients_equals_pooled():
    sc = Scenario(base_year=2025, end_year=2200)
    c = Countries(np.array(["R", "P"]), np.array([5e12, 1e12]),
                  np.array([10.0, 26.0]), np.array([0, 1]))
    coef = {"b_temp": 0.3, "b_temp_sq": -0.02, "vcov": np.diag([0.01, 1e-5])}
    df = constant(0.02, horizon=400).df
    pooled = scc(df, [(c, coef)], sc)
    het = scc(df, [(c.subset(c.poor == 0), coef), (c.subset(c.poor == 1), coef)], sc)
    assert np.isclose(pooled.scc, het.scc)


def test_scc_proportional_to_tcre():
    c = one_country(temp=25.0)
    coef = {"b_temp": 0.3, "b_temp_sq": -0.02, "vcov": np.zeros((2, 2))}
    df = constant(0.02, horizon=400).df
    lo = scc(df, [(c, coef)], Scenario(tcre=0.27e-12)).scc
    hi = scc(df, [(c, coef)], Scenario(tcre=0.63e-12)).scc
    assert np.isclose(hi / lo, 0.63 / 0.27)
