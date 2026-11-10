"""
Unit tests for the discounting engine.  Run from the repo root:

    pytest python/tests -v

Each test checks a method against something we know must be true — a closed
form, a limiting case, or a brute-force simulation — so a bug in the engine
shows up here before it can contaminate the social-cost-of-carbon results.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # python/
from climate_discount import (constant, gamma, newell_pizer,    # noqa: E402
                              ramsey_sdf, ramsey_sdf_montecarlo)


def test_constant_rate_closed_form():
    s = constant(0.03, horizon=100)
    assert s.df[0] == 1.0
    assert np.isclose(s.df[50], np.exp(-1.5))
    # forward and certainty-equivalent rates are flat at 3%
    assert np.allclose(s.forward_rate()[1:], 0.03)
    assert np.allclose(s.ce_rate()[1:], 0.03)


def test_pv_of_annuity_matches_formula():
    r, H = 0.04, 200
    s = constant(r, horizon=H)
    cash = np.r_[0.0, np.ones(H)]                    # $1 in years 1..H
    expected = np.sum(np.exp(-r * np.arange(1, H + 1)))
    assert np.isclose(s.pv(cash), expected)


def test_gamma_matches_monte_carlo():
    mu, sd = 0.04, 0.03
    s = gamma(mu, sd, horizon=300)
    rng = np.random.default_rng(0)
    shape, scale = (mu / sd) ** 2, sd**2 / mu        # Gamma with mean mu, std sd
    r = rng.gamma(shape, scale, size=400_000)
    for t in (10, 100, 300):
        assert np.isclose(s.df[t], np.exp(-r * t).mean(), rtol=0.02)


def test_gamma_forward_rate_declines():
    f = gamma(0.04, 0.03, horizon=300).forward_rate()
    assert f[1] > f[100] > f[300] > 0


def test_uncertainty_raises_discount_factor_jensen():
    """With the same mean rate, any uncertainty must give DF >= constant DF."""
    base = constant(0.04, horizon=300).df
    assert np.all(gamma(0.04, 0.03, horizon=300).df >= base - 1e-12)


def test_newell_pizer_zero_volatility_is_constant():
    s = newell_pizer(0.03, phi=0.9, sigma=0.0, horizon=100, n_paths=10)
    assert np.allclose(s.df, constant(0.03, horizon=100).df)


def test_newell_pizer_paths_have_mean_r0():
    """The mean correction keeps the average simulated rate at r0."""
    from climate_discount.discounting import simulate_np_paths
    for model in ("ar1", "rw"):
        r = simulate_np_paths(0.03, 0.9, 0.1, horizon=100, n_paths=200_000, model=model)
        assert np.allclose(r.mean(axis=0)[[0, 49, 99]], 0.03, rtol=0.03)


def test_newell_pizer_rw_declines_faster_than_ar1():
    ar = newell_pizer(0.03, phi=0.8, sigma=0.1, horizon=300, model="ar1")
    rw = newell_pizer(0.03, phi=0.8, sigma=0.1, horizon=300, model="rw")
    assert rw.ce_rate()[300] < ar.ce_rate()[300] < 0.03


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_ramsey_sdf_closed_form_matches_simulation(beta):
    kw = dict(rho=0.01, eta=1.5, mu_g=0.018, sigma_g=0.02, se_mu=0.004,
              climate_beta=beta)
    exact = ramsey_sdf(horizon=100, **kw)
    mc = ramsey_sdf_montecarlo(horizon=100, n_paths=200_000, **kw)
    for t in (10, 50, 100):
        assert np.isclose(exact.df[t], mc.df[t], rtol=0.02)


def test_ramsey_rule_without_uncertainty():
    """sigma = se = 0 collapses to the textbook Ramsey rule r = rho + eta*g."""
    s = ramsey_sdf(rho=0.015, eta=1.45, mu_g=0.02, sigma_g=0.0, horizon=50)
    assert np.allclose(s.forward_rate()[1:], 0.015 + 1.45 * 0.02)


def test_climate_beta_adds_risk_premium():
    kw = dict(rho=0.01, eta=1.5, mu_g=0.02, sigma_g=0.03, horizon=50)
    r0 = ramsey_sdf(climate_beta=0.0, **kw).forward_rate()[10]
    r1 = ramsey_sdf(climate_beta=1.0, **kw).forward_rate()[10]
    assert np.isclose(r1 - r0, 1.5 * 1.0 * 0.03**2)
