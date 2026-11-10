"""
discounting.py — four ways to turn future dollars into present dollars.

Every method returns a DiscountSchedule: the discount factor DF(t) for
t = 0, 1, ..., horizon years. DF(t) is "the value today of $1 received in
year t". Everything else follows from it:

    certainty-equivalent rate   R(t) = -ln DF(t) / t
        the single constant rate that would give the same DF at horizon t
    forward rate                f(t) = ln DF(t-1) - ln DF(t)
        the rate that applies between year t-1 and year t; THIS is what
        "declining discount rates" refers to
    present value               PV  = sum_t DF(t) * cashflow(t)

Methods
-------
1. constant      DF(t) = exp(-r t)                               (Ramsey / fixed)
2. gamma         Weitzman (2001): r is uncertain and Gamma-distributed
                 DF(t) = E[exp(-r t)] = (1 + t sigma^2/mu)^(-mu^2/sigma^2)
3. newell_pizer  Newell & Pizer (2003): the rate follows a stochastic process
                 (AR(1) or random walk in logs); DF(t) = E[exp(-sum r_s)]
4. ramsey_sdf    consumption-based stochastic discount factor
                 M_t = exp(-rho t) (C_t / C_0)^(-eta), with lognormal
                 consumption growth whose mean is itself uncertain
                 (Gollier 2008), plus an optional "climate beta" for
                 damages that move with consumption.

Why uncertainty lowers long-run rates (the core idea of the paper):
    DF(t) = E[exp(-r t)] >= exp(-E[r] t)        (Jensen's inequality)
    and for large t the expectation is dominated by the LOW-rate scenarios,
    so the effective rate falls towards the lowest plausible rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Common container
# ---------------------------------------------------------------------------
@dataclass
class DiscountSchedule:
    """Discount factors for years 0..horizon, plus derived quantities."""

    name: str
    df: np.ndarray                     # DF(0)=1, DF(1), ..., DF(H)
    params: dict = field(default_factory=dict)

    @property
    def years(self) -> np.ndarray:
        return np.arange(len(self.df))

    def ce_rate(self) -> np.ndarray:
        """Certainty-equivalent (average) rate to each horizon. NaN at t=0."""
        t = self.years.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -np.log(self.df) / t
        r[0] = np.nan
        return r

    def forward_rate(self) -> np.ndarray:
        """Year-on-year forward rate f(t) = ln DF(t-1) - ln DF(t). NaN at t=0."""
        logdf = np.log(self.df)
        f = np.empty_like(logdf)
        f[0] = np.nan
        f[1:] = logdf[:-1] - logdf[1:]
        return f

    def pv(self, cashflows: np.ndarray) -> float:
        """Present value of a cash-flow vector aligned with years 0..H."""
        cashflows = np.asarray(cashflows, dtype=float)
        n = min(len(cashflows), len(self.df))
        return float(np.sum(self.df[:n] * cashflows[:n]))


# ---------------------------------------------------------------------------
# 1. Constant rate
# ---------------------------------------------------------------------------
def constant(r: float, horizon: int = 400, name: str | None = None) -> DiscountSchedule:
    """DF(t) = exp(-r t). r is a decimal (0.03 = 3%)."""
    t = np.arange(horizon + 1)
    return DiscountSchedule(name or f"Constant {r:.1%}", np.exp(-r * t), {"r": r})


# ---------------------------------------------------------------------------
# 2. Gamma discounting (Weitzman 2001)
# ---------------------------------------------------------------------------
def gamma(mu: float, sigma: float, horizon: int = 400,
          name: str | None = None) -> DiscountSchedule:
    """
    If the 'true' constant rate r ~ Gamma with mean mu and std sigma, then
        DF(t) = E[exp(-r t)] = (1 + t * sigma^2 / mu) ** (-(mu / sigma) ** 2)
    (this is the Gamma distribution's moment-generating function).
    The forward rate is f(t) ≈ mu / (1 + t sigma^2 / mu), declining towards 0.
    """
    if mu <= 0 or sigma <= 0:
        raise ValueError("gamma discounting needs mu > 0 and sigma > 0")
    t = np.arange(horizon + 1)
    df = (1.0 + t * sigma**2 / mu) ** (-(mu / sigma) ** 2)
    return DiscountSchedule(name or f"Gamma (μ={mu:.1%}, σ={sigma:.1%})", df,
                            {"mu": mu, "sigma": sigma})


# ---------------------------------------------------------------------------
# 3. Newell & Pizer (2003): stochastic interest-rate paths
# ---------------------------------------------------------------------------
def simulate_np_paths(r0: float, phi: float, sigma: float, horizon: int,
                      n_paths: int = 20_000, model: str = "ar1",
                      seed: int | None = 12345) -> np.ndarray:
    """
    Simulate interest-rate paths the way Newell & Pizer did:
      * work with x_t = log(r_t / r0), so rates stay positive and the
        *uncertainty structure* (persistence phi, volatility sigma) comes from
        the data while the *level* r0 is a choice;
      * 'ar1':  x_t = phi * x_{t-1} + e_t      (mean-reverting to r0)
      * 'rw' :  x_t = x_{t-1} + e_t            (random walk, phi = 1)
      * e_t ~ N(0, sigma^2), x_0 = 0.
      * Mean correction: r_t = r0 * exp(x_t - ½ Var(x_t)), so E[r_t] = r0 in
        every year. Without it, the lognormal shape would push the *average*
        rate above r0 (E[e^x] = e^{Var/2} > 1), mixing "more uncertainty" with
        "higher rates". With it, any fall in the effective rate is due to
        uncertainty alone — the effect we want to isolate.
    Returns an (n_paths, horizon) array of rates for years 1..horizon.
    """
    rng = np.random.default_rng(seed)
    if model == "rw":
        phi = 1.0
    elif model != "ar1":
        raise ValueError("model must be 'ar1' or 'rw'")
    shocks = rng.normal(0.0, sigma, size=(n_paths, horizon))
    x = np.empty((n_paths, horizon))
    prev = np.zeros(n_paths)
    for t in range(horizon):              # loop over time, vectorised over paths
        prev = phi * prev + shocks[:, t]
        x[:, t] = prev

    # Var(x_t) for t = 1..horizon
    t = np.arange(1, horizon + 1)
    if phi == 1.0:
        var_x = sigma**2 * t
    else:
        var_x = sigma**2 * (1 - phi ** (2 * t)) / (1 - phi**2)
    return r0 * np.exp(x - 0.5 * var_x)


def newell_pizer(r0: float, phi: float, sigma: float, horizon: int = 400,
                 n_paths: int = 20_000, model: str = "ar1",
                 seed: int | None = 12345, name: str | None = None) -> DiscountSchedule:
    """
    DF(t) = E[ exp( -sum_{s=1..t} r_s ) ], averaged over simulated paths.
    Uses continuous compounding to match the other methods.
    """
    rates = simulate_np_paths(r0, phi, sigma, horizon, n_paths, model, seed)
    cum = np.cumsum(rates, axis=1)                       # sum of rates to t
    df = np.concatenate([[1.0], np.exp(-cum).mean(axis=0)])
    label = name or ("Newell-Pizer, random walk" if model == "rw"
                     else f"Newell-Pizer, AR(1) φ={phi:.2f}")
    return DiscountSchedule(label, df, {"r0": r0, "phi": 1.0 if model == "rw" else phi,
                                        "sigma": sigma, "model": model,
                                        "n_paths": n_paths})


# ---------------------------------------------------------------------------
# 4. Consumption-based SDF (Ramsey rule under uncertainty)
# ---------------------------------------------------------------------------
def ramsey_sdf(rho: float, eta: float, mu_g: float, sigma_g: float,
               se_mu: float = 0.0, climate_beta: float = 0.0,
               horizon: int = 400, name: str | None = None) -> DiscountSchedule:
    """
    Value of a cash flow D_t = C_t ** climate_beta (normalised) under the SDF
        M_t = exp(-rho t) * (C_t/C_0) ** (-eta),
    with log consumption growth g_s ~ iid N(mu, sigma_g^2) and mean mu itself
    uncertain: mu ~ N(mu_g, se_mu^2)  (Gollier 2008, parameter uncertainty).

    Derivation. With an uncertain mean, log C_t ~ N(a, V) where
        a = mu_g t,   V = sigma_g² t + se_mu² t².
    We price the damage relative to its expected value,
        DF(t) = E[M_t D_t] / E[D_t],
    and the lognormal moment formula E[exp(kX)] = exp(k a + ½ k² V) gives

        ln DF(t) = -(rho + eta*mu_g - ½ eta² sigma_g²) t
                   - eta*climate_beta*sigma_g² t
                   + ½ eta² se_mu² t²  - eta*climate_beta*se_mu² t²

    * First line: the Ramsey rule  r = rho + eta*g - ½ eta² sigma²
      (time preference + wealth effect - precautionary saving).
    * climate_beta > 0 (damages higher when we are richer) adds a risk
      premium eta*beta*sigma² — the "climate beta" of Dietz, Gollier &
      Kessler (2018). climate_beta = 0 gives the risk-free rate.
    * The t² terms: uncertainty about the long-run growth rate makes the
      forward rate decline linearly with t — a declining term structure
      from first principles.
    """
    t = np.arange(horizon + 1, dtype=float)
    r_f = rho + eta * mu_g - 0.5 * eta**2 * sigma_g**2
    log_df = (-(r_f + eta * climate_beta * sigma_g**2) * t
              + (0.5 * eta**2 - eta * climate_beta) * se_mu**2 * t**2)
    label = name or f"Ramsey SDF (ρ={rho:.1%}, η={eta:g}, β={climate_beta:g})"
    return DiscountSchedule(label, np.exp(log_df),
                            {"rho": rho, "eta": eta, "mu_g": mu_g, "sigma_g": sigma_g,
                             "se_mu": se_mu, "climate_beta": climate_beta,
                             "risk_free_rate": r_f})


def ramsey_sdf_montecarlo(rho: float, eta: float, mu_g: float, sigma_g: float,
                          se_mu: float = 0.0, climate_beta: float = 0.0,
                          horizon: int = 200, n_paths: int = 200_000,
                          seed: int | None = 7) -> DiscountSchedule:
    """
    Brute-force check of ramsey_sdf(): simulate consumption paths, compute
    E[M_t D_t] / E[D_t] directly. Used in the unit tests.
    """
    rng = np.random.default_rng(seed)
    mu_draw = mu_g + se_mu * rng.standard_normal(n_paths)          # uncertain trend
    g = mu_draw[:, None] + sigma_g * rng.standard_normal((n_paths, horizon))
    log_c = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(g, axis=1)], axis=1)
    t = np.arange(horizon + 1)
    m = np.exp(-rho * t - eta * log_c)                               # SDF
    d = np.exp(climate_beta * log_c)                                 # damage
    df = (m * d).mean(axis=0) / d.mean(axis=0)
    return DiscountSchedule("Ramsey SDF (Monte Carlo)", df)
