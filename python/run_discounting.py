"""
run_discounting.py — Step 3: calibrate and compare the discounting methods.

Run from the repo root (with the climate_ap environment active):

    python python/run_discounting.py

What it does
  1. Calibrates every method from the project's data (calibrate.py).
  2. Builds a 400-year discount schedule for each method.
  3. Saves:
       output/tables/discounting_calibration.json   estimated parameters
       output/tables/discount_schedules.csv         DF, CE rate, forward rate by year
       output/tables/discounting_summary.csv        headline comparison table
       output/figures/fig4_forward_rates.png        the term structure of rates
       output/figures/fig5_value_of_future_dollar.png
  4. Prints the key results.

The schedules are what Step 4 multiplies by the climate-damage stream to get
the social cost of carbon under each method.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # render to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climate_discount import constant, gamma, newell_pizer, ramsey_sdf  # noqa: E402
from climate_discount.calibrate import calibrate                       # noqa: E402
from climate_discount.paths import DIR_FIGURES, DIR_TABLES             # noqa: E402

HORIZON = 400          # years; Step 4 truncates damages at 2300 (~275 years)

# Chart styling — validated categorical slots 1-4 + neutral ink for the
# constant-rate benchmark, matching the R figures.
COLORS = {"Constant": "#52514e", "Gamma": "#2a78d6", "NP AR(1)": "#eb6834",
          "NP random walk": "#1baf7a", "Ramsey SDF": "#eda100"}
INK, INK2, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def build_schedules(cal: dict) -> dict:
    """Create every schedule from the calibration dictionary."""
    r0 = cal["r0"]["us_mean_real_rate"]
    np_us = cal["newell_pizer"]["US"]
    gw = cal["gamma"]["weitzman_survey"]
    gh = cal["gamma"]["us_historical"]
    g = cal["growth"]
    st, nd = cal["ethics"]["Stern"], cal["ethics"]["Nordhaus"]

    s = {
        "Constant": constant(r0, HORIZON, name=f"Constant {r0:.1%} (US mean real rate)"),
        "Gamma": gamma(gw["mu"], gw["sigma"], HORIZON, name="Gamma (Weitzman survey)"),
        "Gamma (US history)": gamma(gh["mu"], gh["sigma"], HORIZON,
                                    name="Gamma (US real-rate history)"),
        "NP AR(1)": newell_pizer(r0, np_us["phi"], np_us["sigma_ar1"], HORIZON,
                                 model="ar1", name="Newell-Pizer AR(1), US"),
        "NP random walk": newell_pizer(r0, 1.0, np_us["sigma_rw"], HORIZON,
                                       model="rw", name="Newell-Pizer random walk, US"),
        "Ramsey SDF": ramsey_sdf(nd["rho"], nd["eta"], g["mu_g"], g["sigma_g"],
                                 g["se_mu"], 0.0, HORIZON, name="Ramsey SDF, Nordhaus ethics"),
        "Ramsey SDF (Stern)": ramsey_sdf(st["rho"], st["eta"], g["mu_g"], g["sigma_g"],
                                         g["se_mu"], 0.0, HORIZON,
                                         name="Ramsey SDF, Stern ethics"),
        "Ramsey SDF (β=1)": ramsey_sdf(nd["rho"], nd["eta"], g["mu_g"], g["sigma_g"],
                                       g["se_mu"], 1.0, HORIZON,
                                       name="Ramsey SDF, Nordhaus, climate beta = 1"),
    }
    return s


def summary_table(schedules: dict) -> pd.DataFrame:
    rows = []
    stream = np.r_[0.0, np.ones(300)]               # $1 a year for 300 years
    for key, s in schedules.items():
        ce, fwd = s.ce_rate(), s.forward_rate()
        rows.append({
            "method": s.name,
            "fwd rate yr 1 (%)": 100 * fwd[1],
            "fwd rate yr 100 (%)": 100 * fwd[100],
            "fwd rate yr 300 (%)": 100 * fwd[300],
            "CE rate to yr 300 (%)": 100 * ce[300],
            "$1M in 100 yrs, today ($)": 1e6 * s.df[100],
            "$1M in 300 yrs, today ($)": 1e6 * s.df[300],
            "PV of $1/yr for 300 yrs ($)": s.pv(stream),
        })
    return pd.DataFrame(rows)


def long_table(schedules: dict) -> pd.DataFrame:
    frames = []
    for key, s in schedules.items():
        frames.append(pd.DataFrame({"method": s.name, "key": key, "year": s.years,
                                    "df": s.df, "ce_rate": s.ce_rate(),
                                    "forward_rate": s.forward_rate()}))
    return pd.concat(frames, ignore_index=True)


def style(ax, title, subtitle):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=INK2)


def plot_figures(schedules: dict) -> None:
    keys = ["Constant", "Gamma", "NP AR(1)", "NP random walk", "Ramsey SDF"]
    years = np.arange(1, 301)

    # --- Figure 4: forward-rate term structure -------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    ends = {}
    for k in keys:
        f = 100 * schedules[k].forward_rate()[1:301]
        ax.plot(years, f, color=COLORS[k], linewidth=2, label=schedules[k].name)
        ends[k] = f[-1]
    # Direct labels at the line ends, pushed apart so close lines stay legible
    min_gap = 0.06 * (max(ends.values()) - min(min(ends.values()), 0) + 1e-9)
    placed = []
    for k, y in sorted(ends.items(), key=lambda kv: kv[1]):
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(k, (years[-1], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_xlim(0, 345)
    ax.set_xlabel("Years from today", color=INK2)
    ax.set_ylabel("Forward discount rate (% per year)", color=INK2)
    style(ax, "The term structure of discount rates",
          "Rate applied between year t−1 and t. Uncertainty makes long-horizon rates fall.")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(DIR_FIGURES / "fig4_forward_rates.png", dpi=200)
    plt.close(fig)

    # --- Figure 5: value today of $1 received in year t (log scale) ---------
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in keys:
        df = schedules[k].df[1:301]
        ax.plot(years, df, color=COLORS[k], linewidth=2, label=schedules[k].name)
    ax.set_yscale("log")
    ax.set_xlabel("Years from today", color=INK2)
    ax.set_ylabel("Value today of $1 (log scale)", color=INK2)
    style(ax, "What is $1 of future climate damage worth today?",
          "Discount factor by horizon. Small rate differences compound into orders of magnitude.")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(DIR_FIGURES / "fig5_value_of_future_dollar.png", dpi=200)
    plt.close(fig)


def main() -> dict:
    cal = calibrate(save=True)
    schedules = build_schedules(cal)

    long_table(schedules).to_csv(DIR_TABLES / "discount_schedules.csv", index=False)
    summ = summary_table(schedules)
    summ.to_csv(DIR_TABLES / "discounting_summary.csv", index=False)
    plot_figures(schedules)

    np_us, g = cal["newell_pizer"]["US"], cal["growth"]
    print("\n================ STEP 3 KEY RESULTS ================")
    print(f"US real rate {cal['r0']['years'][0]}-{cal['r0']['years'][1]}: "
          f"mean {cal['r0']['us_mean_real_rate']:.2%}, std {cal['r0']['us_real_rate_std']:.2%}")
    print(f"Log US 10y yield AR(1): phi = {np_us['phi']:.3f} (se {np_us['phi_se']:.3f}), "
          f"sigma = {np_us['sigma_ar1']:.3f}; RW sigma = {np_us['sigma_rw']:.3f}")
    ca = cal["newell_pizer"]["Canada"]
    print(f"Log Canada 10y yield AR(1): phi = {ca['phi']:.3f}, sigma = {ca['sigma_ar1']:.3f}")
    print(f"World GDP pc growth {g['years'][0]}-{g['years'][1]}: "
          f"mean {g['mu_g']:.2%}, std {g['sigma_g']:.2%}, se(mean) {g['se_mu']:.2%}")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:,.2f}".format):
        print("\n", summ.to_string(index=False))
    print("=====================================================")
    return {"cal": cal, "schedules": schedules, "summary": summ}


if __name__ == "__main__":
    main()
