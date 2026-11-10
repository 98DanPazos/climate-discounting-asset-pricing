"""
run_scc.py — Step 4: the social cost of carbon under every discounting method.

Run from the repo root, after R/07_damage_regressions.R and
python/run_discounting.py:

    python python/run_scc.py

Inputs   output/tables/damage_coefficients.json   (Module 1, R)
         output/tables/discount_schedules.csv     (Module 2, Python)
         data/processed/panel_country_year.csv    (base-year GDP, temperature)
Outputs  output/tables/scc_results.csv            every method x damage spec
         output/tables/scc_summary.json
         output/figures/fig6_scc_by_method.png
         output/figures/fig7_scc_by_horizon.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climate_discount.paths import DIR_FIGURES, DIR_PROCESSED, DIR_TABLES   # noqa: E402
from climate_discount.scc import (TCRE_BEST, TCRE_LIKELY, Scenario,         # noqa: E402
                                  countries_from_panel, scc)

# Benchmarks (per tonne of CO2 emitted in 2020)
#   US EPA (2023): $190 at a 2% near-term discount rate ($120 at 2.5%, $340
#   at 1.5%), in 2020 US$. Canada (ECCC 2023) adopted these values: C$247 in
#   2021 Canadian dollars. We convert EPA's 2020 US$ to 2015 US$ (the units
#   of our GDP data) with the US GDP deflator, ratio ≈ 1.09 (approximate).
USD2020_PER_USD2015 = 1.09
EPA = {"1.5%": 340, "2.0%": 190, "2.5%": 120}
EPA_2015USD = {k: v / USD2020_PER_USD2015 for k, v in EPA.items()}

METHOD_ORDER = ["Constant", "Gamma", "Gamma (US history)", "NP AR(1)",
                "NP random walk", "Ramsey SDF (Stern)", "Ramsey SDF", "Ramsey SDF (β=1)"]
PERSISTENCE = {"Level effect (λ=0)": 0.0, "Growth effect (λ=1)": 1.0}

INK, INK2, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SPEC_COLORS = {"Pooled": "#2a78d6", "Rich/poor": "#eb6834"}
METHOD_COLORS = {"Constant": "#52514e", "Gamma": "#2a78d6", "NP AR(1)": "#eb6834",
                 "NP random walk": "#1baf7a", "Ramsey SDF": "#eda100"}


def load_inputs():
    with open(DIR_TABLES / "damage_coefficients.json") as f:
        coefs = json.load(f)
    sched = pd.read_csv(DIR_TABLES / "discount_schedules.csv")
    dfs = {k: g.sort_values("year")["df"].to_numpy() for k, g in sched.groupby("key")}
    names = sched.drop_duplicates("key").set_index("key")["method"].to_dict()
    panel = pd.read_csv(DIR_PROCESSED / "panel_country_year.csv")
    countries, base_year = countries_from_panel(panel)
    return coefs, dfs, names, countries, base_year


def damage_specs(coefs, countries):
    """Pooled: one damage function for everyone. Rich/poor: each group its own."""
    rich, poor = countries.subset(countries.poor == 0), countries.subset(countries.poor == 1)
    return {
        "Pooled": [(countries, coefs["main"])],
        "Rich/poor": [(rich, coefs["rich"]), (poor, coefs["poor"])],
    }


def run_all(coefs, dfs, names, countries, base_year):
    specs = damage_specs(coefs, countries)
    rows, streams = [], {}
    for pers_label, lam in PERSISTENCE.items():
        sc = Scenario(base_year=base_year, persistence=lam)
        for spec_label, sets in specs.items():
            for key in [k for k in METHOD_ORDER if k in dfs]:
                r = scc(dfs[key], sets, sc)
                lo, hi = r.ci
                d = dfs[key][1:sc.horizon + 1] * r.damages        # discounted damages
                cum = np.cumsum(d) / d.sum() if d.sum() != 0 else np.full_like(d, np.nan)
                yrs = sc.calendar()
                # Dollar split rather than a share: when early damages are
                # negative (cold, rich countries gain at first) a "share of the
                # total" can exceed 100% and stops meaning anything.
                rows.append({"persistence": pers_label, "damage_spec": spec_label,
                             "method_key": key, "method": names.get(key, key),
                             "scc": r.scc, "se": r.se, "ci_low": lo, "ci_high": hi,
                             "pv_to_2100": float(d[yrs <= 2100].sum()),
                             "pv_after_2100": float(d[yrs > 2100].sum()),
                             "sign_changes": bool((np.diff(np.sign(r.damages)) != 0).any())})
                streams[(pers_label, spec_label, key)] = (yrs, cum)
    return pd.DataFrame(rows), streams


def style(ax, title=None, subtitle=None):
    ax.set_facecolor(BG)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)


def plot_scc(res: pd.DataFrame, base_year: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.set_facecolor(BG)
    methods = [m for m in METHOD_ORDER if m in res["method_key"].unique()]
    ypos = {m: i for i, m in enumerate(reversed(methods))}
    for ax, pers in zip(axes, PERSISTENCE):
        sub = res[res["persistence"] == pers]
        # Axis range from the POINT estimates (plus the benchmark), so one very
        # wide interval can't squash everything else. Intervals running past
        # the edge are clipped and marked with an arrow.
        hi = 1.35 * max(sub["scc"].max(), EPA_2015USD["2.0%"])
        lo = -0.35 * hi
        for j, (spec, col) in enumerate(SPEC_COLORS.items()):
            s = sub[sub["damage_spec"] == spec]
            y = np.array([ypos[k] for k in s["method_key"]]) + (0.15 if j == 0 else -0.15)
            ax.hlines(y, s["ci_low"].clip(lo, hi), s["ci_high"].clip(lo, hi),
                      color=col, linewidth=1.4, alpha=0.8)
            for yy, l, h in zip(y, s["ci_low"], s["ci_high"]):
                if h > hi:
                    ax.plot(hi, yy, marker=">", color=col, markersize=5)
                if l < lo:
                    ax.plot(lo, yy, marker="<", color=col, markersize=5)
            ax.plot(s["scc"], y, "o", color=col, markersize=6,
                    markeredgecolor=BG, markeredgewidth=1.5, label=spec)
        ax.set_xlim(lo * 1.03, hi * 1.03)
        ax.axvline(0, color="#c3c2b7", linewidth=0.8)
        ax.axvline(EPA_2015USD["2.0%"], color=MUTED, linestyle="--", linewidth=0.9)
        ax.annotate("EPA / ECCC\ncentral", (EPA_2015USD["2.0%"], len(methods) - 0.4),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED, va="top")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        style(ax, pers)
        ax.set_xlabel("SCC (2015 US$ per tonne CO₂)", color=INK2)
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([res.loc[res["method_key"] == m, "method"].iloc[0]
                             for m in ypos.keys()], color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Social cost of carbon by discounting method", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.925, f"Tonne emitted in {base_year}. Points = estimate, lines = 95% CI "
             "from the damage regression (arrows: interval extends beyond the axis). Panels have different x-scales.",
             fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(DIR_FIGURES / "fig6_scc_by_method.png", dpi=200, facecolor=BG)
    plt.close(fig)


def plot_horizon(streams: dict, spec="Pooled", pers="Growth effect (λ=1)"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.set_facecolor(BG)
    for key, col in METHOD_COLORS.items():
        if (pers, spec, key) not in streams:
            continue
        yrs, cum = streams[(pers, spec, key)]
        ax.plot(yrs, 100 * cum, color=col, linewidth=2, label=key)
    ax.axvline(2100, color="#c3c2b7", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Share of SCC accumulated (%)", color=INK2)
    ax.set_xlabel("Year damages occur", color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    style(ax)
    ax.set_title("When do the damages that make up the SCC occur?", loc="left",
                 fontsize=13, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, f"Cumulative share of the SCC by year ({spec} damages, {pers.lower()}). "
            "Lower long-run rates push weight past 2100.", transform=ax.transAxes,
            fontsize=9.5, color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(DIR_FIGURES / "fig7_scc_by_horizon.png", dpi=200, facecolor=BG)
    plt.close(fig)


def main():
    coefs, dfs, names, countries, base_year = load_inputs()
    res, streams = run_all(coefs, dfs, names, countries, base_year)
    res.to_csv(DIR_TABLES / "scc_results.csv", index=False)
    plot_scc(res, base_year)
    plot_horizon(streams)

    summary = {
        "base_year": base_year, "n_countries": int(len(countries.iso3c)),
        "world_gdp_base_2015usd_trillion": float(countries.gdp.sum() / 1e12),
        "gdp_weighted_temp": float(np.average(countries.temp, weights=countries.gdp)),
        "tcre_per_1000GtCO2": TCRE_BEST * 1e15,
        "tcre_scaling_low_high": [TCRE_LIKELY[0] / TCRE_BEST, TCRE_LIKELY[1] / TCRE_BEST],
        "epa_2020usd": EPA, "epa_in_2015usd_approx": EPA_2015USD,
    }
    with open(DIR_TABLES / "scc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ STEP 4 KEY RESULTS ================")
    print(f"Base year {base_year}; {summary['n_countries']} countries; world GDP "
          f"${summary['world_gdp_base_2015usd_trillion']:.0f}T (2015 US$); "
          f"GDP-weighted temperature {summary['gdp_weighted_temp']:.1f}°C")
    print(f"Benchmark: EPA/ECCC $190 (2020 US$) ≈ ${EPA_2015USD['2.0%']:.0f} in 2015 US$\n")
    tbl = res.pivot_table(index="method", columns=["persistence", "damage_spec"],
                          values="scc", sort=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:,.1f}".format):
        print(tbl.to_string())
    print("\nPresent value of damages up to 2100 vs after 2100 (growth effect, $ per tonne):")
    g = res[res.persistence.str.startswith("Growth")]
    for spec in ("Pooled", "Rich/poor"):
        print(f"  {spec}:")
        for _, r in g[g.damage_spec == spec].iterrows():
            flag = "  (damage stream changes sign)" if r["sign_changes"] else ""
            print(f"    {r['method']:<42} to 2100: {r['pv_to_2100']:>9,.1f}   after: "
                  f"{r['pv_after_2100']:>10,.1f}{flag}")
    print(f"\nWarming-per-tonne uncertainty (IPCC likely range) scales every SCC by "
          f"x{summary['tcre_scaling_low_high'][0]:.2f} to x{summary['tcre_scaling_low_high'][1]:.2f}")
    print("=====================================================")
    return {"results": res, "streams": streams, "summary": summary,
            "countries": countries, "coefs": coefs}


if __name__ == "__main__":
    main()
