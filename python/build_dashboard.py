"""
build_dashboard.py — Module 5: one interactive page summarising the project.

Run from the repo root after all other modules:

    python python/build_dashboard.py            # docs/index.html (Plotly via CDN)
    python python/build_dashboard.py --inline   # embeds plotly.js (~4 MB), works offline

GitHub Pages serves the docs/ folder: Settings -> Pages -> Source: "Deploy from a
branch" -> Branch: main, folder: /docs. The page then lives at
https://<username>.github.io/climate-discounting-asset-pricing/

Every number on the page is read from output/ at build time, so rebuilding
after re-running the models keeps the dashboard in sync automatically.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climate_discount.paths import DIR_PROCESSED, DIR_TABLES, ROOT   # noqa: E402

DOCS = ROOT / "docs"
REPO_URL = "https://github.com/98DanPazos/climate-discounting-asset-pricing"

# Palette: validated categorical slots + neutral inks (same as every figure)
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, BG, CARD = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#ffffff"
METHOD_COLORS = {
    "Constant": INK2, "Gamma": BLUE, "Gamma (US history)": "#8fb8ea",
    "NP AR(1)": ORANGE, "NP random walk": GREEN,
    "Ramsey SDF (Stern)": "#f5c45e", "Ramsey SDF": YELLOW, "Ramsey SDF (β=1)": "#b77d00",
}
METHOD_ORDER = list(METHOD_COLORS)
SHORT = {"Constant": "Constant rate", "Gamma": "Gamma (survey)", "Gamma (US history)": "Gamma (US history)",
         "NP AR(1)": "Newell–Pizer AR(1)", "NP random walk": "Newell–Pizer RW",
         "Ramsey SDF (Stern)": "Ramsey (Stern)", "Ramsey SDF": "Ramsey (Nordhaus)",
         "Ramsey SDF (β=1)": "Ramsey (Nordhaus, β=1)"}
FONT = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def base_layout(**kw) -> dict:
    layout = dict(
        font=dict(family=FONT, size=13, color=INK2),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        margin=dict(l=60, r=30, t=30, b=55),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=INK)),
        xaxis=dict(gridcolor=GRID, zerolinecolor="#c3c2b7", linecolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor="#c3c2b7", linecolor=GRID, tickfont=dict(color=MUTED)),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=12)),
    )
    layout.update(kw)
    return layout


# =============================================================================
# Load everything the modules produced
# =============================================================================
def load():
    d = {}
    d["coef"] = json.loads((DIR_TABLES / "damage_coefficients.json").read_text())
    d["grid"] = pd.read_csv(DIR_TABLES / "damage_function_grid.csv")
    d["sched"] = pd.read_csv(DIR_TABLES / "discount_schedules.csv")
    d["cal"] = json.loads((DIR_TABLES / "discounting_calibration.json").read_text())
    d["scc"] = pd.read_csv(DIR_TABLES / "scc_results.csv")
    d["scc_sum"] = json.loads((DIR_TABLES / "scc_summary.json").read_text())
    d["betas"] = pd.read_csv(DIR_TABLES / "table4_temperature_betas.csv")
    d["ap"] = json.loads((DIR_TABLES / "asset_pricing_results.json").read_text())
    panel = pd.read_csv(DIR_PROCESSED / "panel_country_year.csv")
    name_col = "country" if "country" in panel else "iso3c"
    d["countries"] = (panel.dropna(subset=["temp"])
                      .groupby(["iso3c", name_col] if name_col != "iso3c" else ["iso3c"])
                      ["temp"].mean().reset_index()
                      .rename(columns={name_col: "name"}))
    if "name" not in d["countries"]:
        d["countries"]["name"] = d["countries"]["iso3c"]
    return d


# =============================================================================
# Figures
# =============================================================================
def fig_scc(d) -> go.Figure:
    """SCC by method with 95% CI; dropdown switches damage spec x persistence."""
    res = d["scc"]
    bench = d["scc_sum"]["epa_in_2015usd_approx"]["2.0%"]
    combos = [(p, s) for p in res["persistence"].unique() for s in res["damage_spec"].unique()]
    fig = go.Figure()
    ranges = []
    for j, (p, s) in enumerate(combos):
        sub = res[(res.persistence == p) & (res.damage_spec == s)].copy()
        sub["order"] = sub["method_key"].map({k: i for i, k in enumerate(METHOD_ORDER)})
        sub = sub.sort_values("order", ascending=False)
        # Axis range from the point estimates (+ benchmark): very wide intervals
        # would otherwise squash every estimate against zero. They are cut at the edge.
        hi = 1.35 * max(sub["scc"].max(), bench)
        lo = min(-0.35 * hi, 1.15 * sub["scc"].min())
        ranges.append([lo, hi])
        fig.add_trace(go.Scatter(
            x=sub["scc"], y=sub["method_key"].map(SHORT), mode="markers", text=sub["method"],
            marker=dict(size=11, color=[METHOD_COLORS.get(k, BLUE) for k in sub["method_key"]],
                        line=dict(color="white", width=1.5)),
            error_x=dict(type="data", symmetric=False, array=sub["ci_high"] - sub["scc"],
                         arrayminus=sub["scc"] - sub["ci_low"], color="#b8b7b0", thickness=1.4, width=0),
            customdata=np.stack([sub["ci_low"], sub["ci_high"]], axis=1),
            hovertemplate="<b>%{text}</b><br>SCC: $%{x:,.0f} per tonne"
                          "<br>95% CI: $%{customdata[0]:,.0f} to $%{customdata[1]:,.0f}<extra></extra>",
            visible=(j == 0), name=f"{s} · {p}", showlegend=False))
    fig.add_vline(x=bench, line=dict(color=MUTED, dash="dash", width=1.2),
                  annotation=dict(text=f"EPA / ECCC ≈ ${bench:,.0f}", font=dict(color=MUTED, size=11),
                                  yanchor="bottom"), annotation_position="top")
    fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
    buttons = [dict(label=f"{s} damages · {p}", method="update",
                    args=[{"visible": [k == j for k in range(len(combos))]},
                          {"xaxis.range": ranges[j]}])
               for j, (p, s) in enumerate(combos)]
    fig.update_layout(**base_layout(
        height=490, margin=dict(l=10, r=20, t=60, b=70),
        xaxis=dict(title="SCC, 2015 US$ per tonne CO₂", gridcolor=GRID, zerolinecolor="#c3c2b7",
                   tickfont=dict(color=MUTED), tickprefix="$", separatethousands=True,
                   range=ranges[0]),
        yaxis=dict(tickfont=dict(color=INK2, size=12), gridcolor=GRID, automargin=True),
        updatemenus=[dict(buttons=buttons, direction="down", x=0, xanchor="left", y=1.14, yanchor="top",
                          bgcolor="white", bordercolor=GRID, font=dict(size=12, color=INK))]))
    return fig


def fig_rates(d) -> go.Figure:
    s = d["sched"]
    s = s[(s.year >= 1) & (s.year <= 300)]
    fig = go.Figure()
    for key in METHOD_ORDER:
        g = s[s.key == key]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g.year, y=100 * g.forward_rate, mode="lines", name=g.method.iloc[0],
            line=dict(color=METHOD_COLORS[key], width=2.2 if key in ("Constant", "Gamma", "NP AR(1)",
                                                                        "NP random walk", "Ramsey SDF") else 1.4),
            hovertemplate="%{fullData.name}<br>Year %{x}: %{y:.2f}%<extra></extra>"))
    fig.update_layout(**base_layout(
        height=470, legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
        xaxis=dict(title="Years from today", gridcolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(title="Forward discount rate (% per year)", gridcolor=GRID, zerolinecolor="#c3c2b7",
                   tickfont=dict(color=MUTED), ticksuffix="%")))
    return fig


def fig_damage(d) -> go.Figure:
    g, c, coef = d["grid"], d["countries"].copy(), d["coef"]["main"]
    lo_t, hi_t = c.temp.min(), c.temp.max()
    g = g[(g.temp >= lo_t - 0.5) & (g.temp <= hi_t + 0.5)]
    c["me"] = coef["b_temp"] + 2 * coef["b_temp_sq"] * c.temp
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.concat([g.temp, g.temp[::-1]]),
                             y=pd.concat([g.me_hi, g.me_lo[::-1]]), fill="toself",
                             fillcolor="rgba(42,120,214,0.13)", line=dict(width=0),
                             hoverinfo="skip", name="95% CI"))
    fig.add_trace(go.Scatter(x=g.temp, y=g.me, mode="lines", line=dict(color=BLUE, width=2.5),
                             name="Effect of +1°C", hovertemplate="%{x:.1f}°C: %{y:+.2f} pp<extra></extra>"))
    fig.add_trace(go.Scatter(x=c.temp, y=c.me, mode="markers", name="Countries",
                             marker=dict(size=6, color=INK2, opacity=0.55, line=dict(color="white", width=0.8)),
                             text=c.name, hovertemplate="<b>%{text}</b><br>Avg temp %{x:.1f}°C"
                                                        "<br>+1°C → %{y:+.2f} pp growth<extra></extra>"))
    t_opt = coef["t_opt"]
    fig.add_vline(x=t_opt, line=dict(color=MUTED, dash="dash", width=1.2),
                  annotation=dict(text=f"T* ≈ {t_opt:.1f}°C", font=dict(color=MUTED, size=11)),
                  annotation_position="top right")
    fig.add_hline(y=0, line=dict(color="#c3c2b7", width=1))
    fig.update_layout(**base_layout(
        height=450,
        xaxis=dict(title="Country's average annual temperature (°C)", gridcolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(title="Change in GDP-per-capita growth (pp)", gridcolor=GRID, tickfont=dict(color=MUTED))))
    return fig


def fig_betas(d) -> go.Figure:
    b = d["betas"].sort_values("beta_temp")
    sig = b.t.abs() > 1.96
    fig = go.Figure(go.Scatter(
        x=b.beta_temp, y=b.industry, mode="markers",
        marker=dict(size=8, color=np.where(sig, BLUE, "#c3c2b7"), line=dict(color="white", width=1)),
        error_x=dict(type="data", array=1.96 * b.nw_se, color="#d5d4cc", thickness=1.2, width=0),
        customdata=np.stack([b.nw_se, b.t], axis=1),
        hovertemplate="<b>%{y}</b><br>β = %{x:.3f}% per °C<br>NW se %{customdata[0]:.3f}, "
                      "t = %{customdata[1]:.2f}<extra></extra>", showlegend=False))
    fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
    fig.update_layout(**base_layout(
        height=max(560, 18 * len(b)), margin=dict(l=80, r=30, t=20, b=60),
        xaxis=dict(title="Monthly return (%) per 1°C temperature surprise", gridcolor=GRID,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(tickfont=dict(size=11, color=INK2), gridcolor=GRID, dtick=1)))
    return fig


# =============================================================================
# Headline numbers (computed, so the text always matches the data)
# =============================================================================
def headlines(d) -> dict:
    coef, cal, res, ap = d["coef"], d["cal"], d["scc"], d["ap"]
    gp = res[(res.persistence.str.startswith("Growth")) & (res.damage_spec == "Pooled")]
    npus = cal["newell_pizer"]["US"]
    s = d["sched"]
    df300 = s[s.year == 300].set_index("key")["df"]
    fm_main = next((r for r in ap["fama_macbeth_temperature"] if "main" in r["spec"]), None)
    lvl = res[(res.persistence.str.startswith("Level")) & (res.damage_spec == "Pooled") &
              (res.method_key == "Constant")].scc.iloc[0]
    grw = gp[gp.method_key == "Constant"].scc.iloc[0]
    return {
        "n_countries": coef["sample"]["countries"],
        "years": f'{coef["sample"]["years"][0]}–{coef["sample"]["years"][1]}',
        "t_opt": coef["main"]["t_opt"], "t_opt_poor": coef["poor"]["t_opt"],
        "curv_ratio": coef["poor"]["b_temp_sq"] / coef["main"]["b_temp_sq"],
        "phi": npus["phi"], "phi_t": (npus["phi"] - 1) / npus["phi_se"],
        "np_ratio": df300.get("NP random walk", np.nan) / df300.get("NP AR(1)", np.nan),
        "scc_min": gp.scc.min(), "scc_max": gp.scc.max(),
        "scc_min_m": gp.loc[gp.scc.idxmin(), "method"], "scc_max_m": gp.loc[gp.scc.idxmax(), "method"],
        "all_ci_zero": bool(((res.ci_low < 0) & (res.ci_high > 0)).all()),
        "lvl": lvl, "grw": grw,
        "bench": d["scc_sum"]["epa_in_2015usd_approx"]["2.0%"],
        "n_sig": ap["n_significant_betas"], "n_ind": ap["n_industries"],
        "fm_t": fm_main["t"] if fm_main else np.nan,
        "fm_prem": fm_main["premium_1sd_annual"] if fm_main else np.nan,
    }


# =============================================================================
# Page
# =============================================================================
CSS = f"""
*{{box-sizing:border-box}} body{{margin:0;background:{BG};color:{INK};font-family:{FONT};line-height:1.55}}
.wrap{{max-width:1080px;margin:0 auto;padding:40px 22px 70px}}
header h1{{font-size:2.1rem;line-height:1.2;margin:0 0 8px;letter-spacing:-0.01em}}
header p.lede{{font-size:1.08rem;color:{INK2};margin:0 0 6px;max-width:760px}}
header p.meta{{color:{MUTED};font-size:.9rem;margin:10px 0 0}}
a{{color:{BLUE};text-decoration:none}} a:hover{{text-decoration:underline}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:30px 0 8px}}
.kpi{{background:{CARD};border:1px solid {GRID};border-radius:12px;padding:16px 18px}}
.kpi .v{{font-size:1.65rem;font-weight:700;color:{INK};letter-spacing:-0.01em}}
.kpi .l{{font-size:.88rem;color:{INK2};margin-top:2px}}
section{{background:{CARD};border:1px solid {GRID};border-radius:14px;padding:24px 24px 14px;margin-top:22px}}
section h2{{font-size:1.25rem;margin:0 0 4px}} section .num{{color:{MUTED};font-weight:600;margin-right:6px}}
section p{{color:{INK2};margin:6px 0 10px;max-width:860px}}
section .finding{{border-left:3px solid {BLUE};padding:6px 0 6px 14px;margin:12px 0 4px;color:{INK}}}
section .src{{font-size:.85rem;color:{MUTED};margin-top:4px}}
footer{{color:{MUTED};font-size:.88rem;margin-top:34px}}
@media (max-width:640px){{header h1{{font-size:1.6rem}} section{{padding:18px 14px 8px}}}}
"""


def section(num, title, intro, fig_html, finding, notebook):
    return f"""
<section>
  <h2><span class="num">{num}</span>{html.escape(title)}</h2>
  <p>{intro}</p>
  {fig_html}
  <p class="finding">{finding}</p>
  <p class="src">Full analysis: <a href="{REPO_URL}/blob/main/notebooks/{notebook}">{notebook}</a></p>
</section>"""


def build(inline: bool = False) -> Path:
    d = load()
    h = headlines(d)
    cfg = {"displaylogo": False, "responsive": True,
           "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}
    figs = {k: f.to_html(full_html=False, include_plotlyjs=False, config=cfg)
            for k, f in {"damage": fig_damage(d), "rates": fig_rates(d),
                         "scc": fig_scc(d), "betas": fig_betas(d)}.items()}

    plotly_tag = ("<script>" + __import__("plotly.offline", fromlist=["x"]).get_plotlyjs() + "</script>"
                  if inline else '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>')
    # Every claim follows the numbers, so a re-run with new data can't
    # leave the page saying something the results no longer support.
    ci_text = ("Every 95% confidence interval includes zero: the damage function is too imprecise "
               "to rule out a zero SCC." if h["all_ci_zero"]
               else "Some estimates are statistically distinguishable from zero; see the notebook for intervals.")
    rw_text = (f"a random walk cannot be rejected (t = {h['phi_t']:.1f} vs. a −2.9 critical value)"
               if h["phi_t"] > -2.9 else
               f"a random walk is rejected (t = {h['phi_t']:.1f} vs. a −2.9 critical value)")
    in_range = h["scc_min"] <= h["bench"] <= h["scc_max"]
    bench_text = (f"the official ≈${h['bench']:,.0f} sits inside that range" if in_range else
                  f"the official ≈${h['bench']:,.0f} lies {'below' if h['bench'] < h['scc_min'] else 'above'} that range")
    chance = 0.05 * h["n_ind"]
    sig_text = (f"Only {h['n_sig']} of {h['n_ind']} industries respond significantly (about {chance:.0f} "
                "expected by chance)" if h["n_sig"] <= 3 * chance else
                f"{h['n_sig']} of {h['n_ind']} industries respond significantly (about {chance:.0f} "
                "expected by chance)")
    if abs(h["fm_t"]) < 1.96:
        fm_text = (f", and Fama–MacBeth finds no significant price of temperature risk (t = {h['fm_t']:.2f}). "
                   "Markets, like the consumption model, attach little premium to short-run weather risk.")
    else:
        fm_text = (f", and Fama–MacBeth finds a significant price of temperature risk (t = {h['fm_t']:.2f}), "
                   f"worth about {h['fm_prem']:.1f}% a year per 1-sd difference in exposure: far above the "
                   "consumption model's near-zero premium.")
    shape_text = ("Warming raises growth in cold countries and lowers it in hot ones; the growth-maximising "
                  f"temperature is about {h['t_opt']:.0f}°C." if d["coef"]["main"]["b_temp_sq"] < 0 else
                  "The estimated response is not hump-shaped in this sample.")

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pricing the Future — climate damages, discounting and the social cost of carbon</title>
<meta name="description" content="Empirical damage function, data-calibrated discounting, social cost of carbon and a Fama-MacBeth test of temperature risk.">
{plotly_tag}
<style>{CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>Pricing the Future</h1>
  <p class="lede">How much is a tonne of CO₂ worth avoiding today? An empirical answer that estimates
  climate damages, calibrates four discounting methods to data, computes the social cost of carbon,
  and tests whether markets price temperature risk.</p>
  <p class="meta">Daniel Pazos Cordero · Extension of my M.Sc. project, University of Bath ·
  <a href="{REPO_URL}">Code and data on GitHub</a> · Built {date.today():%B %Y}</p>
</header>

<div class="kpis">
  <div class="kpi"><div class="v">${h['scc_min']:,.0f} – ${h['scc_max']:,.0f}</div>
    <div class="l">SCC per tonne across discounting methods (pooled damages, growth effects)</div></div>
  <div class="kpi"><div class="v">{h['curv_ratio']:.1f}×</div>
    <div class="l">steeper damage curve in poor countries than the global average</div></div>
  <div class="kpi"><div class="v">{h['np_ratio']:,.0f}×</div>
    <div class="l">value of $1 in 300 years: permanent vs mean-reverting rate shocks</div></div>
  <div class="kpi"><div class="v">{h['n_sig']} / {h['n_ind']}</div>
    <div class="l">industries whose returns respond significantly to temperature shocks</div></div>
</div>

{section("1", "How temperature affects economic growth",
         f"A panel of {h['n_countries']} countries, {h['years']}, with country and year fixed effects: "
         "each country's year-to-year weather shocks identify the effect. Hover over a dot to see a country.",
         figs["damage"],
         f"{shape_text} Poor countries' damage curve is {h['curv_ratio']:.1f}× as steep as the average, "
         f"with an optimum near {h['t_opt_poor']:.0f}°C.",
         "01_damages.ipynb")}

{section("2", "Four ways to discount the far future",
         "Forward discount rates implied by constant, gamma (Weitzman), stochastic (Newell–Pizer) and "
         "consumption-based (Ramsey SDF) discounting, each calibrated to data. Click legend entries to hide or show methods.",
         figs["rates"],
         f"US interest rates are highly persistent (φ = {h['phi']:.2f}); {rw_text}. Under the random walk, "
         f"long-run rates fall towards zero and $1 of damage in 300 years is worth {h['np_ratio']:,.0f}× more "
         "today than under mean reversion.",
         "02_discounting.ipynb")}

{section("3", "The social cost of carbon",
         "Damages from one extra tonne emitted today, discounted under each method, with 95% confidence "
         "intervals (cut off where they run past the axis). Use the menu to switch the damage specification "
         "and whether lost growth is ever recovered.",
         figs["scc"],
         f"The method moves the SCC from ${h['scc_min']:,.0f} ({html.escape(h['scc_min_m'])}) to "
         f"${h['scc_max']:,.0f} ({html.escape(h['scc_max_m'])}); {bench_text}. Whether lost growth is "
         f"recovered matters as much: ${h['lvl']:,.0f} (level effect) vs ${h['grw']:,.0f} (growth effect) "
         f"at a constant rate. {ci_text}",
         "03_scc.ipynb")}

{section("4", "Is temperature risk priced in stock returns?",
         "Response of each of the 49 Fama–French industries to a 1°C temperature surprise, controlling for "
         "market, size and value. Blue = significant at 5%.",
         figs["betas"],
         sig_text + fm_text,
         "04_asset_pricing.ipynb")}

<footer>
  Data: World Bank WDI, Copernicus ERA5 via Our World in Data, NASA GISTEMP v4, Kenneth French Data Library, FRED.
  Methods follow Burke, Hsiang &amp; Miguel (2015), Weitzman (2001), Newell &amp; Pizer (2003),
  Gollier (2008), Bansal, Kiku &amp; Ochoa (2016) and Fama &amp; MacBeth (1973).
  SCC in 2015 US$; benchmark: US EPA (2023) $190 in 2020 US$, adopted by ECCC.
</footer>
</div></body></html>"""

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "index.html"
    out.write_text(page, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")          # serve the file as-is on GitHub Pages
    print(f"Dashboard written to {out}  ({out.stat().st_size / 1024:,.0f} KB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inline", action="store_true", help="embed plotly.js so the page works offline")
    build(inline=ap.parse_args().inline)
