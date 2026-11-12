# Pricing the Future: Climate Damages, Discounting, and Temperature Risk

**[▶ Interactive dashboard](https://98danpazos.github.io/climate-discounting-asset-pricing/)** · notebooks: [damages](notebooks/01_damages.ipynb) · [discounting](notebooks/02_discounting.ipynb) · [social cost of carbon](notebooks/03_scc.ipynb) · [asset pricing](notebooks/04_asset_pricing.ipynb)

An empirical extension of my M.Sc. project at the University of Bath, *"Empirical analysis of asset pricing models: Pricing the future — effective discounting for climate damages."*

The original brief compared fixed, Weitzman certainty-equivalent, stochastic (Newell–Pizer), and stochastic-discount-factor (SDF) approaches to valuing climate damages, and argued that fixed-rate discounting understates the social cost of carbon (SCC). This repo makes that argument **empirical**:

| Module | Question | Method | Language |
|---|---|---|---|
| 1. Damages | How does temperature affect GDP-per-capita growth? | Country-year panel, two-way fixed effects, non-linear temperature (Burke–Hsiang–Miguel 2015) | R |
| 2. Discounting | How sensitive is the SCC to the discounting method? | Discounting engine calibrated to real-rate data: fixed, gamma (Weitzman 2001), stochastic (Newell–Pizer 2003), consumption-based SDF | Python |
| 3. SCC | What do the estimates imply for Canadian carbon pricing? | Damage function × discount schedules, benchmarked to federal estimates | Python |
| 4. Asset pricing | Is temperature risk priced in equity returns? | Temperature betas + Fama–MacBeth on 49 industry portfolios (Bansal, Kiku & Ochoa) | R |
| 5. Dashboard | Interactive summary | Plotly | Python |

> **Status:** complete. All five modules, including the interactive dashboard (`docs/index.html`, built by `python/build_dashboard.py`).

## What changed from the original paper

* **Damages:** the original regressed global GDP growth on one global temperature series (~60 observations) and found no significant effect. Module 1 uses ~180 countries × ~60 years and identifies the effect from within-country weather shocks.
* **Discounting:** the original simulated interest-rate and consumption paths from assumed parameters; Module 2 estimates them from 70 years of US and Canadian rate data.
* **Asset pricing:** the original cited the SDF literature; Module 4 tests it.

## Data

| Source | Series | Script |
|---|---|---|
| World Bank WDI | GDP per capita (level, growth), population, CO₂ | `R/01_get_worldbank.R` |
| Copernicus ERA5 via Our World in Data | Monthly 2 m temperature by country, 1940– | `R/02_get_temperature.R` |
| NASA GISTEMP v4 | Global monthly temperature anomaly, 1880– | `R/03_get_gistemp.R` |
| Kenneth French Data Library | FF3, FF5 factors; 49 industry portfolios | `R/04_get_french.R` |
| FRED | US & Canada 10-y yields, CPI; US 10-y TIPS | `R/05_get_rates.R` |

## Reproduce

```bash
conda env create -f environment.yml
conda activate climate_ap
Rscript R/00_setup.R          # one-time package check
cp .Renviron.example .Renviron   # then paste your FRED key
Rscript R/run_all.R           # downloads, builds data/processed/, runs Modules 1 and 4 (R)
python python/run_discounting.py   # Module 2: calibrate + compare discounting methods
python python/run_scc.py           # Module 3: social cost of carbon under every method
pytest python/tests                # unit tests (discounting engine + SCC)
python python/build_dashboard.py   # Module 5: rebuild docs/index.html from all outputs
```

On Windows (Anaconda Prompt) use `copy` instead of `cp`.

## Repository layout

```
R/                 data pipeline + econometrics (R)
python/            discounting engine (package climate_discount + tests)
notebooks/         narrative notebooks for each module
data/raw/          downloads (git-ignored, rebuilt by run_all.R)
data/processed/    analysis-ready datasets
output/            figures and tables
docs/              interactive dashboard (served by GitHub Pages)
```

## References

Bansal, Kiku & Ochoa (2016) · Burke, Hsiang & Miguel (2015) · Dell, Jones & Olken (2012) · Newell & Pizer (2003) · Weitzman (1998, 2001, 2009) · Rennert et al. (2022) · Cochrane (2005)
