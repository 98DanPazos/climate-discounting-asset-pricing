# =============================================================================
# 00_setup.R — install every R package the project needs (run once)
# -----------------------------------------------------------------------------
# Most packages are already installed by environment.yml (conda-forge). This
# script is the safety net: it checks what is missing and installs only that
# from CRAN. Re-running it is harmless.
# =============================================================================

packages <- c(
  # --- data wrangling ---------------------------------------------------------
  "dplyr", "tidyr", "readr", "purrr", "stringr", "lubridate",
  # --- data sources -----------------------------------------------------------
  "WDI",         # World Bank World Development Indicators API
  "fredr",       # FRED API (needs a free API key in .Renviron)
  "frenchdata",  # Kenneth French data library (factors + industry portfolios)
  # --- econometrics (used from Step 2 onward) --------------------------------
  "fixest",       # fast fixed-effects regressions, clustered/Driscoll-Kraay SEs
  "modelsummary", # publication-quality regression tables
  "sandwich", "lmtest",
  # --- plotting ---------------------------------------------------------------
  "ggplot2", "scales"
)

missing <- packages[!vapply(packages, requireNamespace, logical(1),
                            quietly = TRUE)]

if (length(missing) == 0) {
  message("All ", length(packages), " packages already installed.")
} else {
  message("Installing from CRAN: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
}

# Quick FRED key check — fredr reads FRED_API_KEY from the environment.
if (Sys.getenv("FRED_API_KEY") == "") {
  message("\nNOTE: FRED_API_KEY is not set. Copy .Renviron.example to ",
          ".Renviron in the repo root, paste your key, and restart R.")
} else {
  message("FRED_API_KEY found.")
}
