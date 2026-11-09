# =============================================================================
# 03_get_gistemp.R — NASA GISTEMP v4 global temperature anomaly (monthly)
# -----------------------------------------------------------------------------
# This is the series the original paper used for Figure 1. We keep it for two
# reasons:
#   * Step 5 (asset pricing) needs a long monthly global temperature series
#     back to 1880 to build "temperature shocks";
#   * it lets us reproduce the original global-time-series regression and show
#     side by side why the panel approach is better.
#
# File layout: line 1 is a title, line 2 the header
#   Year, Jan, ..., Dec, J-D, D-N, DJF, MAM, JJA, SON
# Missing values are written as "***". Anomalies are in °C vs 1951-1980.
#
# Output: data/processed/gistemp_global_monthly.csv
# =============================================================================

source(file.path("R", "_utils.R"))
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(readr) })

raw_file <- file.path(DIR_RAW, "gistemp_GLB_Ts_dSST.csv")
url <- "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

if (needs_download(raw_file)) {
  log_step("GISTEMP: downloading ...")
  download_to(url, raw_file)
} else {
  log_step("GISTEMP: cached copy found, skipping download")
}

gis_wide <- read_csv(raw_file, skip = 1, na = c("***", "****", ""),
                     show_col_types = FALSE)

# Reshape wide (one column per month) to long (one row per month).
gis_long <- gis_wide |>
  select(Year, all_of(month.abb)) |>
  pivot_longer(all_of(month.abb), names_to = "month_abb",
               values_to = "anomaly_c") |>
  mutate(year  = as.integer(Year),
         month = match(month_abb, month.abb),
         date  = as.Date(sprintf("%d-%02d-01", year, month)),
         anomaly_c = as.numeric(anomaly_c)) |>
  filter(!is.na(anomaly_c)) |>
  select(date, year, month, anomaly_c) |>
  arrange(date)

save_csv(gis_long, file.path(DIR_PROCESSED, "gistemp_global_monthly.csv"))
