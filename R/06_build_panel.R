# =============================================================================
# 06_build_panel.R — merge WDI + temperature into the analysis panel
# -----------------------------------------------------------------------------
# This is the dataset Step 2's regressions run on. One row = one country-year.
#
# Key constructed variables
#   growth      100 * [ln(GDPpc_t) - ln(GDPpc_{t-1})]  -> dependent variable
#   temp        annual mean temperature, °C
#   temp_sq     temp^2  -> lets the effect of +1°C depend on how hot a country
#               already is (the Burke-Hsiang-Miguel "hump")
#   temp_anom   temp minus the country's own 1951-1980 average, i.e. the
#               "anomaly" concept the original paper used, now per country
#   poor        1 if the country's GDPpc in its first observed year was below
#               the sample median -> heterogeneity test in Step 2
#
# The script ends with data-quality checks printed to the console AND written
# to output/tables/panel_diagnostics.txt so the README can cite them.
#
# Output: data/processed/panel_country_year.csv
# =============================================================================

source(file.path("R", "_utils.R"))
suppressPackageStartupMessages({ library(dplyr); library(readr) })

wdi  <- read_csv(file.path(DIR_RAW, "wdi_country_year.csv"), show_col_types = FALSE)
temp <- read_csv(file.path(DIR_PROCESSED, "temperature_country_year.csv"),
                 show_col_types = FALSE)

# --- 1. Country baseline climate (1951-1980), computed BEFORE trimming years --
baseline <- temp |>
  filter(year >= 1951, year <= 1980) |>
  group_by(iso3c) |>
  summarise(temp_base_5180 = mean(temp_mean), .groups = "drop")

# --- 2. Merge -------------------------------------------------------------------
# inner_join: a country-year must have BOTH GDP and temperature to be useful.
panel <- wdi |>
  inner_join(temp |> select(iso3c, year, temp = temp_mean),
             by = c("iso3c", "year")) |>
  left_join(baseline, by = "iso3c") |>
  arrange(iso3c, year) |>
  group_by(iso3c) |>
  mutate(
    # lag() inside group_by works within country. We also require the previous
    # row to be exactly year-1 so gaps in the data don't create fake growth.
    growth = ifelse(lag(year) == year - 1,
                    100 * (log(gdp_pc) - log(lag(gdp_pc))), NA_real_),
    temp_sq   = temp^2,
    temp_anom = temp - temp_base_5180,
    temp_lag1 = ifelse(lag(year) == year - 1, lag(temp), NA_real_)
  ) |>
  ungroup()

# --- 3. Rich / poor split, by initial income ------------------------------------
initial_income <- panel |>
  filter(!is.na(gdp_pc)) |>
  group_by(iso3c) |>
  slice_min(year, n = 1) |>
  ungroup() |>
  transmute(iso3c, gdp_pc_initial = gdp_pc)

panel <- panel |>
  left_join(initial_income, by = "iso3c") |>
  mutate(poor = as.integer(gdp_pc_initial < median(initial_income$gdp_pc_initial,
                                                   na.rm = TRUE)))

save_csv(panel, file.path(DIR_PROCESSED, "panel_country_year.csv"))

# --- 4. Diagnostics -------------------------------------------------------------
est <- panel |> filter(!is.na(growth), !is.na(temp))

unmatched_wdi  <- setdiff(unique(wdi$iso3c),  unique(temp$iso3c))
growth_check   <- est |> filter(!is.na(gdp_pc_growth_wdi))
growth_corr    <- cor(growth_check$growth, growth_check$gdp_pc_growth_wdi)

diag <- c(
  "PANEL DIAGNOSTICS",
  sprintf("Built: %s", Sys.time()),
  "",
  sprintf("Estimation sample (non-missing growth & temp): %d obs", nrow(est)),
  sprintf("Countries: %d   Years: %d-%d", n_distinct(est$iso3c),
          min(est$year), max(est$year)),
  sprintf("Avg years per country: %.1f", nrow(est) / n_distinct(est$iso3c)),
  "",
  sprintf("Growth (log-diff) vs WDI growth series, correlation: %.4f", growth_corr),
  "  (should be > 0.99; a lower value means a level/growth mismatch to inspect)",
  "",
  sprintf("Temperature range: %.1f to %.1f °C", min(est$temp), max(est$temp)),
  sprintf("Growth range: %.1f to %.1f %%  (extreme values = wars/oil states;",
          min(est$growth), max(est$growth)),
  "  Step 2 checks robustness to trimming the top/bottom 1%)",
  "",
  sprintf("WDI countries with no temperature match (%d): %s",
          length(unmatched_wdi), paste(unmatched_wdi, collapse = ", "))
)

writeLines(diag, file.path(DIR_TABLES, "panel_diagnostics.txt"))
cat(paste(diag, collapse = "\n"), "\n")
