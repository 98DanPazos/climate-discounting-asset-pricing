# =============================================================================
# 05_get_rates.R — long-run interest rates and inflation (for Step 3)
# -----------------------------------------------------------------------------
# The original paper ASSUMED an interest-rate process for stochastic
# (Newell-Pizer) discounting. In Step 3 we ESTIMATE it from data. That needs a
# long history of real long-term interest rates, which we build as
#   real rate = nominal 10-year government yield - year-over-year CPI inflation
# for the US and Canada, plus the market-traded US real yield (10y TIPS) as a
# cross-check from 2003 on.
#
# FRED series (all monthly):
#   IRLTLT01USM156N  US long-term government bond yield (OECD MEI), %
#   IRLTLT01CAM156N  Canada long-term government bond yield (OECD MEI), %
#   CPIAUCSL         US CPI, all items, index (seasonally adjusted)
#   CANCPIALLMINMEI  Canada CPI, all items, index 2015=100, NSA (OECD MEI)
#   DFII10           US 10y TIPS real yield (daily -> monthly average), %
#
# Each series is downloaded separately inside tryCatch: if FRED retires one
# series, the script logs it and carries on instead of failing.
#
# Requires FRED_API_KEY in .Renviron.
# Output: data/processed/rates_monthly.csv
# =============================================================================

source(file.path("R", "_utils.R"))
suppressPackageStartupMessages({ library(fredr); library(dplyr); library(tidyr); library(purrr) })

out_file <- file.path(DIR_PROCESSED, "rates_monthly.csv")

if (!needs_download(out_file)) {
  log_step("FRED: cached, skipping")
} else {
  if (Sys.getenv("FRED_API_KEY") == "") stop("FRED_API_KEY not set — see .Renviron.example")

  series <- c(
    us_lt_yield = "IRLTLT01USM156N",
    ca_lt_yield = "IRLTLT01CAM156N",
    us_cpi      = "CPIAUCSL",
    ca_cpi      = "CANCPIALLMINMEI",
    us_tips10   = "DFII10"
  )

  fetch_one <- function(name, id) {
    log_step("FRED: ", id)
    tryCatch(
      fredr(series_id = id,
            observation_start = as.Date("1950-01-01"),
            frequency = "m",            # FRED aggregates daily -> monthly
            aggregation_method = "avg") |>
        transmute(date, series = name, value),
      error = function(e) {
        log_step("  !! failed: ", conditionMessage(e)); NULL
      }
    )
  }

  rates_long <- imap(series, ~ fetch_one(.y, .x)) |> compact() |> bind_rows()

  rates <- rates_long |>
    pivot_wider(names_from = series, values_from = value) |>
    arrange(date)

  # Build inflation and real rates only for the series that actually arrived,
  # so one retired FRED series doesn't crash the whole script.
  has <- function(...) all(c(...) %in% names(rates))

  if (has("us_cpi")) {
    # Year-over-year inflation from the CPI index: (P_t / P_{t-12} - 1) * 100
    rates <- rates |> mutate(us_infl_yoy = 100 * (us_cpi / lag(us_cpi, 12) - 1))
  }
  if (has("ca_cpi")) {
    rates <- rates |> mutate(ca_infl_yoy = 100 * (ca_cpi / lag(ca_cpi, 12) - 1))
  }
  # Ex-post real long rate = nominal yield - realised inflation. Crude (uses
  # realised, not expected, inflation) but standard for long samples; Step 3
  # discusses the alternatives.
  if (has("us_lt_yield", "us_infl_yoy")) {
    rates <- rates |> mutate(us_real_lt = us_lt_yield - us_infl_yoy)
  }
  if (has("ca_lt_yield", "ca_infl_yoy")) {
    rates <- rates |> mutate(ca_real_lt = ca_lt_yield - ca_infl_yoy)
  }

  missing_series <- setdiff(names(series), names(rates))
  if (length(missing_series) > 0) {
    log_step("FRED: WARNING, missing series: ", paste(missing_series, collapse = ", "))
  }

  save_csv(rates, out_file)
}
