# =============================================================================
# 04_get_french.R — Kenneth French data library (for Step 5, asset pricing)
# -----------------------------------------------------------------------------
# We pull:
#   * Fama/French 3 Factors (monthly): Mkt-RF, SMB, HML, RF
#   * Fama/French 5 Factors (2x3) (monthly): adds RMW, CMA
#   * 49 Industry Portfolios, value-weighted monthly returns
#     -> these are our "test assets": we estimate each industry's exposure
#        (beta) to temperature shocks, then ask whether that beta earns a
#        risk premium (Fama-MacBeth, Step 5).
#
# All returns in the French files are in PERCENT per month. We keep them in
# percent here and convert in the analysis step, so the raw data stay faithful
# to the source.
#
# Outputs: data/processed/ff3_monthly.csv, ff5_monthly.csv,
#          industry49_vw_monthly.csv
# =============================================================================

source(file.path("R", "_utils.R"))
suppressPackageStartupMessages({ library(frenchdata); library(dplyr); library(readr) })

# frenchdata returns an object whose $subsets is a table of (name, data).
# Each dataset contains several sub-tables (monthly, annual, equal-weighted,
# value-weighted, ...). This helper picks the one whose name matches `pattern`.
get_french_subset <- function(dataset, pattern) {
  fd  <- download_french_data(dataset)
  hit <- grepl(pattern, fd$subsets$name, ignore.case = TRUE)
  if (!any(hit)) {
    stop("No subset matching '", pattern, "' in ", dataset, ". Available: ",
         paste(fd$subsets$name, collapse = " | "))
  }
  df <- fd$subsets$data[[which(hit)[1]]]
  # The `date` column is an integer like 196307 (YYYYMM) -> proper Date.
  df |>
    mutate(date = as.Date(paste0(date, "01"), format = "%Y%m%d")) |>
    filter(!is.na(date))
}

jobs <- list(
  list(dataset = "Fama/French 3 Factors",
       pattern = "^$|monthly",                       # first table = monthly
       file    = "ff3_monthly.csv"),
  list(dataset = "Fama/French 5 Factors (2x3)",
       pattern = "^$|monthly",
       file    = "ff5_monthly.csv"),
  list(dataset = "49 Industry Portfolios",
       pattern = "Average Value Weighted Returns -- Monthly",
       file    = "industry49_vw_monthly.csv")
)

for (job in jobs) {
  out <- file.path(DIR_PROCESSED, job$file)
  if (!needs_download(out)) {
    log_step("French: ", job$file, " cached, skipping")
    next
  }
  log_step("French: downloading '", job$dataset, "' ...")
  df <- get_french_subset(job$dataset, job$pattern)

  # The French files code missing returns as -99.99 or -999.
  df <- df |> mutate(across(-date, ~ ifelse(.x <= -99.99, NA, .x)))
  save_csv(df, out)
}
