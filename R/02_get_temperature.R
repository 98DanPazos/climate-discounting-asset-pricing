# =============================================================================
# 02_get_temperature.R — country-level temperature, 1940 to present
# -----------------------------------------------------------------------------
# Source: Our World in Data, "Average monthly surface temperature", built from
#   the Copernicus ERA5 reanalysis (2 m air temperature). OWID clips the global
#   grid to each country's World Bank borders and takes a latitude-weighted
#   average. Unit: degrees Celsius.
#
# Why this source?  The original paper used ONE global temperature series,
# which gives ~60 observations and no cross-country variation. The climate-
# growth literature identifies the effect of temperature from year-to-year
# *within-country* weather shocks, which needs temperature by country and year.
#
# Note on the method: the CSV is monthly (one row per country-month). We build
# annual means ourselves and keep only country-years with all 12 months, so a
# partial current year can't sneak in as a "cold" year.
#
# Outputs:
#   data/raw/owid_temperature_monthly.csv        (untouched download)
#   data/processed/temperature_country_year.csv  (annual country means)
#   data/processed/temperature_monthly_key.csv   (USA, CAN, World monthly —
#                                                 used by the asset-pricing
#                                                 module in Step 5)
# =============================================================================

source(file.path("R", "_utils.R"))
suppressPackageStartupMessages({ library(dplyr); library(readr) })

raw_file <- file.path(DIR_RAW, "owid_temperature_monthly.csv")
url <- paste0("https://ourworldindata.org/grapher/",
              "average-monthly-surface-temperature.csv",
              "?v=1&csvType=full&useColumnShortNames=true")

# --- 1. Download (cached) ----------------------------------------------------
if (needs_download(raw_file)) {
  log_step("Temperature: downloading OWID / ERA5 monthly file ...")
  download_to(url, raw_file)
} else {
  log_step("Temperature: cached copy found, skipping download")
}

# --- 2. Read and standardise column names ------------------------------------
# Expected columns: entity, code, year, day, temperature_2m (+ possibly an
# extra annual-average column we don't need). We lower-case names so small
# changes in OWID capitalisation don't break the script.
temp_raw <- read_csv(raw_file, show_col_types = FALSE)
names(temp_raw) <- tolower(names(temp_raw))

required <- c("entity", "code", "temperature_2m")
if (!all(required %in% names(temp_raw))) {
  stop("OWID file layout changed. Found columns: ",
       paste(names(temp_raw), collapse = ", "))
}

# OWID has changed the time column over time. Older files had `day`
# ("1940-01-15"); newer ones have `month` ("1940-01" or "1940-01-15").
# We take whichever exists and parse it into a proper Date.
time_col <- intersect(c("day", "month", "date"), names(temp_raw))[1]
if (is.na(time_col)) {
  stop("No time column (day/month/date) in OWID file. Found: ",
       paste(names(temp_raw), collapse = ", "))
}

parse_owid_date <- function(x) {
  x <- as.character(x)
  # "YYYY-MM" has no day -> add "-01" so as.Date can read it.
  x <- ifelse(grepl("^\\d{4}-\\d{2}$", x), paste0(x, "-01"), x)
  as.Date(substr(x, 1, 10))
}

temp_monthly <- temp_raw |>
  transmute(
    entity,
    iso3c  = code,
    date   = parse_owid_date(.data[[time_col]]),
    year   = as.integer(format(date, "%Y")),
    month  = as.integer(format(date, "%m")),
    temp_c = temperature_2m
  ) |>
  filter(!is.na(temp_c))

# Safety check: if the date parsing silently failed, stop with a clear message
# rather than producing an empty panel.
if (all(is.na(temp_monthly$date))) {
  stop("Could not parse dates from column '", time_col, "'. First values: ",
       paste(head(temp_raw[[time_col]], 3), collapse = ", "))
}
log_step("Temperature: time column '", time_col, "', ",
         format(min(temp_monthly$date, na.rm = TRUE)), " to ",
         format(max(temp_monthly$date, na.rm = TRUE)))

# --- 3. Country-year annual means ---------------------------------------------
temp_annual <- temp_monthly |>
  # OWID uses "OWID_xxx" codes for aggregates/regions (e.g. OWID_WRL = World)
  # and for a few territories. Keep proper ISO3 codes for the country panel.
  filter(!is.na(iso3c), !grepl("^OWID_", iso3c)) |>
  group_by(iso3c, entity, year) |>
  summarise(n_months = n(),
            temp_mean = mean(temp_c),
            .groups = "drop") |>
  filter(n_months == 12) |>
  select(-n_months) |>
  arrange(iso3c, year)

save_csv(temp_annual, file.path(DIR_PROCESSED, "temperature_country_year.csv"))
log_step("Temperature: ", n_distinct(temp_annual$iso3c), " countries, ",
         min(temp_annual$year), "-", max(temp_annual$year))

# --- 4. Monthly series for the asset-pricing module ---------------------------
temp_key <- temp_monthly |>
  filter(iso3c %in% c("USA", "CAN") | entity == "World") |>
  select(entity, iso3c, date, year, month, temp_c) |>
  arrange(entity, date)

save_csv(temp_key, file.path(DIR_PROCESSED, "temperature_monthly_key.csv"))
