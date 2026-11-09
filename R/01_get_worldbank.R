# =============================================================================
# 01_get_worldbank.R — economic outcomes and controls for the country panel
# -----------------------------------------------------------------------------
# What we pull (World Bank World Development Indicators):
#   NY.GDP.PCAP.KD.ZG   GDP per capita growth, annual %      -> dependent var
#   NY.GDP.PCAP.KD      GDP per capita, constant 2015 US$    -> rebuild growth
#                                                               ourselves + rich/
#                                                               poor split
#   SP.POP.TOTL         Population, total                    -> regression weights
#   SP.POP.GROW         Population growth, annual %          -> control (as in
#                                                               the original paper)
#   EN.GHG.CO2.MT.CE.AR5  CO2 emissions excl. LULUCF, Mt CO2e  -> descriptive /
#                                                               robustness only
#
# Why rebuild growth from levels?  The WDI growth series and the log-difference
# of the level series should agree; checking that they do is a cheap data-
# quality test, and log-differences are what the climate-growth literature
# (Dell-Jones-Olken 2012; Burke-Hsiang-Miguel 2015) actually uses.
#
# Output: data/raw/wdi_country_year.csv
# =============================================================================

source(file.path("R", "_utils.R"))
suppressPackageStartupMessages({ library(WDI); library(dplyr) })

out_file <- file.path(DIR_RAW, "wdi_country_year.csv")

if (!needs_download(out_file)) {
  log_step("WDI: cached copy found, skipping download (set FORCE_DOWNLOAD=TRUE to refresh)")
} else {
  log_step("WDI: downloading indicators ", YEAR_START, "-", YEAR_END, " ...")

  # Named vector: the names become our column names.
  indicators <- c(
    gdp_pc_growth_wdi = "NY.GDP.PCAP.KD.ZG",
    gdp_pc            = "NY.GDP.PCAP.KD",
    population        = "SP.POP.TOTL",
    pop_growth        = "SP.POP.GROW",
    co2_mt            = "EN.GHG.CO2.MT.CE.AR5"
  )

  # extra = TRUE adds region, income group, lat/long, and iso3c.
  wdi <- WDI(country = "all", indicator = indicators,
             start = YEAR_START, end = YEAR_END, extra = TRUE)

  wdi_clean <- wdi |>
    # WDI returns regional aggregates ("World", "Euro area", ...) alongside
    # countries. Aggregates have region == "Aggregates"; drop them so the
    # panel contains countries only.
    filter(!is.na(region), region != "Aggregates") |>
    select(iso3c, country, year, region, income,
           latitude, longitude, all_of(names(indicators))) |>
    mutate(latitude  = as.numeric(latitude),
           longitude = as.numeric(longitude)) |>
    arrange(iso3c, year)

  save_csv(wdi_clean, out_file)
  log_step("WDI: ", n_distinct(wdi_clean$iso3c), " countries")
}
