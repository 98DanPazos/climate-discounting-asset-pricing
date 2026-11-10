# =============================================================================
# 07_damage_regressions.R — Step 2: how does temperature affect growth?
# -----------------------------------------------------------------------------
# PART A  Reproduce the original paper's approach: ONE global time series
#         (world GDP-per-capita growth on the global temperature anomaly).
#         ~60 observations -> little power. This is our baseline.
#
# PART B  The panel approach (Burke, Hsiang & Miguel 2015, "BHM"):
#
#           growth_it = b1*T_it + b2*T_it^2 + country_i + year_t + e_it
#
#         * country fixed effects absorb everything permanent about a country
#           (geography, institutions, average climate), so we only use each
#           country's year-to-year deviations from its own normal;
#         * year fixed effects absorb global shocks hitting everyone at once
#           (oil crises, 2008, COVID);
#         * so b1, b2 are identified from "this country was unusually hot this
#           year" — weather shocks that are as good as random.
#         * T^2 lets the effect of +1°C depend on the starting temperature:
#           dGrowth/dT = b1 + 2*b2*T. If b2 < 0, warming helps cold countries
#           and hurts hot ones, with a growth-maximising temperature
#           T* = -b1 / (2*b2).
#         * Standard errors are clustered by country (allows any correlation
#           of a country's errors over time).
#
# PART C  Heterogeneity (rich vs poor) and robustness checks.
#
# PART D  Figures + export the damage function for the Python engine (Step 3).
#
# Units: growth is 100 x log-difference ≈ percentage points per year.
#
# Outputs
#   output/tables/table1_global_timeseries.{txt,tex}
#   output/tables/table2_panel_main.{txt,tex}
#   output/tables/table3_robustness.{txt,tex}
#   output/figures/fig1_marginal_effect.png
#   output/figures/fig2_response_rich_poor.png
#   output/figures/fig3_optimum_robustness.png
#   output/tables/damage_coefficients.json   <- read by python/ in Step 3
#   output/tables/damage_function_grid.csv   <- for the dashboard (Step 6)
#
# When sourced (e.g. from the notebook), every model and figure object stays
# in memory so it can be displayed without re-running anything.
# =============================================================================

source(file.path("R", "_utils.R"))
suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr)
  library(fixest); library(ggplot2); library(jsonlite)
})

# Readable variable names in every regression table.
setFixest_dict(c(
  growth = "GDP pc growth (pp)", growth_w = "World GDP pc growth (pp)",
  temp = "Temperature (°C)", temp_sq = "Temperature²",
  temp_anom = "Temp. anomaly vs 1951-80 (°C)",
  anom = "Global temp. anomaly (°C)", anom_sq = "Global anomaly²",
  anom_ma5 = "Global anomaly, 5-yr avg",
  pop_growth_w = "World pop. growth (%)", log_co2 = "log global CO2",
  iso3c = "Country", year = "Year", poor = "Poor"
))

# Chart palette (validated default categorical slots 1 & 2 + neutral inks).
COL_MAIN  <- "#2a78d6"   # blue  — main estimate / rich countries
COL_ALT   <- "#eb6834"   # orange — poor countries
COL_INK   <- "#0b0b0b"; COL_INK2 <- "#52514e"; COL_MUTED <- "#898781"
COL_GRID  <- "#e1e0d9"

theme_project <- function() {
  theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = COL_GRID, linewidth = 0.3),
          axis.text  = element_text(colour = COL_MUTED),
          axis.title = element_text(colour = COL_INK2),
          plot.title = element_text(colour = COL_INK, face = "bold"),
          plot.subtitle = element_text(colour = COL_INK2),
          plot.caption  = element_text(colour = COL_MUTED, hjust = 0),
          legend.position = "top", legend.title = element_blank(),
          plot.background = element_rect(fill = "#fcfcfb", colour = NA))
}

# =============================================================================
# Load data
# =============================================================================
panel <- read_csv(file.path(DIR_PROCESSED, "panel_country_year.csv"),
                  show_col_types = FALSE)
est <- panel |> filter(!is.na(growth), !is.na(temp))
log_step("Estimation sample: ", nrow(est), " obs, ",
         n_distinct(est$iso3c), " countries")

# =============================================================================
# PART A — the original global time-series regression
# =============================================================================
# "World" growth = population-weighted mean of country growth each year
# (same countries as the panel, so the comparison is like-for-like).
gis_annual <- read_csv(file.path(DIR_PROCESSED, "gistemp_global_monthly.csv"),
                       show_col_types = FALSE) |>
  group_by(year) |>
  summarise(n = n(), anom = mean(anomaly_c), .groups = "drop") |>
  filter(n == 12) |> select(-n)

world <- est |>
  filter(!is.na(population)) |>
  group_by(year) |>
  summarise(growth_w     = weighted.mean(growth, population),
            pop_growth_w = weighted.mean(pop_growth, population, na.rm = TRUE),
            co2_total    = sum(co2_mt, na.rm = TRUE),
            .groups = "drop")

ts <- world |>
  inner_join(gis_annual, by = "year") |>
  arrange(year) |>
  mutate(unit     = 1L,                         # single "panel unit" so fixest
         anom_sq  = anom^2,                     # can compute Newey-West SEs
         anom_ma5 = as.numeric(stats::filter(anom, rep(1/5, 5), sides = 1)),
         log_co2  = ifelse(co2_total > 0, log(co2_total), NA_real_))

# Newey-West SEs (3 lags): time-series errors are autocorrelated, so plain
# OLS SEs would overstate precision.
ts_vcov <- NW(3) ~ unit + year
m_ts <- list(
  "Same-year" = feols(growth_w ~ anom + pop_growth_w + log_co2, ts, vcov = ts_vcov),
  "5-yr avg"  = feols(growth_w ~ anom_ma5 + pop_growth_w + log_co2, ts, vcov = ts_vcov),
  "Quadratic" = feols(growth_w ~ anom + anom_sq + pop_growth_w + log_co2, ts, vcov = ts_vcov)
)

# =============================================================================
# PART B — the panel (BHM) regressions
# =============================================================================
# Declare the panel structure once so l() (lags) and DK SEs know the time axis.
est <- est |> mutate(year = as.integer(year))
pdat <- panel(est, ~ iso3c + year)

m_main <- feols(growth ~ temp + temp_sq | iso3c + year, pdat, cluster = ~iso3c)

m_panel <- list(
  "Linear"            = feols(growth ~ temp | iso3c + year, pdat, cluster = ~iso3c),
  "Quadratic (main)"  = m_main,
  "+ country trends"  = feols(growth ~ temp + temp_sq | iso3c[year] + year,
                              pdat, cluster = ~iso3c),
  "Driscoll-Kraay SE" = summary(m_main, vcov = "DK")
)

# =============================================================================
# PART C — heterogeneity and robustness
# =============================================================================
# (1) Rich vs poor: same model, estimated separately on each group.
m_split <- feols(growth ~ temp + temp_sq | iso3c + year, pdat,
                 cluster = ~iso3c, split = ~poor)
m_rich <- m_split[[which(names(m_split) == "sample.var: poor; sample: 0")]]
m_poor <- m_split[[which(names(m_split) == "sample.var: poor; sample: 1")]]

# (2) Trim the 1% tails of growth (wars, oil-state swings).
q <- quantile(est$growth, c(0.01, 0.99))
pdat_trim <- panel(est |> filter(growth > q[1], growth < q[2]), ~ iso3c + year)

# (3) Lagged temperature: does the effect reverse next year (a "level" effect
#     that bounces back) or persist (a true growth effect)?
m_lag <- feols(growth ~ temp + temp_sq + l(temp, 1) + l(temp_sq, 1) | iso3c + year,
               pdat, cluster = ~iso3c)

m_robust <- list(
  "Main"             = m_main,
  "Trim 1%"          = feols(growth ~ temp + temp_sq | iso3c + year, pdat_trim, cluster = ~iso3c),
  "Pop.-weighted"    = feols(growth ~ temp + temp_sq | iso3c + year, pdat,
                             cluster = ~iso3c, weights = ~population),
  "Post-1990"        = feols(growth ~ temp + temp_sq | iso3c + year,
                             panel(est |> filter(year >= 1990), ~ iso3c + year),
                             cluster = ~iso3c),
  "+ 1 lag"          = m_lag,
  "Anomaly (linear)" = feols(growth ~ temp_anom | iso3c + year, pdat, cluster = ~iso3c)
)

# =============================================================================
# Helper functions: optimum temperature and the damage function (delta method)
# =============================================================================
# T* = -b1/(2 b2). Its SE comes from the delta method:
#   grad = d T*/d(b1, b2) = ( -1/(2 b2),  b1/(2 b2^2) )
#   Var(T*) = grad' V grad
optimum <- function(m, b1 = "temp", b2 = "temp_sq") {
  b <- coef(m)[c(b1, b2)]; V <- vcov(m)[c(b1, b2), c(b1, b2)]
  t_opt <- -b[1] / (2 * b[2])
  g <- c(-1 / (2 * b[2]), b[1] / (2 * b[2]^2))
  se <- sqrt(as.numeric(t(g) %*% V %*% g))
  c(t_opt = unname(t_opt), se = se,
    lo = unname(t_opt) - 1.96 * se, hi = unname(t_opt) + 1.96 * se)
}

# Marginal effect of +1°C at temperature T:  me(T) = b1 + 2 b2 T
#   Var(me) = V11 + 4 T^2 V22 + 4 T V12
# Response relative to T*:  f(T) = b1 (T - T*) + b2 (T^2 - T*^2), treating T*
#   as fixed at its point estimate (the BHM-style "response function").
damage_grid <- function(m, temps = seq(-10, 30, by = 0.25), label = "Main") {
  b <- coef(m)[c("temp", "temp_sq")]
  V <- vcov(m)[c("temp", "temp_sq"), c("temp", "temp_sq")]
  t_opt <- -b[1] / (2 * b[2])
  tibble(temp = temps) |>
    mutate(
      me     = b[1] + 2 * b[2] * temp,
      me_se  = sqrt(V[1, 1] + 4 * temp^2 * V[2, 2] + 4 * temp * V[1, 2]),
      d1     = temp - t_opt, d2 = temp^2 - t_opt^2,
      resp   = b[1] * d1 + b[2] * d2,
      resp_se = sqrt(d1^2 * V[1, 1] + d2^2 * V[2, 2] + 2 * d1 * d2 * V[1, 2]),
      me_lo = me - 1.96 * me_se,   me_hi = me + 1.96 * me_se,
      resp_lo = resp - 1.96 * resp_se, resp_hi = resp + 1.96 * resp_se,
      model = label
    ) |>
    select(-d1, -d2)
}

opt_main <- optimum(m_main)
grid_main <- damage_grid(m_main, label = "All countries")

# Each country's average temperature in the sample — used to place countries
# on the damage curve.
country_temp <- est |>
  group_by(iso3c, country) |>
  summarise(temp = mean(temp), .groups = "drop")
highlight <- c("CAN", "USA", "FRA", "CHN", "BRA", "IND", "NGA")
labels_df <- country_temp |>
  filter(iso3c %in% highlight) |>
  mutate(me = coef(m_main)["temp"] + 2 * coef(m_main)["temp_sq"] * temp) |>
  arrange(temp)

# =============================================================================
# PART D — tables
# =============================================================================
save_table <- function(models, stem, title) {
  txt <- capture.output(print(etable(models, title = title)))
  writeLines(txt, file.path(DIR_TABLES, paste0(stem, ".txt")))
  etable(models, title = title, tex = TRUE,
         file = file.path(DIR_TABLES, paste0(stem, ".tex")), replace = TRUE)
  log_step("Saved ", stem, ".txt/.tex")
}
save_table(m_ts,     "table1_global_timeseries",
           "Original approach: global time series (Newey-West SEs, 3 lags)")
save_table(m_panel,  "table2_panel_main",
           "Panel: country & year fixed effects (SEs clustered by country unless noted)")
save_table(c(m_robust, list("Rich" = m_rich, "Poor" = m_poor)),
           "table3_robustness", "Robustness and heterogeneity")

# =============================================================================
# PART D — figures
# =============================================================================
temp_hist <- est |> mutate(bin = floor(temp)) |> count(bin)

# Figure 1: marginal effect of +1°C on growth, by starting temperature.
fig1 <- ggplot(grid_main |> filter(temp >= min(est$temp), temp <= max(est$temp)),
               aes(temp, me)) +
  geom_hline(yintercept = 0, colour = "#c3c2b7", linewidth = 0.4) +
  geom_ribbon(aes(ymin = me_lo, ymax = me_hi), fill = COL_MAIN, alpha = 0.15) +
  geom_line(colour = COL_MAIN, linewidth = 0.8) +
  geom_point(data = labels_df, aes(temp, me), colour = COL_INK, size = 2.2) +
  # ggrepel nudges labels apart and draws a thin leader line when it has to
  # move one away from its point, so close neighbours (e.g. USA/CHN) stay legible.
  ggrepel::geom_text_repel(data = labels_df, aes(temp, me, label = iso3c),
                           colour = COL_INK2, size = 3.1, seed = 1,
                           box.padding = 0.5, min.segment.length = 0,
                           segment.colour = COL_MUTED, segment.size = 0.3) +
  geom_vline(xintercept = opt_main["t_opt"], linetype = "dashed",
             colour = COL_MUTED, linewidth = 0.4) +
  labs(title = "Effect of a 1°C warmer year on GDP-per-capita growth",
       subtitle = sprintf("Depends on how hot a country already is. Growth-maximising temperature T* ≈ %.1f°C (dashed).",
                          opt_main["t_opt"]),
       x = "Country's annual mean temperature (°C)",
       y = "Change in growth (percentage points)",
       caption = sprintf("Panel of %d countries, %d-%d. Country and year fixed effects; 95%% CI clustered by country. Points = country sample averages.",
                         n_distinct(est$iso3c), min(est$year), max(est$year))) +
  theme_project()

# Figure 2: response function, rich vs poor countries.
grid_rp <- bind_rows(damage_grid(m_rich, label = "Rich countries"),
                     damage_grid(m_poor, label = "Poor countries")) |>
  mutate(model = factor(model, levels = c("Rich countries", "Poor countries"))) |>
  filter(temp >= min(est$temp), temp <= max(est$temp))

fig2 <- ggplot(grid_rp, aes(temp, resp, colour = model, fill = model)) +
  geom_hline(yintercept = 0, colour = "#c3c2b7", linewidth = 0.4) +
  geom_ribbon(aes(ymin = resp_lo, ymax = resp_hi), alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = c("Rich countries" = COL_MAIN, "Poor countries" = COL_ALT)) +
  scale_fill_manual(values = c("Rich countries" = COL_MAIN, "Poor countries" = COL_ALT)) +
  coord_cartesian(ylim = range(c(grid_rp$resp_lo, 2), na.rm = TRUE)) +
  labs(title = "Growth response to temperature: rich vs poor countries",
       subtitle = "Growth relative to each group's own optimum temperature",
       x = "Annual mean temperature (°C)",
       y = "Growth vs optimum (percentage points)",
       caption = "Rich/poor = above/below median initial GDP per capita. 95% CI clustered by country.") +
  theme_project()

# Figure 3: how stable is T* across specifications?
opt_tbl <- bind_rows(lapply(
  c(m_robust[c("Main", "Trim 1%", "Pop.-weighted", "Post-1990")],
    list("+ country trends" = m_panel[["+ country trends"]],
         "Rich" = m_rich, "Poor" = m_poor)),
  function(m) as_tibble(as.list(optimum(m)))), .id = "spec") |>
  mutate(spec = factor(spec, levels = rev(spec)))

fig3 <- ggplot(opt_tbl, aes(t_opt, spec)) +
  geom_vline(xintercept = opt_main["t_opt"], colour = COL_MUTED, linewidth = 0.4,
             linetype = "dashed") +
  geom_segment(aes(x = lo, xend = hi, yend = spec), colour = COL_MAIN, linewidth = 0.6) +
  geom_point(colour = COL_MAIN, size = 2.5) +
  labs(title = "Growth-maximising temperature across specifications",
       subtitle = "Point estimate and 95% CI (delta method)",
       x = "T* (°C)", y = NULL) +
  theme_project()

ggsave(file.path(DIR_FIGURES, "fig1_marginal_effect.png"),  fig1, width = 8, height = 5, dpi = 200)
ggsave(file.path(DIR_FIGURES, "fig2_response_rich_poor.png"), fig2, width = 8, height = 5, dpi = 200)
ggsave(file.path(DIR_FIGURES, "fig3_optimum_robustness.png"), fig3, width = 7, height = 4.5, dpi = 200)
log_step("Saved figures 1-3")

# =============================================================================
# PART D — export the damage function for Step 3 (Python discounting engine)
# =============================================================================
coef_block <- function(m) {
  b <- coef(m)[c("temp", "temp_sq")]
  V <- vcov(m)[c("temp", "temp_sq"), c("temp", "temp_sq")]
  o <- optimum(m)
  list(b_temp = unname(b[1]), b_temp_sq = unname(b[2]),
       vcov = unname(V), t_opt = unname(o["t_opt"]), t_opt_se = unname(o["se"]),
       n_obs = nobs(m))
}

export <- list(
  description = paste("Panel damage function: growth_it = b_temp*T + b_temp_sq*T^2",
                      "+ country FE + year FE. Growth in percentage points",
                      "(100 x log difference) per year; T in °C. SEs clustered by country."),
  sample = list(countries = n_distinct(est$iso3c),
                years = c(min(est$year), max(est$year))),
  main = coef_block(m_main),
  rich = coef_block(m_rich),
  poor = coef_block(m_poor),
  canada_mean_temp = country_temp$temp[country_temp$iso3c == "CAN"],
  created = format(Sys.time(), "%Y-%m-%d %H:%M")
)
write_json(export, file.path(DIR_TABLES, "damage_coefficients.json"),
           auto_unbox = TRUE, digits = 10, pretty = TRUE)
save_csv(grid_main, file.path(DIR_TABLES, "damage_function_grid.csv"))

# =============================================================================
# Console summary
# =============================================================================
b <- coef(m_main)
cat("\n================ STEP 2 KEY RESULTS ================\n")
cat(sprintf("Main model: b_temp = %.3f, b_temp_sq = %.4f (N = %d)\n",
            b["temp"], b["temp_sq"], nobs(m_main)))
cat(sprintf("Growth-maximising temperature T* = %.1f°C  [95%% CI %.1f, %.1f]\n",
            opt_main["t_opt"], opt_main["lo"], opt_main["hi"]))
for (iso in c("CAN", "USA", "IND", "NGA")) {
  tt <- country_temp$temp[country_temp$iso3c == iso]
  if (length(tt)) cat(sprintf("  +1°C effect at %s (%.1f°C): %+.2f pp\n",
                              iso, tt, b["temp"] + 2 * b["temp_sq"] * tt))
}
cat(sprintf("Global time series, same-year anomaly coef: %.3f (p = %.3f, N = %d)\n",
            coef(m_ts[[1]])["anom"], pvalue(m_ts[[1]])["anom"], nobs(m_ts[[1]])))
cat("=====================================================\n")
