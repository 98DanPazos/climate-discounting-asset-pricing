# =============================================================================
# 08_temperature_risk.R — Module 4: is temperature risk priced in stock returns?
# -----------------------------------------------------------------------------
# Module 3 found that the consumption-based model (Ramsey SDF) implies an
# almost zero risk premium for climate damages. Here we ask what MARKETS say,
# following the logic of Bansal, Kiku & Ochoa (2016):
#
#   STEP 1  Temperature shocks. Monthly temperature contains a seasonal cycle,
#           a warming trend and persistence; investors can anticipate all
#           three. The *shock* is what's left: the residual from
#               anomaly_t = c + d*trend + rho*anomaly_{t-1} + e_t
#           where anomaly = temperature minus that calendar month's average.
#           Built for the US (ERA5) and the globe (NASA GISTEMP).
#
#   STEP 2  Temperature betas. For each of the 49 Fama-French industry
#           portfolios, regress monthly excess returns on the Fama-French
#           factors plus the temperature shock:
#               R_it - RF_t = a_i + b_i'F_t + beta_temp_i * shock_t + e_it
#           beta_temp_i: how much industry i's return moves (% per month) when
#           temperature is 1°C higher than expected. Newey-West SEs.
#
#   STEP 3  Fama-MacBeth (1973). Is temperature exposure *priced*?
#           (a) First pass: rolling 60-month betas for every industry-month,
#               using only past data (no look-ahead).
#           (b) Second pass: every month, regress the 49 industries' returns
#               on their betas -> a price of risk lambda_t for each factor.
#           (c) The premium is the time-series average of lambda_t; its
#               t-statistic uses Newey-West SEs on the lambda_t series.
#           lambda_temp > 0: investors demand extra return to hold industries
#           that do well when it's unexpectedly hot (i.e. temperature shocks
#           are "good news" states); lambda_temp < 0: industries that hedge
#           heat are expensive (heat is a "bad state" investors insure against).
#
# Units: returns and factors in % per month; shocks in °C.
#
# Outputs
#   output/tables/table4_temperature_betas.{csv,txt}
#   output/tables/table5_fama_macbeth.{csv,txt}
#   output/tables/asset_pricing_results.json
#   output/figures/fig8_temperature_betas.png
#   output/figures/fig9_alpha_vs_temp_beta.png
# =============================================================================

source(file.path("R", "_utils.R"))
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr)
  library(ggplot2); library(sandwich); library(jsonlite)
})

SAMPLE_START <- as.Date("1960-01-01")   # 49-industry coverage is complete by then
WINDOW       <- 60                      # months in each rolling first-pass window
MIN_OBS      <- 48                      # min months with data inside a window
NW_LAG       <- 6                       # Newey-West lags (monthly data)

COL_MAIN <- "#2a78d6"; COL_ALT <- "#eb6834"
COL_INK <- "#0b0b0b"; COL_INK2 <- "#52514e"; COL_MUTED <- "#898781"; COL_GRID <- "#e1e0d9"
theme_project <- function() {
  theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = COL_GRID, linewidth = 0.3),
          axis.text = element_text(colour = COL_MUTED), axis.title = element_text(colour = COL_INK2),
          plot.title = element_text(colour = COL_INK, face = "bold"),
          plot.subtitle = element_text(colour = COL_INK2),
          plot.caption = element_text(colour = COL_MUTED, hjust = 0),
          legend.position = "top", legend.title = element_blank(),
          plot.background = element_rect(fill = "#fcfcfb", colour = NA))
}

# Consistent column names: "Mkt-RF" -> "mkt_rf", "Agric " -> "agric"
clean_names <- function(df) {
  names(df) <- tolower(gsub("[^A-Za-z0-9]+", "_", trimws(names(df))))
  names(df) <- gsub("^_|_$", "", names(df))
  df
}
# Put every date on the first of its month so all sources line up
month_start <- function(d) as.Date(format(as.Date(d), "%Y-%m-01"))

# Newey-West t-statistic for the mean of a series
nw_mean <- function(x, lag = NW_LAG) {
  x <- x[!is.na(x)]
  fit <- lm(x ~ 1)
  se <- sqrt(NeweyWest(fit, lag = lag, prewhite = FALSE)[1, 1])
  c(mean = mean(x), se = se, t = mean(x) / se, n = length(x))
}

# =============================================================================
# STEP 1 — temperature shocks
# =============================================================================
make_shock <- function(df) {
  # df: date, value (monthly). Returns date, anomaly, shock.
  df <- df |>
    mutate(date = month_start(date), month = as.integer(format(date, "%m"))) |>
    group_by(date) |> summarise(value = mean(value), month = first(month), .groups = "drop") |>
    arrange(date)
  base <- df |> filter(date >= as.Date("1951-01-01"), date < as.Date("1981-01-01"))
  if (nrow(base) < 120) base <- df                      # fallback: whole sample
  clim <- base |> group_by(month) |> summarise(clim = mean(value), .groups = "drop")
  df <- df |>
    left_join(clim, by = "month") |>
    mutate(anomaly = value - clim,
           trend = row_number(),
           anomaly_lag = lag(anomaly))
  fit <- lm(anomaly ~ trend + anomaly_lag, data = df)
  df$shock <- NA_real_
  df$shock[as.integer(names(residuals(fit)))] <- residuals(fit)
  attr(df, "rho") <- unname(coef(fit)["anomaly_lag"])
  df |> select(date, anomaly, shock)
}

temp_key <- read_csv(file.path(DIR_PROCESSED, "temperature_monthly_key.csv"), show_col_types = FALSE)
shock_us <- make_shock(temp_key |> filter(iso3c == "USA") |> transmute(date, value = temp_c))
rho_us <- attr(shock_us, "rho")

gis <- read_csv(file.path(DIR_PROCESSED, "gistemp_global_monthly.csv"), show_col_types = FALSE)
shock_gl <- make_shock(gis |> transmute(date, value = anomaly_c))
rho_gl <- attr(shock_gl, "rho")

shocks <- full_join(shock_us |> select(date, shock_us = shock, anom_us = anomaly),
                    shock_gl |> select(date, shock_gl = shock, anom_gl = anomaly), by = "date")
log_step(sprintf("Shocks built. Persistence of anomaly: US rho = %.2f, global rho = %.2f", rho_us, rho_gl))

# =============================================================================
# Returns and factors
# =============================================================================
ff3 <- read_csv(file.path(DIR_PROCESSED, "ff3_monthly.csv"), show_col_types = FALSE) |>
  clean_names() |> mutate(date = month_start(date))
ff5 <- read_csv(file.path(DIR_PROCESSED, "ff5_monthly.csv"), show_col_types = FALSE) |>
  clean_names() |> mutate(date = month_start(date)) |> select(date, rmw, cma)
ind_raw <- read_csv(file.path(DIR_PROCESSED, "industry49_vw_monthly.csv"), show_col_types = FALSE)
industry_labels <- trimws(setdiff(names(ind_raw), "date"))
ind <- ind_raw |> clean_names() |> mutate(date = month_start(date))
industries <- setdiff(names(ind), "date")
label_of <- setNames(industry_labels, industries)

panel <- ind |>
  pivot_longer(-date, names_to = "industry", values_to = "ret") |>
  inner_join(ff3, by = "date") |>
  left_join(ff5, by = "date") |>
  inner_join(shocks, by = "date") |>
  filter(date >= SAMPLE_START, !is.na(ret)) |>
  mutate(exret = ret - rf)
log_step("Panel: ", n_distinct(panel$industry), " industries, ",
         format(min(panel$date)), " to ", format(max(panel$date)))

# =============================================================================
# STEP 2 — full-sample temperature betas (time series, per industry)
# =============================================================================
ts_betas <- function(dat, shock_var = "shock_us", factors = c("mkt_rf", "smb", "hml")) {
  f <- reformulate(c(factors, shock_var), response = "exret")
  bind_rows(lapply(split(dat, dat$industry), function(d) {
    d <- d[complete.cases(d[, c("exret", factors, shock_var)]), ]
    fit <- lm(f, data = d)
    V <- NeweyWest(fit, lag = NW_LAG, prewhite = FALSE)
    b <- coef(fit)[shock_var]; se <- sqrt(V[shock_var, shock_var])
    tibble(industry = d$industry[1], beta_temp = b, se = se, t = b / se,
           alpha = coef(fit)["(Intercept)"], n = nrow(d))
  }))
}

betas_main <- ts_betas(panel) |>
  mutate(label = label_of[industry],
         lo = beta_temp - 1.96 * se, hi = beta_temp + 1.96 * se,
         significant = abs(t) > 1.96)

# Alpha relative to FF3 WITHOUT temperature: the return the standard model
# can't explain. If temperature risk is priced, it should line up with beta_temp.
alpha_ff3 <- bind_rows(lapply(split(panel, panel$industry), function(d) {
  fit <- lm(exret ~ mkt_rf + smb + hml, data = d)
  tibble(industry = d$industry[1], alpha_ff3 = unname(coef(fit)[1]))
}))
betas_main <- betas_main |> left_join(alpha_ff3, by = "industry")

# =============================================================================
# STEP 3 — Fama-MacBeth with rolling first-pass betas
# =============================================================================
fama_macbeth <- function(dat, factors, label, start = SAMPLE_START, rolling = TRUE) {
  dat <- dat |> filter(date >= start)
  dates <- sort(unique(dat$date))
  inds  <- sort(unique(dat$industry))
  # Wide matrices: months x industries (returns) and months x factors
  R <- dat |> select(date, industry, exret) |>
    pivot_wider(names_from = industry, values_from = exret) |>
    arrange(date)
  Rm <- as.matrix(R[, inds])
  Fm <- dat |> distinct(date, .keep_all = TRUE) |> arrange(date) |>
    select(all_of(factors)) |> as.matrix()
  Tn <- nrow(Rm); K <- length(factors)

  # (a) First pass: betas[t, i, ] estimated on months t-WINDOW .. t-1
  #     (rolling = FALSE: one full-sample beta per industry, used every month.
  #      Less noisy, so less errors-in-variables bias, but uses future data.)
  betas <- array(NA_real_, dim = c(Tn, length(inds), K))
  if (!rolling) {
    X <- cbind(1, Fm)
    for (i in seq_along(inds)) {
      ok <- complete.cases(X) & !is.na(Rm[, i])
      b <- lm.fit(X[ok, , drop = FALSE], Rm[ok, i])$coefficients[-1]
      for (t in (WINDOW + 1):Tn) betas[t, i, ] <- b
    }
  }
  if (rolling) for (t in (WINDOW + 1):Tn) {
    rows <- (t - WINDOW):(t - 1)
    X <- cbind(1, Fm[rows, , drop = FALSE])
    okF <- complete.cases(X)
    for (i in seq_along(inds)) {
      y <- Rm[rows, i]
      ok <- okF & !is.na(y)
      if (sum(ok) >= MIN_OBS) {
        betas[t, i, ] <- lm.fit(X[ok, , drop = FALSE], y[ok])$coefficients[-1]
      }
    }
  }

  # (b) Second pass: cross-section each month
  lam <- matrix(NA_real_, Tn, K + 1, dimnames = list(NULL, c("const", factors)))
  for (t in (WINDOW + 1):Tn) {
    B <- betas[t, , ]
    y <- Rm[t, ]
    ok <- complete.cases(B) & !is.na(y)
    if (sum(ok) >= 30) lam[t, ] <- lm.fit(cbind(1, B[ok, , drop = FALSE]), y[ok])$coefficients
  }

  # (c) Average prices of risk with Newey-West t-stats
  res <- bind_rows(lapply(colnames(lam), function(k) {
    s <- nw_mean(lam[, k])
    tibble(spec = label, term = k, lambda = s["mean"], se = s["se"], t = s["t"], months = s["n"])
  }))
  # Economic size: premium for a 1-sd difference in temperature beta, % per year
  temp_col <- grep("^shock", factors)
  sd_beta <- mean(apply(betas[, , temp_col], 1, sd, na.rm = TRUE), na.rm = TRUE)
  attr(res, "lambda_ts") <- tibble(date = dates, lambda_temp = lam[, temp_col + 1])
  attr(res, "sd_beta_temp") <- sd_beta
  res
}

log_step("Fama-MacBeth: running 5 specifications (rolling 60-month betas) ...")
fm_specs <- list(
  list(label = "FF3 + US temp (main)",    factors = c("mkt_rf", "smb", "hml", "shock_us")),
  list(label = "FF3 + global temp",       factors = c("mkt_rf", "smb", "hml", "shock_gl")),
  list(label = "FF5 + US temp",           factors = c("mkt_rf", "smb", "hml", "rmw", "cma", "shock_us"),
       start = as.Date("1963-07-01")),
  list(label = "FF3 + US temp, post-1990", factors = c("mkt_rf", "smb", "hml", "shock_us"),
       start = as.Date("1990-01-01")),
  list(label = "FF3 + US temp, full-sample betas", factors = c("mkt_rf", "smb", "hml", "shock_us"),
       rolling = FALSE)
)
fm_results <- lapply(fm_specs, function(s) {
  fama_macbeth(panel, s$factors, s$label,
               start = if (is.null(s$start)) SAMPLE_START else s$start,
               rolling = if (is.null(s$rolling)) TRUE else s$rolling)
})
fm_table <- bind_rows(fm_results)
fm_temp <- fm_table |>
  filter(grepl("^shock", term)) |>
  mutate(sd_beta = vapply(fm_results, function(r) attr(r, "sd_beta_temp"), numeric(1)),
         premium_1sd_annual = 12 * lambda * sd_beta)

# =============================================================================
# Tables
# =============================================================================
tab4 <- betas_main |>
  arrange(desc(beta_temp)) |>
  transmute(industry = label, beta_temp = round(beta_temp, 3), nw_se = round(se, 3),
            t = round(t, 2), months = n)
write_csv(tab4, file.path(DIR_TABLES, "table4_temperature_betas.csv"))
writeLines(capture.output(print(as.data.frame(tab4), row.names = FALSE)),
           file.path(DIR_TABLES, "table4_temperature_betas.txt"))

tab5 <- fm_table |> mutate(across(c(lambda, se, t), ~ round(.x, 3)))
write_csv(tab5, file.path(DIR_TABLES, "table5_fama_macbeth.csv"))
writeLines(capture.output(print(as.data.frame(tab5), row.names = FALSE)),
           file.path(DIR_TABLES, "table5_fama_macbeth.txt"))
log_step("Saved tables 4-5")

# =============================================================================
# Figures
# =============================================================================
plot_b <- betas_main |>
  mutate(label = factor(label, levels = label[order(beta_temp)]),
         sig = ifelse(significant, "Significant at 5%", "Not significant"))

fig8 <- ggplot(plot_b, aes(beta_temp, label, colour = sig)) +
  geom_vline(xintercept = 0, colour = "#c3c2b7", linewidth = 0.4) +
  geom_segment(aes(x = lo, xend = hi, yend = label), linewidth = 0.5) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = c("Significant at 5%" = COL_MAIN, "Not significant" = "#c3c2b7")) +
  labs(title = "How do industries respond to temperature shocks?",
       subtitle = "Monthly excess return (%) when US temperature is 1°C above expectations, controlling for FF3 factors",
       x = "Temperature beta (% return per °C shock)", y = NULL,
       caption = sprintf("49 Fama-French industries, %s-%s. 95%% CI with Newey-West SEs (%d lags).",
                         format(min(panel$date), "%Y"), format(max(panel$date), "%Y"), NW_LAG)) +
  theme_project() + theme(axis.text.y = element_text(size = 7, colour = COL_INK2))

lab_pts <- bind_rows(slice_max(betas_main, beta_temp, n = 4), slice_min(betas_main, beta_temp, n = 4),
                     slice_max(betas_main, abs(alpha_ff3), n = 2)) |> distinct(industry, .keep_all = TRUE)
fig9 <- ggplot(betas_main, aes(beta_temp, 12 * alpha_ff3)) +
  geom_hline(yintercept = 0, colour = "#c3c2b7", linewidth = 0.4) +
  geom_vline(xintercept = 0, colour = "#c3c2b7", linewidth = 0.4) +
  geom_smooth(method = "lm", formula = y ~ x, colour = COL_ALT, fill = COL_ALT, alpha = 0.12, linewidth = 0.7) +
  geom_point(colour = COL_MAIN, size = 2) +
  ggrepel::geom_text_repel(data = lab_pts, aes(label = label), size = 3, colour = COL_INK2,
                           seed = 1, min.segment.length = 0, segment.colour = COL_MUTED) +
  labs(title = "Is temperature exposure rewarded?",
       subtitle = "Return the Fama-French 3-factor model cannot explain vs. each industry's temperature beta",
       x = "Temperature beta (% return per °C shock)", y = "FF3 alpha (% per year)",
       caption = "A sloped relationship would mean temperature risk carries a premium. Line: OLS fit, 95% band.") +
  theme_project()

ggsave(file.path(DIR_FIGURES, "fig8_temperature_betas.png"), fig8, width = 8, height = 9, dpi = 200)
ggsave(file.path(DIR_FIGURES, "fig9_alpha_vs_temp_beta.png"), fig9, width = 8, height = 5.5, dpi = 200)
log_step("Saved figures 8-9")

# =============================================================================
# Export + console summary
# =============================================================================
main <- fm_temp |> filter(spec == "FF3 + US temp (main)")
export <- list(
  sample = c(format(min(panel$date)), format(max(panel$date))),
  shock_persistence = list(us = rho_us, global = rho_gl),
  n_industries = n_distinct(panel$industry),
  n_significant_betas = sum(betas_main$significant),
  fama_macbeth_temperature = fm_temp,
  top_positive = head(tab4, 5), top_negative = tail(tab4, 5)
)
write_json(export, file.path(DIR_TABLES, "asset_pricing_results.json"),
           auto_unbox = TRUE, digits = 8, pretty = TRUE)

cat("\n================ STEP 5 / MODULE 4 KEY RESULTS ================\n")
cat(sprintf("Sample %s to %s, %d industries\n", export$sample[1], export$sample[2], export$n_industries))
cat(sprintf("Temperature anomaly persistence: US rho = %.2f, global rho = %.2f\n", rho_us, rho_gl))
cat(sprintf("Industries with a significant US temperature beta (5%%): %d of %d (expect ~%.0f by chance)\n",
            export$n_significant_betas, export$n_industries, 0.05 * export$n_industries))
cat("\nMost positive temperature betas:\n"); print(as.data.frame(head(tab4, 5)), row.names = FALSE)
cat("\nMost negative temperature betas:\n"); print(as.data.frame(tail(tab4, 5)), row.names = FALSE)
cat("\nFama-MacBeth price of temperature risk (lambda, % per month per unit beta):\n")
print(as.data.frame(fm_temp |> transmute(spec, lambda = round(lambda, 3), t = round(t, 2),
                                         months, premium_1sd_annual = round(premium_1sd_annual, 2))),
      row.names = FALSE)
cat("\nAll Fama-MacBeth prices of risk (main spec):\n")
print(as.data.frame(fm_table |> filter(spec == "FF3 + US temp (main)") |>
                      transmute(term, lambda = round(lambda, 3), t = round(t, 2))), row.names = FALSE)
cat("================================================================\n")
