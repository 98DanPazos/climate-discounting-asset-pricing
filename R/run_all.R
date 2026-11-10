# =============================================================================
# run_all.R — rebuild every dataset from scratch, in order
# -----------------------------------------------------------------------------
# Usage (from the repo root):
#   Rscript R/run_all.R
# or in RStudio / Jupyter (R kernel) with the working directory at the root:
#   source("R/run_all.R")
#
# Each script is independent and cached, so a failure in one source (e.g. FRED
# key missing) doesn't stop the others; the summary at the end shows what ran.
# =============================================================================

scripts <- c(
  "R/01_get_worldbank.R",
  "R/02_get_temperature.R",
  "R/03_get_gistemp.R",
  "R/04_get_french.R",
  "R/05_get_rates.R",
  "R/06_build_panel.R",
  "R/07_damage_regressions.R"
)

status <- vapply(scripts, function(s) {
  cat("\n==============================", s, "==============================\n")
  ok <- tryCatch({ source(s, local = new.env(), encoding = "UTF-8"); TRUE },
                 error = function(e) { message("FAILED: ", conditionMessage(e)); FALSE })
  if (ok) "ok" else "FAILED"
}, character(1))

cat("\n---------- SUMMARY ----------\n")
print(data.frame(script = names(status), status = unname(status)), row.names = FALSE)
