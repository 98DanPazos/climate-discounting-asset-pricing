# =============================================================================
# _utils.R — shared helpers sourced by every script in R/
# -----------------------------------------------------------------------------
# Why a utils file?  Every script needs the same three things:
#   1. to know where the project root is (so paths work from RStudio, Jupyter,
#      or the command line, regardless of the working directory),
#   2. a consistent way to save raw / processed data,
#   3. a "cache" rule: don't re-download a file that already exists unless we
#      explicitly ask for it (saves time and is polite to the data providers).
# =============================================================================

# --- 1. Locate the project root ---------------------------------------------
# We walk up the folder tree from the current working directory until we find
# a marker file that only exists at the repo root (environment.yml).
find_project_root <- function(marker = "environment.yml") {
  dir <- normalizePath(getwd(), winslash = "/")
  repeat {
    if (file.exists(file.path(dir, marker))) return(dir)
    parent <- dirname(dir)
    if (identical(parent, dir)) {
      stop("Could not find project root (no '", marker, "' found above ",
           getwd(), "). Open R from inside the repo folder.")
    }
    dir <- parent
  }
}

ROOT          <- find_project_root()
DIR_RAW       <- file.path(ROOT, "data", "raw")
DIR_PROCESSED <- file.path(ROOT, "data", "processed")
DIR_FIGURES   <- file.path(ROOT, "output", "figures")
DIR_TABLES    <- file.path(ROOT, "output", "tables")

# Create the folders if they are missing (e.g. on a fresh clone, because
# data/raw is git-ignored).
for (d in c(DIR_RAW, DIR_PROCESSED, DIR_FIGURES, DIR_TABLES)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# --- 1b. Load the project's .Renviron explicitly ------------------------------
# R normally reads .Renviron at start-up, but only from the folder R was
# launched in (and conda/Jupyter setups don't always honour it). Loading it
# here makes the FRED key available no matter how the script is started.
renviron_path <- file.path(ROOT, ".Renviron")
if (file.exists(renviron_path)) {
  readRenviron(renviron_path)
} else if (file.exists(paste0(renviron_path, ".txt"))) {
  # Classic Windows/Notepad trap: the file was saved as ".Renviron.txt".
  warning("Found '.Renviron.txt' — rename it to '.Renviron' (no .txt).")
  readRenviron(paste0(renviron_path, ".txt"))
}

# --- 2. Global switches -------------------------------------------------------
# Set FORCE_DOWNLOAD=TRUE in your .Renviron (or Sys.setenv before sourcing)
# to ignore the cache and pull fresh copies of every dataset.
FORCE_DOWNLOAD <- isTRUE(as.logical(Sys.getenv("FORCE_DOWNLOAD", "FALSE")))

# Sample window used across the project. GDP data from the World Bank start
# in 1960/61; ERA5 temperatures start in 1940.
YEAR_START <- 1960
YEAR_END   <- as.integer(format(Sys.Date(), "%Y")) - 1   # last full year

# --- 3. Small helpers ---------------------------------------------------------

# Pretty console logging so the run_all.R output is easy to follow.
log_step <- function(...) {
  cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = "")
}

# Returns TRUE if we need to (re)download `path`.
needs_download <- function(path) {
  FORCE_DOWNLOAD || !file.exists(path)
}

# Write a CSV and report its size, so every script ends with a visible receipt.
save_csv <- function(df, path) {
  readr::write_csv(df, path)
  log_step("Saved ", basename(path), "  (", nrow(df), " rows x ",
           ncol(df), " cols)")
  invisible(path)
}

# Download a URL to disk with a browser-like User-Agent (OWID asks for one)
# and a generous timeout (default R timeout is 60s, too short for big files).
download_to <- function(url, dest) {
  old <- options(timeout = max(600, getOption("timeout")))
  on.exit(options(old))
  utils::download.file(
    url, dest, mode = "wb", quiet = TRUE,
    headers = c("User-Agent" = "Our World In Data data fetch/1.0")
  )
  invisible(dest)
}
