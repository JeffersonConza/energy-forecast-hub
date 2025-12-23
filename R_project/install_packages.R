# R_project/install_packages.R

# 1. Detect User Library Path (Usually Documents/R/win-library/...)
user_lib <- Sys.getenv("R_LIBS_USER")

# If the folder doesn't exist, create it
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE)
  print(paste("📁 Created personal library at:", user_lib))
}

# Tell R to use this library for this session
.libPaths(c(user_lib, .libPaths()))

print(paste("⬇️ Installing packages to:", user_lib))

# 2. Set CRAN Mirror (Required for non-interactive scripts)
r <- getOption("repos")
r["CRAN"] <- "https://cloud.r-project.org/"
options(repos = r)

# 3. List of required packages
required_pkgs <- c(
  "devtools",
  "tidyverse",
  "tidymodels",
  "ranger",
  "xgboost",
  "plumber",
  "shiny",
  "plotly",
  "bslib",
  "jsonlite",
  "httr"
)

# 4. Install only what is missing
installed_pkgs <- installed.packages()[,"Package"]
missing_pkgs <- required_pkgs[!(required_pkgs %in% installed_pkgs)]

if (length(missing_pkgs) > 0) {
  print(paste("📦 Installing:", paste(missing_pkgs, collapse = ", ")))
  install.packages(missing_pkgs, lib = user_lib)
} else {
  print("✅ All packages are already installed!")
}