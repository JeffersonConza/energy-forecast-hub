# R_project/api.R
library(plumber)
library(tidyverse)
library(tidymodels)
library(ranger)
library(xgboost)

source("R/processing.R")

# Load the saved model
model <- readRDS("models/production_model_r.rds")

#* @apiTitle Energy Forecast API (R Version)
#* @apiDescription Serves the production Random Forest model

#* CORS filter & Preflight Handler
#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  res$setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
  res$setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization")
  
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$status <- 200
    return(list())
  }
  forward()
}

#* Logger filter
#* @filter logger
function(req) {
  cat(as.character(Sys.time()), "-", req$REQUEST_METHOD, req$PATH_INFO, "\n")
  forward()
}

#* Health Check
#* @serializer json list(auto_unbox = TRUE)
#* @get /
function() {
  list(status = "online", language = "R")
}

#* Predict Power Consumption
#* @serializer json list(auto_unbox = TRUE)
#* @post /predict
function(req, res, date = NULL) {
  # If date is not in standard query/form args, check JSON body
  if (is.null(date) && !is.null(req$body$date)) {
    date <- req$body$date
  }
  
  if (is.null(date)) {
    res$status <- 400
    return(list(error = "Missing date parameter"))
  }
  
  # Create single-row dataframe
  input_data <- tibble(date = date)
  
  # Add optional overrides from JSON body or request arguments
  opt_fields <- c("year", "month", "semester", "quarter", "day_in_week", "week_in_year", "day_in_year", "power_rolling_mean_7d")
  for (field in opt_fields) {
    val <- req$body[[field]]
    if (is.null(val)) {
      val <- req$args[[field]]
    }
    if (!is.null(val)) {
      if (length(val) > 1) val <- val[1]
      
      if (field %in% c("year", "month", "semester", "quarter", "week_in_year", "day_in_year")) {
        input_data[[field]] <- as.integer(val)
      } else if (field == "power_rolling_mean_7d") {
        input_data[[field]] <- as.numeric(val)
      } else {
        input_data[[field]] <- as.character(val)
      }
    }
  }
  
  # Feature Engineering
  processed_data <- create_features(input_data)
  
  # Predict
  prediction <- predict(model, new_data = processed_data)
  pred_val <- round(as.numeric(prediction$.pred), 2)
  
  list(
    date = date,
    predicted_consumption_kw = pred_val,
    model_used = "Random Forest (Production)"
  )
}