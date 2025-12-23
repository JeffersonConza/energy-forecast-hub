# R_project/api.R
library(plumber)
library(tidyverse)
library(tidymodels)
library(ranger)
library(xgboost)

source("R/processing.R")

# Load the saved model (The winner from step 4)
model <- readRDS("models/production_model_r.rds")

#* @apiTitle Energy Forecast API (R Version)
#* @apiDescription Serves the best performing model (LR, RF, or XGB)

#* Health Check
#* @get /
function() {
  list(status = "online", language = "R", message = "Ready to predict ⚡")
}

#* Predict Power Consumption
#* @param date:string Date in YYYY-MM-DD format
#* @post /predict
function(date) {
  # Create single-row dataframe
  input_data <- tibble(date = date)
  
  # Feature Engineering
  processed_data <- create_features(input_data)
  
  # Predict
  prediction <- predict(model, new_data = processed_data)
  
  list(
    date = date,
    predicted_consumption_kw = round(prediction$.pred, 2),
    model_used = "Best R Model"
  )
}