library(tidyverse)
library(tidymodels)
library(ranger)
library(xgboost)

source("R_project/R/processing.R")

# 1. Load Data
print("Loading Data...")
df_train <- read_csv("data/df_train.csv", show_col_types = FALSE)
df_test <- read_csv("data/df_test.csv", show_col_types = FALSE)

# 2. Preprocessing
print("Feature Engineering...")
train_prep <- create_features(df_train)
test_prep <- create_features(df_test)

# 3. Define Recipes
energy_recipe <- recipe(power_consumption ~ ., data = train_prep) %>%
  step_rm(date) %>%
  step_dummy(day_in_week, one_hot = TRUE)

# 4. Define Models (Matching Python Params)
lm_spec <- linear_reg() %>% set_engine("lm")

rf_spec <- rand_forest(trees = 1000, mode = "regression") %>%
  set_engine("ranger", seed = 42)

xgb_spec <- boost_tree(trees = 1000, tree_depth = 6, mode = "regression") %>%
  set_engine("xgboost", seed = 42)

# 5. Time-Series Cross Validation (5 splits)
print("\nStarting Time-Series Cross Validation (5 splits)...")
n_splits <- 5
N <- nrow(train_prep)
val_size <- floor(N / (n_splits + 1)) # 1202 / 6 = 200

cv_scores <- list(
  "Linear Regression" = numeric(n_splits),
  "Random Forest" = numeric(n_splits),
  "XGBoost" = numeric(n_splits)
)

train_and_eval_fold <- function(spec, recipe, train_fold, val_fold) {
  wf <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(spec)
  
  fit_wf <- wf %>% fit(data = train_fold)
  preds <- predict(fit_wf, new_data = val_fold) %>%
    bind_cols(val_fold %>% select(power_consumption))
  
  rmse_val <- rmse(preds, truth = power_consumption, estimate = .pred)$.estimate
  return(rmse_val)
}

for (fold in 1:n_splits) {
  train_end <- N - (n_splits - fold + 1) * val_size
  val_start <- train_end + 1
  val_end <- val_start + val_size - 1
  
  fold_train <- train_prep[1:train_end, ]
  fold_val <- train_prep[val_start:val_end, ]
  
  cv_scores[["Linear Regression"]][fold] <- train_and_eval_fold(lm_spec, energy_recipe, fold_train, fold_val)
  cv_scores[["Random Forest"]][fold] <- train_and_eval_fold(rf_spec, energy_recipe, fold_train, fold_val)
  cv_scores[["XGBoost"]][fold] <- train_and_eval_fold(xgb_spec, energy_recipe, fold_train, fold_val)
}

mean_cv_rmse <- tibble(
  Model = c("Linear Regression", "Random Forest", "XGBoost"),
  RMSE = c(
    mean(cv_scores[["Linear Regression"]]),
    mean(cv_scores[["Random Forest"]]),
    mean(cv_scores[["XGBoost"]])
  )
)

print("\n📊 Cross-Validation RMSE Results:")
print(mean_cv_rmse)

best_model_name <- mean_cv_rmse$Model[which.min(mean_cv_rmse$RMSE)]
best_cv_rmse <- min(mean_cv_rmse$RMSE)
cat(sprintf("\n🏆 Selected Best Model: %s (Lowest CV RMSE: %.2f kW)\n", best_model_name, best_cv_rmse))

# 6. Retrain best model on the FULL training set
print(sprintf("\nRetraining %s on the entire training set...", best_model_name))
best_spec <- if (best_model_name == "Linear Regression") {
  lm_spec
} else if (best_model_name == "Random Forest") {
  rf_spec
} else {
  xgb_spec
}

final_wf <- workflow() %>%
  add_recipe(energy_recipe) %>%
  add_model(best_spec)

final_fit <- final_wf %>% fit(data = train_prep)

# 7. Evaluate on Hold-Out Test Set
test_preds <- predict(final_fit, new_data = test_prep) %>%
  bind_cols(test_prep %>% select(power_consumption))

test_rmse <- rmse(test_preds, truth = power_consumption, estimate = .pred)$.estimate
cat(sprintf("\nFinal Hold-Out Test set RMSE for production model (%s): %.2f kW\n", best_model_name, test_rmse))

# Save the trained winner model
if(!dir.exists("R_project/models")) dir.create("R_project/models", recursive = TRUE)
saveRDS(final_fit, "R_project/models/production_model_r.rds")
print("Model saved to R_project/models/production_model_r.rds")