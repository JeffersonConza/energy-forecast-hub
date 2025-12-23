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

# 3. Define Recipe
energy_recipe <- recipe(power_consumption ~ ., data = train_prep) %>%
  step_rm(date) %>%
  step_dummy(day_in_week, one_hot = TRUE)

# 4. Define Models (Matching Python Params)

# A. Linear Regression
lm_spec <- linear_reg() %>% set_engine("lm")

# B. Random Forest (1000 trees, seed 42)
rf_spec <- rand_forest(trees = 1000, mode = "regression") %>%
  set_engine("ranger", seed = 42)

# C. XGBoost (1000 trees, depth 6, seed 42)
xgb_spec <- boost_tree(trees = 1000, tree_depth = 6, mode = "regression") %>%
  set_engine("xgboost", seed = 42)

# 5. Training Helper
train_and_eval <- function(spec, name) {
  print(paste("   Training", name, "..."))
  
  wf <- workflow() %>%
    add_recipe(energy_recipe) %>%
    add_model(spec)
  
  set.seed(42)
  fit <- wf %>% fit(data = train_prep)
  
  preds <- predict(fit, new_data = test_prep) %>%
    bind_cols(test_prep %>% select(power_consumption))
  
  rmse_val <- rmse(preds, truth = power_consumption, estimate = .pred)$.estimate
  return(list(model = fit, rmse = rmse_val, name = name))
}

# 6. Run Training
print("Training Models (This may take a minute)...")
res_lm  <- train_and_eval(lm_spec, "Linear Regression")
res_rf  <- train_and_eval(rf_spec, "Random Forest")
res_xgb <- train_and_eval(xgb_spec, "XGBoost")

# 7. Compare & Save Winner
comparison <- bind_rows(
  tibble(Model = res_lm$name, RMSE = res_lm$rmse),
  tibble(Model = res_rf$name, RMSE = res_rf$rmse),
  tibble(Model = res_xgb$name, RMSE = res_xgb$rmse)
)

print("\n📊 Model Results:")
print(comparison)

best <- list(res_lm, res_rf, res_xgb)[[which.min(comparison$RMSE)]]
cat(sprintf("\n🏆 Winner: %s (RMSE: %.2f kW)\n\n", best$name, best$rmse))


if(!dir.exists("R_project/models")) dir.create("R_project/models", recursive = TRUE)
saveRDS(best$model, "R_project/models/production_model_r.rds")