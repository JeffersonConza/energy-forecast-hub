using CSV
using DataFrames
using Dates
using Random
using DecisionTree
using XGBoost
using JLD2
using LinearAlgebra

# Include the Julia preprocessor
include("preprocessing.jl")
using .Preprocessing

# Model Struct Definitions for serialization
struct SimpleLinearRegression
    beta::Vector{Float64}
end

struct SimpleRandomForest
    forest::DecisionTree.Ensemble{Float64, Float64}
end

# Fit/Predict Helpers

# 1. Linear Regression (OLS with tiny L2 regularization for stability)
function fit_lr(X::Matrix{Float64}, y::Vector{Float64})
    X_int = hcat(ones(size(X, 1)), X)
    M = X_int' * X_int
    beta = (M + 1e-8 * I) \ (X_int' * y)
    return SimpleLinearRegression(beta)
end

function predict_lr(model::SimpleLinearRegression, X::Matrix{Float64})
    X_int = hcat(ones(size(X, 1)), X)
    return X_int * model.beta
end

# 2. Random Forest
function fit_rf(X::Matrix{Float64}, y::Vector{Float64})
    forest = build_forest(y, X, -1, 1000, 0.7, -1)
    return SimpleRandomForest(forest)
end

function predict_rf(model::SimpleRandomForest, X::Matrix{Float64})
    return apply_forest(model.forest, X)
end

# 3. XGBoost
function fit_xgb(X::Matrix{Float64}, y::Vector{Float64})
    dtrain = DMatrix(X, label=y)
    booster = Booster(dtrain)
    XGBoost.setparam!(booster, "max_depth", 6)
    XGBoost.setparam!(booster, "objective", "reg:squarederror")
    XGBoost.setparam!(booster, "seed", 42)
    
    # Custom loop to bypass the package bug
    for i in 1:1000
        XGBoost.updateone!(booster, dtrain; round_number=i)
    end
    return booster
end

function predict_xgb(model::Booster, X::Matrix{Float64})
    return XGBoost.predict(model, X)
end

# One-hot encoding categories alphabetically to match Python & R
function get_feature_matrix(df::DataFrame)
    days = ["Friday", "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"]
    n = nrow(df)
    dummies = Matrix{Float64}(undef, n, length(days))
    for j in 1:length(days)
        dummies[:, j] = [d == days[j] ? 1.0 : 0.0 for d in df.day_in_week]
    end

    num_features = hcat(
        Float64.(df.year),
        Float64.(df.semester),
        Float64.(df.quarter),
        Float64.(df.week_in_year),
        Float64.(df.day_in_year),
        Float64.(df.month),
        Float64.(df.power_rolling_mean_7d)
    )

    return hcat(dummies, num_features)
end

# Generic rmse to handle Float32 (XGBoost) and Float64 (LR/RF)
function rmse(p::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    return sqrt(sum((p .- y).^2) / length(y))
end

function main()
    src_dir = @__DIR__
    root_dir = dirname(src_dir)
    train_path = joinpath(root_dir, "data", "df_train.csv")
    test_path = joinpath(root_dir, "data", "df_test.csv")

    println("Loading data...")
    df_train = CSV.read(train_path, DataFrame)
    df_test = CSV.read(test_path, DataFrame)

    println("Preprocessing data...")
    df_train_p = Preprocessing.preprocess_data(df_train)
    df_test_p = Preprocessing.preprocess_data(df_test)

    # Build exact feature matrices
    X_train = get_feature_matrix(df_train_p)
    y_train = Float64.(df_train_p.power_consumption)
    X_test = get_feature_matrix(df_test_p)
    y_test = Float64.(df_test_p.power_consumption)

    # Time-Series Cross Validation (5 splits)
    println("\nStarting Time-Series Cross Validation (5 splits)...")
    n_splits = 5
    N = size(X_train, 1)
    val_size = floor(Int, N / (n_splits + 1)) # 1202 / 6 = 200

    cv_scores = Dict(
        "Linear Regression" => Float64[],
        "Random Forest" => Float64[],
        "XGBoost" => Float64[]
    )

    for fold in 1:n_splits
        train_end = N - (n_splits - fold + 1) * val_size
        val_start = train_end + 1
        val_end = val_start + val_size - 1
        
        X_tr = X_train[1:train_end, :]
        y_tr = y_train[1:train_end]
        X_val = X_train[val_start:val_end, :]
        y_val = y_train[val_start:val_end]
        
        # Linear Regression
        model_lr = fit_lr(X_tr, y_tr)
        p_lr = predict_lr(model_lr, X_val)
        push!(cv_scores["Linear Regression"], rmse(p_lr, y_val))
        
        # Random Forest (with fixed seed)
        Random.seed!(42)
        model_rf = fit_rf(X_tr, y_tr)
        p_rf = predict_rf(model_rf, X_val)
        push!(cv_scores["Random Forest"], rmse(p_rf, y_val))
        
        # XGBoost (with fixed seed)
        Random.seed!(42)
        model_xgb = fit_xgb(X_tr, y_tr)
        p_xgb = predict_xgb(model_xgb, X_val)
        push!(cv_scores["XGBoost"], rmse(p_xgb, y_val))
    end

    mean_cv_rmse = Dict(
        "Linear Regression" => sum(cv_scores["Linear Regression"]) / n_splits,
        "Random Forest" => sum(cv_scores["Random Forest"]) / n_splits,
        "XGBoost" => sum(cv_scores["XGBoost"]) / n_splits
    )

    println("\n📊 Cross-Validation RMSE Results:")
    for (name, val) in mean_cv_rmse
        println("   $name: ", round(val, digits=2), " kW")
    end

    best_model_name = "Linear Regression"
    min_val = mean_cv_rmse[best_model_name]
    for (name, val) in mean_cv_rmse
        if val < min_val
            best_model_name = name
            min_val = val
        end
    end

    println("\n🏆 Selected Best Model: $best_model_name (Lowest CV RMSE: ", round(min_val, digits=2), " kW)")

    # Retrain on the FULL training set
    println("\nRetraining $best_model_name on the entire training set...")
    final_model = if best_model_name == "Linear Regression"
        fit_lr(X_train, y_train)
    elseif best_model_name == "Random Forest"
        Random.seed!(42)
        fit_rf(X_train, y_train)
    else
        Random.seed!(42)
        fit_xgb(X_train, y_train)
    end

    # Save the trained model
    models_dir = joinpath(root_dir, "models")
    if !isdir(models_dir)
        mkdir(models_dir)
    end
    
    if best_model_name == "XGBoost"
        xgb_path = joinpath(models_dir, "production_model_julia.xgb")
        XGBoost.save(final_model, xgb_path)
        println("XGBoost model saved to $xgb_path")
    end
    
    model_path = joinpath(models_dir, "production_model_julia.jld2")
    # Save the final model (for LR/RF) and best_model_name to JLD2
    @save model_path final_model best_model_name
    println("Model metadata saved to $model_path")

    # Evaluate on Hold-Out Test Set
    p_final = if best_model_name == "Linear Regression"
        predict_lr(final_model, X_test)
    elseif best_model_name == "Random Forest"
        predict_rf(final_model, X_test)
    else
        predict_xgb(final_model, X_test)
    end

    test_rmse = rmse(p_final, y_test)
    println("\nFinal Hold-Out Test set RMSE for production model ($best_model_name): ", round(test_rmse, digits=2), " kW")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
