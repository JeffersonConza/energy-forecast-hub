# src/api.jl
using Oxygen
using HTTP
using JSON3
using DataFrames
using JLD2
using Dates
using DecisionTree

# Include the Julia preprocessor
include("preprocessing.jl")
using .Preprocessing

# Model Struct Definitions for deserialization
struct SimpleLinearRegression
    beta::Vector{Float64}
end

struct SimpleRandomForest
    forest::DecisionTree.Ensemble{Float64, Float64}
end

# Predict Helpers
function predict_lr(model::SimpleLinearRegression, X::Matrix{Float64})
    X_int = hcat(ones(size(X, 1)), X)
    return X_int * model.beta
end

function predict_rf(model::SimpleRandomForest, X::Matrix{Float64})
    return apply_forest(model.forest, X)
end

# Feature formatting
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

# Load model metadata and weights
const src_dir = @__DIR__
const root_dir = dirname(src_dir)
const model_path = joinpath(root_dir, "models", "production_model_julia.jld2")

if !isfile(model_path)
    error("Model file not found at $model_path. Run Julia training script first.")
end

println("Loading Julia production model from: ", model_path)
@load model_path final_model best_model_name
println("Loaded model: ", best_model_name)

# Custom CORS middleware
function CorsHandler(handler)
    return function(req::HTTP.Request)
        if req.method == "OPTIONS"
            return HTTP.Response(200, [
                "Access-Control-Allow-Origin" => "*",
                "Access-Control-Allow-Headers" => "Content-Type, Authorization",
                "Access-Control-Allow-Methods" => "GET, POST, OPTIONS"
            ], "")
        end
        res = handler(req)
        HTTP.setheader(res, "Access-Control-Allow-Origin" => "*")
        HTTP.setheader(res, "Access-Control-Allow-Headers" => "Content-Type, Authorization")
        HTTP.setheader(res, "Access-Control-Allow-Methods" => "GET, POST, OPTIONS")
        return res
    end
end

# Logger middleware
function LogHandler(handler)
    return function(req::HTTP.Request)
        println(Dates.now(), " - ", req.method, " - ", req.target)
        return handler(req)
    end
end

# Routes
@get "/" function()
    return Dict("status" => "online", "language" => "Julia")
end

@post "/predict" function(req::HTTP.Request)
    try
        data = json(req)
        if !haskey(data, "date")
            return HTTP.Response(400, "Missing 'date' key in JSON payload")
        end
        date_str = data["date"]
        
        # Preprocess input date
        df = DataFrame(date = [date_str])
        df_p = Preprocessing.preprocess_data(df)
        X = get_feature_matrix(df_p)
        
        # Perform prediction
        pred = if best_model_name == "Linear Regression"
            predict_lr(final_model, X)[1]
        elseif best_model_name == "Random Forest"
            predict_rf(final_model, X)[1]
        else
            error("Unsupported model: $best_model_name")
        end
        
        pred_val = round(pred, digits=2)
        
        return Dict(
            "date" => date_str,
            "predicted_consumption_kw" => pred_val,
            "model_used" => "Random Forest (Production)"
        )
    catch e
        println("Error during prediction: ", e)
        return HTTP.Response(500, "Internal Server Error: $(e)")
    end
end

# Start the Oxygen server
serve(host="0.0.0.0", port=8002, middleware=[CorsHandler, LogHandler])
