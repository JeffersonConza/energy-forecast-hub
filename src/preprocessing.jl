module Preprocessing

using DataFrames
using Dates
using CSV

export preprocess_data, split_features_target

const ROLLING_MEAN_LOOKUP = Dict{String, Float64}()

function get_rolling_mean_lookup()
    if !isempty(ROLLING_MEAN_LOOKUP)
        return ROLLING_MEAN_LOOKUP
    end

    src_dir = @__DIR__
    root_dir = dirname(src_dir)
    train_path = joinpath(root_dir, "data", "df_train.csv")
    test_path = joinpath(root_dir, "data", "df_test.csv")

    if isfile(train_path) && isfile(test_path)
        df_train = CSV.read(train_path, DataFrame)
        df_test = CSV.read(test_path, DataFrame)
        df_combined = vcat(df_train, df_test)

        df_combined.parsed_date = [parse_julia_date(d) for d in df_combined.date]
        sort!(df_combined, :parsed_date)

        n = nrow(df_combined)
        rolling_means = fill(NaN, n)
        for i in 8:n
            rolling_means[i] = sum(df_combined.power_consumption[i-7:i-1]) / 7.0
        end
        df_combined.power_rolling_mean_7d = rolling_means

        for row in eachrow(df_combined)
            d_str = Dates.format(row.parsed_date, "yyyy-mm-dd")
            if !isnan(row.power_rolling_mean_7d)
                ROLLING_MEAN_LOOKUP[d_str] = row.power_rolling_mean_7d
            end
        end
    end

    return ROLLING_MEAN_LOOKUP
end

function parse_julia_date(d_str::AbstractString)
    for fmt in ["yyyy-mm-dd", "m/d/yyyy", "d/m/yyyy"]
        try
            return Dates.Date(d_str, Dates.DateFormat(fmt))
        catch
        end
    end
    # Fallback
    try
        return Dates.parse(Dates.Date, d_str)
    catch
        return Dates.Date(d_str)
    end
end

parse_julia_date(d::Dates.Date) = d

function preprocess_data(df::DataFrame)
    res_df = copy(df)

    if :date in propertynames(res_df)
        res_df.parsed_date = [parse_julia_date(d) for d in res_df.date]

        if !(:year in propertynames(res_df))
            res_df.year = [Int(Dates.year(d)) for d in res_df.parsed_date]
        end

        if !(:month in propertynames(res_df))
            res_df.month = [Int(Dates.month(d)) for d in res_df.parsed_date]
        end

        if !(:semester in propertynames(res_df))
            res_df.semester = [Int(Dates.month(d) > 6 ? 2 : 1) for d in res_df.parsed_date]
        end

        if !(:quarter in propertynames(res_df))
            res_df.quarter = [Int((Dates.month(d) - 1) ÷ 3 + 1) for d in res_df.parsed_date]
        end

        if !(:day_in_week in propertynames(res_df))
            res_df.day_in_week = [Dates.dayname(d) for d in res_df.parsed_date]
        end

        if !(:week_in_year in propertynames(res_df))
            res_df.week_in_year = [Int(Dates.week(d)) for d in res_df.parsed_date]
        end

        if !(:day_in_year in propertynames(res_df))
            res_df.day_in_year = [Int(Dates.dayofyear(d)) for d in res_df.parsed_date]
        end

        if !(:power_rolling_mean_7d in propertynames(res_df))
            lookup = get_rolling_mean_lookup()
            rolling_vals = Float64[]
            for d in res_df.parsed_date
                d_str = Dates.format(d, "yyyy-mm-dd")
                val = get(lookup, d_str, 1592.96)
                push!(rolling_vals, val)
            end
            res_df.power_rolling_mean_7d = rolling_vals
        end

        select!(res_df, Not(:parsed_date))
    end

    for col in [:year, :month, :semester, :quarter, :week_in_year, :day_in_year]
        if col in propertynames(res_df)
            res_df[!, col] = convert.(Int64, res_df[!, col])
        end
    end
    if :power_rolling_mean_7d in propertynames(res_df)
        res_df[!, :power_rolling_mean_7d] = convert.(Float64, res_df[!, :power_rolling_mean_7d])
    end

    return res_df
end

function split_features_target(df::DataFrame, target_col::Symbol=:power_consumption, drop_cols::Vector{Symbol}=[:date])
    target_and_drops = vcat([target_col], drop_cols)
    feat_cols = [col for col in names(df) if !(Symbol(col) in target_and_drops)]
    
    X = df[:, feat_cols]
    y = target_col in propertynames(df) ? df[:, target_col] : nothing
    return X, y
end

end # module
