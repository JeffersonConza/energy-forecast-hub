import subprocess
import json
import pandas as pd
import sys
from src.preprocessing import preprocess_data

def run_python_preprocessing(dates):
    df = pd.DataFrame({'date': dates})
    df_p = preprocess_data(df)
    return df_p.to_dict(orient='records')

def run_r_preprocessing(dates):
    dates_str = ', '.join([f"'{d}'" for d in dates])
    r_code = f"source('R_project/R/processing.R'); df <- tibble::tibble(date = c({dates_str})); df_p <- create_features(df); cat(jsonlite::toJSON(df_p, auto_unbox=TRUE))"
    cmd = ["Rscript", "-e", r_code]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

def run_julia_preprocessing(dates):
    dates_str = ', '.join([f'"{d}"' for d in dates])
    julia_code = f'include("src/preprocessing.jl"); using DataFrames, JSON3; df = DataFrame(date = [{dates_str}]); df_p = Preprocessing.preprocess_data(df); dicts = [Dict(String(k) => r[k] for k in keys(r)) for r in eachrow(df_p)]; print(JSON3.write(dicts))'
    cmd = ["julia", "--project=.", "-e", julia_code]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

def main():
    test_dates = ["2006-12-25", "2010-04-05", "2026-06-23"]
    print("Running Python preprocessing...")
    try:
        py_res = run_python_preprocessing(test_dates)
    except Exception as e:
        print(f"Python failed: {e}")
        sys.exit(1)
    
    print("Running R preprocessing...")
    try:
        r_res = run_r_preprocessing(test_dates)
    except Exception as e:
        print(f"R failed: {e}")
        sys.exit(1)
    
    print("Running Julia preprocessing...")
    try:
        ju_res = run_julia_preprocessing(test_dates)
    except Exception as e:
        print(f"Julia failed: {e}")
        sys.exit(1)
    
    features = ["year", "month", "semester", "quarter", "day_in_week", "week_in_year", "day_in_year", "power_rolling_mean_7d"]
    
    all_ok = True
    print("\n--- Preprocessing Feature Alignment Validation ---")
    for i, date in enumerate(test_dates):
        print(f"\nDate: {date}")
        py_row = py_res[i]
        r_row = r_res[i]
        ju_row = ju_res[i]
        
        print(f"{'Feature':<25} | {'Python':<15} | {'R':<15} | {'Julia':<15} | {'Match'}")
        print("-" * 80)
        for feat in features:
            py_val = py_row.get(feat)
            r_val = r_row.get(feat)
            ju_val = ju_row.get(feat)
            
            c_py = round(py_val, 2) if isinstance(py_val, (int, float)) and feat == "power_rolling_mean_7d" else py_val
            c_r = round(r_val, 2) if isinstance(r_val, (int, float)) and feat == "power_rolling_mean_7d" else r_val
            c_ju = round(ju_val, 2) if isinstance(ju_val, (int, float)) and feat == "power_rolling_mean_7d" else ju_val
            
            match = (c_py == c_r == c_ju)
            if not match:
                all_ok = False
            match_str = "OK" if match else f"FAIL (Py: {c_py}, R: {c_r}, Ju: {c_ju})"
            print(f"{feat:<25} | {str(c_py):<15} | {str(c_r):<15} | {str(c_ju):<15} | {match_str}")
            
    if all_ok:
        print("\nSuccess! Preprocessing features match exactly across Python, R, and Julia!")
    else:
        print("\nFailure! Some features do not match across languages.")
        sys.exit(1)

if __name__ == "__main__":
    main()
