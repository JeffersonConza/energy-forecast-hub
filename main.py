import os
from src.preprocessing import load_data, preprocess_data, split_features_target
from src.training import train_models, cross_validate_models, save_model
from src.evaluation import evaluate_models
from src.visualization import plot_predictions, plot_residuals

def main():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    # 1. Load and Preprocess
    print("Loading data...")
    try:
        df_train, df_test = load_data(DATA_DIR)
    except FileNotFoundError as e:
        print(e)
        return

    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)

    X_train, y_train = split_features_target(df_train)
    X_test, y_test = split_features_target(df_test)

    # 2. Time-Series Cross Validation on Train Set
    print("\nStarting Time-Series Cross Validation (5 splits)...")
    cv_results = cross_validate_models(X_train, y_train, n_splits=5)
    
    print("\n--- Cross-Validation RMSE Results ---")
    for name, score in cv_results.items():
        print(f"   {name}: {score:.2f} kW")

    # 3. Find best model based on validation RMSE
    best_model_name = min(cv_results, key=cv_results.get)
    print(f"\nSelected Best Model: {best_model_name} (Lowest CV RMSE: {cv_results[best_model_name]:.2f} kW)")

    # 4. Retrain the selected best model on the FULL training set
    print(f"\nRetraining {best_model_name} on the entire training set...")
    full_trained_models = train_models(X_train, y_train, verbose=False)
    best_model = full_trained_models[best_model_name]

    # Save the best model as production_model.pkl.joblib
    save_model(best_model, "production_model.pkl")

    # 5. Final evaluation on the hold-out test set
    test_results, predictions = evaluate_models(full_trained_models, X_test, y_test)
    
    print(f"\nFinal Hold-Out Test set RMSE for production model ({best_model_name}): {test_results[best_model_name]:.2f} kW")

    # 6. Visualize
    print(f"Visualizing results for {best_model_name}...")
    plot_predictions(y_test, predictions, best_model_name)
    plot_residuals(y_test, predictions, best_model_name)

if __name__ == "__main__":
    main()