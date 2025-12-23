import os
from src.preprocessing import load_data, preprocess_data, split_features_target
from src.training import train_models
from src.evaluation import evaluate_models
from src.visualization import plot_predictions, plot_residuals

# Model saving functionality
from src.training import train_models, save_model

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

    # 2. Train
    print("\nStarting training...")
    models = train_models(X_train, y_train)

    # 3. Evaluate
    results, predictions = evaluate_models(models, X_test, y_test)

    # 4. Find best model
    best_model_name = min(results, key=results.get)
    print(f"\nBest Model: {best_model_name}")

    # Save the best model to a static filename
    best_model = models[best_model_name]
    # We use a fixed name so predict.py always knows what to load
    save_model(best_model, "production_model.pkl")

    # 5. Visualize
    print(f"Visualizing results for {best_model_name}...")
    plot_predictions(y_test, predictions, best_model_name)
    plot_residuals(y_test, predictions, best_model_name)

if __name__ == "__main__":
    main()