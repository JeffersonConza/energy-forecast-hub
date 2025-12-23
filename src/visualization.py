import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_test, predictions, best_model_name):
    """Plots a subset of Actual vs Predicted values for clarity."""
    y_pred = predictions[best_model_name]
    
    plt.figure(figsize=(12, 6))
    # We plot the first 200 points to make the graph readable
    plt.plot(y_test.values[:200], label='Actual Consumption', color='blue', alpha=0.6)
    plt.plot(y_pred[:200], label=f'Predicted ({best_model_name})', color='orange', alpha=0.8, linestyle='--')
    
    plt.title(f'Energy Forecast: Actual vs {best_model_name} (First 200 Days)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Power Consumption (kW)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, predictions, best_model_name):
    """Plots residuals to check for bias."""
    y_pred = predictions[best_model_name]
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residual Plot ({best_model_name})')
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.show()