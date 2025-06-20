# Energy Consumption Forecasting Project

## 📌 Overview
This project focuses on forecasting daily household energy consumption using machine learning models implemented in both **R** and **Python**. Leveraging real-world data spanning four years, the goal is to predict future energy needs, optimize consumption patterns, and analyze trends to support energy efficiency.

---

## 📂 Dataset
The dataset, sourced from the **UCI Machine Learning Repository**, includes the following features:
- **Date**: Measurement date (`mm/dd/YYYY` format).  
- **Power Consumption**: Daily consumption in kilowatts (target variable).  
- **Temporal Features**: Year, semester, quarter, day of the week, week number, day number, and month.  

Files:  
- `df_train.csv`: Training data (1,202 entries).  
- `df_test.csv`: Test data for validation.  

---

## 🛠️ Methodology

### Data Preprocessing
- **Date Handling**: Converted to datetime format and extracted temporal features.  
- **Categorical Encoding**: One-hot encoding for `day_in_week`.  
- **Train-Test Split**: Separated features (`year`, `month`, etc.) and target (`power_consumption`).  

### Models Implemented
1. **Linear Regression**: Baseline model.  
2. **Random Forest**: 1,000 trees for robust performance.  
3. **XGBoost**: Optimized with 500 rounds, learning rate 0.1, and categorical support.  

### Evaluation Metric
- **Root Mean Squared Error (RMSE)** to compare model accuracy.  

---

## 📊 Results

### Performance Comparison (RMSE)
| Model               | Python RMSE (kW) | R RMSE (kW) |
|---------------------|------------------|-------------|
| Linear Regression   | 504.30           | 504.30      |
| Random Forest       | 431.94           | **392.74**  |
| XGBoost             | **405.29**       | 403.55      |

**Best Model**:  
- **Python**: XGBoost (RMSE: 405.29).  
- **R**: Random Forest (RMSE: 392.74).  

### Key Insights
- **Trend Similarity**: Moderate correlation (~0.53–0.60) between actual and predicted values.  
- **Error Analysis**: Residual plots highlight consistent under/over-prediction patterns.  

---

## 📈 Visualizations
- **Actual vs. Predicted**: Line plots comparing model forecasts.  
- **Residual Analysis**: Scatter plots with error magnitude gradients.  

![Example Plot](https://via.placeholder.com/600x400?text=Actual+vs+Predicted+Plot)  
*(Replace with actual plot from the project)*  

---

## 💻 Code Implementation
- **Python**: Uses `pandas`, `scikit-learn`, and `xgboost`.  
- **R**: Leverages `dplyr`, `ranger`, and `ggplot2`.  

### Dependencies
```bash
# Python
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# R
install.packages(c("dplyr", "lubridate", "ranger", "xgboost", "ggplot2"))
