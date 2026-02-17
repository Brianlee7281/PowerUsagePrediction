# ⚡ Industrial Electricity Consumption Forecasting

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![PyCaret](https://img.shields.io/badge/PyCaret-AutoML-green.svg)](https://pycaret.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Optimized-blueviolet.svg)](https://optuna.org/)

## 📌 Objective
The primary objective of this project is to accurately forecast electricity consumption to ensure a stable, efficient, and cost-effective energy supply. By leveraging temporal data and meteorological features, this machine learning pipeline is designed to discover power demand algorithms that can be actively deployed in industrial settings. This project validates the practical applicability of regression models in real-world energy management.

## 📊 Data Pipeline & Preprocessing
Energy demand is highly dependent on both time-based and weather-based factors. To prepare the raw dataset for predictive modeling, a rigorous preprocessing pipeline was established:

* **Categorical Encoding:** Converted all non-numerical and categorical data types (e.g., seasons, day types) into numerical formats.
* **Feature Scaling:** Because meteorological features (e.g., temperature, humidity, wind speed) operate on vastly different scales, `StandardScaler` was applied to normalize the dataset and ensure stable gradient updates.
* **Validation Strategy:** The dataset was partitioned into training and testing sets to evaluate the model’s forecasting accuracy on unseen future data.

## 🧠 Modeling Strategy
A multi-stage approach was utilized, beginning with unsupervised learning for feature engineering, followed by AutoML benchmarking, and concluding with a highly tuned stacked ensemble.

### 1. Feature Engineering (Unsupervised Learning)
* **K-Means Clustering:** Before training regression models, K-Means clustering was applied to discover hidden groupings and operational regimes within the historical data.
* **Elbow Method:** Utilized the Elbow Method to determine the optimal number of clusters, appending the resulting cluster labels as an additional predictive feature to help the models differentiate between distinct consumption patterns.

### 2. Baseline Benchmarking
* Leveraged **PyCaret** to rapidly evaluate a broad spectrum of regression algorithms. This step allowed for a data-driven comparison of different model performances and guided the selection of our base learners.

### 3. Bayesian Hyperparameter Optimization
Based on PyCaret insights, three models were selected and subjected to 100 trials of Bayesian optimization via **Optuna**:
* **Decision Tree Regressor:** Tuned `max_depth` (to prevent overgrowing) and `min_samples_split` (to control node splitting thresholds).
* **K-Nearest Neighbors (KNN) Regressor:** Tuned `n_neighbors` and distance `weights` to map the most relevant historical data points.
* **Huber Regressor:** Tuned `alpha` (regularization penalty) and `epsilon` (threshold for outlier robustness).

### 4. Ensemble Learning (Stacking)
To maximize forecasting accuracy and robustness against anomalies, a **Stacking Regressor** was implemented:
* **Level-0 (Base Estimators):** The tuned `DecisionTreeRegressor` and `KNeighborsRegressor` act as the foundational models, capturing complex non-linear and proximity-based relationships.
* **Level-1 (Meta-Learner):** The tuned `HuberRegressor` learns to optimally combine the predictions of the base estimators, applying its built-in robustness to minimize the impact of extreme weather outliers.

<br>

> <p align="center">
>   <img width="663" alt="Stacking Architecture" src="https://github.com/user-attachments/assets/a2f7d1bf-7162-4751-abf5-685ae6c8e83c">
> </p>

## 📈 Results & Evaluation
The final stacked ensemble demonstrated strong forecasting accuracy on the hold-out validation set. The use of clustering as an additional feature, combined with the stacking approach, contributed significantly to capturing the complex patterns in power demand:

| Metric | Score |
| :--- | :--- |
| **MAPE (Mean Absolute Percentage Error)** | 0.747 |
| **MAE (Mean Absolute Error)** | 2,470.91 |
| **RMSE (Root Mean Squared Error)** | 3,625.80 |

*Note: The model successfully captures underlying temporal/meteorological patterns, prioritizing robust predictions over extreme weather fluctuations.*

## 🚀 Challenges & Future Work
* **Time-Series Dynamics:** While the current model leverages temporal features, future iterations will treat the data strictly sequentially by introducing lag features (e.g., $t-1, t-24$ consumption) and rolling window statistics to capture momentum.
* **Advanced Outlier Handling:** Industrial electricity consumption can spike due to unrecorded external factors. Implementing localized anomaly detection prior to training could further improve the signal fed to the regression models.
* **Industrial Deployment:** The pipeline is architected for real-world scaling. The next phase involves packaging the final model as an API endpoint to feed real-time energy management dashboards.
