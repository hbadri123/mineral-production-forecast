### Forecasting Monthly Mineral Production in South Africa Using Multivariate Time-Series Models (1990–2002)

#### 1. Introduction and Background

The mining sector has historically been a cornerstone of South Africa’s economy, contributing significantly to exports, employment, and industrial activity. The period from 1990 to 2002 is of particular interest due to structural economic changes, commodity price volatility, and evolving global demand conditions. Forecasting mineral production during this period provides valuable insights into production dynamics and allows for rigorous comparison of econometric and machine learning approaches under data-constrained conditions.

#### 2. Problem Statement and Objectives

**Primary Objective**: This project aims to demonstrate whether machine learning models significantly outperform or not baseline forecasting methods for mineral production forecasting in South Africa.

The objective is to forecast monthly mineral production in South Africa over 3-, 6-, and 12-month horizons for gold, coal, iron ore, and copper. The analysis is restricted to the period 1990–2002 due to data availability constraints. Each mineral is modeled independently to capture commodity-specific production behavior. **The key research question is whether modern ML approaches (Random Forest, XGBoost, LightGBM, CatBoost) can consistently outperform simple baseline methods (naive and historical mean forecasts) across different minerals and forecast horizons.**

For each mineral ( m ), the target variable is defined as:

[
y_t^m = \text{Monthly production level or production index at time } t
]

#### 3. Data Collection and Preprocessing

Monthly production data are sourced from Statistics South Africa. Exogenous regressors include international commodity prices, CPI inflation, industrial production, and the ZAR/USD exchange rate. All series are aggregated to monthly frequency and aligned by calendar month. Missing values are handled using linear interpolation for short gaps and forward filling for slow-moving macroeconomic indicators.

#### 4. Feature Engineering

The feature set includes lagged production values (1, 3, 6, and 12 months), lagged commodity prices (1–6 months), lagged macroeconomic variables (1–3 months), rolling window statistics (moving averages, rolling standard deviations, and growth rates), and seasonal indicators based on month-of-year effects.

#### 5. Forecasting Strategy

Forecasts are generated using a direct multi-horizon approach for horizons of 3, 6, and 12 months. Model performance is evaluated using a rolling-origin backtesting framework that simulates real-time forecasting conditions.

#### 6. Models

Baseline models include naive and historical mean forecasts. Classical time-series models include ARIMA, SARIMA, and SARIMAX. Machine learning models include Random Forest, XGBoost, LightGBM, and CatBoost regressors.

#### 7. Evaluation

Forecast accuracy is assessed using RMSE, MAE, MAPE (or sMAPE), and directional accuracy. **The primary evaluation metric is the skill score relative to the naive baseline**, which quantifies the improvement of ML models over baseline methods. A positive skill score indicates that a model outperforms the naive baseline. Cross-mineral comparisons are conducted using normalized RMSE and skill scores to systematically demonstrate ML superiority.

#### 8. Outputs and Reproducibility

Outputs include forecast plots, model comparison tables, and feature importance visualizations. All code is version-controlled, random seeds are fixed, and preprocessing and evaluation steps are fully documented.
