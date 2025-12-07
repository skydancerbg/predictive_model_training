# Soil Moisture Forecasting with XGBoost

This repository contains reproducible materials for short‑horizon soil moisture forecasting using environmental time series data and gradient‑boosted decision trees (XGBoost).

---

## 📊 Data Sources and Preprocessing
- **Period:** 1 September 2024 – 30 September 2025  
- **Variables:**  
  - Air temperature (°C)  
  - Wind speed (m/s)  
  - Soil temperature (°C)  
  - Volumetric soil moisture (%)  
- **Database:** Stored in MariaDB, extracted via SQLAlchemy.  
- **Preprocessing:**  
  - Reconstructed timestamps from DATE + TIME fields  
  - Strictly monotonic temporal index  
  - No missing values or duplicates  
  - Values within sensor‑appropriate ranges  

---

## 🪟 Sliding‑Window Supervised Learning
- **Forecast horizon:** 24 hours  
- **Target:** Soil moisture at `y(t+24h)`  
- **Predictors:** 24 hourly lag values `y(t) … y(t−23)`  
- **Samples:**  
  - Total: **9,433**  
  - Training set: **7,546** (80%)  
  - Test set: **1,887** (20%)  
- **Split:** Chronological (avoids look‑ahead bias)  

---

## ⚙️ Feature Engineering
- 24 soil moisture lags  
- Meteorological variables: air temperature, wind speed, soil temperature  
- Cyclical encodings:  
  - Hour‑of‑day → `sin(2πh/24)`, `cos(2πh/24)`  
  - Day‑of‑year → `sin(2πd/365)`, `cos(2πd/365)`  
- Captures diurnal and seasonal evapotranspiration patterns  

---

## 🌲 Model Architecture (XGBoost)
- Framework: **XGBoost** (reg:squarederror, hist tree method)  
- Ensemble: **500 trees**, max depth = 3  
- Regularization: L1 + L2  
- Shrinkage: learning rate  
- Subsampling: tree‑level  

---

## 🔧 Hyperparameter Optimization
- **Method:** RandomizedSearchCV (30 candidates, 3 TimeSeriesSplit folds)  
- **Search space:**  
  - `learning_rate ∈ [0.01, 0.20]`  
  - `n_estimators ∈ [200, 800]`  
  - `max_depth ∈ {2,3,4,5,6}`  
  - `subsample ∈ [0.6, 1.0]`  
  - `colsample_bytree ∈ [0.6, 1.0]`  
  - `gamma ∈ [0, 2]`  
  - `reg_alpha ∈ [0, 10]`  
  - `reg_lambda ∈ [0, 10]`  
- **Optimal configuration:**  
  - `learning_rate = 0.07`  
  - `n_estimators = 500`  
  - `max_depth = 3`  
  - `subsample = 1.0`  
  - `colsample_bytree = 0.90`  
  - `gamma = 0.0`  
  - `reg_alpha = 2.0`  
  - `reg_lambda = 10.0`  

---

## 📈 Evaluation Metrics
- **RMSE:** 0.825  
- **MAE:** 0.249  
- **R²:** 0.639  
- **Cross‑validated RMSE:** 2.97  
- Results indicate reliable short‑horizon forecasting with sub‑1% average error magnitude.  

---

## 🔍 Feature Importance Analysis
Top predictors (gain‑based importance):  
| Feature   | Importance |
|-----------|------------|
| sm_lag_0  | 0.2687     |
| doy_sin   | 0.2371     |
| sm_lag_1  | 0.1085     |
| sm_lag_10 | 0.0492     |
| sm_lag_7  | 0.0446     |

- Soil moisture lags dominate predictive power.  
- Seasonal encodings (day‑of‑year) strongly influence dynamics.  
- Meteorological forcings contribute less due to short horizon and synthetic dataset smoothness.  

---

## 🔄 Reproducibility
- **Language:** Python 3.12  
- **Libraries:** XGBoost 2.x, scikit‑learn 1.5, pandas, numpy, SQLAlchemy, joblib  
- **Logging:**  
  - Training timestamps  
  - Feature lists  
  - Hyperparameters  
  - Evaluation outputs → `metricsandfeatures.txt`  
- **Code:** Full source and configs available in the associated GitHub repository.  

---

## 📂 Repository Structure
