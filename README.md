S1. Data Sources and Preprocessing
The training dataset consisted of hourly environmental measurements collected over the period 1 September 2024 to 30 September 2025. 
Four variables were included: air temperature (°C), wind speed (m/s), soil temperature (°C), and volumetric soil moisture (%). All data 
were stored in a MariaDB instance. After extraction, timestamps were reconstructed by combining the DATE and TIME fields, ensuring a strictly 
monotonic temporal index. No imputation was required due to the complete synthetic dataset. Duplicate timestamps were verified to be absent, 
and values remained within sensor-appropriate physical ranges.
S2. Sliding-Window Supervised Learning
A 24‑hour forecasting horizon was used. To convert the time series into a supervised learning dataset, a sliding‑window approach 
was applied such that the target variable y(t+24h) was predicted using 24 hourly lag values of soil moisture: y(t) … y(t−23). 
The total number of supervised samples after windowing was N = 9,433. The first 80% (7,546 samples) formed the training set, 
and the remaining 1,887 samples formed the test set. This chronological split avoids look‑ahead bias common in environmental data.
S3. Feature Engineering
In addition to the 24 moisture lags, the following predictors were engineered:
• Air temperature (°C)
• Wind speed (m/s)
• Soil temperature (°C)
• Hour‑of‑day encoded cyclically as sin(2πh/24) and cos(2πh/24)
• Day‑of‑year encoded cyclically as sin(2πd/365) and cos(2πd/365)
These cyclical features capture diurnal and seasonal evapotranspiration patterns that strongly influence soil water dynamics.
S4. Model Architecture (XGBoost)
The forecasting model used the XGBoost gradient‑boosted decision tree framework (reg:squarederror objective; hist tree method). 
This architecture is well suited to tabular environmental data and nonlinear autoregressive dependencies. The model integrates L1 and L2 
regularization, shrinkage via the learning rate, and tree‑level subsampling to mitigate overfitting. The final model learned an ensemble 
of 500 trees of maximum depth 3.
S5. Hyperparameter Optimization
Hyperparameter tuning was performed using RandomizedSearchCV with 30 candidate configurations evaluated via three forward‑chaining 
(TimeSeriesSplit) folds. The search space included:
• learning_rate ∈ [0.01, 0.20]
• n_estimators ∈ [200, 800]
• max_depth ∈ {2,3,4,5,6}
• subsample ∈ [0.6, 1.0]
• colsample_bytree ∈ [0.6, 1.0]
• gamma ∈ [0, 2]
• reg_alpha ∈ [0, 10]
• reg_lambda ∈ [0, 10]
The optimal configuration was:
learning_rate = 0.07, n_estimators = 500, max_depth = 3, subsample = 1.0,
colsample_bytree = 0.90, gamma = 0.0, reg_alpha = 2.0, reg_lambda = 10.0.
S6. Evaluation Metrics
Performance was assessed using root mean squared error (RMSE), mean absolute error (MAE), and the coefficient of determination (R²).
The model achieved:
• RMSE = 0.825
• MAE  = 0.249
• R²   = 0.639
The cross‑validated RMSE from the randomized search was 2.97. Results indicate reliable short‑horizon forecasting performance with 
sub‑1% average error magnitude.
S7. Feature Importance Analysis
XGBoost feature importance (gain-based) showed that soil moisture lags dominated predictive power. The most influential features were:
sm_lag_0 (0.2687), doy_sin (0.2371), sm_lag_1 (0.1085), sm_lag_10 (0.0492), sm_lag_7 (0.0446). Seasonal encodings (doy_sin, doy_cos) 
were especially strong, confirming that moisture trends reflect long-term seasonal evapotranspiration cycles. Meteorological forcings 
contributed less due to the short 24‑hour prediction window and the synthetic dataset's smooth dynamics.
S8. Reproducibility Materials
The complete training pipeline is implemented in Python 3.12 using XGBoost 2.x, scikit‑learn 1.5, pandas, numpy, SQLAlchemy, and joblib. 
Model training start and end timestamps, feature lists, hyperparameters, and evaluation outputs are programmatically logged in 
metricsandfeatures.txt. The full source code and configuration files are provided in the associated GitHub repository for full reproducibility.
