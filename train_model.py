# version: v0.1.2
"""
Training pipeline for 24-hour-ahead soil moisture forecasting using XGBoost.
Fixed compatibility with XGBoost 2.x (eval_metric & early_stopping_rounds).
Added start/end timestamps and duration logging.
"""

import os
from datetime import datetime, timezone
from configparser import ConfigParser
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib


# ================================================================
# Load configuration
# ================================================================
config = ConfigParser()
config.read("config.ini")

DB_HOST = config.get("database", "host")
DB_PORT = config.get("database", "port")
DB_USER = config.get("database", "user")
DB_PASSWORD = config.get("database", "password")
DB_NAME = config.get("database", "database")
TABLE_NAME = config.get("database", "table")

TRAIN_START_DATE = config.get("training", "start_date")
TRAIN_END_DATE = config.get("training", "end_date")

# ================================================================
# Constants
# ================================================================
LAGS = 24
HORIZON = 24
MODEL_DIR = "models"
METRICS_FILE = "metricsandfeatures.txt"

RANDOM_STATE = 42
N_JOBS = -1     # use all cores


# ================================================================
# Load data
# ================================================================
def load_data_from_db():
    password_encoded = quote_plus(DB_PASSWORD)
    engine_url = f"mysql+pymysql://{DB_USER}:{password_encoded}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_url)

    query = f"""
        SELECT
            id, date, time,
            air_temperature, wind_speed,
            soil_temperature, soil_moisture
        FROM {TABLE_NAME}
        WHERE date >= '{TRAIN_START_DATE}' AND date <= '{TRAIN_END_DATE}'
        ORDER BY date, time, id
    """

    df = pd.read_sql(query, engine)

    df["time_str"] = df["time"].apply(lambda x: str(x).split()[-1])
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time_str"],
        format="%Y-%m-%d %H:%M:%S"
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    df.rename(columns={
        "air_temperature": "air_temp",
        "wind_speed": "wind_speed",
        "soil_temperature": "soil_temp",
        "soil_moisture": "soil_moisture"
    }, inplace=True)

    return df


# ================================================================
# Feature engineering
# ================================================================
def add_time_features(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["doy"] = df["datetime"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

    return df


def build_supervised(df, lags=LAGS, horizon=HORIZON):
    df = df.copy()

    for lag in range(lags):
        df[f"sm_lag_{lag}"] = df["soil_moisture"].shift(lag)

    df["target_sm"] = df["soil_moisture"].shift(-horizon)
    df = df.dropna().reset_index(drop=True)

    feature_cols = (
        [f"sm_lag_{lag}" for lag in range(lags)] +
        ["air_temp", "wind_speed", "soil_temp",
         "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    )

    X = df[feature_cols].values
    y = df["target_sm"].values

    return df, X, y, feature_cols


# ================================================================
# Model training: XGBoost 2.x SAFE VERSION
# ================================================================
def train_xgb_model(X, y, feature_cols):
    n = len(X)
    train_size = int(0.8 * n)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # hyperparameter search space
    param_distributions = {
        "learning_rate": np.linspace(0.01, 0.2, 20),
        "n_estimators": np.arange(200, 801, 50),
        "max_depth": np.arange(2, 7),
        "subsample": np.linspace(0.6, 1.0, 9),
        "colsample_bytree": np.linspace(0.6, 1.0, 9),
        "gamma": np.linspace(0, 2, 9),
        "reg_alpha": np.linspace(0, 10, 11),
        "reg_lambda": np.linspace(0, 10, 11),
    }

    # Base model for RandomizedSearchCV (CANNOT use early stopping here)
    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    search.fit(X_train, y_train)
    best_params = search.best_params_

    # FINAL MODEL with early stopping (allowed)
    final_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=30,   # <---- FIXED!!
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        **best_params
    )

    val_size = int(0.2 * len(X_train))
    X_sub, X_val = X_train[:-val_size], X_train[-val_size:]
    y_sub, y_val = y_train[:-val_size], y_train[-val_size:]

    # FIT WORKS NOW (XGB 2.x compatible)
    final_model.fit(
        X_sub, y_sub,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Test prediction
    y_pred = final_model.predict(X_test)

    metrics = {
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "cv_rmse": -search.best_score_,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_total": n,
    }

    # Feature importances
    booster = final_model.get_booster()
    raw = booster.get_score(importance_type="gain")
    total_gain = sum(raw.values()) or 1

    feature_importances = {
        col: raw.get(f"f{idx}", 0) / total_gain
        for idx, col in enumerate(feature_cols)
    }

    feature_importances = dict(
        sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    )

    return final_model, metrics, feature_importances, best_params


# ================================================================
# Write summary
# ================================================================
def write_summary(metrics, feature_importances, best_params, feature_cols):
    with open(METRICS_FILE, "w") as f:
        f.write(f"Start time: {metrics['start_time']}\n")
        f.write(f"End time:   {metrics['end_time']}\n")
        f.write(f"Duration (seconds): {metrics['duration_seconds']:.2f}\n\n")

        f.write("=== DATASET INFO ===\n")
        f.write(f"Training range: {TRAIN_START_DATE} → {TRAIN_END_DATE}\n")
        f.write(f"Total samples: {metrics['n_total']}\n")
        f.write(f"Train samples: {metrics['n_train']}\n")
        f.write(f"Test samples:  {metrics['n_test']}\n\n")

        f.write("=== CROSS-VALIDATION ===\n")
        f.write(f"Best CV RMSE: {metrics['cv_rmse']:.4f}\n\n")

        f.write("=== TEST PERFORMANCE ===\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"MAE : {metrics['mae']:.4f}\n")
        f.write(f"R²  : {metrics['r2']:.4f}\n\n")

        f.write("=== BEST HYPERPARAMETERS ===\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("=== FEATURE IMPORTANCES ===\n")
        for col, val in feature_importances.items():
            f.write(f"{col}: {val:.4f}\n")
        f.write("\nFeature columns:\n")
        f.write(", ".join(feature_cols) + "\n")


# ================================================================
# Main
# ================================================================
def main():
    start_time = datetime.now(timezone.utc)
    print(f"Training started at: {start_time.isoformat()}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data_from_db()
    print(f"Loaded {len(df)} rows.")

    df = add_time_features(df)
    _, X, y, feature_cols = build_supervised(df)
    print(f"Supervised samples: {len(X)}")

    model, metrics, feature_importances, best_params = train_xgb_model(X, y, feature_cols)

    joblib.dump(model, f"{MODEL_DIR}/xgb_soil_moisture.pkl")
    with open(f"{MODEL_DIR}/feature_columns.txt", "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    end_time = datetime.now(timezone.utc)

    metrics["start_time"] = start_time.isoformat()
    metrics["end_time"] = end_time.isoformat()
    metrics["duration_seconds"] = (end_time - start_time).total_seconds()

    write_summary(metrics, feature_importances, best_params, feature_cols)

    print(f"Training finished at: {end_time.isoformat()}")
    print(f"Duration: {metrics['duration_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
