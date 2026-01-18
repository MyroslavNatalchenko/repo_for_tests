import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def header(title):
    print(f"\n{'=' * 60}\n{title.center(60)}\n{'=' * 60}")

def load_data(file_path, target_col='popularity'):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial, X_train, y_train, X_test, y_test):
    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'tree_method': 'auto'
    }

    model = xgb.XGBRegressor(**param, early_stopping_rounds=20)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse


def train_evaluate(X_train, X_test, y_train, y_test):
    header("HYPERPARAMETER TUNING (OPTUNA)")

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)

    best_params = study.best_params

    final_model = xgb.XGBRegressor(
        **best_params,
        early_stopping_rounds=50,
        n_jobs=-1
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    header("BEST HYPERPARAMETERS")
    print(best_params)

    header("TEST RESULTS")
    preds = final_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"- RMSE (Root Mean Sq. Error): {rmse:>10.4f}")
    print(f"- MAE  (Mean Absolute Error): {mae:>10.4f}")
    print(f"- R2   (R-Squared Score):     {r2:>10.4f}")

    return final_model

def main():
    X_train, X_test, y_train, y_test = load_data('../models/train_data.csv')
    model = train_evaluate(X_train, X_test, y_train, y_test)
    joblib.dump(model, '../models/xgb_model.joblib')

if __name__ == "__main__":
    main()