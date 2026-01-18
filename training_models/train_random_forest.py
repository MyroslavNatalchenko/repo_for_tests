import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def header(title):
    print(f"\n{'=' * 60}\n{title.center(60)}\n{'=' * 60}")

def load_data(file_path, target_col='popularity'):
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def tune_model(X_train, y_train):
    header("HYPERPARAMETER TUNING")

    search_params = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=search_params,
        n_iter=20,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )

    search.fit(X_train, y_train)

    print(f"Best Parameters: \n{search.best_params_}")
    return search.best_estimator_

def train_evaluate(X_train, X_test, y_train, y_test):
    rf = tune_model(X_train, y_train)

    preds = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    header("TEST RESULTS")
    print(f"- RMSE (Root Mean Sq. Error): {rmse:>10.4f}")
    print(f"- MAE  (Mean Absolute Error): {mae:>10.4f}")
    print(f"- R2   (R-Squared Score):     {r2:>10.4f}")

    header("TOP 10 FEATURE IMPORTANCE")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    for i in indices:
        feat_name = X_train.columns[i]
        score = importances[i]
        bar_len = int(score * 30)
        bar = '█' * bar_len
        print(f"{feat_name:<20} | {bar:<30} {score:.4f}")

    return rf

def main():
    X_train, X_test, y_train, y_test = load_data('../models/train_data.csv')
    model = train_evaluate(X_train, X_test, y_train, y_test)
    joblib.dump(model, '../models/rf_model.joblib')

if __name__ == "__main__":
    main()