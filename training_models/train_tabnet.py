import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tabnet_keras import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

def header(title):
    print(f"\n{'=' * 60}\n{title.center(60)}\n{'=' * 60}")

def load_data(file_path, target_col='popularity'):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col]).astype('float32')
    y = df[target_col].astype('float32') / 100.0
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, '../models/tabnet_scaler.joblib')
    print("Scaler saved to ../models/tabnet_scaler.joblib")

    tabnet_params = {
        "decision_dim": 64,
        "attention_dim": 64,
        "n_steps": 8,
        "n_shared_glus": 2,
        "n_dependent_glus": 2,
        "relaxation_factor": 1.5,
        "epsilon": 1e-15,
        "momentum": 0.98,
        "mask_type": "softmax",
        "lambda_sparse": 1e-4,
    }

    model = TabNetRegressor(n_regressors=1, **tabnet_params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')],
        run_eagerly=True
    )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )

    print("Fitting model...")
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=512,
        validation_split=0.2,
        callbacks=[stop_early, reduce_lr],
        verbose=1
    )

    header("TEST RESULTS")

    preds_real = model.predict(X_test) * 100.0
    y_test = y_test * 100.0

    rmse = np.sqrt(mean_squared_error(y_test, preds_real))
    mae = mean_absolute_error(y_test, preds_real)
    r2 = r2_score(y_test, preds_real)

    print(f"- RMSE (Root Mean Sq. Error): {rmse:>10.4f}")
    print(f"- MAE  (Mean Absolute Error): {mae:>10.4f}")
    print(f"- R2   (R-Squared Score):     {r2:>10.4f}")

    return model

def main():
    X_train, X_test, y_train, y_test = load_data('../models/train_data.csv')
    model = train_evaluate(X_train, X_test, y_train, y_test)
    model.save('../models/tabnet_model.keras')

if __name__ == "__main__":
    main()