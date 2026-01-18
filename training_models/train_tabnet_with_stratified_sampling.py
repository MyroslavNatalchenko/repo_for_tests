import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tabnet_keras import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def header(title):
    print(f"\n{'=' * 60}\n{title.center(60)}\n{'=' * 60}")

def load_data(file_path, target_col='popularity'):
    df = pd.read_csv(file_path)

    bins = [-1, 0, 20, 40, 60, 80, 100]
    labels = [0, 1, 2, 3, 4, 5]
    df['strata'] = pd.cut(df[target_col], bins=bins, labels=labels)

    df_zeros = df[df['strata'] == 0].sample(n=5000, random_state=42)
    df_others = df[df['strata'] != 0]
    df_balanced = pd.concat([df_zeros, df_others])

    X = df_balanced.drop(columns=[target_col, 'strata']).astype('float32')
    y = df_balanced[target_col].astype('float32') / 100.0

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_balanced['strata'])

def train_evaluate(X_train, X_test, y_train, y_test):
    scaler = joblib.load("../models/tabnet_scaler.joblib")
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
        loss=tf.keras.losses.Huber(),
        optimizer=optimizer,
        metrics=['mae', 'mse'],
        run_eagerly=True
    )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
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
        epochs=150,
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
    model.save('../models/tabnet_after_stratified_sampling_model.keras')

if __name__ == "__main__":
    main()