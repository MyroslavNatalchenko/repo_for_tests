import pandas as pd
import numpy as np
import tensorflow as tf
from tabnet_keras import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


def header(title):
    print(f"\n{'=' * 60}\n{title.center(60)}\n{'=' * 60}")


def load_data(file_path, target_col='popularity'):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_col]).astype('float32')
    # Using raw values (0-100) similar to the XGBoost example, rather than scaling to 0-1
    y = df[target_col].astype('float32')
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_evaluate(X_train, X_test, y_train, y_test):
    header("TRAINING TABNET MODEL")

    # TabNet Hyperparameters from documentation
    tabnet_params = {
        "decision_dim": 16,
        "attention_dim": 16,
        "n_steps": 3,
        "n_shared_glus": 2,
        "n_dependent_glus": 2,
        "relaxation_factor": 1.3,
        "epsilon": 1e-15,
        "momentum": 0.98,
        "mask_type": "sparsemax",  # can be 'sparsemax' or 'softmax'
        "lambda_sparse": 1e-3,
        "virtual_batch_splits": 8  # number of splits for ghost batch normalization
    }

    # Initialize TabNetRegressor
    # n_regressors=1 for single target regression
    model = TabNetRegressor(n_regressors=1, **tabnet_params)

    # Compile the model
    # Using RootMeanSquaredError as a metric to track performance easily
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    # Early stopping to prevent overfitting
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    print("Fitting model...")
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=1024,
        validation_split=0.2,
        callbacks=[stop_early],
        verbose=1
    )

    header("TEST RESULTS")
    preds = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"- RMSE (Root Mean Sq. Error): {rmse:>10.4f}")
    print(f"- MAE  (Mean Absolute Error): {mae:>10.4f}")
    print(f"- R2   (R-Squared Score):     {r2:>10.4f}")

    return model


def main():
    X_train, X_test, y_train, y_test = load_data('../models/train_data.csv')
    model = train_evaluate(X_train, X_test, y_train, y_test)

    # Save the model
    # Note: tabnet-keras models are Keras models, so they can be saved normally
    model.save('../models/tabnet_model.keras')
    print("\nModel saved to ../models/tabnet_model.keras")


if __name__ == "__main__":
    main()