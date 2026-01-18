import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

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

def build_model(hp, X_train_sample):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train_sample))

    model = tf.keras.Sequential()
    model.add(normalizer)

    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation='relu'
        ))
        if hp.Boolean(f'dropout_{i}'):
            model.add(tf.keras.layers.Dropout(rate=hp.Float(f'dropout_rate_{i}', 0.1, 0.4, step=0.1)))
        if hp.Boolean(f'batch_norm_{i}'):
            model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(),
        metrics=['mae', 'mse']
    )
    return model

def train_evaluate(X_train, X_test, y_train, y_test):
    header("HYPERPARAMETER TUNING (KERAS TUNER)")

    if os.path.exists('models/kt_dir'):
        import shutil
        shutil.rmtree('models/kt_dir')

    tuner = kt.Hyperband(
        lambda hp: build_model(hp, X_train),
        objective='val_loss',
        max_epochs=30,
        factor=3,
        hyperband_iterations=1,
        directory='models/kt_dir',
        project_name='music_popularity'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    tuner.search(
        np.array(X_train), np.array(y_train),
        epochs=30,
        validation_split=0.2,
        callbacks=[stop_early],
        batch_size=256,
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    model.fit(
        np.array(X_train), np.array(y_train),
        epochs=100,
        batch_size=256,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        ],
        verbose=1
    )

    header("BEST HYPERPARAMETERS DETAILED")
    print(f"Learning Rate: {best_hps.get('learning_rate'):>10.4f}")
    print(f"Num Layers:    {best_hps.get('num_layers'):>10.4f}")

    for i in range(best_hps.get('num_layers')):
        print(f"\n\t[Layer {i + 1} INFORMATION]")
        print(f"Units:           {best_hps.get(f'units_{i}'):>10.4f}")
        print(f"Dropout Enabled: {best_hps.get(f'dropout_{i}'):>10.4f}")
        if best_hps.get(f'dropout_{i}'):
            print(f"Dropout Rate:    {best_hps.get(f'dropout_rate_{i}'):>10.4f}")
        print(f"Batch Norm:      {best_hps.get(f'batch_norm_{i}'):>10.4f}")

    header("TEST RESULTS")
    raw_predictions = model.predict(np.array(X_test))
    predictions_original = raw_predictions * 100.0
    y_test_original = y_test * 100.0

    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)

    print(f"- RMSE (Root Mean Sq. Error): {rmse:>10.4f}")
    print(f"- MAE  (Mean Absolute Error): {mae:>10.4f}")
    print(f"- R2   (R-Squared Score):     {r2:>10.4f}")

    return model

def main():
    X_train, X_test, y_train, y_test = load_data('../models/train_data.csv')
    model = train_evaluate(X_train, X_test, y_train, y_test)
    model.save('../models/tf_after_stratified_sampling_model.keras')

if __name__ == "__main__":
    main()