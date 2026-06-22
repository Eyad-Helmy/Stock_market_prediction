import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np

def build_model(input_shape: tuple) -> tf.keras.Model:

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
 
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
 
    print("[model] Model built:")
    model.summary()
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32
) -> tuple:
    """
    Parameters
    ----------
    model                      : Compiled model from build_model()
    X_train, y_train           : Training sequences (already scaled)
    X_test, y_test             : Test sequences (already scaled, using
                                  train-fitted scaler — never re-fit)
    epochs, batch_size         : Training hyperparameters
 
    Returns
    -------
    model   : Trained model (best weights restored)
    history : Keras History object (loss curves for plotting)
    """
    print(f"[model] Training samples : {len(X_train)}")
    print(f"[model] Test samples     : {len(X_test)}")
 
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
 
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
 
    return model, history
