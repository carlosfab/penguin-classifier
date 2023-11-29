
import argparse
import os

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


def train(model_directory: str, train_path: str, validation_path: str, epochs: int =50, batch_size: int=32) -> None:
    """
    Train a model using the training and validation data sets.

    Args:
        model_directory: Directory where the model will be saved.
        train_path: Path to the training data set.
        validation_path: Path to the validation data set.
        epochs: Number of epochs to train the model.
        batch_size: Batch size used during training.
    """
    # Load training and validation data sets
    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train.drop(X_train.columns[-1], axis=1, inplace=True)

    X_validation = pd.read_csv(Path(validation_path) / "validation.csv")
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation.drop(X_validation.columns[-1], axis=1, inplace=True)

    # Build a Sequential model
    model = Sequential([
        Dense(10, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(8, activation="relu"),
        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    # Make predictions
    predictions = model.predict(X_validation)
    predictions = np.argmax(predictions, axis=-1)
    print(f"Validation accuracy: {accuracy_score(y_validation, predictions)}")

    # Save the model
    model_filepath = Path(model_directory) / "001"
    model.save(model_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)

    args, _ = parser.parse_known_args()

    train(
        model_directory=os.environ["SM_MODEL_DIR"],
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        validation_path=os.environ["SM_CHANNEL_VALIDATION"],

        epochs=args.epochs,
        batch_size=args.batch_size
    )
