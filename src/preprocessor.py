"""
This module preprocesses data for machine learning tasks. It includes functions to read data from CSV files,
split the data into training, validation, and test sets, save baseline data, transform data, and save the
processed data and models.
"""

import os
import tarfile
import tempfile
from pathlib import Path

# Import statements...
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def preprocess(base_directory: str) -> None:
    """
    Preprocess the data by loading, splitting, transforming, saving the splits, and saving the model.

    Args:
        base_directory: The base directory where the input data and outputs will be managed.
    """
    # 1. Load supplied data
    df = _read_data_from_csv_files(base_directory)

    # 2. Split data into train and test sets
    df_train, df_validation, df_test = _split_data(df)

    # 3. Save baseline data
    _save_baselines(base_directory, df_train, df_test)

    # 3. Transform the train and test sets
    target_transformer = ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), [0])]
    )

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder()
    )

    features_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                make_column_selector(dtype_exclude="object"),
            ),
            ("categorical", categorical_transformer, ["island"]),
        ]
    )

    y_train = target_transformer.fit_transform(
        np.array(df_train.species.values).reshape(-1, 1)
    )
    y_validation = target_transformer.transform(
        np.array(df_validation.species.values).reshape(-1, 1)
    )
    y_test = target_transformer.transform(
        np.array(df_test.species.values).reshape(-1, 1)
    )

    df_train = df_train.drop(columns=["species"], axis=1)
    df_validation = df_validation.drop(columns=["species"], axis=1)
    df_test = df_test.drop(columns=["species"], axis=1)

    X_train = features_transformer.fit_transform(df_train)
    X_validation = features_transformer.transform(df_validation)
    X_test = features_transformer.transform(df_test)

    # 4. Save the train and test splits
    _save_splits(
        base_directory, X_train, y_train, X_validation, y_validation, X_test, y_test
    )

    # 5. Save the model (transformers) in tar.gz format
    _save_model(base_directory, target_transformer, features_transformer)


def _read_data_from_csv_files(base_directory: str) -> pd.DataFrame:
    """
    Read and concatenate data from CSV files located in the input directory.

    Args:
        base_directory: The directory where CSV files are located.

    Returns:
        A DataFrame containing the concatenated data.
    """
    input_directory = Path(base_directory) / "input"
    files = [file for file in input_directory.glob("*.csv")]

    if len(files) == 0:
        raise ValueError(f"No csv files found in {input_directory}")

    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data)

    # Shuffle the data
    return df.sample(frac=1, random_state=42)


def _split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training, validation, and test sets.

    Args:
        df: The DataFrame to be split.

    Returns:
        A tuple containing the training, validation, and test DataFrames.
    """
    df_train, temp = train_test_split(df, test_size=0.3)
    df_validation, df_test = train_test_split(temp, test_size=0.5)

    return df_train, df_validation, df_test


def _save_baselines(
    base_directory: str, df_train: pd.DataFrame = None, df_test: pd.DataFrame = None
) -> None:
    """
    Save baseline versions of the training and test data sets.

    Args:
        base_directory: Directory where the baseline data will be saved.
        df_train: Training data DataFrame.
        df_test: Test data DataFrame.
    """
    for split, data in [("train", df_train), ("test", df_test)]:
        baseline_path = Path(base_directory) / f"{split}-baseline"
        baseline_path.mkdir(parents=True, exist_ok=True)

        df = data.copy().dropna()

        # Save header only for the train baseline
        header = True if split == "train" else False
        df.to_csv(baseline_path / f"{split}-baseline.csv", index=False, header=header)


def _save_splits(
    base_directory: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Save the training, validation, and test sets after concatenating features with their respective targets.

    Args:
        base_directory: Directory where the data splits will be saved.
        X_train: Features of the training set.
        y_train: Target of the training set.
        X_validation: Features of the validation set.
        y_validation: Target of the validation set.
        X_test: Features of the test set.
        y_test: Target of the test set.
    """
    train = np.concatenate((X_train, y_train), axis=1)
    validation = np.concatenate((X_validation, y_validation), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", index=False, header=False)
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv", index=False, header=False
    )
    pd.DataFrame(test).to_csv(test_path / "test.csv", index=False, header=False)


def _save_model(base_directory: str, target_transformer, features_transformers) -> None:
    """
    Save the preprocessing model (transformers) in tar.gz format.

    Args:
        base_directory: Directory where the model will be saved.
        target_transformer: The transformer used for the target variable.
        features_transformers: The transformers used for the feature variables.
    """
    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(target_transformer, os.path.join(directory, "target.joblib"))
        joblib.dump(features_transformers, os.path.join(directory, "features.joblib"))

        model_path = Path(base_directory) / "model"
        model_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(f"{str(model_path / 'model.tar.gz')}", "w:gz") as tar:
            tar.add(os.path.join(directory, "target.joblib"), arcname="target.joblib")
            tar.add(
                os.path.join(directory, "features.joblib"), arcname="features.joblib"
            )


if __name__ == "__main__":
    preprocess(base_directory="/opt/ml/processing")
