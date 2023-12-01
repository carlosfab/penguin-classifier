
import json
import tarfile
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import accuracy_score
from tensorflow import keras


def evaluate(model_path: str, test_path: str, output_path: str) -> None:
    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test.drop(X_test.columns[-1], axis=1, inplace=True)

    # Let's now extract the model package so we can load 
    # it in memory.
    with tarfile.open(Path(model_path) / "model.tar.gz") as tar:
        tar.extractall(path=Path(model_path))

    model = keras.models.load_model(Path(model_path) / "001")

    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy}")

    evaluation_report = {
        "metrics": {
            "accuracy": {
                "value": accuracy
            },
        },
    }

    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "evaluation.json", "w") as f:
        f.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    evaluate(
        model_path="/opt/ml/processing/model/", 
        test_path="/opt/ml/processing/test/",
        output_path="/opt/ml/processing/evaluation/"
    )
