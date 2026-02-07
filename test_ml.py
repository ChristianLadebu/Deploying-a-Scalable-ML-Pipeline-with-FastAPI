from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

# Keep these aligned with training scirpt
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="session")
def census_df():
    """
    Load a small sample of the census dataset for fast, stable tests.
    """
    data_path = Path(__file__).resolve().parent / "data" / "census.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find census.csv at {data_path}")

    df = pd.read_csv(data_path)

    # Small deterministic sample so tests run quickly
    df = df.sample(n=500, random_state=42).reset_index(drop=True)
    return df


def test_train_model_returns_logreg(census_df):
    """Model uses expected algorithm."""
    train_df, _ = train_test_split(
        census_df,
        test_size=0.2,
        random_state=42,
        stratify=census_df["salary"],
    )

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    model = train_model(X_train, y_train)

    assert isinstance(model, LogisticRegression)
    assert encoder is not None
    assert lb is not None


def test_inference_output_shape(census_df):
    """Inference returns predictions with correct shape and valid values."""
    train_df, test_df = train_test_split(
        census_df,
        test_size=0.2,
        random_state=42,
        stratify=census_df["salary"],
    )

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == y_test.shape[0]
    # LogisticRegression predict outputs class labels
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_known_case():
    """Metrics return expected values on a simple, known input."""
    y = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])

    # For positive class = 1:
    # TP=2 (positions 2,3), FP=1 (position 1), FN=0
    # precision = 2/3, recall = 2/2 = 1, F1 = 2*(2/3)*1 / ((2/3)+1) = 0.8
    p, r, fb = compute_model_metrics(y, preds)

    assert p == pytest.approx(2 / 3, abs=1e-6)
    assert r == pytest.approx(1.0, abs=1e-6)
    assert fb == pytest.approx(0.8, abs=1e-6)


def test_train_test_split_sizes(census_df):
    """Train/test split has expected sizes."""
    train_df, test_df = train_test_split(
        census_df,
        test_size=0.2,
        random_state=42,
        stratify=census_df["salary"],
    )

    # 500 rows total -> 400 train, 100 test
    assert train_df.shape[0] == 400
    assert test_df.shape[0] == 100
