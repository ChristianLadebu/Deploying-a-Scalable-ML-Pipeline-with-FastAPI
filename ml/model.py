"""
Model-related functions: training, inference, persistence, and slice metrics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(
    X_train: Union[np.ndarray, Any],
    y_train: Union[np.ndarray, Any],
) -> Any:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray or sparse matrix
        Training data.
    y_train : np.ndarray
        Labels.

    Returns
    -------
    model : Any
        Trained model.
    """
    # TODO: implement training
    # A strong, simple baseline:
    model = LogisticRegression(max_iter=2000)

    model.fit(X_train, y_train)
    return model


def compute_model_metrics(
    y: np.ndarray,
    preds: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and fbeta.

    Returns (precision, recall, fbeta).
    """
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)
    return precision, recall, fbeta


def inference(
    model: Any,
    X: Union[np.ndarray, Any],
) -> np.ndarray:
    """
    Run model inference and return predictions.

    Inputs
    ------
    model : Any
        Trained model object with .predict().
    X : np.ndarray or sparse matrix
        Data for inference.

    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    # TODO: implement inference
    return model.predict(X)


def save_model(
    model: Any,
    encoder: Optional[Any] = None,
    lb: Optional[Any] = None,
    model_path: str = "model/model.joblib",
    encoder_path: str = "model/encoder.joblib",
    lb_path: str = "model/lb.joblib",
) -> None:
    """
    Save trained model and optional preprocessing artifacts.

    Inputs
    ------
    model : Any
        Trained model.
    encoder : Any, optional
        Fitted OneHotEncoder (or similar).
    lb : Any, optional
        Fitted LabelBinarizer (or similar).
    model_path : str
    encoder_path : str
    lb_path : str
    """
    # TODO: implement save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)

    if encoder is not None:
        joblib.dump(encoder, encoder_path)

    if lb is not None:
        joblib.dump(lb, lb_path)


def load_model(
    model_path: str = "model/model.joblib",
    encoder_path: str = "model/encoder.joblib",
    lb_path: str = "model/lb.joblib",
) -> Tuple[Any, Optional[Any], Optional[Any]]:
    """
    Load trained model and optional preprocessing artifacts.

    Returns
    -------
    model, encoder, lb
    """
    # TODO: implement load
    model = joblib.load(model_path)

    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    lb = joblib.load(lb_path) if os.path.exists(lb_path) else None

    return model, encoder, lb


def performance_on_categorical_slice(
    model: Any,
    data,  # pd.DataFrame
    categorical_features: List[str],
    label: str,
    slice_feature: str,
    encoder: Any,
    lb: Any,
    process_data_fn,
) -> List[Dict[str, Any]]:
    """
    Compute model performance on slices of the data where `slice_feature` is held fixed.

    Inputs
    ------
    model : trained model
    data : pd.DataFrame
        Raw (unprocessed) dataframe including label column.
    categorical_features : list[str]
        List of categorical feature names (for process_data).
    label : str
        Label column name.
    slice_feature : str
        Feature to slice on (must be a column in `data`).
    encoder : fitted encoder used during training
    lb : fitted label binarizer used during training
    process_data_fn : callable
        Typically ml.data.process_data

    Returns
    -------
    results : list[dict]
        Each dict has {feature, value, n, precision, recall, fbeta}
    """
    # TODO: implement slice performance

    if slice_feature not in data.columns:
        raise ValueError(f"slice_feature '{slice_feature}' not found in data columns.")

    results: List[Dict[str, Any]] = []

    # Ensure we don't get weird ordering from pandas categoricals
    unique_values = sorted(data[slice_feature].dropna().unique().tolist())

    for v in unique_values:
        slice_df = data[data[slice_feature] == v]

        # Skip empty slices just in case
        if slice_df.shape[0] == 0:
            continue

        X_slice, y_slice, _, _ = process_data_fn(
            slice_df,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        results.append(
            {
                "feature": slice_feature,
                "value": v,
                "n": int(slice_df.shape[0]),
                "precision": float(precision),
                "recall": float(recall),
                "fbeta": float(fbeta),
            }
        )

    return results
