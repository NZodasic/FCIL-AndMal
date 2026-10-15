"""Leak-free feature scaling for prepared tabular datasets.

The scaler is fitted on the training split only and then reused for validation,
test, and all client partitions.  Keeping this transform in Stage 1 avoids
client-specific preprocessing and prevents held-out statistics from entering
training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.schema import get_feature_columns


def fit_transform_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a ``StandardScaler`` on train features and transform all splits.

    Non-feature columns are preserved byte-for-byte.  Non-numeric feature
    values are coerced to NaN and imputed with the *training* column means
    before scaling.  The fitted scaler and ordered feature list are persisted
    alongside the prepared split files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_columns = get_feature_columns(train_df)
    if not feature_columns:
        raise ValueError("Cannot fit a scaler because no feature columns were found")

    def numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Split is missing feature columns: {missing[:5]}")
        sub = frame.loc[:, feature_columns]
        if all(pd.api.types.is_numeric_dtype(t) for t in sub.dtypes):
            return sub.astype(np.float32)
        return sub.apply(pd.to_numeric, errors="coerce").astype(np.float32)

    train_numeric = numeric_frame(train_df)
    train_means = train_numeric.mean(axis=0).fillna(0.0)
    train_numeric = train_numeric.fillna(train_means)

    scaler = StandardScaler()
    scaler.fit(train_numeric.to_numpy(dtype=np.float32, copy=False))

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        values = numeric_frame(frame).fillna(train_means)
        scaled = scaler.transform(values.to_numpy(dtype=np.float32, copy=False)).astype(
            np.float32
        )
        # Rebuild the frame so integer-typed source columns cannot reject the
        # floating-point standardized values under modern pandas.
        scaled_frame = pd.DataFrame(scaled, columns=feature_columns, index=frame.index)
        non_features = [column for column in frame.columns if column not in feature_columns]
        return pd.concat([frame.loc[:, non_features].copy(), scaled_frame], axis=1)

    joblib.dump(scaler, out / "scaler.joblib")
    with open(out / "scaler_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "transform": "sklearn.preprocessing.StandardScaler",
                "fit_split": "train",
                "feature_columns": feature_columns,
                "feature_count": len(feature_columns),
                "train_imputation": "column_mean_fit_on_train",
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            handle,
            indent=2,
        )

    return transform(train_df), transform(val_df), transform(test_df)


def load_scaler(output_dir: str):
    """Load a persisted Stage 1 scaler."""
    path = Path(output_dir) / "scaler.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Fitted scaler not found: {path}")
    return joblib.load(path)
