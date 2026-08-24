"""Few-shot session construction for malware class-incremental learning."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FewShotSession:
    """Support/query data derived only from the selected K examples per class."""

    support_X: np.ndarray
    support_y: np.ndarray
    query_X: np.ndarray
    query_y: np.ndarray
    support_indices: np.ndarray

    @property
    def train_X(self) -> np.ndarray:
        if len(self.query_X) == 0:
            return self.support_X
        return np.concatenate((self.support_X, self.query_X), axis=0)

    @property
    def train_y(self) -> np.ndarray:
        if len(self.query_y) == 0:
            return self.support_y
        return np.concatenate((self.support_y, self.query_y), axis=0)


def build_few_shot_session(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k_shot: int,
    query_per_class: int = 0,
    mask_probability: float = 0.0,
    expected_classes: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> FewShotSession:
    """Build an N-way K-shot session with masking-derived query examples.

    The paper does not permit access to previous-session training samples. Only
    K current-session examples per class are selected. Optional query examples
    are generated from those support examples through random feature masking.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2-D and y must be a matching 1-D array")
    if k_shot <= 0:
        raise ValueError("k_shot must be positive")
    if query_per_class < 0:
        raise ValueError("query_per_class cannot be negative")
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must be in [0, 1]")

    classes = (
        np.asarray(list(expected_classes), dtype=y.dtype)
        if expected_classes is not None
        else np.unique(y)
    )
    if len(classes) == 0:
        raise ValueError("A few-shot session must contain at least one class")

    rng = np.random.default_rng(seed)
    selected_indices = []
    for class_id in classes:
        class_indices = np.flatnonzero(y == class_id)
        if len(class_indices) < k_shot:
            raise ValueError(
                f"Class {class_id} has {len(class_indices)} samples; "
                f"{k_shot} are required"
            )
        selected_indices.extend(
            rng.choice(class_indices, size=k_shot, replace=False).tolist()
        )

    support_indices = np.asarray(selected_indices, dtype=np.int64)
    support_X = X[support_indices].copy()
    support_y = y[support_indices].copy()

    query_parts = []
    query_label_parts = []
    for class_id in classes:
        class_support = support_X[support_y == class_id]
        if query_per_class == 0:
            continue
        source_indices = rng.choice(
            len(class_support), size=query_per_class, replace=True
        )
        augmented = class_support[source_indices].copy()
        if mask_probability > 0.0:
            mask = rng.random(augmented.shape) < mask_probability
            augmented[mask] = 0.0
        query_parts.append(augmented)
        query_label_parts.append(
            np.full(query_per_class, class_id, dtype=y.dtype)
        )

    query_X = (
        np.concatenate(query_parts, axis=0)
        if query_parts
        else np.empty((0, X.shape[1]), dtype=X.dtype)
    )
    query_y = (
        np.concatenate(query_label_parts, axis=0)
        if query_label_parts
        else np.empty((0,), dtype=y.dtype)
    )
    return FewShotSession(
        support_X=support_X,
        support_y=support_y,
        query_X=query_X,
        query_y=query_y,
        support_indices=support_indices,
    )
