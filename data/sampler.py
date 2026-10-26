"""
Class-Balanced Sampling Strategies for Imbalanced Intrusion and Malware Datasets.
Implements:
1. BalancedBatchSampler: Yields batches with an equal (or near-equal) quota per class
   (Algorithm 1 in Falcon; Yang et al., IEEE TCE 2026).
2. build_weighted_random_sampler: Builds PyTorch WeightedRandomSampler with inverse-frequency weights.
"""

from typing import Dict, Iterator, List, Optional, Sequence, Union
import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


class BalancedBatchSampler(Sampler[List[int]]):
    """
    Per-class balanced mini-batch sampler.
    
    Given batch size B and K active classes:
    - Each class is assigned a quota q_k = floor(B / K), with remainder (B mod K)
      distributed across the first classes.
    - Mini-batches contain exactly B samples where each class has >= 1 sample.
    - If a minority class has fewer available samples than q_k, sampling with
      replacement is applied to that class.
    
    Args:
        labels: Sequence of class labels (ints or 1D tensor/array) for each dataset index.
        batch_size: Total batch size B.
        drop_last: If True, drop the final incomplete batch when epoch ends.
        shuffle: Whether to shuffle sample selection within each class.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        labels: Union[Sequence[int], np.ndarray, torch.Tensor],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # 1. Group sample indices by class
        self.unique_classes = np.unique(self.labels)
        self.unique_classes = self.unique_classes[self.unique_classes >= 0]  # ignore -1 invalid
        self.num_classes = len(self.unique_classes)

        if self.num_classes == 0:
            raise ValueError("No valid classes found in labels array.")

        self.class_indices: Dict[int, np.ndarray] = {
            int(cls): np.where(self.labels == cls)[0] for cls in self.unique_classes
        }

        # 2. Compute per-class quotas q_k
        self.quotas: Dict[int, int] = {}
        base_quota = self.batch_size // self.num_classes
        remainder = self.batch_size % self.num_classes

        for i, cls in enumerate(self.unique_classes):
            cls_int = int(cls)
            self.quotas[cls_int] = base_quota + (1 if i < remainder else 0)
            if self.quotas[cls_int] == 0:
                self.quotas[cls_int] = 1

        # Total samples per batch should equal batch_size
        self.actual_batch_size = sum(self.quotas.values())

        # Determine number of batches in one epoch based on dataset size
        # Ensures all samples from largest class or full dataset are roughly covered
        max_samples = max(len(idx) for idx in self.class_indices.values())
        max_quota = max(self.quotas.values())
        self.num_batches = max(1, max_samples // max(1, max_quota))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # For each class, prepare shuffled index pools
        class_pools: Dict[int, List[int]] = {}
        for cls, indices in self.class_indices.items():
            idx_copy = indices.copy()
            if self.shuffle:
                self.rng.shuffle(idx_copy)
            class_pools[cls] = list(idx_copy)

        for _ in range(self.num_batches):
            batch: List[int] = []
            for cls in self.unique_classes:
                cls_int = int(cls)
                quota = self.quotas[cls_int]
                pool = class_pools[cls_int]

                # If pool does not have enough samples, refill with replacement
                if len(pool) < quota:
                    idx_copy = self.class_indices[cls_int].copy()
                    if self.shuffle:
                        self.rng.shuffle(idx_copy)
                    pool.extend(list(idx_copy))
                    # If class is extremely scarce (e.g., 1 sample and quota is 4)
                    while len(pool) < quota:
                        pool.extend(list(idx_copy))

                # Draw quota samples
                drawn = pool[:quota]
                class_pools[cls_int] = pool[quota:]
                batch.extend(drawn)

            if self.shuffle:
                self.rng.shuffle(batch)

            yield batch


def build_weighted_random_sampler(
    labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    replacement: bool = True,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Build a PyTorch WeightedRandomSampler with weights inversely proportional
    to class frequency (Algorithm 1, lines 5-7).
    
    Args:
        labels: Class labels for each sample.
        replacement: Draw samples with replacement.
        num_samples: Total number of samples to draw per epoch. If None, len(labels).
        
    Returns:
        WeightedRandomSampler instance.
    """
    if isinstance(labels, torch.Tensor):
        labels_arr = labels.cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    classes, counts = np.unique(labels_arr, return_counts=True)
    class_weights = {cls: 1.0 / max(1, count) for cls, count in zip(classes, counts)}

    sample_weights = np.array([class_weights[lbl] for lbl in labels_arr], dtype=np.float64)
    weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

    n_samples = num_samples if num_samples is not None else len(labels_arr)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=n_samples, replacement=replacement)
