"""Task configuration for CIC-AndMal-2020 dataset.

Defines the 5-task incremental learning setup with 15 labels.
"""

from typing import Dict, List

# Task label mapping: 5 tasks, 3 labels per task
# Benign only appears in Task 1 (task_id=0)
TASK_LABEL_MAP: Dict[int, List[str]] = {
    0: ["Benign", "PUA", "Backdoor"],
    1: ["Adware", "TrojanBanker", "TrojanSpy"],
    2: ["NoCategory", "Trojan", "Riskware"],
    3: ["FileInfector", "Ransomware", "TrojanDropper"],
    4: ["Scareware", "ZeroDay", "TrojanSMS"],
}

# All 15 labels in order of appearance
ALL_LABELS: List[str] = [
    "Benign",      # Task 0
    "PUA",         # Task 0
    "Backdoor",    # Task 0
    "Adware",      # Task 1
    "TrojanBanker",# Task 1
    "TrojanSpy",   # Task 1
    "NoCategory",  # Task 2
    "Trojan",      # Task 2
    "Riskware",    # Task 2
    "FileInfector",# Task 3
    "Ransomware",  # Task 3
    "TrojanDropper",# Task 3
    "Scareware",   # Task 4
    "ZeroDay",     # Task 4
    "TrojanSMS",   # Task 4
]

# Label to index mapping
LABEL_TO_IDX: Dict[str, int] = {label: idx for idx, label in enumerate(ALL_LABELS)}
LABEL2ID = LABEL_TO_IDX

# Index to label mapping
IDX_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(ALL_LABELS)}
ID2LABEL = IDX_TO_LABEL

# Task to label indices mapping
TASK_TO_INDICES: Dict[int, List[int]] = {
    task_id: [LABEL_TO_IDX[label] for label in labels]
    for task_id, labels in TASK_LABEL_MAP.items()
}

# Label to task mapping (first appearance)
LABEL_TO_TASK: Dict[str, int] = {}
for task_id, labels in TASK_LABEL_MAP.items():
    for label in labels:
        if label not in LABEL_TO_TASK:
            LABEL_TO_TASK[label] = task_id


def get_labels_for_task(task_id: int) -> List[str]:
    """Get labels for a specific task.

    Args:
        task_id: Task ID (0-4).

    Returns:
        List of label names.
    """
    return TASK_LABEL_MAP.get(task_id, [])


def get_indices_for_task(task_id: int) -> List[int]:
    """Get label indices for a specific task.

    Args:
        task_id: Task ID (0-4).

    Returns:
        List of label indices.
    """
    return TASK_TO_INDICES.get(task_id, [])


def get_cumulative_labels(task_id: int) -> List[str]:
    """Get all labels from task 0 up to and including task_id.

    Args:
        task_id: Task ID (0-4).

    Returns:
        List of all labels seen up to task_id.
    """
    labels = []
    for t in range(task_id + 1):
        labels.extend(TASK_LABEL_MAP.get(t, []))
    return labels


def get_cumulative_indices(task_id: int) -> List[int]:
    """Get all label indices from task 0 up to and including task_id.

    Args:
        task_id: Task ID (0-4).

    Returns:
        List of label indices.
    """
    indices = []
    for t in range(task_id + 1):
        indices.extend(TASK_TO_INDICES.get(t, []))
    return indices


def get_num_classes_for_task(task_id: int) -> int:
    """Get total number of classes seen up to task_id.

    Args:
        task_id: Task ID (0-4).

    Returns:
        Total number of classes.
    """
    return len(get_cumulative_labels(task_id))


def get_task_for_label(label: str) -> int:
    """Get the task ID where a label first appears.

    Args:
        label: Label name.

    Returns:
        Task ID where label first appears.
    """
    return LABEL_TO_TASK.get(label, -1)


def is_label_in_task(label: str, task_id: int) -> bool:
    """Check if a label is in a specific task.

    Args:
        label: Label name.
        task_id: Task ID (0-4).

    Returns:
        True if label is in task.
    """
    return label in TASK_LABEL_MAP.get(task_id, [])


# Client participation configuration
# Number of active clients increases gradually per task
def get_client_counts(n_clients: int) -> List[int]:
    """Get number of active clients per task.

    Participation increases from base_ratio to 100%.
    Formula: int(min(0.6 + t * 0.1, 1.0) * K)

    Args:
        n_clients: Total number of clients (K).

    Returns:
        List of client counts for each of 5 tasks.
    """
    counts = []
    for t in range(5):
        ratio = min(0.6 + t * 0.1, 1.0)
        counts.append(int(ratio * n_clients))
    return counts


# Feature dimensions
STATIC_FEATURE_DIM = 9503
DYNAMIC_FEATURE_DIM = 141  # or 282 if using before+after
FUSED_FEATURE_DIM = STATIC_FEATURE_DIM + DYNAMIC_FEATURE_DIM

# Minimum samples per client per label
MIN_SAMPLES_PER_LABEL_CLIENT = 30

# Default batch size (minimum)
DEFAULT_BATCH_SIZE = 256
