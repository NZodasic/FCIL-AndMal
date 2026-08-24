""" Configuration Module.

This module contains all configurations for the Federated Class-Incremental
Learning framework for Android malware detection.

"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json


@dataclass
class ScenarioConfig:
    """Configuration for FL data partitioning scenario."""
    n_clients: int = 20
    feature_type: str = 'dynamic'  # 'static', 'dynamic', or 'fused'
    dirichlet_alpha: float = 0.5
    base_ratio: float = 0.6
    step: float = 0.1
    seed: int = 42
    min_samples_per_label_client: int = 30
    output_dir: str = './fl_data_partitions'
    partition_output_dir: Optional[str] = None
    raw_data_dir: str = './raw_data'
    prepared_data_dir: str = './prepared_data'
    label_col: str = 'label'

    def __post_init__(self):
        if self.partition_output_dir is not None:
            self.output_dir = self.partition_output_dir
        else:
            self.partition_output_dir = self.output_dir

    def get_scenario_dir(self) -> str:
        """Get output directory path for current scenario."""
        import os
        return os.path.join(self.output_dir, self.feature_type, f"{self.n_clients}clients")

    def get_active_client_count(self, task_id: int) -> int:
        """Get number of active clients participating in task_id."""
        ratio = min(self.base_ratio + task_id * self.step, 1.0)
        return int(ratio * self.n_clients)


@dataclass
class ModelConfig:
    """Configuration for model architecture.

    Attributes:
        backbone_type: Type of neural backbone ('mlp', 'cnn1d', 'tcn', 'capsule', 'fused').
        input_dim: Dimension of input features.
        latent_dim: Feature embedding latent dimension before classification head.
        classes_per_task: Number of new classes introduced per task.
        num_total_classes: Total cumulative number of classes across all tasks.
        feature_type: Type of input features ('static', 'dynamic', 'fused').
        static_input_dim: Dimension of static features (after reduction).
        dynamic_input_dim: Dimension of dynamic features.
        hidden_dims: List of hidden layer dimensions.
        dropout_rate: Dropout probability.
        dropout: Alias for dropout_rate.
        use_tcn: Whether to use TCN for dynamic features.
        tcn_layers: Number of TCN layers.
        tcn_kernel_size: Kernel size for TCN.
        tcn_dilations: List of dilation rates for TCN.
        use_capsule: Whether to use Capsule Network.
        capsule_dim: Dimension of capsule vectors.
        capsule_routings: Number of routing iterations.
        fusion_method: How to fuse static and dynamic ('concat', 'attention', 'gated').
    """
    backbone_type: str = 'mlp'
    input_dim: int = 141
    latent_dim: int = 128
    classes_per_task: int = 3
    num_total_classes: int = 15
    feature_type: str = 'dynamic'
    static_input_dim: int = 500  # After dimensionality reduction
    dynamic_input_dim: int = 141  # or 282 for before+after
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.5
    dropout: float = 0.5
    use_tcn: bool = False
    tcn_layers: int = 3
    tcn_kernel_size: int = 3
    tcn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    use_capsule: bool = False
    capsule_dim: int = 16
    capsule_routings: int = 3
    fusion_method: str = 'concat'

    def __post_init__(self):
        if self.dropout != 0.5 and self.dropout_rate == 0.5:
            self.dropout_rate = self.dropout
        else:
            self.dropout = self.dropout_rate


@dataclass
class FLConfig:
    n_rounds: int = 50
    rounds_per_task: int = 50
    local_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    lr: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = 'adam'
    aggregator: str = 'fedavg'
    client_fraction: float = 1.0
    eval_every: int = 5
    n_tasks: int = 5
    device: str = 'cpu'

    def __post_init__(self):
        if self.rounds_per_task != 50 and self.n_rounds == 50:
            self.n_rounds = self.rounds_per_task
        if self.lr != 0.001 and self.learning_rate == 0.001:
            self.learning_rate = self.lr


@dataclass
class IncrementalConfig:
    """Configuration for incremental learning strategies.

    Attributes:
        method_name: Incremental strategy name ('finetune', 'joint', 'ewc', 'lwf', 'replay', 'spcil', 'malfscil').
        strategy: Alias for method_name.
        ewc_lambda: EWC regularization strength.
        lwf_alpha: LwF distillation weight.
        lwf_temperature: Temperature for LwF distillation.
        replay_buffer_size: Number of samples per class for replay.
        replay_buffer_size_per_class: Alias for replay_buffer_size.
        replay_selection: Replay sample selection ('random', 'herding').
        herding: Alias for replay_selection == 'herding'.
        spcil_mu: SPCIL pacing parameter.
        spcil_lambda_init: Initial SPCIL lambda.
        spcil_lambda_step: SPCIL step increment.
        fscil_k_shot: Labeled support samples per new class.
        fscil_query_per_class: Mask-augmented query samples per new class.
        malfscil_vae_weight: Weight for base-session VAE loss.
        malfscil_arc_weight: ArcFace share of the incremental objective.
    """
    method_name: str = 'finetune'
    strategy: str = 'finetune'
    ewc_lambda: float = 1000.0
    lwf_alpha: float = 1.0
    lwf_temperature: float = 2.0
    replay_buffer_size: int = 20
    replay_buffer_size_per_class: int = 20
    replay_selection: str = 'herding'
    herding: bool = True
    spcil_mu: float = 0.5
    spcil_lambda_init: float = 0.5
    spcil_lambda_step: float = 0.1
    fscil_n_way: int = 3
    fscil_k_shot: int = 5
    fscil_query_per_class: int = 5
    fscil_mask_probability: float = 0.1
    malfscil_vae_weight: float = 1.0
    malfscil_kl_weight: float = 1.0
    malfscil_arc_weight: float = 0.5
    malfscil_arc_scale: float = 30.0
    malfscil_arc_margin: float = 0.5
    malfscil_graph_attention_dim: int = 64
    # Deprecated MALFSIL fields retained for configuration compatibility.
    malfsil_distill_weight: float = 1.0
    malfsil_proto_weight: float = 0.1
    malfsil_margin: float = 1.0
    malfsil_prototype_weight: float = 0.1

    def __post_init__(self):
        if self.method_name.lower() == 'malfsil':
            self.method_name = 'malfscil'
        if self.strategy.lower() == 'malfsil':
            self.strategy = 'malfscil'
        if self.fscil_n_way <= 0 or self.fscil_k_shot <= 0:
            raise ValueError('FSCIL n_way and k_shot must be positive')
        if self.fscil_query_per_class < 0:
            raise ValueError('fscil_query_per_class cannot be negative')
        if not 0.0 <= self.fscil_mask_probability <= 1.0:
            raise ValueError('fscil_mask_probability must be in [0, 1]')


# Alias for compatibility
ILConfig = IncrementalConfig


@dataclass
class ExperimentConfig:
    """Configuration for complete experiment."""
    exp_name: str = 'fcil_andmal'
    experiment_name: str = 'fcil_andmal'
    output_root: str = './EXPERIMENT'
    scenario: Optional[ScenarioConfig] = None
    model: Optional[ModelConfig] = None
    il: Optional[ILConfig] = None
    fl: Optional[FLConfig] = None
    seed: int = 42
    n_tasks: int = 5
    n_seeds: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    device: str = 'cuda'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    results_dir: str = './results'

    def __post_init__(self):
        if self.exp_name != 'fcil_andmal':
            self.experiment_name = self.exp_name
        else:
            self.exp_name = self.experiment_name

    def get_exp_dir(self) -> str:
        import os
        return os.path.join(self.output_root, self.exp_name)

    def save_json(self, path: str) -> None:
        import json
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# Task label mapping for CIC-AndMal-2020 (5 tasks, 15 labels)
# Benign only appears in Task 1
task_config={
    "n_tasks": 5,
    "n_classes": 15,
    "task_labels": {
        0: ["Benign", "PUA", "Backdoor"],
        1: ["Adware", "TrojanBanker", "TrojanSpy"],
        2: ["NoCategory", "Trojan", "Riskware"],
        3: ["FileInfector", "Ransomware", "TrojanDropper"],
        4: ["Scareware", "ZeroDay", "TrojanSMS"]
    },
    "all_labels": [
        "Benign", "PUA", "Backdoor", "Adware", "TrojanBanker",
        "TrojanSpy", "NoCategory", "Trojan", "Riskware",
        "FileInfector", "Ransomware", "TrojanDropper",
        "Scareware", "ZeroDay", "TrojanSMS"
    ]
}


# Label mappings for dataset preparation
STATIC_LABEL_MAP = {
    'Ben0': 'Benign',
    'Ben1': 'Benign',
    'Ben2': 'Benign',
    'Ben3': 'Benign',
    'Ben4': 'Benign',
    'Adware': 'Adware',
    'Backdoor': 'Backdoor',
    'Banker': 'TrojanBanker',
    'Dropper': 'TrojanDropper',
    'FileInfector': 'FileInfector',
    'NoCategory': 'NoCategory',
    'PUA': 'PUA',
    'Ransomware': 'Ransomware',
    'Riskware': 'Riskware',
    'Scareware': 'Scareware',
    'SMS': 'TrojanSMS',
    'Spy': 'TrojanSpy',
    'Trojan': 'Trojan',
    'Zeroday': 'ZeroDay',
}

DYNAMIC_LABEL_MAP = {
    'adware': 'Adware',
    'backdoor': 'Backdoor',
    'trojan_banker': 'TrojanBanker',
    'trojan_dropper': 'TrojanDropper',
    'fileinfector': 'FileInfector',
    'no_category': 'NoCategory',
    'pua': 'PUA',
    'ransomware': 'Ransomware',
    'riskware': 'Riskware',
    'scareware': 'Scareware',
    'trojan_sms': 'TrojanSMS',
    'trojan_spy': 'TrojanSpy',
    'trojan': 'Trojan',
    'zero_day': 'ZeroDay',
}


def get_task_label_map(task_id: int) -> List[str]:
    """Get list of labels for a specific task.

    Args:
        task_id: Task ID (0-indexed).

    Returns:
        List of label names for the task.
    """
    return task_config["task_labels"].get(task_id, [])


def get_cumulative_labels(task_id: int) -> List[str]:
    """Get all labels up to and including the given task.

    Args:
        task_id: Task ID (0-indexed).

    Returns:
        List of all labels from task 0 to task_id.
    """
    labels = []
    for t in range(task_id + 1):
        labels.extend(task_config["task_labels"].get(t, []))
    return labels


def get_label_to_task_mapping() -> Dict[str, int]:
    """Get mapping from label to first task it appears in.

    Returns:
        Dictionary mapping label name to task ID.
    """
    mapping = {}
    for task_id, labels in task_config["task_labels"].items():
        for label in labels:
            if label not in mapping:
                mapping[label] = task_id
    return mapping


def save_config(config: Union[ScenarioConfig, ModelConfig, FLConfig,
                              IncrementalConfig, ExperimentConfig],
                path: str) -> None:
    """Save configuration to JSON file.

    Args:
        config: Configuration dataclass instance.
        path: Path to save JSON file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)


def load_config(config_class, path: str):
    """Load configuration from JSON file.

    Args:
        config_class: Configuration dataclass type.
        path: Path to JSON file.

    Returns:
        Configuration instance.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return config_class(**data)
