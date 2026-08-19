# FCIL-AndroidMalware

**Federated Class-Incremental Learning for Android Malware Family Detection on CIC-AndMal-2020**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Academic-green.svg)]()

## Overview

This repository implements a comprehensive research framework for **Federated Class-Incremental Learning (FCIL)** applied to Android malware family detection. The framework addresses two key challenges in real-world malware detection:

1. **Temporal Evolution**: New malware families emerge continuously, requiring models to learn sequentially without catastrophic forgetting
2. **Spatial Distribution**: Data cannot be centralized due to privacy, ownership, and bandwidth constraints

## Key Features

- **5-Task Incremental Learning**: 15 malware families (Benign + 14 families) split across 5 sequential tasks
- **Federated Learning**: Multiple clients with non-IID (Dirichlet α=0.5) data distribution
- **Multiple Feature Types**: Static (9503D), Dynamic (141/282D), and Fused
- **Comprehensive Strategies**: Fine-tune, Joint, EWC, LwF, Replay, SPCIL, and MALFSIL (proposed)
- **Aggregation Methods**: FedAvg and FedNova
- **Model Architectures**: CNN, TCN, and Fused models

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/fcil-android-malware.git
cd fcil-android-malware

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### Stage 1: Merge and Normalize

```bash
python data/prepare_dataset.py \
    --root /path/to/CIC-AndMal-2020 \
    --output_dir ./prepared_data \
    --type both \
    --summary
```

### Stage 2: Create FL Partitions

```bash
# Static features
python -m data.fl_data_partition.run \
    --dataset ./prepared_data/static/static_all.csv \
    --feature_type static \
    --n_clients 20 50

# Dynamic features
python -m data.fl_data_partition.run \
    --dataset ./prepared_data/dynamic/dynamic_all.csv \
    --feature_type dynamic \
    --n_clients 20 50

# Fused features
python -m data.fl_data_partition.run \
    --static_dataset ./prepared_data/static/static_all.csv \
    --dynamic_dataset ./prepared_data/dynamic/dynamic_all.csv \
    --feature_type fused \
    --n_clients 20 50
```

### Visualize Partitions

```bash
python -m data.visualization.visualize_partitions \
    --base_dir ./fl_data_partitions
```

## Project Structure

```
fcil_android_malware/
├── config/                 # Configuration and task definitions
│   ├── config.py          # Scenario, Model, FL, Experiment configs
│   ├── task_config.py     # 5-task label mapping (15 labels)
│   └── paths.py           # Path utilities
├── data/                   # Data pipeline
│   ├── prepare_dataset.py # Stage 1: Merge and normalize
│   ├── fl_data_partition/ # Stage 2: FL partitioning
│   └── visualization/     # Partition visualization
├── models/                 # Model architectures
│   ├── base_model.py      # Incremental model base class
│   ├── static_cnn.py      # Static feature models
│   ├── dynamic_cnn.py     # Dynamic feature models (CNN/TCN)
│   ├── fused_model.py     # Static + Dynamic fusion
│   └── layers/            # Custom layers (TCN, Capsule)
├── incremental/            # Incremental learning strategies
│   ├── base_strategy.py   # Base strategy class
│   ├── fine_tune.py       # Baseline: sequential fine-tuning
│   ├── joint.py           # Upper bound: joint training
│   ├── ewc.py             # Elastic Weight Consolidation
│   ├── lwf.py             # Learning without Forgetting
│   ├── replay.py          # Experience Replay
│   ├── spcil.py           # Self-Paced CIL
│   └── malfsil.py         # Proposed method
├── federated/              # Federated learning components
│   ├── client.py          # FL Client
│   ├── server.py          # FL Server
│   └── aggregators/       # FedAvg, FedNova
├── training/               # Training utilities
│   ├── metrics.py         # Evaluation metrics
│   └── evaluator.py       # Model evaluator
├── utils/                  # Utilities
│   ├── checkpoint.py      # Checkpoint management
│   └── logging.py         # Structured logging
├── experiments/            # Experiment runners
├── scripts/                # Utility scripts
├── requirements.txt
└── README.md
```

## Task Configuration

| Task | Labels | Cumulative Classes |
|------|--------|-------------------|
| T1   | Benign, PUA, Backdoor | 3 |
| T2   | Adware, TrojanBanker, TrojanSpy | 6 |
| T3   | NoCategory, Trojan, Riskware | 9 |
| T4   | FileInfector, Ransomware, TrojanDropper | 12 |
| T5   | Scareware, ZeroDay, TrojanSMS | 15 |

## Usage Example

```python
from config import ScenarioConfig, ModelConfig, FLConfig
from data.fl_data_partition.dataset_api import FLTaskDataset, get_participating_clients
from models.dynamic_cnn import DynamicCNN
from incremental.ewc import EWC
from federated.client import FLClient
from federated.server import FLServer

# Load data
scenario_dir = "./fl_data_partitions/dynamic/20clients"
for task_id in range(5):
    active_clients = get_participating_clients(scenario_dir, task_id)

    for cid in active_clients:
        dataset = FLTaskDataset(scenario_dir, task_id, cid)
        dataloader = dataset.as_dataloader(batch_size=256)

        # Create model and strategy
        model = DynamicCNN(input_dim=141, initial_classes=3)
        strategy = EWC(model, optimizer_fn=lambda p: torch.optim.Adam(p, lr=0.001))

        # Create client
        client = FLClient(cid, model, strategy)
```

## Running Experiments

```bash
# Run all experiments
bash scripts/run_experiments.sh

# Export source code
bash scripts/export_code.sh
```

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Macro-F1**: Macro-averaged F1 score (important for imbalanced malware data)
- **Forgetting**: Drop in performance on old classes
- **Backward Transfer (BWT)**: Influence of learning new tasks on old tasks
- **Communication Cost**: MB transmitted per task

## Citation

```bibtex
@article{fcil_andmal2026,
  title={Federated Class-Incremental Learning for Android Malware Family Detection},
  author={Research Team},
  journal={arXiv preprint},
  year={2026}
}
```

## References

1. Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)
2. Li \& Hoiem. "Learning without Forgetting" (TPAMI 2017)
3. Rebuffi et al. "iCaRL: Incremental Classifier and Representation Learning" (CVPR 2017)
4. McMahan et al. "Communication-Efficient Learning of Deep Networks" (AISTATS 2017)
5. Wang et al. "FedNova: Normalized Averaging" (NeurIPS 2020)

## License

This project is for academic research purposes.

## Contact

For questions or issues, please open a GitHub issue or contact the research team.
