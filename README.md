# FCIL-AndroidMalware

**Federated Class-Incremental Learning for Android Malware Family Detection on CIC-AndMal-2020**

## Overview

This repository implements a comprehensive research framework for **Federated Class-Incremental Learning (FCIL)** applied to Android malware family detection. The framework addresses two key challenges in real-world malware detection:

1. **Temporal Evolution**: New malware families emerge continuously, requiring models to learn sequentially without catastrophic forgetting
2. **Spatial Distribution**: Data cannot be centralized due to privacy, ownership, and bandwidth constraints

## Key Features

- **5-Task Incremental Learning**: 15 malware families (Benign + 14 families) split across 5 sequential tasks
- **Federated Learning**: Multiple clients with non-IID (Dirichlet α=0.5) data distribution
- **Multiple Feature Types**: Static (9503D), Dynamic (141/282D), and Fused
- **Comprehensive Strategies**: Fine-tune, Joint, EWC, LwF, Replay, SPCIL, and MalFSCIL
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
python3 -m data.prepare_dataset \
    --root /home/raymond/Desktop/AndMal-IDS/Dataset \
    --output_dir ./prepared_data \
    --type dynamic \
    --summary
```

### Stage 2: Create FL Partitions

```bash
# Static features
python3 -m data.partition \
    --dataset ./prepared_data/static/train.parquet \
    --feature_type static \
    --n_clients 20 50

# Dynamic features
python3 -m data.partition \
    --dataset ./prepared_data/dynamic/train.parquet \
    --feature_type dynamic \
    --n_clients 20 50

# Fused features
python3 -m data.partition \
    --static_dataset ./prepared_data/static/train.parquet \
    --dynamic_dataset ./prepared_data/dynamic/train.parquet \
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
│   └── malfsil.py         # Deprecated legacy compatibility method
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

The main entry point prepares and partitions missing data automatically. For a
data-only validation against the local dynamic dataset:

```bash
python3 main.py \
    --raw_root /home/raymond/Desktop/AndMal-IDS/Dataset \
    --feature_type dynamic \
    --prepare_only
```

Then run a short centralized training check:

```bash
python3 main.py \
    --raw_root /home/raymond/Desktop/AndMal-IDS/Dataset \
    --feature_type dynamic \
    --mode centralized \
    --method finetune \
    --rounds_per_task 1 \
    --device cpu
```

The supplied dynamic files contain 14 malware families but no Benign samples.
The pipeline reports this explicitly; such a run is useful for development but
is not the complete 15-class benchmark described in this README.

```bash
# Run all experiments
bash scripts/run_experiments.sh

# Export source code
bash scripts/export_code.sh
```

### MalFSCIL

`MalFSCIL` implements the two-stage method from Chai et al., adapted from
malware images to this project's tabular static/dynamic features:

1. The base session jointly optimizes classification and a variational
   reconstruction objective.
2. Each later session is restricted to an `N`-way `K`-shot support set.
3. The feature extractor is frozen after the base session.
4. Support features initialize class prototypes, which are evolved through
   graph attention and optimized on masking-derived query samples using
   softmax and additive angular-margin losses.

The initial implementation targets the centralized FSCIL trainer. The paper
does not define federated optimization; the older `malfsil` federated strategy
is retained only as a compatibility experiment and is not the paper method.

```bash
python3 main.py \
    --mode centralized \
    --method malfscil \
    --feature_type dynamic \
    --fscil_n_way 3 \
    --fscil_k_shot 5 \
    --fscil_query_per_class 5 \
    --fscil_mask_probability 0.1
```

The deprecated CLI spelling `--method malfsil` resolves to `malfscil` in the
current trainer so existing centralized commands remain usable.

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Macro, micro, and weighted averages
- **Recall**: Macro, micro, and weighted averages
- **F1-score**: Macro, micro, and weighted averages
- **Forgetting**: Drop in performance on old classes
- **Backward Transfer (BWT)**: Influence of learning new tasks on old tasks
- **Communication Cost**: MB transmitted per task

Task-final classification results are written to `EXPERIMENT/evaluation_results.xlsx`.
Each experiment case has its own sheet, and each task has one row with the held-out
test location, confusion-matrix path, and checkpoint path. Federated rows use
cumulative global rounds (50, 100, 150, 200, and 250 with the default setup);
centralized rows use the final epoch. The requested `patch_size` column records the
configured batch size because this project evaluates tabular features rather than
image patches.

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
6. Chai et al. "MalFSCIL: A Few-Shot Class-Incremental Learning Approach for Malware Detection" (IEEE TIFS 2024), DOI: 10.1109/TIFS.2024.3516565
