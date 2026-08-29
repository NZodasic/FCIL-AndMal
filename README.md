# FCIL-AndMal: Federated Class-Incremental Learning for Android Malware Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: FCIL](https://img.shields.io/badge/Framework-FCIL%20%7C%20MalFSCIL-green.svg)](#)

A comprehensive academic research platform for **Federated Class-Incremental Learning (FCIL)** and **Few-Shot Class-Incremental Learning (MalFSCIL)** applied to Android malware family detection on the **CIC-AndMal-2020** benchmark.

---

## 📌 Overview

Modern Android malware detection faces two core operational challenges:
1. **Temporal Evolution**: Malware variants evolve continuously, introducing novel family classes sequentially over time. Static models suffer from **catastrophic forgetting** when updated.
2. **Spatial Privacy & Distribution**: Malware telemetry is fragmented across edge clients (devices, enterprise nodes, security vendors) and cannot be centralized due to strict data privacy regulations.

**FCIL-AndMal** unifies state-of-the-art Continual Learning (CIL), Few-Shot Learning (FSCIL), and Federated Aggregation (FedAvg, FedNova) under realistic Dirichlet non-IID label skew.

---

## ✨ Key Features

- **5-Task Class-Incremental Benchmark**: 15 standardized malware and benign classes split across 5 sequential tasks.
- **Multi-Modal Feature Fusion**: Supports **Static**, **Dynamic**, and **Fused** (multi-modal static + dynamic alignment) representations.
- **Non-IID Federated Partitioning**: Dirichlet distribution ($\alpha = 0.5$) with progressive client participation ($12 \to 14 \to 16 \to 18 \to 20$ active clients).
- **State-of-the-Art Baselines**:
  - *Centralized CIL*: Fine-tuning, Joint Training, Elastic Weight Consolidation (**EWC**), Learning without Forgetting (**LwF**), Exemplar Replay, and **SPCIL**.
  - *Few-Shot CIL*: **MalFSCIL** (Variational Reconstruction + Angular Margin ArcFace + Graph Attention Prototypes).
  - *Federated Aggregation*: **FedAvg** and **FedNova**.
- **Automated Logging & Excel Reporting**: Exports standard metrics (Macro-F1, Accuracy, Average Forgetting, Malware F1) into auto-formatted Excel reports (`evaluation_results.xlsx`) and PyTorch model checkpoints.

---

## 📂 Project Architecture

```
FCIL-AndMal/
├── config/                 # Central configuration schemas & task mappings
│   ├── config.py           # Dataclass configs (ScenarioConfig, ModelConfig, FLConfig, etc.)
│   ├── task_config.py      # 5-Task label map (15 malware/benign classes)
│   └── paths.py            # Path resolution utilities
├── data/                   # Data pipeline & dataset loaders
│   ├── prepare_dataset.py  # Stage 1: Merge, normalize & stratify held-out test splits
│   ├── partition.py        # Stage 2: Dirichlet non-IID client partitioning
│   ├── dataset.py          # PyTorch TabularMalwareDataset & FLTaskDataset loaders
│   ├── schema.py           # Feature schema validation & metadata cleaning
│   └── synthetic_generator.py # Synthetic data generator for development/testing
├── models/                 # Neural architectures & backbones
│   ├── fcil_model.py       # Primary FCIL model router & feature extractors
│   ├── backbones.py        # MLP, 1D-CNN, TCN, and Fused backbones
│   └── fused_model.py      # Multi-modal fusion networks
├── methods/                # Continual & Few-Shot CIL algorithms
│   ├── base.py             # Abstract base class for CIL methods
│   ├── ewc.py              # Elastic Weight Consolidation
│   ├── lwf.py              # Learning without Forgetting
│   ├── replay.py           # Exemplar Replay
│   ├── spcil.py            # Self-Paced Class-Incremental Learning
│   └── malfscil.py         # MalFSCIL Few-Shot CIL strategy
├── federated/              # Federated simulation environment
│   ├── client.py           # Local FL client optimizer
│   ├── server.py           # Global FL server orchestrator
│   └── aggregators/        # FedAvg & FedNova aggregation logic
├── training/               # Centralized & Federated trainers
│   ├── trainer.py          # Centralized continual trainer
│   ├── evaluator.py        # Continual matrix evaluator & metric tracking
│   └── checkpoint.py       # PyTorch model checkpoint manager
├── main.py                 # Primary entry point & CLI orchestrator
├── requirements.txt        # Python dependencies
├── README.md               # English documentation
└── README_VN.md            # Vietnamese documentation
```

---

## 📅 5-Task Benchmark Schedule

The 15 standardized classes are partitioned across 5 incremental tasks as follows:

| Task | Introduced Classes | Active Clients ($K=20$) | Cumulative Classes |
| :---: | :--- | :---: | :---: |
| **Task 1** | `Benign`, `PUA`, `Backdoor` | 12 / 20 | 3 |
| **Task 2** | `Adware`, `TrojanBanker`, `TrojanSpy` | 14 / 20 | 6 |
| **Task 3** | `NoCategory`, `Trojan`, `Riskware` | 16 / 20 | 9 |
| **Task 4** | `FileInfector`, `Ransomware`, `TrojanDropper` | 18 / 20 | 12 |
| **Task 5** | `Scareware`, `ZeroDay`, `TrojanSMS` | 20 / 20 | 15 |

---

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-org/FCIL-AndMal.git
   cd FCIL-AndMal
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Data Pipeline Workflow

### **Stage 1: Prepare Raw Datasets**
Process raw CSV files, normalize feature columns, align static/dynamic features, and generate stratified held-out test splits:

```bash
python3 -m data.prepare_dataset \
  --root /path/to/raw_dataset \
  --output_dir /path/to/prepared_data \
  --type all \
  --summary
```

### **Stage 2: Generate Non-IID Dirichlet Partitions**
Partition the prepared data into client slices across the 5 incremental tasks:

```bash
# Partition Fused Modality (20 Clients)
python3 -m data.partition \
  --dataset /path/to/prepared_data/fused/train.parquet \
  --feature_type fused \
  --n_clients 20 \
  --dirichlet_alpha 0.5 \
  --output_dir /path/to/fl_data_partitions

# Partition Dynamic Modality (20 Clients)
python3 -m data.partition \
  --dataset /path/to/prepared_data/dynamic/train.parquet \
  --feature_type dynamic \
  --n_clients 20 \
  --dirichlet_alpha 0.5 \
  --output_dir /path/to/fl_data_partitions

# Partition Static Modality (20 Clients)
python3 -m data.partition \
  --dataset /path/to/prepared_data/static/train.parquet \
  --feature_type static \
  --n_clients 20 \
  --dirichlet_alpha 0.5 \
  --output_dir /path/to/fl_data_partitions
```

---

## 💻 Running Experiments

Execute experiments using `main.py`. Change `--device cuda` to `--device cpu` or `--device mps` as appropriate for your computing environment.

### **1. Centralized Continual Learning Experiments**

#### **MalFSCIL (Primary Proposed Method - Fused Features)**
```bash
python3 main.py \
  --mode centralized \
  --feature_type fused \
  --method malfscil \
  --prepared_dir /path/to/prepared_data \
  --partition_dir /path/to/fl_data_partitions \
  --raw_root /path/to/raw_dataset \
  --rounds_per_task 50 \
  --device cpu
```

#### **Centralized Baselines (Finetune, EWC, LwF, Replay)**
```bash
# Fine-Tuning Baseline
python3 main.py --mode centralized --feature_type fused --method finetune \
  --prepared_dir /path/to/prepared_data --partition_dir /path/to/fl_data_partitions --device cpu

# Elastic Weight Consolidation (EWC)
python3 main.py --mode centralized --feature_type fused --method ewc --ewc_lambda 5000.0 \
  --prepared_dir /path/to/prepared_data --partition_dir /path/to/fl_data_partitions --device cpu

# Learning without Forgetting (LwF)
python3 main.py --mode centralized --feature_type fused --method lwf --lwf_temp 2.0 --lwf_alpha 1.0 \
  --prepared_dir /path/to/prepared_data --partition_dir /path/to/fl_data_partitions --device cpu

# Exemplar Replay
python3 main.py --mode centralized --feature_type fused --method replay --buffer_size 20 \
  --prepared_dir /path/to/prepared_data --partition_dir /path/to/fl_data_partitions --device cpu
```

---

### **2. Federated Class-Incremental Learning (FCIL)**

#### **FedAvg + Exemplar Replay**
```bash
python3 main.py \
  --mode federated \
  --feature_type fused \
  --method replay \
  --aggregator fedavg \
  --n_clients 20 \
  --local_epochs 5 \
  --rounds_per_task 50 \
  --prepared_dir /path/to/prepared_data \
  --partition_dir /path/to/fl_data_partitions \
  --device cpu
```

#### **FedNova + Exemplar Replay**
```bash
python3 main.py \
  --mode federated \
  --feature_type fused \
  --method replay \
  --aggregator fednova \
  --n_clients 20 \
  --local_epochs 5 \
  --rounds_per_task 50 \
  --prepared_dir /path/to/prepared_data \
  --partition_dir /path/to/fl_data_partitions \
  --device cpu
```

---

## 📊 Evaluation & Artifact Output

All execution metrics, configs, logs, and checkpoints are stored automatically inside `./EXPERIMENT/`:

```
EXPERIMENT/
├── evaluation_results.xlsx    # Central Excel report detailing Accuracy, Macro-F1, Malware F1, and Forgetting
└── <EXPERIMENT_CASE_NAME>/
    ├── experiment_config.json # Hyperparameter snapshot
    ├── academic_experiment.log # Complete console execution log
    └── checkpoints/           # Saved PyTorch model weights (.pt) per task session
```

---

## 📝 Citation

If you use this benchmark or codebase in your research, please cite:

```bibtex
@article{fcil_andmal2026,
  title={Federated Class-Incremental Learning for Android Malware Family Detection},
  author={Research Team},
  journal={Academic Research Platform},
  year={2026}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).
