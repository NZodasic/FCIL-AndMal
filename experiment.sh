#!/usr/bin/env bash
# ==============================================================================
# Master Automated Experiment Runner for FCIL on CIC-AndMal-2020
# 10 Required Experiments:
#   Centralized (4): EWCDR, MES, SPCIL, MALFSIL
#   FL (6): EWCDR×{K20,K50}, MES×{K20,K50}, SPCIL×{K20,K50},
#            MALFSIL×{K20,K50}, FedAvg×{K20,K50}, FedNova×{K20,K50}
#
# Backbone: hybrid_tcn_cnn (TCN + CNN) for all experiments.
# Batch sizes: FL=256, Centralized=1024.
# FL:    5 tasks | 50 rounds/task | 5 epochs/round | K=20 and K=50
# Central: 5 tasks | 250 epochs/task | test every 5 epochs
# ==============================================================================

set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$(pwd)/.." && export PYTHONPATH

# ──────────────────────────────────────────────────────────────────────────────
# Global experiment hyper-parameters (fixed per spec)
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR="./raw_data"
PREPARED_DIR="./prepared_data"
PARTITION_DIR="./fl_data_partitions"
OUTPUT_ROOT="./EXPERIMENT"

FEATURE="dynamic"        # TCN+CNN backbone requires dynamic features
BACKBONE="hybrid_tcn_cnn"
SEED=42

# Centralized
CENTRAL_EPOCHS=250       # 250 epochs/task; eval every 5 epochs (handled inside trainer)
CENTRAL_BATCH=1024

# Federated
FL_ROUNDS=50             # 50 rounds/task
FL_LOCAL_EPOCHS=5        # 5 epochs/round
FL_BATCH=256

echo "=============================================================================="
echo "  FCIL on CIC-AndMal-2020  ·  10-Experiment Benchmark Runner"
echo "  Backbone : $BACKBONE"
echo "  Feature  : $FEATURE"
echo "  Seed     : $SEED"
echo "=============================================================================="

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 0: Dataset preparation & Dirichlet partitioning
# ──────────────────────────────────────────────────────────────────────────────
echo -e "\n[Stage 0] Preparing dataset and partitions..."

python3 -m data.prepare_dataset \
    --root "$DATA_DIR" \
    --output_dir "$PREPARED_DIR" \
    --type all \
    --seed "$SEED"

for K in 20 50; do
    python3 -m data.partition \
        --dataset "$PREPARED_DIR/$FEATURE/train.parquet" \
        --feature_type "$FEATURE" \
        --n_clients "$K" \
        --dirichlet_alpha 0.5 \
        --seed "$SEED" \
        --output_dir "$PARTITION_DIR"
done

echo -e "\n[Stage 0] ✅ Dataset and partitions ready."

# ──────────────────────────────────────────────────────────────────────────────
# Helper: run a centralized experiment
#   $1 = IL method name   $2 = extra args (optional)
# ──────────────────────────────────────────────────────────────────────────────
run_central() {
    local METHOD="$1"
    shift
    echo -e "\n[Centralized] Method=${METHOD^^} | Epochs=${CENTRAL_EPOCHS} | Batch=${CENTRAL_BATCH}"
    python3 main.py \
        --mode centralized \
        --feature_type "$FEATURE" \
        --backbone "$BACKBONE" \
        --method "$METHOD" \
        --rounds_per_task "$CENTRAL_EPOCHS" \
        --output_root "$OUTPUT_ROOT" \
        --seed "$SEED" \
        "$@"
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper: run a federated experiment
#   $1 = IL method   $2 = aggregator   $3 = n_clients   $4+ = extra args
# ──────────────────────────────────────────────────────────────────────────────
run_fl() {
    local METHOD="$1"
    local AGG="$2"
    local K="$3"
    shift 3
    echo -e "\n[FL] Method=${METHOD^^} | Agg=${AGG^^} | K=${K} | Rounds=${FL_ROUNDS} | LocalEpochs=${FL_LOCAL_EPOCHS} | Batch=${FL_BATCH}"
    python3 main.py \
        --mode federated \
        --feature_type "$FEATURE" \
        --backbone "$BACKBONE" \
        --method "$METHOD" \
        --aggregator "$AGG" \
        --n_clients "$K" \
        --dirichlet_alpha 0.5 \
        --rounds_per_task "$FL_ROUNDS" \
        --local_epochs "$FL_LOCAL_EPOCHS" \
        --output_root "$OUTPUT_ROOT" \
        --seed "$SEED" \
        "$@"
}

# ==============================================================================
# CENTRALIZED EXPERIMENTS (4 cases)
# ==============================================================================
echo -e "\n======================================================================"
echo "  CENTRALIZED EXPERIMENTS"
echo "======================================================================"

# Case 1: Centralized_EWCDR  (EWC with Distillation+Replay = ewc method)
run_central ewc \
    --ewc_lambda 5000.0

# Case 2: Centralized_MES  (Method: replay / Memory Experience Sampling)
run_central replay \
    --buffer_size 20

# Case 3: Centralized_SPCIL
run_central spcil

# Case 4: Centralized_MALFSIL
run_central malfsil

# ==============================================================================
# FEDERATED EXPERIMENTS (6 cases × 2 client settings = 12 runs)
# ==============================================================================
echo -e "\n======================================================================"
echo "  FEDERATED EXPERIMENTS"
echo "======================================================================"

for K in 20 50; do
    echo -e "\n--- FL with K=${K} clients ---"

    # Case 5 (K=20) / Case 5b (K=50): FL_EWCDR
    run_fl ewc fedavg "$K" --ewc_lambda 5000.0

    # Case 6 (K=20) / Case 6b (K=50): FL_MES
    run_fl replay fedavg "$K" --buffer_size 20

    # Case 7 (K=20) / Case 7b (K=50): FL_SPCIL
    run_fl spcil fedavg "$K"

    # Case 8 (K=20) / Case 8b (K=50): FL_MALFSIL
    run_fl malfsil fedavg "$K"

    # Case 9 (K=20) / Case 9b (K=50): FL_FedAvg  (finetune + FedAvg aggregation)
    run_fl finetune fedavg "$K"

    # Case 10 (K=20) / Case 10b (K=50): FL_FedNova
    run_fl finetune fednova "$K"
done

echo -e "\n======================================================================"
echo "  ✅ ALL 10 EXPERIMENTS COMPLETED  ·  Results in: $OUTPUT_ROOT"
echo "======================================================================"
