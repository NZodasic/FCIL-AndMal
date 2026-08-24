#!/usr/bin/env bash
# ==============================================================================
# Master Automated Experiment Benchmark Runner for FCIL on CIC-AndMal-2020
# Implements Centralized Baselines, Federated Aggregation (FedAvg, FedNova),
# Feature Modalities (Dynamic, Static, Fused), and IL Algorithms (MalFSCIL, SPCIL, Replay, EWC, LwF)
# ==============================================================================

set -e

export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=$(pwd)/.. export PYTHONPATH

echo "=============================================================================="
echo "  FEDERATED CLASS-INCREMENTAL LEARNING (FCIL) ON CIC-ANDMAL-2020 BENCHMARK"
echo "=============================================================================="

DATA_DIR="./raw_data"
PREPARED_DIR="./prepared_data"
PARTITION_DIR="./fl_data_partitions"
OUTPUT_ROOT="./EXPERIMENT"
ROUNDS=50
EPOCHS=5
BATCH_SIZE=256
SEED=42

# ------------------------------------------------------------------------------
# STEP 1: PREPARATION & PARTITIONING (If not already present)
# ------------------------------------------------------------------------------
echo -e "\n[Pipeline Stage 1 & 2] Ensuring Datasets and Dirichlet Non-IID Partitions..."
python3 -m data.prepare_dataset \
    --root "$DATA_DIR" \
    --output_dir "$PREPARED_DIR" \
    --type all \
    --seed $SEED

python3 -m data.partition \
    --dataset "$PREPARED_DIR/dynamic/train.parquet" \
    --feature_type dynamic \
    --n_clients 20 50 \
    --dirichlet_alpha 0.5 \
    --seed $SEED \
    --output_dir "$PARTITION_DIR"

python3 -m data.partition \
    --dataset "$PREPARED_DIR/static/train.parquet" \
    --feature_type static \
    --n_clients 20 50 \
    --dirichlet_alpha 0.5 \
    --seed $SEED \
    --output_dir "$PARTITION_DIR"

python3 -m data.partition \
    --static_dataset "$PREPARED_DIR/static/train.parquet" \
    --dynamic_dataset "$PREPARED_DIR/dynamic/train.parquet" \
    --feature_type fused \
    --n_clients 20 50 \
    --dirichlet_alpha 0.5 \
    --seed $SEED \
    --output_dir "$PARTITION_DIR"

# Visualize Dirichlet Heatmaps
python3 -m data.visualize_partitions --base_dir "$PARTITION_DIR"

# ------------------------------------------------------------------------------
# STEP 2: CENTRALIZED CONTINUAL LEARNING BASELINES (Upper & Lower Bounds)
# ------------------------------------------------------------------------------
echo -e "\n[Experiment Step 2] Running Centralized Lower Bound (Fine-tune) and Upper Bound (Joint)..."
python3 main.py \
    --mode centralized \
    --feature_type dynamic \
    --method finetune \
    --rounds_per_task $ROUNDS \
    --batch_size $BATCH_SIZE \
    --output_root "$OUTPUT_ROOT" \
    --seed $SEED

echo -e "\n[Experiment Step 2] Running paper-aligned MalFSCIL (3-Way 5-Shot)..."
python3 main.py \
    --mode centralized \
    --feature_type dynamic \
    --method malfscil \
    --fscil_k_shot 5 \
    --fscil_query_per_class 5 \
    --fscil_mask_probability 0.1 \
    --rounds_per_task $ROUNDS \
    --batch_size $BATCH_SIZE \
    --output_root "$OUTPUT_ROOT" \
    --seed $SEED

python3 main.py \
    --mode centralized \
    --feature_type dynamic \
    --method joint \
    --rounds_per_task $ROUNDS \
    --batch_size $BATCH_SIZE \
    --output_root "$OUTPUT_ROOT" \
    --seed $SEED

# ------------------------------------------------------------------------------
# STEP 3: FEDERATED CLASS-INCREMENTAL BENCHMARKS (K=20 Clients, FedAvg)
# ------------------------------------------------------------------------------
METHODS=("finetune" "ewc" "lwf" "replay" "spcil")

for METHOD in "${METHODS[@]}"; do
    echo -e "\n[Experiment Step 3] Running FL FCIL: Method=${METHOD^^} | Aggregator=FedAvg | K=20..."
    python3 main.py \
        --mode federated \
        --feature_type dynamic \
        --method "$METHOD" \
        --aggregator fedavg \
        --n_clients 20 \
        --dirichlet_alpha 0.5 \
        --rounds_per_task $ROUNDS \
        --local_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --buffer_size 20 \
        --output_root "$OUTPUT_ROOT" \
        --seed $SEED
done

# ------------------------------------------------------------------------------
# STEP 4: MALFSCIL FEATURE ABLATION
# ------------------------------------------------------------------------------
for FEAT in "static" "fused"; do
    echo -e "\n[Experiment Step 4] Running MalFSCIL on Feature=${FEAT^^}..."
    python3 main.py \
        --mode centralized \
        --feature_type "$FEAT" \
        --method malfscil \
        --fscil_k_shot 5 \
        --rounds_per_task $ROUNDS \
        --batch_size $BATCH_SIZE \
        --output_root "$OUTPUT_ROOT" \
        --seed $SEED
done

# ------------------------------------------------------------------------------
# STEP 5: MALFSCIL SHOT ABLATION
# ------------------------------------------------------------------------------
for SHOTS in 1 10; do
    echo -e "\n[Experiment Step 5] Running MalFSCIL ${SHOTS}-shot ablation..."
    python3 main.py \
        --mode centralized \
        --feature_type dynamic \
        --method malfscil \
        --fscil_k_shot "$SHOTS" \
        --rounds_per_task $ROUNDS \
        --batch_size $BATCH_SIZE \
        --output_root "$OUTPUT_ROOT" \
        --seed $SEED
done

echo -e "\n=============================================================================="
echo "  ✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY! RESULTS IN: $OUTPUT_ROOT"
echo "=============================================================================="
