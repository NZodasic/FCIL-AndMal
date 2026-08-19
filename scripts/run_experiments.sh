#!/bin/bash
# Run FCIL-AndroidMalware experiments

set -e

echo "=========================================="
echo "FCIL-AndroidMalware Experiment Runner"
echo "=========================================="

# Default configurations
N_CLIENTS="20 50"
FEATURE_TYPES="dynamic static fused"
STRATEGIES="finetune joint ewc lwf replay spcil malfsil"
AGGREGATORS="fedavg fednova"
N_ROUNDS=50
N_EPOCHS=5
N_TASKS=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_clients)
            N_CLIENTS="$2"
            shift 2
            ;;
        --feature_types)
            FEATURE_TYPES="$2"
            shift 2
            ;;
        --strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        --aggregators)
            AGGREGATORS="$2"
            shift 2
            ;;
        --n_rounds)
            N_ROUNDS="$2"
            shift 2
            ;;
        --n_epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n_clients CLIENTS      Number of clients (default: '20 50')"
            echo "  --feature_types TYPES    Feature types (default: 'dynamic static fused')"
            echo "  --strategies STRATEGIES  Incremental strategies"
            echo "  --aggregators AGGS       Aggregation methods"
            echo "  --n_rounds N             Number of FL rounds per task"
            echo "  --n_epochs N             Number of local epochs"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Clients: $N_CLIENTS"
echo "  Feature types: $FEATURE_TYPES"
echo "  Strategies: $STRATEGIES"
echo "  Aggregators: $AGGREGATORS"
echo "  Rounds per task: $N_ROUNDS"
echo "  Local epochs: $N_EPOCHS"
echo ""

# Check if data is prepared
if [ ! -d "./prepared_data" ]; then
    echo "Error: Prepared data not found. Run data preparation first:"
    echo "  python data/prepare_dataset.py --root /path/to/CIC-AndMal-2020 --output_dir ./prepared_data"
    exit 1
fi

# Run experiments
for n_clients in $N_CLIENTS; do
    for feature_type in $FEATURE_TYPES; do
        for strategy in $STRATEGIES; do
            for aggregator in $AGGREGATORS; do
                exp_name="fcil_${feature_type}_k${n_clients}_${strategy}_${aggregator}"

                echo ""
                echo "------------------------------------------"
                echo "Running experiment: $exp_name"
                echo "------------------------------------------"

                python experiments/run_fl.py \
                    --experiment_name "$exp_name" \
                    --feature_type "$feature_type" \
                    --n_clients "$n_clients" \
                    --strategy "$strategy" \
                    --aggregator "$aggregator" \
                    --n_rounds "$N_ROUNDS" \
                    --n_epochs "$N_EPOCHS" \
                    --n_tasks "$N_TASKS" \
                    2>&1 | tee "logs/${exp_name}.log"

                echo "Experiment $exp_name completed"
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
