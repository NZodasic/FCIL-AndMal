#!/usr/bin/env bash
# ==============================================================================
# Runner for Failed EWCDR Benchmark Experiments on Mac (Apple Silicon M-Series) / Linux
#
# Runs:
#   1. Centralized_EWCDR (K=20)
#   2. FL_EWCDR_K20      (K=20)
#   3. Centralized_EWCDR (K=50)
#   4. FL_EWCDR_K50      (K=50)
#
# Usage:
#   ./run_ewcdr.sh                      # Default: cpu (recommended for Mac M3)
#   ./run_ewcdr.sh mps                  # Use Apple Silicon Metal (MPS)
#   ./run_ewcdr.sh cuda                 # Use NVIDIA GPU (if on CUDA machine)
#
# Additional options:
#   ./run_ewcdr.sh cpu --dry_run        # Preview commands without executing
#   ./run_ewcdr.sh cpu --clients 20     # Run only K=20 scenario
#   ./run_ewcdr.sh cpu --clients 50     # Run only K=50 scenario
# ==============================================================================

set -euo pipefail

# Determine device: command-line argument $1, or environment variable DEVICE, or default to "cpu"
DEVICE="${1:-${DEVICE:-cpu}}"
shift || true

CLIENTS_ARGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clients)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CLIENTS_ARGS+=("$1")
                shift
            done
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Default clients if not specified: 20 and 50
if [ ${#CLIENTS_ARGS[@]} -eq 0 ]; then
    CLIENTS_ARGS=(20 50)
fi

echo "======================================================================"
echo "  FCIL-AndMal: Running Failed EWCDR Cases"
echo "  Target Device : ${DEVICE}"
echo "  Clients       : ${CLIENTS_ARGS[*]}"
echo "======================================================================"

# Run via run_all.py filtering specifically to EWCDR
python3 run_all.py \
    --only EWCDR \
    --clients "${CLIENTS_ARGS[@]}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo -e "\n======================================================================"
echo "  ✅ EWCDR experiments completed."
echo "  Check logs and results in: ./EXPERIMENT/"
echo "======================================================================"
