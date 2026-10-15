#!/usr/bin/env bash
# ==============================================================================
# Master Scenario Runner for FCIL-AndMal Research Benchmark
# ==============================================================================
# Hỗ trợ chạy toàn bộ các kịch bản:
#   1. dynamic: 141-dim behavioral features (Backbone: hybrid_tcn_cnn, TCN + CNN)
#   2. static:  300-dim manifest & permission features (Backbone: mlp)
#   3. fused:   441-dim multi-modal features (Backbone: fused)
#   4. all:     Chạy tuần tự toàn bộ cả 3 dạng đặc trưng
#
# Đối với mỗi dạng đặc trưng, script sẽ chạy:
#   - Toàn bộ 5 bài toán Centralized (FineTune, EWCDR, MES, SPCIL, MALFSIL)
#   - Toàn bộ 6 bài toán Federated Learning (FedAvg, FedNova, EWCDR, MES, SPCIL, MALFSIL)
#   - Cả 2 kịch bản phân tán: 20 clients (K=20) và 50 clients (K=50)
# ==============================================================================

set -euo pipefail

# Chuyển về thư mục gốc của repository
cd "$(dirname "$0")"

TARGET_SCENARIO="${1:-dynamic}"
DEVICE="${2:-auto}"
OUTPUT_ROOT="${3:-./EXPERIMENT}"
# Consume only positional arguments that were actually supplied.  A bare
# `shift 3` fails when callers omit the optional output root; bash then leaves
# every argument in "$@", causing the scenario and device to be forwarded to
# run_all.py as invalid extra arguments.
if (( $# >= 3 )); then
    shift 3
else
    shift "$#"
fi
EXTRA_ARGS=("$@")

echo "=============================================================================="
echo "  FCIL-AndMal2020 Master Scenario Training Pipeline"
echo "  Target Scenario : $TARGET_SCENARIO"
echo "  Device          : $DEVICE"
echo "  Output Root     : $OUTPUT_ROOT"
if (( ${#EXTRA_ARGS[@]} )); then
    echo "  Extra Flags     : ${EXTRA_ARGS[*]}"
fi
echo "=============================================================================="

run_scenario() {
    local FEATURE="$1"
    local BACKBONE="$2"
    echo -e "\n=============================================================================="
    echo "  STARTING BENCHMARK: FEATURE=$FEATURE | BACKBONE=$BACKBONE"
    echo "=============================================================================="

    python3 run_all.py \
        --feature_type "$FEATURE" \
        --backbone "$BACKBONE" \
        --clients 20 50 \
        --mode all \
        --device "$DEVICE" \
        --output_root "$OUTPUT_ROOT" \
        --allow_incomplete_benchmark \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

case "$TARGET_SCENARIO" in
    dynamic)
        run_scenario "dynamic" "hybrid_tcn_cnn"
        ;;
    static)
        run_scenario "static" "mlp"
        ;;
    fused)
        run_scenario "fused" "fused"
        ;;
    all)
        echo ">>> Running ALL Scenarios: Dynamic -> Static -> Fused..."
        run_scenario "dynamic" "hybrid_tcn_cnn"
        run_scenario "static" "mlp"
        run_scenario "fused" "fused"
        ;;
    *)
        echo "Lựa chọn không hợp lệ: $TARGET_SCENARIO"
        echo "Sử dụng: $0 [dynamic|static|fused|all] [device: cpu|cuda] [output_root]"
        exit 1
        ;;
esac

echo -e "\n=============================================================================="
echo "  TẤT CẢ KỊCH BẢN THỰC NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG!"
echo "  Báo cáo Excel: report/FCIL_AndMal.xlsx"
echo "  Biểu đồ tổng kết: report/figures/ và Paper/"
echo "=============================================================================="
