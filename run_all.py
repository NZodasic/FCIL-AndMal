#!/usr/bin/env python3
"""
FCIL-AndMal Full Pipeline Runner
=================================
Tự động:
  1. Kiểm tra / chuẩn bị dataset (Stage 1 + Stage 2) cho các kịch bản Client (20, 50 clients).
     Hỗ trợ đầy đủ các dạng đặc trưng: dynamic (141d), static (300d), hoặc fused multi-modal (441d).
  2. Chạy toàn bộ 10 bài toán thực nghiệm (4 Centralized + 6 FL) cho từng kịch bản Client.
  3. In đầy đủ metrics (Accuracy, Macro/Micro/Weighted P/R/F1) mỗi lần test.
  4. Confusion matrix chỉ ở round/epoch cuối mỗi task.
  5. Phân tách kết quả rõ ràng theo thư mục từng kịch bản client:
     ./EXPERIMENT/20clients/ và ./EXPERIMENT/50clients/

Usage:
    python run_all.py [--device cpu/cuda] [--clients 20 50]
                      [--feature_type dynamic/static/fused] [--backbone hybrid_tcn_cnn/fused/...]
                      [--output_root ./EXPERIMENT] [--skip_data_prep] [--dry_run]
                      [--only CASE_NAME] [--list]
"""

import sys, os
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ──────────────────────────────────────────────────────────────────────────────
GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
BLD  = "\033[1m"
RST  = "\033[0m"

def _c(color: str, text: str) -> str:
    return f"{color}{text}{RST}"

def banner(text: str) -> None:
    line = "=" * 70
    print(f"\n{BLD}{line}\n  {text}\n{line}{RST}")

def section(text: str) -> None:
    print(f"\n{CYN}{BLD}── {text} {'─'*(64-len(text))}{RST}")

def ok(text: str)   -> None: print(f"  {GRN}✔{RST}  {text}")
def warn(text: str) -> None: print(f"  {YLW}⚠{RST}  {text}")
def err(text: str)  -> None: print(f"  {RED}✘{RST}  {text}")
def info(text: str) -> None: print(f"  {CYN}→{RST}  {text}")


# ──────────────────────────────────────────────────────────────────────────────
# Core 10 Experiments suite
# ──────────────────────────────────────────────────────────────────────────────
# Each entry: (case_name, mode, method, aggregator_or_None, extra_kwargs)
CORE_EXPERIMENTS: List[Tuple] = [
    # ── Centralized (4 cases) ──────────────────────────────────────────────
    ("Centralized_EWCDR",   "centralized", "ewc",     None,      {"ewc_lambda": 5000.0}),
    ("Centralized_MES",     "centralized", "replay",  None,      {"buffer_size": 20}),
    ("Centralized_SPCIL",   "centralized", "spcil",   None,      {}),
    ("Centralized_MALFSIL", "centralized", "malfsil", None,      {}),
    # ── Federated (6 cases) ────────────────────────────────────────────────
    ("FL_EWCDR",            "federated",   "ewc",     "fedavg",  {"ewc_lambda": 5000.0}),
    ("FL_MES",              "federated",   "replay",  "fedavg",  {"buffer_size": 20}),
    ("FL_SPCIL",            "federated",   "spcil",   "fedavg",  {}),
    ("FL_MALFSIL",          "federated",   "malfsil", "fedavg",  {}),
    ("FL_FedAvg",           "federated",   "finetune","fedavg",  {}),
    ("FL_FedNova",          "federated",   "finetune","fednova", {}),
]

# Fixed hyper-parameters per spec
CENTRAL_EPOCHS  = 250
FL_ROUNDS       = 50
FL_LOCAL_EPOCHS = 5
SEED            = 42


# ──────────────────────────────────────────────────────────────────────────────
# Stage 0: Dataset check
# ──────────────────────────────────────────────────────────────────────────────
def check_dataset(prepared_dir: Path, partition_dir: Path, feature_type: str, clients_list: List[int]) -> Dict[str, bool]:
    """Return dict of readiness flags for stage 1 and each requested stage 2 partition."""
    test_parquet  = prepared_dir / feature_type / "test.parquet"
    train_parquet = prepared_dir / feature_type / "train.parquet"
    res = {
        "stage1_test":  test_parquet.is_file(),
        "stage1_train": train_parquet.is_file(),
    }
    for k in clients_list:
        part_k = partition_dir / feature_type / f"{k}clients" / "partition_table.csv"
        res[f"stage2_k{k}"] = part_k.is_file()
    return res


def run_stage1(raw_dir: Path, prepared_dir: Path, seed: int, dry: bool) -> bool:
    cmd = [
        sys.executable, "-m", "data.prepare_dataset",
        "--root", str(raw_dir),
        "--output_dir", str(prepared_dir),
        "--type", "all",
        "--seed", str(seed),
    ]
    info(f"Stage 1: {' '.join(cmd)}")
    if dry:
        return True
    r = subprocess.run(cmd)
    return r.returncode == 0


def run_stage2(prepared_dir: Path, partition_dir: Path, feature_type: str, k: int, seed: int, dry: bool) -> bool:
    train = prepared_dir / feature_type / "train.parquet"
    if not train.is_file():
        train = prepared_dir / feature_type / "train.csv"
    cmd = [
        sys.executable, "-m", "data.partition",
        "--dataset", str(train),
        "--feature_type", feature_type,
        "--n_clients", str(k),
        "--dirichlet_alpha", "0.5",
        "--seed", str(seed),
        "--output_dir", str(partition_dir),
    ]
    info(f"Stage 2 (K={k}, Feature={feature_type}): {' '.join(cmd)}")
    if dry:
        return True
    r = subprocess.run(cmd)
    return r.returncode == 0


def ensure_dataset(
    raw_dir: Path,
    prepared_dir: Path,
    partition_dir: Path,
    feature_type: str,
    clients_list: List[int],
    generate_synthetic: bool,
    skip: bool,
    dry: bool,
) -> None:
    section("STAGE 0 — Dataset Integrity Check")

    flags = check_dataset(prepared_dir, partition_dir, feature_type, clients_list)
    for key, ready in flags.items():
        (ok if ready else warn)(f"{key}: {'ready' if ready else 'MISSING'}")

    if skip:
        warn("--skip_data_prep set; skipping preparation steps.")
        return

    # Stage 1
    if not (flags["stage1_test"] and flags["stage1_train"]):
        info("Running Stage 1 data preparation...")
        if generate_synthetic and not raw_dir.is_dir():
            info("Generating synthetic raw dataset...")
            if not dry:
                from data.synthetic_generator import generate_synthetic_raw_andmal2020
                generate_synthetic_raw_andmal2020(
                    root_dir=str(raw_dir), samples_per_class=350,
                    static_dim=300, dynamic_dim=141, seed=SEED,
                )
        if not run_stage1(raw_dir, prepared_dir, SEED, dry):
            raise RuntimeError("Stage 1 preparation FAILED. Aborting.")
        ok("Stage 1 complete.")
    else:
        ok("Stage 1 already complete — skipping.")

    # Stage 2 for each requested K
    for k in clients_list:
        key = f"stage2_k{k}"
        flags = check_dataset(prepared_dir, partition_dir, feature_type, clients_list)
        if not flags[key]:
            info(f"Running Stage 2 partitioning for K={k} ({feature_type})...")
            if not run_stage2(prepared_dir, partition_dir, feature_type, k, SEED, dry):
                raise RuntimeError(f"Stage 2 partitioning (K={k}, feature={feature_type}) FAILED. Aborting.")
            ok(f"Stage 2 K={k} ({feature_type}) complete.")
        else:
            ok(f"Stage 2 K={k} ({feature_type}) already complete — skipping.")


# ──────────────────────────────────────────────────────────────────────────────
# Build & Run command for individual experiment
# ──────────────────────────────────────────────────────────────────────────────
def build_cmd(
    case_name: str,
    mode: str,
    method: str,
    aggregator: Optional[str],
    n_clients: int,
    extra: Dict,
    output_root: str,
    device: str,
    prepared_dir: str,
    partition_dir: str,
    feature_type: str,
    backbone: str,
) -> List[str]:
    cmd = [
        sys.executable, "main.py",
        "--mode", mode,
        "--feature_type", feature_type,
        "--backbone", backbone,
        "--method", method,
        "--output_root", output_root,
        "--prepared_dir", prepared_dir,
        "--partition_dir", partition_dir,
        "--n_clients", str(n_clients),
        "--seed", str(SEED),
        "--device", device,
        "--exp_name", case_name,
    ]
    if mode == "centralized":
        cmd += ["--rounds_per_task", str(CENTRAL_EPOCHS)]
    else:
        cmd += [
            "--aggregator", aggregator,
            "--rounds_per_task", str(FL_ROUNDS),
            "--local_epochs", str(FL_LOCAL_EPOCHS),
        ]
    for k, v in extra.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def run_experiment(
    case_name: str,
    cmd: List[str],
    log_dir: Path,
    dry: bool,
) -> Tuple[bool, float]:
    log_file = log_dir / f"{case_name}.stdout.log"
    t0 = time.time()
    info(f"Command: {' '.join(cmd)}")
    info(f"Log    : {log_file}")
    if dry:
        ok(f"[DRY RUN] Would run: {case_name}")
        return True, 0.0
    with open(log_file, "w") as fp:
        result = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return result.returncode == 0, elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(scenario_title: str, results: List[Dict], scenario_root: Path) -> None:
    banner(f"EXPERIMENT SUMMARY — {scenario_title}")
    total   = len(results)
    passed  = sum(1 for r in results if r["success"])
    failed  = total - passed
    total_t = sum(r["elapsed"] for r in results)

    print(f"\n  {'Case':<30} {'Mode':<14} {'Status':<10} {'Elapsed':>10}")
    print(f"  {'-'*30} {'-'*14} {'-'*10} {'-'*10}")
    for r in results:
        status = _c(GRN, "PASSED") if r["success"] else _c(RED, "FAILED")
        elapsed_str = str(timedelta(seconds=int(r["elapsed"])))
        print(f"  {r['case']:<30} {r['mode']:<14} {status:<20} {elapsed_str:>10}")

    print(f"\n  Total  : {total}")
    print(f"  {_c(GRN, 'Passed')}: {passed}")
    if failed:
        print(f"  {_c(RED, 'Failed')}: {failed}")
    print(f"  Wall-clock: {timedelta(seconds=int(total_t))}")

    # Point to outputs
    excel = scenario_root / "evaluation_results.xlsx"
    if excel.is_file():
        ok(f"Excel results: {excel}")
    json_path = scenario_root / "run_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    ok(f"JSON summary: {json_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Arg parse
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FCIL-AndMal full pipeline runner (dataset check + 10 experiments per client scenario)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clients",          type=int, nargs="+", default=[20, 50],
                   help="Client count scenarios to run (e.g. --clients 20 50 or --clients 20)")
    p.add_argument("--feature_type",     type=str, choices=["dynamic", "static", "fused"], default="dynamic",
                   help="Feature representation type: dynamic (141d), static (300d), or fused multi-modal (441d)")
    p.add_argument("--backbone",         type=str, choices=["hybrid_tcn_cnn", "cnn1d", "tcn", "mlp", "fused"], default=None,
                   help="Neural backbone architecture (auto-selects based on feature_type if None)")
    p.add_argument("--raw_root",        default="./raw_data",          help="Raw CIC-AndMal-2020 directory")
    p.add_argument("--prepared_dir",    default="./prepared_data",     help="Stage 1 prepared data dir")
    p.add_argument("--partition_dir",   default="./fl_data_partitions",help="Stage 2 FL partitions dir")
    p.add_argument("--output_root",     default="./EXPERIMENT",        help="Output root for results")
    p.add_argument("--device",          default="cpu",                 help="PyTorch device (cpu / cuda)")
    p.add_argument("--generate_synthetic", action="store_true",
                   help="Generate synthetic data if raw_root is absent")
    p.add_argument("--skip_data_prep",  action="store_true",
                   help="Skip Stage 1 & 2 preparation (assume data is ready)")
    p.add_argument("--dry_run",         action="store_true",
                   help="Print commands without executing")
    p.add_argument("--only",            default=None,
                   help="Run only experiments whose case name contains this substring")
    p.add_argument("--list",            action="store_true",
                   help="List all experiment cases and exit")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Determine backbone default if not specified
    if args.backbone is None:
        if args.feature_type == "fused":
            backbone = "fused"
        elif args.feature_type == "static":
            backbone = "mlp"
        else:
            backbone = "hybrid_tcn_cnn"
    else:
        backbone = args.backbone

    banner("FCIL-AndMal2020 — Full Pipeline Runner")
    info(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info(f"Device   : {args.device}")
    info(f"Output   : {args.output_root}")
    info(f"Feature  : {args.feature_type}")
    info(f"Backbone : {backbone}")
    info(f"Clients  : {args.clients} clients scenario(s)")

    if args.list:
        print(f"\nFeature Type    : {args.feature_type}")
        print(f"Backbone        : {backbone}")
        print(f"Client Scenarios: {args.clients}")
        print("\nCore Experiment Suite (10 cases executed per client scenario):")
        for idx, (name, mode, method, agg, _) in enumerate(CORE_EXPERIMENTS, 1):
            print(f"  [{idx:02d}] {name:<26}  mode={mode:<12} method={method:<10} agg={agg or 'N/A'}")
        total_runs = len(args.clients) * len(CORE_EXPERIMENTS)
        print(f"\nTotal runs across all client scenarios: {total_runs} runs "
              f"({len(CORE_EXPERIMENTS)} cases × {len(args.clients)} scenario(s): {args.clients})")
        return

    # Auto-detect raw data directory if default ./raw_data does not exist but ./Dataset exists
    raw_dir = Path(args.raw_root)
    if not raw_dir.is_dir() and Path("./Dataset").is_dir():
        raw_dir = Path("./Dataset")
        info(f"Auto-detected dataset directory: {raw_dir}")

    prepared_dir  = Path(args.prepared_dir)
    partition_dir = Path(args.partition_dir)
    output_root   = Path(args.output_root)

    # ── Dataset readiness for requested client counts and feature_type ────
    ensure_dataset(
        raw_dir, prepared_dir, partition_dir,
        feature_type=args.feature_type,
        clients_list=args.clients,
        generate_synthetic=args.generate_synthetic,
        skip=args.skip_data_prep,
        dry=args.dry_run,
    )

    # ── Filter experiments if --only is passed ────────────────────────────
    experiments = CORE_EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if args.only.lower() in e[0].lower()]
        if not experiments:
            err(f"No experiments match --only '{args.only}'")
            sys.exit(1)
        warn(f"Filtered to {len(experiments)} core experiment(s) matching '{args.only}'")

    overall_all_passed = True
    overall_summary = {}

    # ── Iterate over Client Scenarios ─────────────────────────────────────
    for k_clients in args.clients:
        scenario_title = f"{k_clients} Clients Scenario ({args.feature_type.upper()})"
        banner(f"RUNNING SCENARIO: {scenario_title.upper()} ({len(experiments)} Cases)")
        
        scenario_output_dir = output_root / f"{k_clients}clients"
        log_dir = scenario_output_dir / "_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        scenario_results = []

        for idx, (case_name, mode, method, agg, extra) in enumerate(experiments, 1):
            full_case_tag = f"{case_name}_K{k_clients}" if mode == "federated" else case_name
            section(f"[{idx}/{len(experiments)}] {full_case_tag} (K={k_clients})")
            info(f"mode={mode}  method={method}  aggregator={agg}  clients={k_clients}")
            info(f"{'epochs' if mode=='centralized' else 'rounds'}/task="
                 f"{CENTRAL_EPOCHS if mode=='centralized' else FL_ROUNDS}"
                 + (f"  local_epochs={FL_LOCAL_EPOCHS}" if mode == "federated" else ""))

            cmd = build_cmd(
                case_name=full_case_tag,
                mode=mode,
                method=method,
                aggregator=agg,
                n_clients=k_clients,
                extra=extra,
                output_root=str(scenario_output_dir),
                device=args.device,
                prepared_dir=str(prepared_dir),
                partition_dir=str(partition_dir),
                feature_type=args.feature_type,
                backbone=backbone,
            )

            success, elapsed = run_experiment(full_case_tag, cmd, log_dir, args.dry_run)
            elapsed_fmt = str(timedelta(seconds=int(elapsed)))

            if success:
                ok(f"Finished {full_case_tag} in {elapsed_fmt}")
            else:
                err(f"FAILED {full_case_tag} after {elapsed_fmt} — see {log_dir / f'{full_case_tag}.stdout.log'}")
                overall_all_passed = False

            scenario_results.append({
                "case":    full_case_tag,
                "mode":    mode,
                "method":  method,
                "clients": k_clients,
                "success": success,
                "elapsed": elapsed,
            })

        print_summary(scenario_title, scenario_results, scenario_output_dir)
        overall_summary[f"{k_clients}clients"] = scenario_results

    # Save overall summary JSON at root
    with open(output_root / "run_summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2, default=str)

    if not overall_all_passed:
        err("Some experiments FAILED. Check per-experiment log files in --output_root.")
        sys.exit(1)


if __name__ == "__main__":
    main()
