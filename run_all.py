#!/usr/bin/env python3
"""
FCIL-AndMal Full Pipeline Runner
=================================
Tự động:
  1. Kiểm tra / chuẩn bị dataset (Stage 1 + Stage 2)
  2. Chạy đủ 10 experiment theo spec
  3. In đầy đủ metrics (Accuracy, Macro/Micro/Weighted P/R/F1) mỗi lần test
  4. Confusion matrix chỉ ở round/epoch cuối mỗi task
  5. Xuất kết quả ra Excel + JSON summary

Usage:
    python run_all.py [--device cuda] [--output_root ./EXPERIMENT]
                      [--skip_data_prep] [--dry_run] [--only CASE_NAME]
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
# Experiment definitions
# ──────────────────────────────────────────────────────────────────────────────
# Each entry: (case_name, mode, method, aggregator_or_None, n_clients_or_None, extra_kwargs)
EXPERIMENTS: List[Tuple] = [
    # ── Centralized (4 cases) ──────────────────────────────────────────────
    ("Centralized_EWCDR",   "centralized", "ewc",     None,       None, {"ewc_lambda": 5000.0}),
    ("Centralized_MES",     "centralized", "replay",  None,       None, {"buffer_size": 20}),
    ("Centralized_SPCIL",   "centralized", "spcil",   None,       None, {}),
    ("Centralized_MALFSIL", "centralized", "malfsil", None,       None, {}),
    # ── FL K=20 (6 cases) ─────────────────────────────────────────────────
    ("FL_EWCDR_K20",        "federated",   "ewc",     "fedavg",   20,   {"ewc_lambda": 5000.0}),
    ("FL_MES_K20",          "federated",   "replay",  "fedavg",   20,   {"buffer_size": 20}),
    ("FL_SPCIL_K20",        "federated",   "spcil",   "fedavg",   20,   {}),
    ("FL_MALFSIL_K20",      "federated",   "malfsil", "fedavg",   20,   {}),
    ("FL_FedAvg_K20",       "federated",   "finetune","fedavg",   20,   {}),
    ("FL_FedNova_K20",      "federated",   "finetune","fednova",  20,   {}),
    # ── FL K=50 (6 cases) ─────────────────────────────────────────────────
    ("FL_EWCDR_K50",        "federated",   "ewc",     "fedavg",   50,   {"ewc_lambda": 5000.0}),
    ("FL_MES_K50",          "federated",   "replay",  "fedavg",   50,   {"buffer_size": 20}),
    ("FL_SPCIL_K50",        "federated",   "spcil",   "fedavg",   50,   {}),
    ("FL_MALFSIL_K50",      "federated",   "malfsil", "fedavg",   50,   {}),
    ("FL_FedAvg_K50",       "federated",   "finetune","fedavg",   50,   {}),
    ("FL_FedNova_K50",      "federated",   "finetune","fednova",  50,   {}),
]

# Fixed hyper-parameters per spec
CENTRAL_EPOCHS  = 250
FL_ROUNDS       = 50
FL_LOCAL_EPOCHS = 5
BACKBONE        = "hybrid_tcn_cnn"
FEATURE_TYPE    = "dynamic"
SEED            = 42


# ──────────────────────────────────────────────────────────────────────────────
# Stage 0: Dataset check
# ──────────────────────────────────────────────────────────────────────────────
def check_dataset(prepared_dir: Path, partition_dir: Path, feature_type: str) -> Dict[str, bool]:
    """Return dict of readiness flags for each stage."""
    test_parquet  = prepared_dir / feature_type / "test.parquet"
    train_parquet = prepared_dir / feature_type / "train.parquet"
    part_k20 = partition_dir / feature_type / "20clients" / "partition_table.csv"
    part_k50 = partition_dir / feature_type / "50clients" / "partition_table.csv"
    return {
        "stage1_test":  test_parquet.is_file(),
        "stage1_train": train_parquet.is_file(),
        "stage2_k20":   part_k20.is_file(),
        "stage2_k50":   part_k50.is_file(),
    }


def run_stage1(raw_dir: Path, prepared_dir: Path, seed: int, dry: bool) -> bool:
    cmd = [
        sys.executable, "-m", "data.prepare_dataset",
        "--root", str(raw_dir),
        "--output_dir", str(prepared_dir),
        "--type", "dynamic",
        "--seed", str(seed),
    ]
    info(f"Stage 1: {' '.join(cmd)}")
    if dry:
        return True
    r = subprocess.run(cmd)
    return r.returncode == 0


def run_stage2(prepared_dir: Path, partition_dir: Path, k: int, seed: int, dry: bool) -> bool:
    train = prepared_dir / FEATURE_TYPE / "train.parquet"
    if not train.is_file():
        train = prepared_dir / FEATURE_TYPE / "train.csv"
    cmd = [
        sys.executable, "-m", "data.partition",
        "--dataset", str(train),
        "--feature_type", FEATURE_TYPE,
        "--n_clients", str(k),
        "--dirichlet_alpha", "0.5",
        "--seed", str(seed),
        "--output_dir", str(partition_dir),
    ]
    info(f"Stage 2 (K={k}): {' '.join(cmd)}")
    if dry:
        return True
    r = subprocess.run(cmd)
    return r.returncode == 0


def ensure_dataset(
    raw_dir: Path,
    prepared_dir: Path,
    partition_dir: Path,
    generate_synthetic: bool,
    skip: bool,
    dry: bool,
) -> None:
    section("STAGE 0 — Dataset Integrity Check")

    flags = check_dataset(prepared_dir, partition_dir, FEATURE_TYPE)
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

    # Stage 2 for K=20 and K=50
    for k, key in [(20, "stage2_k20"), (50, "stage2_k50")]:
        flags = check_dataset(prepared_dir, partition_dir, FEATURE_TYPE)
        if not flags[key]:
            info(f"Running Stage 2 partitioning for K={k}...")
            if not run_stage2(prepared_dir, partition_dir, k, SEED, dry):
                raise RuntimeError(f"Stage 2 partitioning (K={k}) FAILED. Aborting.")
            ok(f"Stage 2 K={k} complete.")
        else:
            ok(f"Stage 2 K={k} already complete — skipping.")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1-N: Run individual experiment
# ──────────────────────────────────────────────────────────────────────────────
def build_cmd(
    case_name: str,
    mode: str,
    method: str,
    aggregator: Optional[str],
    n_clients: Optional[int],
    extra: Dict,
    output_root: str,
    device: str,
    prepared_dir: str,
    partition_dir: str,
) -> List[str]:
    cmd = [
        sys.executable, "main.py",
        "--mode", mode,
        "--feature_type", FEATURE_TYPE,
        "--backbone", BACKBONE,
        "--method", method,
        "--output_root", output_root,
        "--prepared_dir", prepared_dir,
        "--partition_dir", partition_dir,
        "--seed", str(SEED),
        "--device", device,
        "--exp_name", case_name,
    ]
    if mode == "centralized":
        cmd += ["--rounds_per_task", str(CENTRAL_EPOCHS)]
    else:
        cmd += [
            "--aggregator", aggregator,
            "--n_clients", str(n_clients),
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
def print_summary(results: List[Dict], output_root: Path) -> None:
    banner("EXPERIMENT SUMMARY")
    total   = len(results)
    passed  = sum(1 for r in results if r["success"])
    failed  = total - passed
    total_t = sum(r["elapsed"] for r in results)

    print(f"\n  {'Case':<30} {'Status':<10} {'Elapsed':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    for r in results:
        status = _c(GRN, "PASSED") if r["success"] else _c(RED, "FAILED")
        elapsed_str = str(timedelta(seconds=int(r["elapsed"])))
        print(f"  {r['case']:<30} {status:<20} {elapsed_str:>10}")

    print(f"\n  Total  : {total}")
    print(f"  {_c(GRN, 'Passed')}: {passed}")
    if failed:
        print(f"  {_c(RED, 'Failed')}: {failed}")
    print(f"  Wall-clock: {timedelta(seconds=int(total_t))}")

    # Point to outputs
    excel = output_root / "evaluation_results.xlsx"
    if excel.is_file():
        ok(f"Excel results: {excel}")
    json_path = output_root / "run_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    ok(f"JSON summary: {json_path}")

    if failed:
        err("Some experiments FAILED. Check per-experiment log files in --output_root.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Arg parse
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FCIL-AndMal full pipeline runner (dataset check + 10/16 experiments)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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

    banner("FCIL-AndMal2020 — Full Pipeline Runner")
    info(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info(f"Device   : {args.device}")
    info(f"Output   : {args.output_root}")
    info(f"Backbone : {BACKBONE}  (TCN + CNN)")
    info(f"Feature  : {FEATURE_TYPE}")

    if args.list:
        print("\nAvailable experiments:")
        for name, mode, method, agg, k, _ in EXPERIMENTS:
            k_str = f"K={k}" if k else "N/A"
            print(f"  {name:<30}  mode={mode:<12} method={method:<10} agg={agg or 'N/A':<8} {k_str}")
        return

    # Auto-detect raw data directory if default ./raw_data does not exist but ./Dataset exists
    raw_dir = Path(args.raw_root)
    if not raw_dir.is_dir() and Path("./Dataset").is_dir():
        raw_dir = Path("./Dataset")
        info(f"Auto-detected dataset directory: {raw_dir}")

    prepared_dir  = Path(args.prepared_dir)
    partition_dir = Path(args.partition_dir)
    output_root   = Path(args.output_root)
    log_dir       = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset readiness ──────────────────────────────────────────────────
    ensure_dataset(
        raw_dir, prepared_dir, partition_dir,
        generate_synthetic=args.generate_synthetic,
        skip=args.skip_data_prep,
        dry=args.dry_run,
    )

    # ── Filter experiments ─────────────────────────────────────────────────
    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if args.only.lower() in e[0].lower()]
        if not experiments:
            err(f"No experiments match --only '{args.only}'")
            sys.exit(1)
        warn(f"Filtered to {len(experiments)} experiment(s) matching '{args.only}'")

    # ── Run experiments ────────────────────────────────────────────────────
    banner(f"RUNNING {len(experiments)} EXPERIMENT(S)")
    results = []

    for idx, (case_name, mode, method, agg, k, extra) in enumerate(experiments, 1):
        section(f"[{idx}/{len(experiments)}] {case_name}")
        info(f"mode={mode}  method={method}  aggregator={agg}  clients={k}")
        info(f"{'epochs' if mode=='centralized' else 'rounds'}/task="
             f"{CENTRAL_EPOCHS if mode=='centralized' else FL_ROUNDS}"
             + (f"  local_epochs={FL_LOCAL_EPOCHS}" if mode == "federated" else ""))

        cmd = build_cmd(
            case_name=case_name,
            mode=mode,
            method=method,
            aggregator=agg,
            n_clients=k,
            extra=extra,
            output_root=str(output_root),
            device=args.device,
            prepared_dir=str(prepared_dir),
            partition_dir=str(partition_dir),
        )

        success, elapsed = run_experiment(case_name, cmd, log_dir, args.dry_run)
        elapsed_fmt = str(timedelta(seconds=int(elapsed)))

        if success:
            ok(f"Finished in {elapsed_fmt}")
        else:
            err(f"FAILED after {elapsed_fmt} — see {log_dir / f'{case_name}.stdout.log'}")

        results.append({
            "case":    case_name,
            "mode":    mode,
            "method":  method,
            "success": success,
            "elapsed": elapsed,
        })

    # ── Summary ────────────────────────────────────────────────────────────
    print_summary(results, output_root)


if __name__ == "__main__":
    main()
