"""Synthetic tabular data generator for development and unit tests only.

The generated values are simulations; they are not measured CIC-AndMal-2020
telemetry and must not be used as evidence for scientific benchmark claims.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from config import STATIC_LABEL_MAP, DYNAMIC_LABEL_MAP, ALL_LABELS

SYNTHETIC_SAMPLE_ID_RE = re.compile(
    r"^(?:BENIGN_\d+_\d{5}|(?:ADWARE|BACKDOOR|BANKER|DROPPER|FILEINFECTOR|NOCATEGORY|PUA|RANSOMWARE|RISKWARE|SCAREWARE|SMS|SPY|TROJAN|ZERODAY)_\d{5})$"
)
SYNTHETIC_DYNAMIC_COL_RE = re.compile(r"^dyn_metric_\d+$")
SYNTHETIC_STATIC_COL_RE = re.compile(r"^static_feat_\d+$")

_STATIC_REL_DIRS = [
    os.path.join("Static", "CCCS-CIC-Benign-CSVs"),
    "CCCS-CIC-Benign-CSVs",
    os.path.join("Static", "CCCS-CIC-Malicious-CSVs"),
    "CCCS-CIC-Malicious-CSVs",
]

_DYNAMIC_REL_DIRS = [
    os.path.join("Dynamic", "AndMal2020-dynamic-BeforeAndAfterReboot"),
    "AndMal2020-dynamic-BeforeAndAfterReboot",
]


def _glob_csvs(root_dir: str, rel_dirs: List[str]) -> List[str]:
    """Collect sorted unique CSV file paths across relative subdirectories."""
    files: List[str] = []
    for rel_dir in rel_dirs:
        target = os.path.join(root_dir, rel_dir) if rel_dir else root_dir
        if os.path.isdir(target):
            for entry in sorted(os.listdir(target)):
                if entry.lower().endswith(".csv"):
                    full_path = os.path.join(target, entry)
                    if os.path.isfile(full_path):
                        files.append(full_path)
    return sorted(set(files))


def is_synthetic_raw_csv(file_path: str) -> Tuple[bool, Optional[str]]:
    """Inspect the first two non-empty lines of a CSV to detect synthetic generator artifacts."""
    non_empty: List[str] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    non_empty.append(stripped)
                    if len(non_empty) >= 2:
                        break
    except OSError:
        return False, None

    if not non_empty:
        return False, None

    header = [t.strip().strip('"').strip("'") for t in non_empty[0].split(",")]
    row1 = (
        [t.strip().strip('"').strip("'") for t in non_empty[1].split(",")]
        if len(non_empty) > 1
        else []
    )

    if len(header) >= 2 and header[0] == "Sample_ID" and header[1] == "dyn_metric_0":
        return True, "synthetic dynamic header (Sample_ID,dyn_metric_0,...)"
    if len(header) >= 2 and header[0] == "Sample_ID" and header[1] == "static_feat_0":
        return True, "synthetic static header (Sample_ID,static_feat_0,...)"
    if (
        len(header) >= 1
        and header[0] == "Sample_ID"
        and len(row1) >= 1
        and SYNTHETIC_SAMPLE_ID_RE.match(row1[0])
    ):
        return True, f"synthetic Sample_ID row ({row1[0]})"

    return False, None


def detect_synthetic_raw_dir(root_dir: str, feature_type: str = "all") -> Tuple[bool, List[str]]:
    """Detect whether a raw dataset directory contains synthetic generator files."""
    reasons: List[str] = []
    if os.path.isfile(os.path.join(root_dir, "SYNTHETIC_DATASET.json")):
        reasons.append(f"SYNTHETIC_DATASET.json marker file present in {root_dir}")

    if feature_type in ("static", "both", "all", "fused"):
        static_csvs = _glob_csvs(root_dir, _STATIC_REL_DIRS)
        syn_static: List[Tuple[str, Optional[str]]] = []
        for csv_path in static_csvs:
            is_syn, reason = is_synthetic_raw_csv(csv_path)
            if is_syn:
                syn_static.append((csv_path, reason))
        if syn_static:
            for csv_path, reason in syn_static[:5]:
                reasons.append(
                    f"Synthetic static CSV: {os.path.relpath(csv_path, root_dir)} ({reason})"
                )
            if len(syn_static) > 5:
                reasons.append(f"... and {len(syn_static) - 5} additional synthetic static CSV(s)")

    if feature_type in ("dynamic", "both", "all", "fused"):
        dyn_dirs = list(_DYNAMIC_REL_DIRS)
        if os.path.basename(os.path.normpath(root_dir)).casefold() == (
            "AndMal2020-dynamic-BeforeAndAfterReboot".casefold()
        ):
            dyn_dirs.append("")
        dyn_csvs = _glob_csvs(root_dir, dyn_dirs)
        real_dyn: List[str] = []
        syn_dyn: List[Tuple[str, Optional[str]]] = []
        for csv_path in dyn_csvs:
            is_syn, reason = is_synthetic_raw_csv(csv_path)
            if is_syn:
                syn_dyn.append((csv_path, reason))
            else:
                real_dyn.append(csv_path)
        if syn_dyn and len(real_dyn) == 0:
            first_path, first_reason = syn_dyn[0]
            reasons.append(
                f"All {len(syn_dyn)} dynamic CSV(s) under {root_dir} are synthetic "
                f"(e.g., {os.path.basename(first_path)}: {first_reason})"
            )

    return len(reasons) > 0, reasons


def detect_synthetic_prepared_dir(prepared_dir: str, feature_type: str) -> Tuple[bool, List[str]]:
    """Detect whether prepared Stage 1 artifacts originate from synthetic data."""
    reasons: List[str] = []
    prov_path = os.path.join(prepared_dir, "data_provenance.json")
    if os.path.isfile(prov_path):
        try:
            with open(prov_path, "r", encoding="utf-8") as handle:
                prov = json.load(handle)
            if isinstance(prov, dict) and prov.get("source") == "synthetic_development":
                reasons.append("data_provenance.json marks prepared directory as synthetic_development")
        except Exception:
            pass

    def _inspect_modality(mod: str) -> None:
        mod_dir = os.path.join(prepared_dir, mod)
        schema_path = os.path.join(mod_dir, "feature_schema.json")
        if os.path.isfile(schema_path):
            try:
                with open(schema_path, "r", encoding="utf-8") as handle:
                    schema = json.load(handle)
                feature_columns = schema.get("feature_columns", []) or []
                if any(SYNTHETIC_DYNAMIC_COL_RE.match(str(col)) for col in feature_columns[:10]):
                    reasons.append(
                        f"{mod}/feature_schema.json contains synthetic dynamic features (dyn_metric_*)"
                    )
                if (
                    mod == "fused"
                    and schema.get("static_dim") == 300
                    and len(feature_columns) >= 3
                    and feature_columns[:3] == ["static_feat_0", "static_feat_1", "static_feat_2"]
                ):
                    reasons.append(
                        "fused/feature_schema.json uses synthetic 300-dim static schema (static_feat_0..299)"
                    )
            except Exception:
                pass

        sample_ids: List[str] = []
        test_parquet = os.path.join(mod_dir, "test.parquet")
        test_csv = os.path.join(mod_dir, "test.csv")
        if os.path.isfile(test_parquet):
            try:
                import pyarrow.parquet as pq

                pq_schema = pq.read_schema(test_parquet)
                if "Sample_ID" in pq_schema.names:
                    table = pq.read_table(test_parquet, columns=["Sample_ID"])
                    sample_ids = [str(v) for v in table["Sample_ID"].slice(0, 25).to_pylist()]
            except Exception:
                try:
                    df_head = pd.read_parquet(test_parquet)
                    if "Sample_ID" in df_head.columns:
                        sample_ids = [str(v) for v in df_head["Sample_ID"].head(25).tolist()]
                except Exception:
                    pass
        elif os.path.isfile(test_csv):
            try:
                df_head = pd.read_csv(test_csv, nrows=25)
                if "Sample_ID" in df_head.columns:
                    sample_ids = [str(v) for v in df_head["Sample_ID"].tolist()]
            except Exception:
                pass

        for sid in sample_ids:
            if SYNTHETIC_SAMPLE_ID_RE.match(sid):
                reasons.append(f"{mod} split contains synthetic Sample_IDs (e.g. {sid})")
                break

    _inspect_modality(feature_type)
    if feature_type == "fused":
        for sub_mod in ("static", "dynamic"):
            if os.path.isdir(os.path.join(prepared_dir, sub_mod)):
                _inspect_modality(sub_mod)

    return len(reasons) > 0, reasons


def generate_synthetic_raw_andmal2020(
    root_dir: str = "./.synthetic_dev_data",
    samples_per_class: int = 300,
    static_dim: int = 300,
    dynamic_dim: int = 141,
    seed: int = 42
) -> None:
    """
    Generate a CIC-AndMal-shaped directory tree for development tests.
    
    Structure:
    raw_data/
      ├── Static/
      │   ├── CCCS-CIC-Benign-CSVs/
      │   │   ├── Ben0.csv ... Ben4.csv
      │   └── CCCS-CIC-Malicious-CSVs/
      │       ├── Adware.csv, Backdoor.csv, Banker.csv, Dropper.csv, FileInfector.csv,
      │       ├── NoCategory.csv, PUA.csv, Ransomware.csv, Riskware.csv, Scareware.csv,
      │       ├── SMS.csv, Spy.csv, Trojan.csv, Zeroday.csv
      └── Dynamic/
          └── AndMal2020-dynamic-BeforeAndAfterReboot/
              ├── {label}_before_reboot_Cat.csv
              └── {label}_after_reboot_Cat.csv
    """
    abs_root = os.path.abspath(root_dir)
    if os.path.basename(abs_root).casefold() == "dataset":
        raise ValueError(
            "Refusing to generate synthetic data inside a 'Dataset' directory. "
            "Use a dedicated temporary path for development tests."
        )
    if os.path.isdir(abs_root):
        existing_csvs = _glob_csvs(abs_root, _STATIC_REL_DIRS + _DYNAMIC_REL_DIRS)
        for existing_csv in existing_csvs:
            is_syn, _ = is_synthetic_raw_csv(existing_csv)
            if not is_syn:
                raise ValueError(
                    f"Refusing to overwrite directory '{root_dir}' containing "
                    f"non-synthetic dataset files (e.g. {existing_csv})."
                )

    np.random.seed(seed)
    static_benign_dir = os.path.join(root_dir, "Static", "CCCS-CIC-Benign-CSVs")
    static_malware_dir = os.path.join(root_dir, "Static", "CCCS-CIC-Malicious-CSVs")
    dynamic_dir = os.path.join(root_dir, "Dynamic", "AndMal2020-dynamic-BeforeAndAfterReboot")

    os.makedirs(static_benign_dir, exist_ok=True)
    os.makedirs(static_malware_dir, exist_ok=True)
    os.makedirs(dynamic_dir, exist_ok=True)

    print(
        f"[Synthetic Generator] Generating DEVELOPMENT-ONLY simulated files in {root_dir}..."
    )

    # 1. Generate Static Benign Files (Ben0..Ben4)
    benign_samples_per_file = max(samples_per_class // 5, 60)
    for b_idx in range(5):
        filename = f"Ben{b_idx}.csv"
        file_path = os.path.join(static_benign_dir, filename)
        sample_ids = [f"BENIGN_{b_idx}_{i:05d}" for i in range(benign_samples_per_file)]
        # Binary sparse static features (permissions/intents/API calls)
        features = np.random.binomial(n=1, p=0.08, size=(benign_samples_per_file, static_dim))
        cols = ["Sample_ID"] + [f"static_feat_{j}" for j in range(static_dim)]
        df = pd.DataFrame(features, columns=cols[1:])
        df.insert(0, "Sample_ID", sample_ids)
        df.to_csv(file_path, index=False)

    # 2. Generate Static Malicious Files
    static_malware_stems = [
        "Adware", "Backdoor", "Banker", "Dropper", "FileInfector",
        "NoCategory", "PUA", "Ransomware", "Riskware", "Scareware",
        "SMS", "Spy", "Trojan", "Zeroday"
    ]
    for stem in static_malware_stems:
        file_path = os.path.join(static_malware_dir, f"{stem}.csv")
        n_samples = samples_per_class if stem != "Riskware" else samples_per_class * 2  # Riskware is large
        sample_ids = [f"{stem.upper()}_{i:05d}" for i in range(n_samples)]
        # Slightly denser binary features with class-specific bias
        stable_hash = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16)
        bias = (stable_hash % 100) / 1000.0
        features = np.random.binomial(n=1, p=min(0.12 + bias, 0.4), size=(n_samples, static_dim))
        cols = ["Sample_ID"] + [f"static_feat_{j}" for j in range(static_dim)]
        df = pd.DataFrame(features, columns=cols[1:])
        df.insert(0, "Sample_ID", sample_ids)
        df.to_csv(file_path, index=False)

    # 3. Generate Dynamic Files (Before and After Reboot)
    # Dynamic feature names (system calls, network, battery, memory, cpu)
    dynamic_stems = list(DYNAMIC_LABEL_MAP.keys())
    for d_stem in dynamic_stems:
        for phase in ["before", "after"]:
            filename = f"{d_stem}_{phase}_reboot_Cat.csv"
            file_path = os.path.join(dynamic_dir, filename)
            # Find matching static stem for aligned Sample_IDs
            if d_stem == "benign":
                n_samples = max(samples_per_class // 5, 60) * 5
                sample_ids = []
                for b_idx in range(5):
                    for i in range(max(samples_per_class // 5, 60)):
                        sample_ids.append(f"BENIGN_{b_idx}_{i:05d}")
            else:
                matching_static = [s for s, can in STATIC_LABEL_MAP.items() if can == DYNAMIC_LABEL_MAP[d_stem] and not s.startswith("Ben")]
                matched_stem = matching_static[0] if matching_static else d_stem.capitalize()
                n_samples = samples_per_class if matched_stem != "Riskware" else samples_per_class * 2
                sample_ids = [f"{matched_stem.upper()}_{i:05d}" for i in range(n_samples)]

            # Dynamic numerical features (counts, rates, memory bytes) with class-specific gaussian modes
            stable_hash = int(
                hashlib.sha256(d_stem.encode("utf-8")).hexdigest()[:8], 16
            )
            class_mean = (stable_hash % 50) / 10.0 + (1.5 if phase == "after" else 0.5)
            features = np.random.exponential(scale=class_mean, size=(n_samples, dynamic_dim))
            cols = ["Sample_ID"] + [f"dyn_metric_{j}" for j in range(dynamic_dim)]
            df = pd.DataFrame(features, columns=cols[1:])
            df.insert(0, "Sample_ID", sample_ids)
            df.to_csv(file_path, index=False)

    with open(os.path.join(root_dir, "SYNTHETIC_DATASET.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "synthetic": True,
                "intended_use": "development_and_unit_tests_only",
                "not_valid_for": "CIC-AndMal-2020 benchmark claims",
                "seed": seed,
                "samples_per_class": samples_per_class,
                "static_dim": static_dim,
                "dynamic_dim": dynamic_dim,
            },
            handle,
            indent=2,
        )
    print("[Synthetic Generator] Generated development-only simulated data tree.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic development-only CIC-AndMal-shaped CSVs."
    )
    parser.add_argument(
        "--confirm_dev_only",
        action="store_true",
        help="Required confirmation that outputs are for development tests only.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./.synthetic_dev_data",
        help="Destination directory for synthetic development data.",
    )
    parser.add_argument("--samples_per_class", type=int, default=300)
    parser.add_argument("--static_dim", type=int, default=300)
    parser.add_argument("--dynamic_dim", type=int, default=141)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.confirm_dev_only:
        print(
            "ERROR: Generating synthetic data requires --confirm_dev_only "
            "(intended strictly for development tests, never benchmark results).",
            file=sys.stderr,
        )
        sys.exit(2)

    if os.path.basename(os.path.abspath(args.output_dir)).casefold() in {"raw_data", "dataset"}:
        print(
            f"ERROR: Refusing synthetic output in '{args.output_dir}'. "
            "Use an isolated dev directory such as ./.synthetic_dev_data.",
            file=sys.stderr,
        )
        sys.exit(2)

    generate_synthetic_raw_andmal2020(
        root_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        static_dim=args.static_dim,
        dynamic_dim=args.dynamic_dim,
        seed=args.seed,
    )
