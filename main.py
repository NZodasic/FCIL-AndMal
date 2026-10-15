"""
Main Scientific Experiment Orchestrator for FCIL on CIC-AndMal-2020.
Integrates Data Pipeline, Neural Models, Continual Learning Methods,
Federated Aggregators (FedAvg, FedNova), Logging, Checkpointing, and Visualization.
"""

import os
import sys
import argparse
import json
import gc
import pandas as pd

# Prevent generation of __pycache__ byte-code files
sys.dont_write_bytecode = True

import torch
import numpy as np
from torch.utils.data import DataLoader

# Ensure package path is recognized
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.device import resolve_device, get_device_info, configure_mps_environment

from config import (
    ExperimentConfig,
    ScenarioConfig,
    ModelConfig,
    ILConfig,
    FLConfig,
    DriftConfig,
    TASK_LABEL_MAP,
    ALL_LABELS,
    LABEL2ID,
    ID2LABEL,
    get_batch_size_for_mode,
)
from utils.seed import set_seed
from utils.logger import AcademicLogger, get_logger
from utils.device import clear_device_cache
from data.synthetic_generator import generate_synthetic_raw_andmal2020
from data.prepare_dataset import AndMal2020DataPreparer
from data.partition import FCILDataPartitioner
from data.dataset import (
    load_heldout_test_set,
    load_prepared_split,
    TabularMalwareDataset,
    get_participating_clients,
)
from data.schema import get_feature_columns
from training.evaluator import ContinualEvaluator
from training.trainer import CentralizedTrainer
from federated.server import FLServer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated Class-Incremental Learning for Android Malware Family Detection (CIC-AndMal-2020)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core experiment mode
    parser.add_argument("--mode", type=str, choices=["federated", "centralized"], default="centralized",
                        help="Execution mode: federated multi-client simulation or centralized continual baseline")
    parser.add_argument("--exp_name", type=str, default="FCIL_AndMal2020",
                        help="Identifier tag for the experimental run")

    # Feature representation & Dataset
    parser.add_argument("--feature_type", type=str, choices=["dynamic", "static", "fused"], default="dynamic",
                        help="Feature representation: dynamic (141/282d), static (reduced 300d), or fused (441d)")
    parser.add_argument("--backbone", type=str, choices=["hybrid_tcn_cnn", "cnn1d", "tcn", "mlp", "fused"], default="hybrid_tcn_cnn",
                        help="Neural backbone architecture: 'hybrid_tcn_cnn' (Primary TCN+CNN), 'cnn1d', 'tcn', 'mlp', or 'fused'")
    parser.add_argument("--raw_root", type=str, default="./raw_data",
                        help="Path to raw CIC-AndMal-2020 directory")
    parser.add_argument("--prepared_dir", type=str, default="./prepared_data",
                        help="Directory for prepared Stage 1 datasets and held-out test splits")
    parser.add_argument("--partition_dir", type=str, default="./fl_data_partitions",
                        help="Directory for Stage 2 client partition parquet files")
    parser.add_argument("--generate_synthetic", action="store_true", default=False,
                        help="Generate synthetic data for development tests only; outputs are not a real CIC-AndMal benchmark")
    parser.add_argument("--allow_incomplete_benchmark", action="store_true", default=False,
                        help="Permit missing configured classes (development only; default is fail-fast)")
    parser.add_argument("--prepare_only", action="store_true", default=False,
                        help="Prepare, validate, and partition data without starting training")

    # Incremental Learning & Methods
    parser.add_argument("--method", type=str, choices=["finetune", "joint", "ewc", "lwf", "replay", "spcil", "malfscil", "malfsil"],
                        default="malfscil", help="Class-Incremental Learning method")
    parser.add_argument("--ewc_lambda", type=float, default=5000.0, help="Fisher information loss weight for EWC")
    parser.add_argument("--lwf_temp", type=float, default=2.0, help="Distillation temperature for LwF")
    parser.add_argument("--lwf_alpha", type=float, default=1.0, help="Distillation loss scale for LwF")
    parser.add_argument("--buffer_size", type=int, default=20, choices=[5, 20, 50],
                        help="Exemplar replay buffer size m samples/class for Replay")
    parser.add_argument("--fscil_k_shot", type=int, default=5,
                        help="Labeled examples available per new class")
    parser.add_argument("--fscil_n_way", type=int, default=3,
                        help="New malware classes introduced per incremental session")
    parser.add_argument("--fscil_query_per_class", type=int, default=5,
                        help="Mask-augmented query examples generated per new class")
    parser.add_argument("--fscil_mask_probability", type=float, default=0.1,
                        help="Feature masking probability for query augmentation")
    parser.add_argument("--malfscil_vae_weight", type=float, default=1.0,
                        help="Base-session VAE objective weight")
    parser.add_argument("--malfscil_arc_weight", type=float, default=0.5,
                        help="ArcFace share of the incremental objective")
    parser.add_argument("--malfscil_arc_scale", type=float, default=30.0)
    parser.add_argument("--malfscil_arc_margin", type=float, default=0.5)

    # Federated Learning
    parser.add_argument("--aggregator", type=str, choices=["fedavg", "fednova"], default="fedavg",
                        help="Federated aggregation algorithm")
    parser.add_argument("--n_clients", type=int, default=20, choices=[20, 50, 100],
                        help="Total candidate FL clients (participation scales dynamically per task)")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5,
                        help="Dirichlet distribution concentration parameter (lower = stronger non-IID label skew)")
    parser.add_argument("--rounds_per_task", type=int, default=50,
                        help="Communication rounds per incremental task")
    parser.add_argument("--local_epochs", type=int, default=5, choices=[1, 5],
                        help="Local training epochs per client per round (E in {1, 5})")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Fixed by mode: federated=256, centralized=1024",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Computation device ('auto', 'mps', 'cuda', 'cpu')")

    # Reproducibility & Output
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--output_root", type=str, default="./EXPERIMENT",
                        help="Directory to save experimental logs, metrics, plots, and checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to checkpoint file (.pt) to resume training from")

    # FL-MalDrift Drift Detection & Admission Control
    # (Patel et al. 2026 — Algorithm 1 & 2)
    drift_grp = parser.add_argument_group("FL-MalDrift drift control (Patel et al. 2026)")
    drift_grp.add_argument(
        "--enable_drift", action="store_true", default=False,
        help="Activate FL-MalDrift drift detection, admission filtering, and recovery (Algorithm 1 & 2)"
    )
    drift_grp.add_argument(
        "--drift_detector", type=str, default="hddm_w",
        choices=["hddm_w", "hddm_a", "adwin", "ddm", "eddm"],
        help="Drift detector algorithm used client-side (Algorithm 1)"
    )
    drift_grp.add_argument(
        "--drift_k", type=float, default=1.5,
        help="σ multiplier k for dynamic server threshold τ_t (Algorithm 2)"
    )
    drift_grp.add_argument(
        "--drift_alpha", type=float, default=0.8,
        help="EWMA smoothing factor α for server threshold τ_t (Algorithm 2)"
    )
    drift_grp.add_argument(
        "--drift_p_star", type=float, default=0.7,
        help="Target admission rate p* for server admission control (Algorithm 2)"
    )
    drift_grp.add_argument(
        "--drift_warmup", type=int, default=3,
        help="Warm-up rounds T₀ before drift filtering activates (Algorithm 2)"
    )
    drift_grp.add_argument(
        "--drift_theta_base", type=float, default=0.1,
        help="Floor θ_base for client local stability threshold (Algorithm 1)"
    )
    drift_grp.add_argument(
        "--drift_beta", type=float, default=0.9,
        help="EWMA β for client local threshold update (Algorithm 1)"
    )
    drift_grp.add_argument(
        "--drift_eta", type=float, default=0.05,
        help="Admission-rate correction gain η for server threshold (Algorithm 2)"
    )
    drift_grp.add_argument(
        "--drift_window", type=int, default=10,
        help="Sliding window size W for server μ/σ estimation (Algorithm 2)"
    )

    return parser.parse_args()


def build_configs_from_args(args) -> ExperimentConfig:
    """Build unified ExperimentConfig hierarchy from parsed CLI arguments."""
    if args.mode == "federated" and args.method == "malfscil":
        raise ValueError(
            "The cited MalFSCIL paper defines a centralized FSCIL protocol, "
            "not federated optimization. Use --mode centralized for malfscil. "
            "For federated MALFSIL (three-tier: replay+distillation+prototype), "
            "use --method malfsil which is FL-compatible."
        )
    raw_root = args.raw_root
    if raw_root == "./raw_data" and not os.path.isdir(raw_root) and os.path.isdir("./Dataset"):
        raw_root = "./Dataset"

    scenario_cfg = ScenarioConfig(
        feature_type=args.feature_type,
        n_clients=args.n_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
        raw_data_dir=raw_root,
        prepared_data_dir=args.prepared_dir,
        partition_output_dir=args.partition_dir,
    )

    selected_backbone = "fused" if args.feature_type == "fused" else args.backbone

    model_cfg = ModelConfig(
        backbone_type=selected_backbone,
        input_dim=141 if args.feature_type == "dynamic" else (300 if args.feature_type == "static" else 441),
        num_total_classes=15,
        classes_per_task=3
    )

    method_name = args.method  # malfsil and malfscil are now distinct methods

    il_cfg = ILConfig(
        method_name=method_name,
        ewc_lambda=args.ewc_lambda,
        lwf_temperature=args.lwf_temp,
        lwf_alpha=args.lwf_alpha,
        replay_buffer_size_per_class=args.buffer_size,
        fscil_n_way=args.fscil_n_way,
        fscil_k_shot=args.fscil_k_shot,
        fscil_query_per_class=args.fscil_query_per_class,
        fscil_mask_probability=args.fscil_mask_probability,
        malfscil_vae_weight=args.malfscil_vae_weight,
        malfscil_arc_weight=args.malfscil_arc_weight,
        malfscil_arc_scale=args.malfscil_arc_scale,
        malfscil_arc_margin=args.malfscil_arc_margin,
    )

    batch_size = get_batch_size_for_mode(args.mode)
    if args.batch_size is not None and args.batch_size != batch_size:
        raise ValueError(
            f"Batch size for {args.mode} is fixed at {batch_size}; "
            f"received {args.batch_size}."
        )

    configure_mps_environment()
    resolved_dev = resolve_device(args.device)

    fl_cfg = FLConfig(
        aggregator=args.aggregator,
        n_tasks=5,
        rounds_per_task=args.rounds_per_task,
        local_epochs=args.local_epochs,
        batch_size=batch_size,
        lr=args.lr,
        device=str(resolved_dev),
    )

    # --- FL-MalDrift drift config (additive; None when drift is disabled) ---
    drift_cfg: Optional[DriftConfig] = None
    if getattr(args, 'enable_drift', False):
        drift_cfg = DriftConfig(
            enabled=True,
            detector_type=getattr(args, 'drift_detector', 'hddm_w'),
            window_size=getattr(args, 'drift_window', 10),
            warmup_rounds=getattr(args, 'drift_warmup', 3),
            alpha=getattr(args, 'drift_alpha', 0.8),
            k=getattr(args, 'drift_k', 1.5),
            p_star=getattr(args, 'drift_p_star', 0.7),
            eta=getattr(args, 'drift_eta', 0.05),
            beta=getattr(args, 'drift_beta', 0.9),
            theta_base=getattr(args, 'drift_theta_base', 0.1),
        )

    if args.mode == "federated":
        case_name = (
            f"{args.exp_name}_FEDERATED_{args.feature_type.upper()}_"
            f"{method_name.upper()}_{args.aggregator.upper()}_K{args.n_clients}_"
            f"E{args.local_epochs}_S{args.fscil_k_shot}"
        )
    else:
        method_case = method_name.upper()
        if method_name == "malfscil":
            method_case = (
                f"{method_case}_N{args.fscil_n_way}_S{args.fscil_k_shot}_"
                f"Q{args.fscil_query_per_class}"
            )
        case_name = (
            f"{args.exp_name}_CENTRALIZED_{args.feature_type.upper()}_"
            f"{method_case}"
        )

    exp_cfg = ExperimentConfig(
        exp_name=case_name,
        output_root=args.output_root,
        scenario=scenario_cfg,
        model=model_cfg,
        il=il_cfg,
        fl=fl_cfg,
        drift=drift_cfg,
        seed=args.seed
    )
    return exp_cfg


def ensure_dataset_ready(
    exp_cfg: ExperimentConfig,
    generate_synthetic: bool = False,
    allow_incomplete_benchmark: bool = False,
    mode: str = "centralized",
) -> None:
    """Ensure raw data, prepared splits, and partitions exist.

    Synthetic data is generated only after explicit opt-in and is marked as a
    development artifact. Scientific runs fail fast when raw data or benchmark
    classes are missing.
    """
    raw_root = exp_cfg.scenario.raw_data_dir
    prepared_dir = exp_cfg.scenario.prepared_data_dir
    scenario_dir = exp_cfg.scenario.get_scenario_dir()
    feature_dir = os.path.join(prepared_dir, exp_cfg.scenario.feature_type)
    test_parquet = os.path.join(feature_dir, "test.parquet")
    test_csv = os.path.join(feature_dir, "test.csv")
    scaler_path = os.path.join(feature_dir, "scaler.joblib")
    partition_table = os.path.join(scenario_dir, "partition_table.csv")

    # Synthetic generation is an explicit development-only action. Never
    # silently replace a missing scientific dataset with simulated samples.
    if generate_synthetic:
        print(
            "\n[Data Pipeline] Generating explicitly requested SYNTHETIC "
            "development data. Metrics from this data are not CIC-AndMal "
            "benchmark evidence."
        )
        generate_synthetic_raw_andmal2020(
            root_dir=raw_root,
            samples_per_class=350,
            static_dim=300,
            dynamic_dim=141,
            seed=exp_cfg.seed,
        )
    elif not os.path.isdir(raw_root):
        raise FileNotFoundError(
            f"Raw dataset directory not found: {raw_root}. Provide the real "
            "dataset or pass --generate_synthetic for a development-only run."
        )

    # Check if Stage 1 prepared data exists
    if not (os.path.isfile(test_parquet) or os.path.isfile(test_csv)) or not os.path.isfile(scaler_path):
        print(
            f"\n[Data Pipeline] Prepared '{exp_cfg.scenario.feature_type}' data or "
            "train-fitted scaler is missing. Running preparer..."
        )
        preparer = AndMal2020DataPreparer(
            raw_root=raw_root,
            output_dir=prepared_dir,
            seed=exp_cfg.seed,
            strict_class_coverage=not allow_incomplete_benchmark,
            data_provenance=(
                "synthetic_development" if generate_synthetic else "user_supplied"
            ),
        )
        preparer.run_all(data_type=exp_cfg.scenario.feature_type)

    # Check if Stage 2 partition files exist (only required for federated mode)
    if mode == "federated":
        if not os.path.isfile(partition_table):
            print(f"\n[Data Pipeline] Stage 2 partition table not found at {partition_table}. Running partitioner...")
            train_path = os.path.join(prepared_dir, exp_cfg.scenario.feature_type, "train.parquet")
            if os.path.isfile(train_path):
                df = pd.read_parquet(train_path)
            else:
                df = pd.read_csv(os.path.join(prepared_dir, exp_cfg.scenario.feature_type, "train.csv"))

            partitioner = FCILDataPartitioner(exp_cfg.scenario)
            partitioner.partition_dataframe(df)


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_cfg = build_configs_from_args(args)
    exp_dir = exp_cfg.get_exp_dir()
    os.makedirs(exp_dir, exist_ok=True)

    # Save experiment config JSON for reproducibility
    config_json_path = os.path.join(exp_dir, "experiment_config.json")
    exp_cfg.save_json(config_json_path)

    # Initialize Academic Logger
    logger = get_logger(log_dir=exp_dir, exp_name=exp_cfg.exp_name)
    logger.section("ACADEMIC RESEARCH PLATFORM: FCIL FOR ANDROID MALWARE (CIC-AndMal-2020)")
    logger.info(f"Configuration written to: {config_json_path}")
    logger.info(f"Target Experiment Directory: {exp_dir}")

    # Prepare datasets & partitions
    ensure_dataset_ready(
        exp_cfg,
        generate_synthetic=args.generate_synthetic,
        allow_incomplete_benchmark=args.allow_incomplete_benchmark,
        mode=args.mode,
    )

    # Load Held-Out Global Test Set
    logger.info("Loading central held-out test split for multi-task evaluation...")
    test_X, test_y = load_heldout_test_set(
        prepared_data_dir=exp_cfg.scenario.prepared_data_dir,
        feature_type=exp_cfg.scenario.feature_type
    )
    available_class_ids = sorted(np.unique(test_y[test_y >= 0]).tolist())
    available_labels = [ID2LABEL[class_id] for class_id in available_class_ids]
    missing_labels = [
        label for label in ALL_LABELS if LABEL2ID[label] not in available_class_ids
    ]
    exp_cfg.model.input_dim = int(test_X.shape[1])
    if args.feature_type == "dynamic":
        exp_cfg.model.dynamic_input_dim = exp_cfg.model.input_dim
    elif args.feature_type == "static":
        exp_cfg.model.static_input_dim = exp_cfg.model.input_dim
    elif args.feature_type == "fused":
        fused_schema_file = os.path.join(exp_cfg.scenario.prepared_data_dir, "fused", "feature_schema.json")
        static_meta_file = os.path.join(exp_cfg.scenario.prepared_data_dir, "static", "scaler_metadata.json")
        stat_dim = None
        if os.path.isfile(fused_schema_file):
            try:
                with open(fused_schema_file, "r", encoding="utf-8") as f:
                    schema_data = json.load(f)
                if "static_dim" in schema_data:
                    stat_dim = int(schema_data["static_dim"])
            except Exception:
                pass
        if stat_dim is None and os.path.isfile(static_meta_file):
            try:
                with open(static_meta_file, "r", encoding="utf-8") as f:
                    stat_cols = json.load(f).get("feature_columns", [])
                if stat_cols:
                    stat_dim = len(stat_cols)
            except Exception:
                pass
        if stat_dim is None:
            fused_meta_file = os.path.join(exp_cfg.scenario.prepared_data_dir, "fused", "scaler_metadata.json")
            if os.path.isfile(fused_meta_file):
                try:
                    with open(fused_meta_file, "r", encoding="utf-8") as f:
                        fused_cols = json.load(f).get("feature_columns", [])
                    stat_dim = len([c for c in fused_cols if c.startswith("static_feat_")])
                except Exception:
                    pass
        if stat_dim is None or stat_dim <= 0 or stat_dim >= exp_cfg.model.input_dim:
            stat_dim = 9503 if exp_cfg.model.input_dim > 9503 else exp_cfg.model.static_input_dim
        exp_cfg.model.static_input_dim = stat_dim
        exp_cfg.model.dynamic_input_dim = exp_cfg.model.input_dim - stat_dim
        logger.info(
            f"Fused model feature partition: static_dim={exp_cfg.model.static_input_dim}, "
            f"dynamic_dim={exp_cfg.model.dynamic_input_dim} (total={exp_cfg.model.input_dim})"
        )
    exp_cfg.save_json(config_json_path)
    logger.info(
        f"Held-out test set loaded: {len(test_y):,} samples, "
        f"{exp_cfg.model.input_dim} features, {len(available_labels)} observed classes."
    )
    if missing_labels:
        message = (
            f"Configured classes absent from the dataset: {missing_labels}. "
            "Results are not a complete 15-class benchmark."
        )
        provenance_path = os.path.join(exp_cfg.scenario.prepared_data_dir, "data_provenance.json")
        provenance_non_strict = False
        if os.path.isfile(provenance_path):
            try:
                with open(provenance_path, "r", encoding="utf-8") as f:
                    prov_meta = json.load(f)
                provenance_non_strict = (prov_meta.get("strict_class_coverage") is False)
            except Exception:
                pass

        if not args.allow_incomplete_benchmark and not provenance_non_strict:
            raise ValueError(
                f"{message} Re-run only with --allow_incomplete_benchmark for "
                "an explicitly labeled development experiment."
            )
        logger.warning(message)
    if args.prepare_only:
        logger.info(
            "Dataset preparation completed; training was skipped because "
            "--prepare_only was specified."
        )
        return

    configure_mps_environment()
    resolved_device = resolve_device(args.device)
    dev_info = get_device_info()
    logger.info(f"Target Device: {args.device} -> Active: {resolved_device}")
    if dev_info.get("mps_available"):
        logger.info("Apple Silicon GPU (MPS / Metal) acceleration is ACTIVE.")
    elif dev_info.get("cuda_available"):
        logger.info(f"Nvidia CUDA acceleration is ACTIVE ({dev_info.get('cuda_device_name', '')}).")
    else:
        logger.info("Computation running on CPU.")

    evaluator = ContinualEvaluator(
        test_X=test_X,
        test_y=test_y,
        batch_size=exp_cfg.fl.batch_size,
        device=resolved_device,
    )
    val_X, val_y = load_prepared_split(
        prepared_data_dir=exp_cfg.scenario.prepared_data_dir,
        feature_type=exp_cfg.scenario.feature_type,
        split="val",
    )
    validation_evaluator = ContinualEvaluator(
        test_X=val_X,
        test_y=val_y,
        batch_size=exp_cfg.fl.batch_size,
        device=resolved_device,
    )

    # Run selected mode
    if args.mode == "federated":
        from models.fcil_model import FCILNet
        from methods import build_il_method
        from federated.client import FLClient
        from federated.aggregators import build_aggregator
        from training.checkpoint import CheckpointManager
        from federated.il_strategy_adapter import ILMethodStrategyAdapter

        agg = build_aggregator(args.aggregator)
        ckpt_mgr = CheckpointManager(
            checkpoint_dir=os.path.join(exp_cfg.get_exp_dir(), "checkpoints"),
            logger=logger,
        )
        global_model = FCILNet(exp_cfg.model).to(resolved_device)
        server = FLServer(
            global_model=global_model,
            aggregator=agg,
            device=resolved_device,
            config=exp_cfg,
            evaluator=evaluator,
            logger=logger,
            checkpoint_manager=ckpt_mgr,
            enable_drift_filtering=(exp_cfg.drift is not None and exp_cfg.drift.enabled),
        )

        scenario_dir = exp_cfg.scenario.get_scenario_dir()

        def ensure_client_registered(client_id: int) -> None:
            if client_id in server.clients:
                return
            client_model = FCILNet(exp_cfg.model)
            if server.global_model.current_classes > client_model.current_classes:
                client_model.expand_classes(
                    server.global_model.current_classes - client_model.current_classes
                )
            client_il = build_il_method(exp_cfg.il)
            adapter = ILMethodStrategyAdapter(
                model=client_model,
                il_method=client_il,
                lr=exp_cfg.fl.lr,
                device=resolved_device,
                classes_per_task=exp_cfg.model.classes_per_task,
            )
            # --- FL-MalDrift: per-client drift detector & normalizer (Algorithm 1) ---
            _drift_detector = None
            _drift_normalizer = None
            if exp_cfg.drift is not None and exp_cfg.drift.enabled:
                from federated.drift import build_drift_detector, DriftScoreNormalizer
                _drift_detector = build_drift_detector(exp_cfg.drift.detector_type)
                _drift_normalizer = DriftScoreNormalizer()
            client = FLClient(
                client_id=client_id,
                model=client_model,
                strategy=adapter,
                device=resolved_device,
                drift_detector=_drift_detector,
                drift_normalizer=_drift_normalizer,
            )
            server.register_client(client)

        active_cids = get_participating_clients(scenario_dir, 0)
        for cid in active_cids:
            ensure_client_registered(cid)
        logger.info(f"Registered {len(active_cids)} clients | IL: {args.method.upper()} | Agg: {args.aggregator.upper()}")

        all_task_results = []
        checkpoint_paths = []
        from utils.results import export_experiment_results, resolve_test_location

        for task_id in range(5):
            task_cids = get_participating_clients(scenario_dir, task_id)
            train_loaders = {}
            for cid in task_cids:
                ensure_client_registered(cid)
                p_file = os.path.join(scenario_dir, f"task_{task_id}", f"client_{cid:02d}.parquet")
                c_file = os.path.join(scenario_dir, f"task_{task_id}", f"client_{cid:02d}.csv")
                df_c = pd.read_parquet(p_file) if os.path.isfile(p_file) else pd.read_csv(c_file)
                f_cols = get_feature_columns(df_c)
                X_c = df_c[f_cols].to_numpy(dtype=np.float32, copy=False)
                y_c = np.array([LABEL2ID.get(lbl, -1) for lbl in df_c["label"].values], dtype=np.int64)
                del df_c
                ds_c = TabularMalwareDataset(X_c, y_c)
                drop_last = len(ds_c) > exp_cfg.fl.batch_size and (len(ds_c) % exp_cfg.fl.batch_size == 1)
                train_loaders[cid] = DataLoader(
                    ds_c,
                    batch_size=exp_cfg.fl.batch_size,
                    shuffle=True,
                    drop_last=drop_last,
                )

            task_metrics = server.run_task(
                task_id=task_id,
                n_new_classes=3,
                client_ids=task_cids,
                train_loaders=train_loaders,
                n_rounds=args.rounds_per_task,
                n_epochs=args.local_epochs,
            )
            task_metrics["task_id"] = task_id
            task_metrics["task_name"] = f"Task_{task_id + 1}"
            all_task_results.append(task_metrics)

            ck_path = task_metrics["final_checkpoint_path"]
            if ck_path is None:
                raise RuntimeError(f"Task {task_id + 1} produced no round checkpoint")
            checkpoint_paths.append(ck_path)

            train_loaders.clear()
            del train_loaders
            clear_device_cache()

        final_results = {
            "all_task_results": all_task_results,
            "continual_matrix": evaluator.continual_matrix.get_summary_dict(),
        }
        export_experiment_results(
            workbook_path=os.path.join(exp_cfg.output_root, "evaluation_results.xlsx"),
            experiment_name=exp_cfg.exp_name,
            method=args.method,
            setting="federated",
            rounds_per_task=args.rounds_per_task,
            client_num=args.n_clients,
            patch_size=exp_cfg.fl.batch_size,
            test_location=resolve_test_location(
                exp_cfg.scenario.prepared_data_dir,
                exp_cfg.scenario.feature_type,
            ),
            task_results=all_task_results,
            checkpoint_paths=checkpoint_paths,
            artifact_dir=os.path.join(exp_cfg.get_exp_dir(), "confusion_matrices"),
        )


    elif args.mode == "centralized":
        # Load centralized task data directly from prepared training split
        train_parquet = os.path.join(exp_cfg.scenario.prepared_data_dir, exp_cfg.scenario.feature_type, "train.parquet")
        train_csv = os.path.join(exp_cfg.scenario.prepared_data_dir, exp_cfg.scenario.feature_type, "train.csv")
        if os.path.isfile(train_parquet):
            train_df = pd.read_parquet(train_parquet)
        else:
            train_df = pd.read_csv(train_csv)

        feature_cols = get_feature_columns(train_df)
        full_train_X, full_train_y = {}, {}

        for t in range(5):
            task_classes = TASK_LABEL_MAP[t]
            t_df = train_df[train_df["label"].isin(task_classes)]
            if len(t_df) == 0:
                logger.warning(f"Task {t+1} has 0 samples matching {task_classes} in centralized training set.")
            X_t = t_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
            y_t = np.array([LABEL2ID.get(lbl, -1) for lbl in t_df["label"].values], dtype=np.int64)
            full_train_X[t] = X_t
            full_train_y[t] = y_t
            logger.info(f"Centralized Task {t+1}: {len(y_t):,} samples across classes {task_classes}")

        del train_df
        gc.collect()

        trainer = CentralizedTrainer(
            config=exp_cfg,
            full_train_X=full_train_X,
            full_train_y=full_train_y,
            evaluator=evaluator,
            logger=logger,
            validation_evaluator=validation_evaluator,
        )
        final_results = trainer.train_all_tasks(epochs_per_task=args.rounds_per_task)

    # Post-training evaluation visualization pipeline
    generate_post_training_plots(exp_dir, exp_cfg.exp_name, args.mode)

    logger.info("Scientific execution pipeline completed cleanly.")


def generate_post_training_plots(exp_dir: str, exp_name: str, mode: str) -> None:
    """Generate academic evaluation plots (learning curves, F1 bar chart, confusion matrix, forgetting matrix)
    saved into {exp_dir}/plots/.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = os.path.join(exp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        jsonl_candidates = [
            os.path.join(exp_dir, f"{exp_name}.jsonl"),
            os.path.join(exp_dir, "metrics.jsonl"),
        ]
        for f in os.listdir(exp_dir):
            if f.endswith(".jsonl"):
                jsonl_candidates.insert(0, os.path.join(exp_dir, f))

        jsonl_path = next((p for p in jsonl_candidates if os.path.isfile(p)), None)
        if not jsonl_path:
            return

        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        if not records:
            return

        flattened_records = []
        for r in records:
            if not isinstance(r, dict):
                continue
            row = dict(r)
            if "metrics" in row and isinstance(row["metrics"], dict):
                row.update(row["metrics"])
            flattened_records.append(row)

        if not flattened_records:
            return

        df_metrics = pd.DataFrame(flattened_records)

        # 1. Learning Curves
        if "accuracy" in df_metrics.columns:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            if "step" in df_metrics.columns and df_metrics["step"].notna().any():
                x_col = "step"
                x_label = "Communication Round" if mode == "federated" else "Epoch"
            elif "round" in df_metrics.columns and mode == "federated":
                x_col = "round"
                x_label = "Round"
            elif "round_id" in df_metrics.columns and mode == "federated":
                x_col = "round_id"
                x_label = "Round"
            elif "epoch" in df_metrics.columns:
                x_col = "epoch"
                x_label = "Epoch"
            else:
                x_col = "task_id"
                x_label = "Task ID"

            ax.plot(df_metrics[x_col], df_metrics["accuracy"] * 100, marker="o", label="Accuracy (%)", color="#1f77b4")
            f1_col = "f1_macro" if "f1_macro" in df_metrics.columns else ("macro_f1" if "macro_f1" in df_metrics.columns else None)
            if f1_col:
                ax.plot(df_metrics[x_col], df_metrics[f1_col] * 100, marker="s", label="Macro-F1 (%)", color="#ff7f0e")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Metric (%)")
            ax.set_title(f"Evaluation Trajectory: {exp_name}")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.savefig(os.path.join(plots_dir, "learning_curves.png"), bbox_inches="tight", dpi=300)
            fig.savefig(os.path.join(plots_dir, "learning_curves.pdf"), bbox_inches="tight")
            plt.close(fig)

        # 2. Final Confusion Matrix Heatmap
        cm_records = [r for r in records if "confusion_matrix" in r]
        if cm_records:
            from sklearn.metrics import ConfusionMatrixDisplay
            last_record = cm_records[-1]
            last_cm = np.array(last_record["confusion_matrix"])
            labels = last_record.get("confusion_matrix_labels", list(range(len(last_cm))))
            display_labels = [ID2LABEL.get(int(lbl), str(lbl)) for lbl in labels]

            fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
            disp = ConfusionMatrixDisplay(confusion_matrix=last_cm, display_labels=display_labels)
            disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format="d", colorbar=True)
            ax.set_title(f"Final Task Confusion Matrix: {exp_name}", fontsize=12, fontweight="bold", pad=12)
            fig.savefig(os.path.join(plots_dir, "final_confusion_matrix.png"), bbox_inches="tight", dpi=300)
            fig.savefig(os.path.join(plots_dir, "final_confusion_matrix.pdf"), bbox_inches="tight")
            plt.close(fig)

        # 3. Per-Family F1 Score Bar Chart
        if "per_class_f1" in records[-1]:
            per_f1 = records[-1]["per_class_f1"]
            if isinstance(per_f1, dict):
                classes = list(per_f1.keys())
                vals = [per_f1[c] * 100 for c in classes]
                fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
                ax.bar(classes, vals, color="#2ca02c", edgecolor="black", alpha=0.85)
                ax.set_ylabel("F1 Score (%)", fontsize=11)
                ax.set_title(f"Final Per-Family F1 Score: {exp_name}", fontsize=12, fontweight="bold", pad=12)
                ax.set_xticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
                ax.grid(True, axis="y", linestyle="--", alpha=0.5)
                fig.savefig(os.path.join(plots_dir, "per_family_f1.png"), bbox_inches="tight", dpi=300)
                fig.savefig(os.path.join(plots_dir, "per_family_f1.pdf"), bbox_inches="tight")
                plt.close(fig)

        # 4. Forgetting Matrix Heatmap
        continual_records = [r for r in records if "continual_matrix" in r or "task_matrix" in r]
        if continual_records:
            mat_key = "continual_matrix" if "continual_matrix" in continual_records[-1] else "task_matrix"
            raw_mat = continual_records[-1][mat_key]
            if isinstance(raw_mat, (list, np.ndarray)):
                mat = np.array(raw_mat)
                fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
                im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Metric Score", fontsize=10)
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        val = mat[i, j]
                        if not np.isnan(val):
                            tc = "white" if val > 0.5 else "black"
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=tc, fontsize=9)
                ax.set_title(f"Task Evaluation Matrix $(t, \\tau)$: {exp_name}", fontsize=11, fontweight="bold", pad=10)
                ax.set_xlabel(r"Evaluated Task $\tau$", fontsize=10)
                ax.set_ylabel(r"Current Task $t$", fontsize=10)
                ax.set_xticks(range(mat.shape[1]))
                ax.set_xticklabels([f"T{j+1}" for j in range(mat.shape[1])])
                ax.set_yticks(range(mat.shape[0]))
                ax.set_yticklabels([f"T{i+1}" for i in range(mat.shape[0])])
                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, "forgetting_matrix.png"), bbox_inches="tight", dpi=300)
                fig.savefig(os.path.join(plots_dir, "forgetting_matrix.pdf"), bbox_inches="tight")
                plt.close(fig)

    except Exception as e:
        print(f"[Warning] Failed to generate post-training plots: {e}")


if __name__ == "__main__":
    main()
