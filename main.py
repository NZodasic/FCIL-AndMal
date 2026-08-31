"""
Main Scientific Experiment Orchestrator for FCIL on CIC-AndMal-2020.
Integrates Data Pipeline, Neural Models, Continual Learning Methods,
Federated Aggregators (FedAvg, FedNova), Logging, Checkpointing, and Visualization.
"""

import os
import sys
import argparse
import json

# Prevent generation of __pycache__ byte-code files
sys.dont_write_bytecode = True

import torch
import numpy as np

# Ensure package path is recognized
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    ExperimentConfig,
    ScenarioConfig,
    ModelConfig,
    ILConfig,
    FLConfig,
    TASK_LABEL_MAP,
    ALL_LABELS,
    LABEL2ID,
    ID2LABEL,
    get_batch_size_for_mode,
)
from utils.seed import set_seed
from utils.logger import AcademicLogger, get_logger
from data.synthetic_generator import generate_synthetic_raw_andmal2020
from data.prepare_dataset import AndMal2020DataPreparer
from data.partition import FCILDataPartitioner
from data.dataset import load_heldout_test_set, TabularMalwareDataset, get_participating_clients
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
                        help="Generate synthetic CIC-AndMal-2020 raw files if real dataset is not present")
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
    parser.add_argument("--device", type=str, default="cpu", help="Computation device ('cpu', 'cuda')")

    # Reproducibility & Output
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--output_root", type=str, default="./EXPERIMENT",
                        help="Directory to save experimental logs, metrics, plots, and checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to checkpoint file (.pt) to resume training from")

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
    scenario_cfg = ScenarioConfig(
        feature_type=args.feature_type,
        n_clients=args.n_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
        raw_data_dir=args.raw_root,
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

    fl_cfg = FLConfig(
        aggregator=args.aggregator,
        n_tasks=5,
        rounds_per_task=args.rounds_per_task,
        local_epochs=args.local_epochs,
        batch_size=batch_size,
        lr=args.lr,
        device=args.device
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
        seed=args.seed
    )
    return exp_cfg


def ensure_dataset_ready(exp_cfg: ExperimentConfig, generate_synthetic: bool = False) -> None:
    """
    Ensure raw data, Stage 1 prepared files, and Stage 2 partitions exist.
    Generates synthetic data and automated partitions if missing.
    """
    raw_root = exp_cfg.scenario.raw_data_dir
    prepared_dir = exp_cfg.scenario.prepared_data_dir
    scenario_dir = exp_cfg.scenario.get_scenario_dir()
    test_parquet = os.path.join(prepared_dir, exp_cfg.scenario.feature_type, "test.parquet")
    partition_table = os.path.join(scenario_dir, "partition_table.csv")

    # Check if raw data exists or if synthetic generation is requested
    if not os.path.isdir(raw_root) or generate_synthetic:
        print("\n[Data Pipeline] Raw data directory missing or synthetic flag provided. Generating synthetic dataset...")
        generate_synthetic_raw_andmal2020(
            root_dir=raw_root,
            samples_per_class=350,
            static_dim=300,
            dynamic_dim=141,
            seed=exp_cfg.seed
        )

    # Check if Stage 1 prepared data exists
    if not os.path.isfile(test_parquet):
        print(f"\n[Data Pipeline] Stage 1 prepared data not found for '{exp_cfg.scenario.feature_type}'. Running preparer...")
        preparer = AndMal2020DataPreparer(
            raw_root=raw_root,
            output_dir=prepared_dir,
            seed=exp_cfg.seed
        )
        preparer.run_all(data_type=exp_cfg.scenario.feature_type)

    # Check if Stage 2 partition files exist
    if not os.path.isfile(partition_table):
        print(f"\n[Data Pipeline] Stage 2 partition table not found at {partition_table}. Running partitioner...")
        import pandas as pd
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
    ensure_dataset_ready(exp_cfg, generate_synthetic=args.generate_synthetic)

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
    exp_cfg.save_json(config_json_path)
    logger.info(
        f"Held-out test set loaded: {len(test_y):,} samples, "
        f"{exp_cfg.model.input_dim} features, {len(available_labels)} observed classes."
    )
    if missing_labels:
        logger.warning(
            f"Configured classes absent from the dataset: {missing_labels}. "
            "Results are not a complete 15-class benchmark."
        )
    if args.prepare_only:
        logger.info(
            "Dataset preparation completed; training was skipped because "
            "--prepare_only was specified."
        )
        return

    evaluator = ContinualEvaluator(
        test_X=test_X,
        test_y=test_y,
        batch_size=exp_cfg.fl.batch_size,
        device=torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    )

    # Run selected mode
    if args.mode == "federated":
        import pandas as pd
        from models.fcil_model import FCILNet
        from methods import build_il_method
        from federated.server import FLServer
        from federated.client import FLClient
        from federated.aggregators import build_aggregator
        from training.checkpoint import CheckpointManager

        agg = build_aggregator(args.aggregator)
        ckpt_mgr = CheckpointManager(
            checkpoint_dir=os.path.join(exp_cfg.get_exp_dir(), "checkpoints"),
            logger=logger,
        )
        global_model = FCILNet(exp_cfg.model).to(
            torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
        )
        server = FLServer(
            global_model=global_model,
            aggregator=agg,
            device=args.device,
            config=exp_cfg,
            evaluator=evaluator,
            logger=logger,
            checkpoint_manager=ckpt_mgr,
        )

        from federated.il_strategy_adapter import ILMethodStrategyAdapter
        scenario_dir = exp_cfg.scenario.get_scenario_dir()
        active_cids = get_participating_clients(scenario_dir, 0)
        for cid in active_cids:
            client_model = FCILNet(exp_cfg.model)
            client_il = build_il_method(exp_cfg.il)
            adapter = ILMethodStrategyAdapter(
                model=client_model,
                il_method=client_il,
                lr=exp_cfg.fl.lr,
                device=args.device,
                classes_per_task=exp_cfg.model.classes_per_task,
            )
            client = FLClient(
                client_id=cid,
                model=client_model,
                strategy=adapter,
                device=args.device,
            )
            server.register_client(client)
        logger.info(f"Registered {len(active_cids)} clients | IL: {args.method.upper()} | Agg: {args.aggregator.upper()}")

        all_task_results = []
        checkpoint_paths = []
        from utils.results import export_experiment_results, resolve_test_location

        for task_id in range(5):
            task_cids = get_participating_clients(scenario_dir, task_id)
            train_loaders = {}
            for cid in task_cids:
                if cid not in server.clients:
                    client_model = FCILNet(exp_cfg.model)
                    client_il = build_il_method(exp_cfg.il)
                    adapter = ILMethodStrategyAdapter(
                        model=client_model,
                        il_method=client_il,
                        lr=exp_cfg.fl.lr,
                        device=args.device,
                        classes_per_task=exp_cfg.model.classes_per_task,
                    )
                    client = FLClient(
                        client_id=cid,
                        model=client_model,
                        strategy=adapter,
                        device=args.device,
                    )
                    server.register_client(client)
                p_file = os.path.join(scenario_dir, f"task_{task_id}", f"client_{cid:02d}.parquet")
                c_file = os.path.join(scenario_dir, f"task_{task_id}", f"client_{cid:02d}.csv")
                df_c = pd.read_parquet(p_file) if os.path.isfile(p_file) else pd.read_csv(c_file)
                from data.schema import get_feature_columns
                f_cols = get_feature_columns(df_c)
                X_c = df_c[f_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
                y_c = np.array([LABEL2ID.get(lbl, -1) for lbl in df_c["label"].values], dtype=np.int64)
                from data.dataset import TabularMalwareDataset
                from torch.utils.data import DataLoader
                ds_c = TabularMalwareDataset(X_c, y_c)
                train_loaders[cid] = DataLoader(ds_c, batch_size=exp_cfg.fl.batch_size, shuffle=True)

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
        # Load centralized task data
        import pandas as pd
        full_train_X, full_train_y = {}, {}
        scenario_dir = exp_cfg.scenario.get_scenario_dir()
        
        for t in range(5):
            t_task_dfs = []
            active_cids = get_participating_clients(scenario_dir, t)
            for cid in active_cids:
                p_file = os.path.join(scenario_dir, f"task_{t}", f"client_{cid:02d}.parquet")
                c_file = os.path.join(scenario_dir, f"task_{t}", f"client_{cid:02d}.csv")
                df_c = pd.read_parquet(p_file) if os.path.isfile(p_file) else pd.read_csv(c_file)
                t_task_dfs.append(df_c)

            t_df = pd.concat(t_task_dfs, ignore_index=True)
            f_cols = get_feature_columns(t_df)
            X_t = t_df[f_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
            y_t = np.array([LABEL2ID.get(lbl, -1) for lbl in t_df["label"].values], dtype=np.int64)
            full_train_X[t] = X_t
            full_train_y[t] = y_t

        trainer = CentralizedTrainer(
            config=exp_cfg,
            full_train_X=full_train_X,
            full_train_y=full_train_y,
            evaluator=evaluator,
            logger=logger
        )
        final_results = trainer.train_all_tasks(epochs_per_task=args.rounds_per_task)

    logger.info("Scientific execution pipeline completed cleanly.")


if __name__ == "__main__":
    main()
