"""Main experiment runner for FL experiments.

Orchestrates federated class-incremental learning experiments.

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TASK_LABEL_MAP, get_labels_for_task, get_num_classes_for_task,
    PathBuilder, ExperimentConfig
)
from data.fl_data_partition.dataset_api import (
    FLTaskDataset, get_participating_clients, recommend_batch_size
)
from models.dynamic_cnn import DynamicCNN, DynamicTCN
from models.static_cnn import StaticCNN
from models.fused_model import FusedModel
from incremental.fine_tune import FineTune
from incremental.joint import JointTraining
from incremental.ewc import EWC
from incremental.lwf import LwF
from incremental.replay import Replay
from incremental.spcil import SPCIL
from incremental.malfsil import MALFSIL
from federated.client import FLClient
from federated.server import FLServer
from federated.aggregators.base import FedAvg, FedNova
from utils.checkpoint import CheckpointManager
from utils.logging import ExperimentLogger
from training.evaluator import Evaluator
from data.dataset import TabularMalwareDataset, load_heldout_test_set
from utils.results import export_experiment_results, resolve_test_location


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run FL experiments')

    # Experiment settings
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--feature_type', type=str, default='dynamic',
                       choices=['static', 'dynamic', 'fused'])
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--strategy', type=str, default='finetune',
                       choices=['finetune', 'joint', 'ewc', 'lwf', 'replay', 'spcil', 'malfsil'])
    parser.add_argument('--aggregator', type=str, default='fedavg',
                       choices=['fedavg', 'fednova'])

    # Training settings
    parser.add_argument('--n_tasks', type=int, default=5)
    parser.add_argument('--n_rounds', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)

    # Paths
    parser.add_argument('--data_dir', type=str, default='./fl_data_partitions')
    parser.add_argument('--prepared_dir', type=str, default='./prepared_data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--output_root', type=str, default='./EXPERIMENT')

    # Device
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()


def create_model(feature_type: str, initial_classes: int = 3):
    """Create model based on feature type.

    Args:
        feature_type: 'static', 'dynamic', or 'fused'.
        initial_classes: Number of classes in first task.

    Returns:
        Model instance.
    """
    if feature_type == 'static':
        return StaticCNN(input_dim=500, initial_classes=initial_classes)
    elif feature_type == 'dynamic':
        return DynamicCNN(input_dim=141, initial_classes=initial_classes)
    elif feature_type == 'fused':
        return FusedModel(
            static_input_dim=500,
            dynamic_input_dim=141,
            initial_classes=initial_classes
        )
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def create_strategy(strategy_name: str, model, lr: float, device: str):
    """Create incremental learning strategy.

    Args:
        strategy_name: Strategy name.
        model: Model instance.
        lr: Learning rate.
        device: Device.

    Returns:
        Strategy instance.
    """
    optimizer_fn = lambda p: optim.Adam(p, lr=lr)

    strategies = {
        'finetune': FineTune,
        'joint': JointTraining,
        'ewc': EWC,
        'lwf': LwF,
        'replay': Replay,
        'spcil': SPCIL,
        'malfsil': MALFSIL,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name](model, optimizer_fn, device)


def create_aggregator(aggregator_name: str):
    """Create aggregation algorithm.

    Args:
        aggregator_name: 'fedavg' or 'fednova'.

    Returns:
        Aggregator instance.
    """
    if aggregator_name == 'fedavg':
        return FedAvg()
    elif aggregator_name == 'fednova':
        return FedNova()
    else:
        raise ValueError(f"Unknown aggregator: {aggregator_name}")


def run_experiment(args):
    """Run a single experiment.

    Args:
        args: Command line arguments.
    """
    # Setup paths
    path_builder = PathBuilder('.')
    scenario_dir = Path(args.data_dir) / args.feature_type / f"{args.n_clients}clients"

    # Setup logging
    logger = ExperimentLogger(args.log_dir, args.experiment_name)
    logger.info(f"Starting experiment: {args.experiment_name}")
    logger.log_config(vars(args))

    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(
        args.checkpoint_dir,
        args.experiment_name,
        keep_last_n=max(3, args.n_tasks),
    )

    # Create global model
    initial_classes = len(get_labels_for_task(0))
    global_model = create_model(args.feature_type, initial_classes)
    global_model.to(args.device)

    logger.info(f"Model created: {global_model.__class__.__name__}")
    logger.info(f"Initial classes: {initial_classes}")

    # Create aggregator
    aggregator = create_aggregator(args.aggregator)

    # Create server
    server = FLServer(global_model, aggregator, args.device)

    test_X, test_y = load_heldout_test_set(args.prepared_dir, args.feature_type)

    # Create clients
    for cid in range(args.n_clients):
        client_model = create_model(args.feature_type, initial_classes)
        strategy = create_strategy(args.strategy, client_model, args.lr, args.device)
        client = FLClient(cid, client_model, strategy, args.device)
        server.register_client(client)

    logger.info(f"Created {args.n_clients} clients with strategy: {args.strategy}")

    # Run tasks
    all_results = []
    checkpoint_paths = []

    for task_id in range(args.n_tasks):
        # Get participating clients for this task
        active_clients = get_participating_clients(str(scenario_dir), task_id)

        if not active_clients:
            logger.warning(f"No active clients for task {task_id}")
            continue

        logger.info(f"Task {task_id}: {len(active_clients)} active clients")

        # Create data loaders
        train_loaders = {}
        for cid in active_clients:
            dataset = FLTaskDataset(str(scenario_dir), task_id, cid)
            train_loaders[cid] = dataset.as_dataloader(
                batch_size=args.batch_size,
                shuffle=True
            )

        # Get number of new classes
        n_new_classes = len(get_labels_for_task(task_id))

        # Run FL for this task
        task_metrics = server.run_task(
            task_id=task_id,
            n_new_classes=n_new_classes,
            client_ids=active_clients,
            train_loaders=train_loaders,
            n_rounds=args.n_rounds,
            n_epochs=args.n_epochs
        )

        seen_mask = np.isin(test_y, range(get_num_classes_for_task(task_id)))
        test_loader = DataLoader(
            TabularMalwareDataset(test_X[seen_mask], test_y[seen_mask]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        final_metrics = Evaluator(
            server.global_model,
            device=args.device,
            n_classes=get_num_classes_for_task(task_id),
        ).evaluate(test_loader, task_id=task_id)
        task_metrics.update(final_metrics)
        all_results.append(task_metrics)

        logger.info(f"Task {task_id} completed")

        # Save checkpoint
        global_round = (task_id + 1) * args.n_rounds
        checkpoint_path = checkpoint_mgr.save_model(
            server.global_model,
            task_id=task_id,
            round_id=global_round,
            metadata={'metrics': task_metrics}
        )
        checkpoint_paths.append(checkpoint_path)

        workbook_path = export_experiment_results(
            workbook_path=str(Path(args.output_root) / "evaluation_results.xlsx"),
            experiment_name=args.experiment_name,
            method=args.strategy,
            setting="federated",
            rounds_per_task=args.n_rounds,
            client_num=args.n_clients,
            patch_size=args.batch_size,
            test_location=resolve_test_location(
                args.prepared_dir, args.feature_type
            ),
            task_results=all_results,
            checkpoint_paths=checkpoint_paths,
            artifact_dir=str(
                Path(args.output_root)
                / args.experiment_name
                / "confusion_matrices"
            ),
        )
        logger.info(f"Task-final results written to: {workbook_path}")

    logger.info("Experiment completed successfully")

    return all_results


def main():
    """Main entry point."""
    args = parse_args()

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Run experiment
    try:
        results = run_experiment(args)
        print(f"\nExperiment completed: {args.experiment_name}")
        print(f"Results saved to: {args.log_dir}/{args.experiment_name}")
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
