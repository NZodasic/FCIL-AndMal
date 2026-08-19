"""Run script for FL data partitioning (Stage 2).

CLI entry point for creating federated learning partitions.

"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ScenarioConfig, TASK_LABEL_MAP
from data.fl_data_partition.partitioner import FLDataPartitioner, save_partitions


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create FL partitions for CIC-AndMal-2020'
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Path to prepared dataset CSV'
    )
    parser.add_argument(
        '--static_dataset', type=str, default=None,
        help='Path to static features CSV (for fused)'
    )
    parser.add_argument(
        '--dynamic_dataset', type=str, default=None,
        help='Path to dynamic features CSV (for fused)'
    )

    # Scenario arguments
    parser.add_argument(
        '--feature_type', type=str, default='dynamic',
        choices=['static', 'dynamic', 'fused'],
        help='Type of features'
    )
    parser.add_argument(
        '--n_clients', type=int, nargs='+', default=[20, 50],
        help='Number of clients (can specify multiple)'
    )
    parser.add_argument(
        '--dirichlet_alpha', type=float, default=0.5,
        help='Dirichlet concentration parameter (lower = more non-IID)'
    )
    parser.add_argument(
        '--base_ratio', type=float, default=0.6,
        help='Initial client participation ratio'
    )
    parser.add_argument(
        '--step', type=float, default=0.1,
        help='Client participation increment per task'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./fl_data_partitions',
        help='Output directory'
    )
    parser.add_argument(
        '--label_col', type=str, default='label',
        help='Name of label column'
    )

    return parser.parse_args()


def load_fused_data(static_path: str, dynamic_path: str) -> pd.DataFrame:
    """Load and fuse static and dynamic features.

    Args:
        static_path: Path to static features CSV.
        dynamic_path: Path to dynamic features CSV.

    Returns:
        Fused DataFrame with combined features.
    """
    print("Loading static features...")
    static_df = pd.read_csv(static_path)

    print("Loading dynamic features...")
    dynamic_df = pd.read_csv(dynamic_path)

    # Ensure Sample_ID exists in both
    if 'Sample_ID' not in static_df.columns:
        raise ValueError("Sample_ID column not found in static data")
    if 'Sample_ID' not in dynamic_df.columns:
        raise ValueError("Sample_ID column not found in dynamic data")

    # Get feature columns (exclude label, Sample_ID, reboot_phase)
    static_features = [c for c in static_df.columns
                       if c not in ['label', 'Sample_ID']]
    dynamic_features = [c for c in dynamic_df.columns
                        if c not in ['label', 'Sample_ID', 'reboot_phase']]

    # Rename features to avoid collision
    static_df = static_df.rename(columns={
        c: f'static_{c}' for c in static_features
    })
    dynamic_df = dynamic_df.rename(columns={
        c: f'dynamic_{c}' for c in dynamic_features
    })

    # Merge on Sample_ID (inner join)
    print("Fusing features...")
    fused_df = pd.merge(
        static_df,
        dynamic_df,
        on='Sample_ID',
        how='inner',
        suffixes=('', '_dyn')
    )

    # Resolve label columns
    if 'label_dyn' in fused_df.columns:
        # Use static label as primary
        fused_df = fused_df.drop(columns=['label_dyn'])

    print(f"Fused dataset shape: {fused_df.shape}")
    print(f"  Static features: {len(static_features)}")
    print(f"  Dynamic features: {len(dynamic_features)}")
    print(f"  Total samples: {len(fused_df)}")

    return fused_df


def run_scenario(
    df: pd.DataFrame,
    config: ScenarioConfig,
    task_label_map: dict = TASK_LABEL_MAP
) -> None:
    """Run partitioning scenario.

    Args:
        df: Input DataFrame.
        config: Scenario configuration.
        task_label_map: Task to labels mapping.
    """
    print(f"\n{'='*60}")
    print(f"Running FL Data Partitioning")
    print(f"{'='*60}")
    print(f"Feature type: {config.feature_type}")
    print(f"Number of clients: {config.n_clients}")
    print(f"Dirichlet alpha: {config.dirichlet_alpha}")
    print(f"Seed: {config.seed}")
    print(f"Dataset shape: {df.shape}")

    # Create partitioner
    partitioner = FLDataPartitioner(config)

    # Partition data
    partitions = partitioner.partition_data(df, task_label_map)

    # Save partitions
    output_dir = Path(config.output_dir) / config.feature_type / f"{config.n_clients}clients"
    save_partitions(partitions, str(output_dir), config, task_label_map)

    print(f"\n{'='*60}")
    print(f"Partitioning complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    args = parse_args()

    # Determine dataset path(s)
    if args.feature_type == 'fused':
        if args.static_dataset is None or args.dynamic_dataset is None:
            print("Error: --static_dataset and --dynamic_dataset required for fused")
            sys.exit(1)
        df = load_fused_data(args.static_dataset, args.dynamic_dataset)
    else:
        dataset_path = args.dataset
        if dataset_path is None:
            # Try to load from dataset_paths.json
            paths_file = Path('./prepared_data/dataset_paths.json')
            if paths_file.exists():
                import json
                with open(paths_file, 'r') as f:
                    paths = json.load(f)
                if args.feature_type == 'static':
                    dataset_path = paths.get('static_dataset')
                else:
                    dataset_path = paths.get('dynamic_dataset')

        if dataset_path is None:
            print(f"Error: --dataset required for {args.feature_type}")
            sys.exit(1)

        print(f"Loading {args.feature_type} features from {dataset_path}...")
        df = pd.read_csv(dataset_path)

    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}...")
    print(f"Labels: {df[args.label_col].unique()}")

    # Run for each client count
    for n_clients in args.n_clients:
        config = ScenarioConfig(
            n_clients=n_clients,
            feature_type=args.feature_type,
            dirichlet_alpha=args.dirichlet_alpha,
            base_ratio=args.base_ratio,
            step=args.step,
            seed=args.seed,
            min_samples_per_label_client=30,
            output_dir=args.output_dir,
            label_col=args.label_col
        )

        run_scenario(df, config)


if __name__ == '__main__':
    main()
