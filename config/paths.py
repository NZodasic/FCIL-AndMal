from pathlib import Path
from typing import Optional
import os


class PathBuilder:
    """Builder for project paths."""

    def __init__(self, base_dir: str = "."):
        """Initialize path builder.

        Args:
            base_dir: Base directory for the project.
        """
        self.base_dir = Path(base_dir)

    def get_prepared_data_dir(self) -> Path:
        """Get prepared data directory."""
        return self.base_dir / "prepared_data"

    def get_static_data_path(self) -> Path:
        """Get path to static features CSV."""
        return self.get_prepared_data_dir() / "static" / "static_all.csv"

    def get_dynamic_data_path(self) -> Path:
        """Get path to dynamic features CSV."""
        return self.get_prepared_data_dir() / "dynamic" / "dynamic_all.csv"

    def get_partition_dir(self, feature_type: str, n_clients: int) -> Path:
        """Get partition directory for a specific configuration.

        Args:
            feature_type: 'static', 'dynamic', or 'fused'.
            n_clients: Number of clients.

        Returns:
            Path to partition directory.
        """
        return self.base_dir / "fl_data_partitions" / feature_type / f"{n_clients}clients"

    def get_task_dir(self, feature_type: str, n_clients: int, task_id: int) -> Path:
        """Get directory for a specific task.

        Args:
            feature_type: 'static', 'dynamic', or 'fused'.
            n_clients: Number of clients.
            task_id: Task ID (0-4).

        Returns:
            Path to task directory.
        """
        return self.get_partition_dir(feature_type, n_clients) / f"task_{task_id}"

    def get_client_data_path(self, feature_type: str, n_clients: int,
                            task_id: int, client_id: int) -> Path:
        """Get path to client data file.

        Args:
            feature_type: 'static', 'dynamic', or 'fused'.
            n_clients: Number of clients.
            task_id: Task ID (0-4).
            client_id: Client ID.

        Returns:
            Path to client data file (parquet or csv).
        """
        task_dir = self.get_task_dir(feature_type, n_clients, task_id)
        parquet_path = task_dir / f"client_{client_id:02d}.parquet"
        csv_path = task_dir / f"client_{client_id:02d}.csv"

        if parquet_path.exists():
            return parquet_path
        return csv_path

    def get_metadata_path(self, feature_type: str, n_clients: int) -> Path:
        """Get path to partition metadata."""
        return self.get_partition_dir(feature_type, n_clients) / "metadata.json"

    def get_checkpoint_dir(self, experiment_name: str) -> Path:
        """Get checkpoint directory for an experiment."""
        return self.base_dir / "checkpoints" / experiment_name

    def get_task_checkpoint_dir(self, experiment_name: str, task_id: int) -> Path:
        """Get checkpoint directory for a specific task."""
        return self.get_checkpoint_dir(experiment_name) / f"task_{task_id}"

    def get_round_checkpoint_dir(self, experiment_name: str, task_id: int,
                                 round_id: int) -> Path:
        """Get checkpoint directory for a specific round."""
        return self.get_task_checkpoint_dir(experiment_name, task_id) / f"round_{round_id}"

    def get_log_dir(self, experiment_name: str) -> Path:
        """Get log directory for an experiment."""
        return self.base_dir / "logs" / experiment_name

    def get_results_dir(self, experiment_name: str) -> Path:
        """Get results directory for an experiment."""
        return self.base_dir / "results" / experiment_name

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        dirs = [
            self.get_prepared_data_dir(),
            self.base_dir / "fl_data_partitions",
            self.base_dir / "checkpoints",
            self.base_dir / "logs",
            self.base_dir / "results",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def get_default_data_root() -> Path:
    """Get default data root directory."""
    return get_project_root() / "data"


def get_default_checkpoint_root() -> Path:
    """Get default checkpoint root directory."""
    return get_project_root() / "checkpoints"


def get_default_log_root() -> Path:
    """Get default log root directory."""
    return get_project_root() / "logs"
