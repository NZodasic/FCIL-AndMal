"""Logging utilities for experiments.

Provides structured logging with multiple backends.

"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import csv


class ExperimentLogger:
    """Structured logger for experiments.

    Supports both file logging (JSON lines format) and console output.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        log_level: int = logging.INFO
    ):
        """Initialize experiment logger.

        Args:
            log_dir: Directory for log files.
            experiment_name: Name of the experiment.
            log_level: Logging level.
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler for structured logs
        log_file = self.log_dir / f"{experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # JSON lines log for structured data
        self.json_log_path = self.log_dir / f"{experiment_name}.jsonl"

        # Metrics CSV
        self.metrics_csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.metrics_writer = None
        self.metrics_file = None

    def log(self, level: int, message: str, **kwargs) -> None:
        """Log a message.

        Args:
            level: Logging level.
            message: Log message.
            **kwargs: Additional structured data.
        """
        self.logger.log(level, message)

        # Also write to JSON lines
        if kwargs:
            self.log_structured(kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(logging.INFO, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(logging.DEBUG, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(logging.ERROR, message, **kwargs)

    def section(self, title: str) -> None:
        """Log a top-level experiment section."""
        self.info(f"{'=' * 12} {title} {'=' * 12}")

    def subsection(self, title: str) -> None:
        """Log a task-level experiment section."""
        self.info(f"{'-' * 8} {title} {'-' * 8}")

    def log_task_evaluation(
        self, task_id: int, task_name: str, metrics: Dict[str, Any]
    ) -> None:
        """Write a structured task-final evaluation record."""
        self.log_structured({
            "event": "task_evaluation",
            "task_id": task_id,
            "task_name": task_name,
            "metrics": metrics,
        })

    def log_final_results(self, results: Dict[str, Any]) -> None:
        """Write the final structured experiment summary."""
        self.log_structured({"event": "final_results", "results": results})

    def log_structured(self, data: Dict[str, Any]) -> None:
        """Log structured data as JSON line.

        Args:
            data: Dictionary of data to log.
        """
        data['timestamp'] = datetime.now().isoformat()
        data['experiment'] = self.experiment_name

        with open(self.json_log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        task_id: Optional[int] = None,
        round_id: Optional[int] = None
    ) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metrics.
            step: Global step number.
            task_id: Current task ID.
            round_id: Current round ID.
        """
        # Add context
        log_data = {
            'metrics': metrics,
            'step': step,
            'task_id': task_id,
            'round_id': round_id
        }

        # Log to JSON
        self.log_structured(log_data)

        # Log to CSV
        self._write_metrics_csv(metrics, step, task_id, round_id)

        # Log to console
        metric_str = ', '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in metrics.items()])
        self.info(f"Metrics: {metric_str}")

    def _write_metrics_csv(
        self,
        metrics: Dict[str, Any],
        step: Optional[int],
        task_id: Optional[int],
        round_id: Optional[int]
    ) -> None:
        """Write metrics to CSV file."""
        # Initialize CSV writer if needed
        if self.metrics_file is None:
            self.metrics_file = open(self.metrics_csv_path, 'w', newline='')
            fieldnames = ['timestamp', 'step', 'task_id', 'round_id'] + list(metrics.keys())
            self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=fieldnames)
            self.metrics_writer.writeheader()

        # Write row
        row = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'task_id': task_id,
            'round_id': round_id
        }
        row.update(metrics)
        self.metrics_writer.writerow(row)
        self.metrics_file.flush()

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration dictionary.
        """
        config_path = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.info(f"Configuration saved to {config_path}")

    def close(self) -> None:
        """Close logger and cleanup."""
        if self.metrics_file:
            self.metrics_file.close()
            self.metrics_file = None
            self.metrics_writer = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def get_logger(log_dir: str = "./logs", experiment_name: str = "fcil_andmal", exp_name: Optional[str] = None) -> ExperimentLogger:
    """Helper function to get an ExperimentLogger instance."""
    name = exp_name or experiment_name or "fcil_andmal"
    return ExperimentLogger(log_dir=log_dir, experiment_name=name)
