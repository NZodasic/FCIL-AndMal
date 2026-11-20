"""Tests for the complete classification metric and result-reporting schema."""

import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.logging import ExperimentLogger
from utils.metrics import (
    CLASSIFICATION_METRIC_KEYS,
    compute_classification_metrics,
    format_classification_metrics,
    format_confusion_matrix,
)
from utils.results import (
    RESULT_COLUMNS,
    build_result_rows,
    final_round_for_task,
    save_task_confusion_matrix,
    write_experiment_sheet,
)


OPENPYXL_AVAILABLE = importlib.util.find_spec("openpyxl") is not None


class TestMetricsReporting(unittest.TestCase):
    def test_all_ten_classification_metrics_are_reported(self):
        metrics = compute_classification_metrics(
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([0, 1, 1, 1, 2, 0]),
            seen_classes=[0, 1, 2],
        )

        expected = {
            "accuracy",
            "precision_macro",
            "precision_micro",
            "precision_weighted",
            "recall_macro",
            "recall_micro",
            "recall_weighted",
            "f1_macro",
            "f1_micro",
            "f1_weighted",
        }
        self.assertTrue(expected.issubset(metrics))
        self.assertEqual(len(metrics["confusion_matrix"]), 3)
        self.assertEqual(metrics["confusion_matrix_labels"], [0, 1, 2])

        formatted = format_classification_metrics(metrics)
        for label in (
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Micro Precision",
            "Micro Recall",
            "Micro F1",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1",
        ):
            self.assertIn(label, formatted)

    def test_evaluation_logger_records_all_metrics_and_final_matrix(self):
        metrics = compute_classification_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            seen_classes=[0, 1],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ExperimentLogger(temp_dir, "reporting")
            logger.log_evaluation(
                metrics,
                context="Task 1 | Epoch 5/10 | Test",
                task_id=0,
                step=5,
            )
            intermediate_log = Path(temp_dir, "reporting.log").read_text()
            self.assertNotIn("Confusion Matrix", intermediate_log)

            logger.log_evaluation(
                metrics,
                context="Task 1 | Epoch 10/10 | Final Test",
                task_id=0,
                step=10,
                include_confusion_matrix=True,
                label_names={0: "Benign", 1: "Malware"},
            )
            logger.close()

            log_text = Path(temp_dir, "reporting.log").read_text()
            self.assertIn("Confusion Matrix", log_text)
            self.assertIn("Benign", format_confusion_matrix(
                metrics, label_names={0: "Benign", 1: "Malware"}
            ))
            with Path(temp_dir, "reporting_metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(set(CLASSIFICATION_METRIC_KEYS).issubset(rows[0]))

    def test_federated_rows_use_cumulative_task_final_rounds(self):
        task_results = []
        for task_id in range(5):
            result = {column: 0.5 for column in RESULT_COLUMNS[8:18]}
            result["task_id"] = task_id
            task_results.append(result)

        rows = build_result_rows(
            task_results,
            experiment_name="case",
            method="malfsil",
            setting="federated",
            rounds_per_task=50,
            client_num=20,
            patch_size=256,
            test_location="test.parquet",
            confusion_matrix_paths=[f"cm_{i}.png" for i in range(5)],
            checkpoint_paths=[f"checkpoint_{i}.pt" for i in range(5)],
        )

        self.assertEqual([row["round"] for row in rows], [50, 100, 150, 200, 250])
        self.assertTrue(all(list(row) == RESULT_COLUMNS for row in rows))

    def test_centralized_rows_use_final_epoch(self):
        metrics = {column: 1.0 for column in RESULT_COLUMNS[8:18]}
        metrics["task_id"] = 0
        row = build_result_rows(
            [metrics],
            experiment_name="central",
            method="joint",
            setting="centralized",
            rounds_per_task=25,
            client_num="N/A",
            patch_size=128,
            test_location="test.csv",
            confusion_matrix_paths=["cm.png"],
            checkpoint_paths=["checkpoint.pt"],
        )[0]
        self.assertEqual(row["round"], 25)
        self.assertEqual(row["task_id"], 1)

    def test_final_round_validation(self):
        self.assertEqual(final_round_for_task(4, 50), 250)
        with self.assertRaises(ValueError):
            final_round_for_task(-1, 50)

    def test_confusion_matrix_artifact_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_task_confusion_matrix(
                {
                    "confusion_matrix": [[2, 1], [0, 3]],
                    "confusion_matrix_labels": [0, 1],
                },
                task_id=0,
                output_dir=temp_dir,
            )
            self.assertTrue(Path(path).is_file())

    @unittest.skipUnless(OPENPYXL_AVAILABLE, "openpyxl is not installed")
    def test_workbook_preserves_one_sheet_per_experiment(self):
        from openpyxl import load_workbook

        row = {column: "value" for column in RESULT_COLUMNS}
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = str(Path(temp_dir) / "results.xlsx")
            write_experiment_sheet(workbook_path, "case_a", [row])
            write_experiment_sheet(workbook_path, "case_b", [row])
            write_experiment_sheet(workbook_path, "case_a", [row, row])

            workbook = load_workbook(workbook_path, read_only=True)
            self.assertEqual(set(workbook.sheetnames), {"case_a", "case_b"})
            self.assertEqual(
                [cell.value for cell in workbook["case_a"][1]], RESULT_COLUMNS
            )
            self.assertEqual(workbook["case_a"].max_row, 3)

    @unittest.skipUnless(OPENPYXL_AVAILABLE, "openpyxl is not installed")
    def test_custom_exp_dir_reporting_and_summary_graphs(self):
        import json
        from openpyxl import load_workbook
        from report.generate_master_report import generate_master_excel
        from report.generate_summary_graphs import generate_all_summary_graphs

        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_exp = Path(tmp_dir) / "EXPERIMENT_seed123"
            runs = [
                ("centralized", "Centralized_Falcon_CENTRALIZED_DYNAMIC_FALCON"),
                ("20clients", "FL_Falcon_K20_FEDERATED_DYNAMIC_FALCON_FEDAVG_K20_E5_S5"),
                ("50clients", "FL_MalDrift_K50_FEDERATED_DYNAMIC_FINETUNE_FEDSGD_K50_E5_S5"),
            ]
            for folder, bname in runs:
                run_dir = custom_exp / folder / bname
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / f"{bname}_metrics.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "step", "task_id", "round_id",
                        "accuracy", "precision_macro", "recall_macro", "f1_macro",
                        "precision_micro", "recall_micro", "f1_micro",
                        "precision_weighted", "recall_weighted", "f1_weighted",
                    ])
                    for t in range(5):
                        writer.writerow([
                            "2026-11-20T00:00:00", (t + 1) * 50, t, (t + 1) * 50,
                            0.7 - 0.05 * t, 0.65, 0.65, 0.65 - 0.05 * t,
                            0.7 - 0.05 * t, 0.7 - 0.05 * t, 0.7 - 0.05 * t,
                            0.7, 0.7, 0.7,
                        ])
                with (run_dir / f"{bname}.jsonl").open("w") as f:
                    for t in range(5):
                        n_cls = (t + 1) * 3
                        rec = {
                            "event": "test_evaluation",
                            "task_id": t,
                            "step": (t + 1) * 50,
                            "confusion_matrix": np.eye(n_cls, dtype=int).tolist(),
                        }
                        f.write(json.dumps(rec) + "\n")

            generate_master_excel(exp_dir=str(custom_exp))
            excel_path = custom_exp / "report" / "FCIL_AndMal.xlsx"
            copy_path = custom_exp / "report" / "MASTER_overall_all_configs_FCIL.xlsx"
            self.assertTrue(excel_path.is_file())
            self.assertTrue(copy_path.is_file())

            wb = load_workbook(excel_path, read_only=True)
            self.assertIn("falcon", wb.sheetnames)
            self.assertIn("maldrift", wb.sheetnames)
            self.assertEqual(wb["results_all"].max_row, 16)

            generate_all_summary_graphs(exp_dir=str(custom_exp))
            fig_dir = custom_exp / "report" / "figures"
            self.assertTrue((fig_dir / "summary_fig1_accuracy_trajectories.png").is_file())
            self.assertTrue((fig_dir / "summary_fig8_master_overview_dashboard.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
