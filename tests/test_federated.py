"""
Unit Tests for Federated Aggregators, Clients, and Checkpoint Management.
"""

import os
import logging
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

from config import (
    ExperimentConfig,
    FLConfig,
    ILConfig,
    ModelConfig,
    ScenarioConfig,
    get_batch_size_for_mode,
)
from models.fcil_model import FCILNet
from federated.aggregators import FedAvgAggregator, FedNovaAggregator
from federated.server import FLServer
from federated.prototype_manager import GlobalPrototypeManager
from training.checkpoint import CheckpointManager
from training.trainer import CentralizedTrainer
from utils.logging import ExperimentLogger


class TestFederated(unittest.TestCase):
    """Test federated aggregation algorithms, prototype coordination, and checkpoint recovery."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = "./tmp_test_fl"
        os.makedirs(cls.tmp_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)

    def test_fedavg_aggregation(self):
        agg = FedAvgAggregator()
        global_state = {"w": torch.tensor([1.0, 1.0]), "b": torch.tensor([0.0])}
        client1_state = {"w": torch.tensor([2.0, 2.0]), "b": torch.tensor([1.0])}
        client2_state = {"w": torch.tensor([4.0, 4.0]), "b": torch.tensor([3.0])}

        # 100 samples from client1, 300 samples from client2 -> weighted average
        # w = 0.25 * [2, 2] + 0.75 * [4, 4] = [0.5 + 3.0, 0.5 + 3.0] = [3.5, 3.5]
        new_state = agg.aggregate(global_state, [client1_state, client2_state], [100, 300])
        self.assertTrue(torch.allclose(new_state["w"], torch.tensor([3.5, 3.5])))
        self.assertTrue(torch.allclose(new_state["b"], torch.tensor([2.5])))

    def test_fednova_aggregation(self):
        agg = FedNovaAggregator()
        global_state = {"w": torch.tensor([1.0, 1.0])}
        client1_state = {"w": torch.tensor([0.5, 0.5])}
        client2_state = {"w": torch.tensor([0.8, 0.8])}

        new_state = agg.aggregate(global_state, [client1_state, client2_state], [100, 200], [5, 10])
        self.assertTrue("w" in new_state)
        self.assertEqual(new_state["w"].shape, (2,))

    def test_prototype_manager(self):
        mgr = GlobalPrototypeManager()
        c1_protos = {0: (torch.tensor([1.0, 1.0]), 10), 1: (torch.tensor([2.0, 2.0]), 20)}
        c2_protos = {0: (torch.tensor([3.0, 3.0]), 10), 2: (torch.tensor([4.0, 4.0]), 30)}

        mgr.update_prototypes([c1_protos, c2_protos])
        protos = mgr.get_prototypes()
        # class 0 mean = (1.0*10 + 3.0*10)/20 = 2.0
        self.assertTrue(torch.allclose(protos[0], torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.allclose(protos[1], torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.allclose(protos[2], torch.tensor([4.0, 4.0])))

    def test_checkpoint_manager_save_and_load(self):
        cfg = ModelConfig(input_dim=10, latent_dim=8, classes_per_task=3, num_total_classes=15)
        model = FCILNet(cfg)
        ckpt_dir = os.path.join(self.tmp_dir, "checkpoints")
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

        save_path = mgr.save_task_checkpoint(
            task_id=0,
            round_id=5,
            global_model=model,
            continual_matrix_dict={"final_acc": 0.85},
            client_states={0: {"step": 5}}
        )
        self.assertTrue(os.path.isfile(save_path))

        # Test recovery
        new_model = FCILNet(cfg)
        state = mgr.load_checkpoint(save_path, model=new_model)
        self.assertEqual(state["task_id"], 0)
        self.assertEqual(state["current_classes"], 3)
        self.assertEqual(state["checkpoint_type"], "weights_only")
        self.assertIn("model_state_dict", state)
        self.assertNotIn("model", state)
        self.assertNotIn("optimizer_state_dict", state)
        self.assertTrue(
            all(tensor.device.type == "cpu" for tensor in state["model_state_dict"].values())
        )

    def test_checkpoint_manager_keeps_every_epoch_weight_file(self):
        cfg = ModelConfig(
            input_dim=10, latent_dim=8, classes_per_task=3, num_total_classes=15
        )
        model = FCILNet(cfg)
        ckpt_dir = os.path.join(self.tmp_dir, "epoch_checkpoints")
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

        paths = [
            mgr.save_weights_checkpoint(
                model,
                task_id=0,
                step_type="epoch",
                step_id=epoch,
                global_step=epoch,
            )
            for epoch in range(1, 4)
        ]

        self.assertEqual(len(set(paths)), 3)
        self.assertTrue(all(Path(path).is_file() for path in paths))
        self.assertEqual(
            [Path(path).name for path in paths],
            [
                "epoch_0001_weights.pt",
                "epoch_0002_weights.pt",
                "epoch_0003_weights.pt",
            ],
        )

    def test_fl_server_saves_weights_after_every_round(self):
        cfg = ModelConfig(
            input_dim=10, latent_dim=8, classes_per_task=3, num_total_classes=15
        )
        model = FCILNet(cfg)
        ckpt_dir = os.path.join(self.tmp_dir, "round_checkpoints")
        mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
        server = FLServer(
            global_model=model,
            device="cpu",
            checkpoint_manager=mgr,
        )
        server.run_round = lambda client_ids, train_loaders, n_epochs, **kwargs: {
            "client_metrics": []
        }

        result = server.run_task(
            task_id=0,
            n_new_classes=3,
            client_ids=[],
            train_loaders={},
            n_rounds=3,
            n_epochs=1,
        )

        paths = result["round_checkpoint_paths"]
        self.assertEqual(len(paths), 3)
        self.assertEqual(result["final_checkpoint_path"], paths[-1])
        self.assertTrue(all(Path(path).is_file() for path in paths))
        self.assertEqual(
            [metrics["global_round"] for metrics in result["round_metrics"]],
            [1, 2, 3],
        )

        task_two = server.run_task(
            task_id=1,
            n_new_classes=3,
            client_ids=[],
            train_loaders={},
            n_rounds=1,
            n_epochs=1,
        )
        task_two_state = mgr.load_checkpoint(task_two["final_checkpoint_path"])
        self.assertEqual(task_two_state["current_classes"], 6)

    def test_batch_sizes_are_fixed_by_mode(self):
        self.assertEqual(get_batch_size_for_mode("federated"), 256)
        self.assertEqual(get_batch_size_for_mode("centralized"), 1024)
        with self.assertRaises(ValueError):
            get_batch_size_for_mode("unknown")

    def test_centralized_trainer_saves_weights_after_every_epoch(self):
        class ContinualMatrixStub:
            @staticmethod
            def get_summary_dict():
                return {}

        class EvaluatorStub:
            def __init__(self):
                self.continual_matrix = ContinualMatrixStub()

            @staticmethod
            def evaluate_all_seen_tasks(model, task_id):
                n_classes = (task_id + 1) * 3
                metrics = {
                    "accuracy": 0.5,
                    "precision_macro": 0.5,
                    "recall_macro": 0.5,
                    "f1_macro": 0.5,
                    "precision_micro": 0.5,
                    "recall_micro": 0.5,
                    "f1_micro": 0.5,
                    "precision_weighted": 0.5,
                    "recall_weighted": 0.5,
                    "f1_weighted": 0.5,
                    "macro_f1": 0.5,
                    "f1_malware_avg": 0.5,
                    "average_forgetting": 0.0,
                    "continual_avg_accuracy": 0.5,
                    "confusion_matrix_labels": list(range(n_classes)),
                    "confusion_matrix": np.eye(n_classes, dtype=int).tolist(),
                }
                return metrics

        output_root = os.path.join(self.tmp_dir, "central_output")
        config = ExperimentConfig(
            exp_name="central_checkpoint_test",
            output_root=output_root,
            scenario=ScenarioConfig(prepared_data_dir=self.tmp_dir),
            model=ModelConfig(
                backbone_type="mlp",
                input_dim=4,
                latent_dim=4,
                hidden_dims=[8],
                classes_per_task=3,
                num_total_classes=15,
                dropout=0.0,
            ),
            il=ILConfig(method_name="finetune"),
            fl=FLConfig(batch_size=1024, device="cpu"),
        )
        full_train_X = {
            task_id: np.random.randn(6, 4).astype(np.float32)
            for task_id in range(5)
        }
        full_train_y = {
            task_id: np.repeat(np.arange(task_id * 3, task_id * 3 + 3), 2)
            for task_id in range(5)
        }
        logger = ExperimentLogger(
            os.path.join(output_root, "logs"),
            "central_checkpoint_test",
            log_level=logging.CRITICAL,
        )
        trainer = CentralizedTrainer(
            config=config,
            full_train_X=full_train_X,
            full_train_y=full_train_y,
            evaluator=EvaluatorStub(),
            logger=logger,
        )

        with patch(
            "training.trainer.export_experiment_results",
            return_value=os.path.join(output_root, "results.xlsx"),
        ):
            trainer.train_all_tasks(epochs_per_task=2)
        logger.close()

        checkpoint_root = Path(config.get_exp_dir(), "checkpoints")
        epoch_paths = sorted(checkpoint_root.glob("task_*/epoch_*_weights.pt"))
        self.assertEqual(len(epoch_paths), 10)
        for task_id in range(1, 6):
            task_paths = sorted((checkpoint_root / f"task_{task_id:02d}").glob("*.pt"))
            self.assertEqual(
                [path.name for path in task_paths],
                ["epoch_0001_weights.pt", "epoch_0002_weights.pt"],
            )


if __name__ == "__main__":
    unittest.main()
