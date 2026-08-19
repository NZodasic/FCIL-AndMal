"""
Unit Tests for Federated Aggregators, Clients, and Checkpoint Management.
"""

import os
import shutil
import unittest
import torch
import numpy as np

from config import ModelConfig
from models.fcil_model import FCILNet
from federated.aggregators import FedAvgAggregator, FedNovaAggregator
from federated.prototype_manager import GlobalPrototypeManager
from training.checkpoint import CheckpointManager


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


if __name__ == "__main__":
    unittest.main()
