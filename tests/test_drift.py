"""Unit Tests for FL-MalDrift Concept Drift Detection & Client Stability Assessment.

Tests:
1. Streaming drift detectors (ADWIN, DDM, EDDM, HDDM-A, HDDM-W)
2. Unified Z-score normalizer with sliding window
3. Client parameter divergence computation (d_{c,t})
4. Client adaptive local stability threshold (theta_{c,t})
5. Client local adaptation (parameter shrinkage towards global state)
6. Integration of drift detection inside FLClient.local_train()
"""

import unittest
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from federated.drift.detectors import (
    ADWINDetector,
    DDMDetector,
    EDDMDetector,
    HDDMDetector,
    build_drift_detector,
)
from federated.drift.normalizer import DriftScoreNormalizer
from federated.drift.controller import ServerDriftController
from federated.client import FLClient
from incremental.base_strategy import IncrementalStrategy


class DummyStrategy(IncrementalStrategy):
    """Minimal dummy incremental strategy for testing client training loop."""

    def __init__(self, model: nn.Module):
        super().__init__(
            model=model,
            optimizer_fn=lambda p: torch.optim.SGD(p, lr=0.01),
            device="cpu",
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train_task(self, train_loader: DataLoader, task_id: int, n_epochs: int, **kwargs):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        last_loss = 0.0
        for _ in range(n_epochs):
            for bx, by in train_loader:
                self.optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                self.optimizer.step()
                last_loss = float(loss.item())
        return {"loss": last_loss, "accuracy": 0.85}

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        self.current_task = task_id
        self.n_classes_so_far += n_new_classes

    def after_task(self, task_id: int, **kwargs) -> None:
        pass


class TestDriftDetection(unittest.TestCase):
    """Test suite for concept drift detectors and client-side stability assessment."""

    def test_hddm_w_detection(self):
        detector = build_drift_detector("hddm_w", alpha=0.01, min_samples=15)
        # Low error rate stream (stationary benign phase)
        for _ in range(30):
            detector.update(0.0)
        self.assertFalse(detector.drift_detected)
        self.assertLess(detector.get_statistic(), 0.3)

        # Inject sudden surge of errors (concept drift event)
        drift_flag = False
        for _ in range(30):
            d, _ = detector.update(1.0)
            if d:
                drift_flag = True
                break
        self.assertTrue(drift_flag)

    def test_adwin_detection(self):
        detector = build_drift_detector("adwin", min_subwindow_size=5)
        # Stationary stream
        for _ in range(30):
            detector.update(0.0)
        self.assertFalse(detector.drift_detected)

        # Severe distribution shift
        drift_flag = False
        for _ in range(50):
            d, _ = detector.update(1.0)
            if d:
                drift_flag = True
                break
        self.assertTrue(drift_flag)

    def test_ddm_detection(self):
        detector = build_drift_detector("ddm", min_samples=25, warning_level=1.5, drift_level=2.5)
        # Initially low error rate
        for _ in range(50):
            detector.update(0.0)
        self.assertFalse(detector.drift_detected)

        # Shift to high error rate
        drift_flag = False
        for _ in range(80):
            d, _ = detector.update(1.0)
            if d:
                drift_flag = True
                break
        self.assertTrue(drift_flag)

    def test_eddm_detection(self):
        detector = build_drift_detector("eddm", min_errors=10)
        # Stationary: errors spaced far apart
        for i in range(250):
            detector.update(1.0 if i % 25 == 0 else 0.0)
        self.assertFalse(detector.drift_detected)

        # Sudden frequent errors: error every sample
        drift_flag = False
        for _ in range(40):
            d, _ = detector.update(1.0)
            if d:
                drift_flag = True
                break
        self.assertTrue(drift_flag)

    def test_drift_score_normalizer_bounds(self):
        norm = DriftScoreNormalizer(window_size=5)
        for z in [0.01, 0.05, 0.02, 0.50, 0.90, 1.20]:
            s = norm.normalize(z, state="stable")
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

        # Symbolic mapping test
        s_drift = norm.normalize(0.1, state="drift")
        self.assertGreaterEqual(s_drift, 0.8)
        s_warn = norm.normalize(0.1, state="warning")
        self.assertGreaterEqual(s_warn, 0.4)

    def test_client_divergence_computation(self):
        model = nn.Sequential(nn.Linear(10, 4))
        strategy = DummyStrategy(model)
        client = FLClient(client_id=1, model=model, strategy=strategy, device="cpu")

        global_state = {k: v.clone() for k, v in model.state_dict().items()}
        client.set_model_state(global_state)

        # Before modification, divergence is 0.0
        d_initial = client.compute_divergence()
        self.assertAlmostEqual(d_initial, 0.0, places=5)

        # Perturb local model weights
        with torch.no_grad():
            model[0].weight.add_(torch.ones_like(model[0].weight) * 2.0)

        d_modified = client.compute_divergence()
        self.assertGreater(d_modified, 0.0)

    def test_client_local_adaptation(self):
        model = nn.Sequential(nn.Linear(10, 4))
        strategy = DummyStrategy(model)
        client = FLClient(client_id=1, model=model, strategy=strategy, device="cpu")

        global_state = {k: v.clone() for k, v in model.state_dict().items()}
        client.set_model_state(global_state)

        # Perturb weights
        with torch.no_grad():
            model[0].weight.copy_(torch.ones_like(model[0].weight) * 10.0)

        # Apply local adaptation with shrink factor 0.5
        client.apply_local_adaptation(shrink_factor=0.5)

        adapted_weight = model[0].weight.data
        expected_weight = 0.5 * 10.0 + 0.5 * global_state["0.weight"]
        self.assertTrue(torch.allclose(adapted_weight, expected_weight))

    def test_client_local_stability_threshold_update(self):
        model = nn.Sequential(nn.Linear(10, 4))
        strategy = DummyStrategy(model)
        client = FLClient(
            client_id=1,
            model=model,
            strategy=strategy,
            device="cpu",
            theta_base=0.1,
            beta=0.8,
            lambda1=0.1,
            lambda2=0.1,
        )
        self.assertEqual(client.local_theta, 0.1)

        # Update with drift score 0.5 and val error 0.2
        # theta = 0.8 * 0.1 + 0.2 * (0.1 + 0.1 * 0.5 + 0.1 * 0.2)
        #       = 0.08 + 0.2 * (0.1 + 0.05 + 0.02) = 0.08 + 0.2 * 0.17 = 0.08 + 0.034 = 0.114
        updated = client.update_local_stability_threshold(s_ct=0.5, val_error=0.2)
        self.assertAlmostEqual(updated, 0.114, places=4)

    def test_client_train_with_drift_detection(self):
        model = nn.Sequential(nn.Linear(10, 3))
        strategy = DummyStrategy(model)
        detector = build_drift_detector("hddm_w", min_samples=5)
        client = FLClient(
            client_id=1,
            model=model,
            strategy=strategy,
            device="cpu",
            drift_detector=detector,
            theta_base=0.1,
        )

        global_state = {k: v.clone() for k, v in model.state_dict().items()}
        client.set_model_state(global_state)

        # Create dummy dataloader
        X = torch.randn(32, 10)
        y = torch.randint(0, 3, (32,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=16)

        metrics = client.local_train(loader, n_epochs=1)

        # Verify FL-MalDrift metrics are populated
        self.assertIn("drift_score", metrics)
        self.assertIn("raw_drift_statistic", metrics)
        self.assertIn("divergence", metrics)
        self.assertIn("local_threshold", metrics)
        self.assertIn("drift_detected", metrics)
        self.assertIn("drift_state", metrics)
        self.assertIn("drift_adapted", metrics)
        self.assertGreaterEqual(metrics["drift_score"], 0.0)
        self.assertLessEqual(metrics["drift_score"], 1.0)


    def test_server_drift_controller_warmup_and_adaptation(self):
        controller = ServerDriftController(
            warmup_rounds=2,
            window_size=5,
            alpha=0.8,
            k=1.5,
            target_participation_budget=0.7,
            gain_eta=0.05,
            tau_min=0.1,
            tau_max=0.9,
            default_tau_0=0.5,
        )

        # Round 1 & 2: Warm-up phase -> all clients admitted
        scores_r1 = {1: 0.1, 2: 0.2, 3: 0.3}
        adm1, tau1, info1 = controller.step(1, scores_r1)
        self.assertEqual(len(adm1), 3)
        self.assertTrue(info1["warmup_active"])

        scores_r2 = {1: 0.15, 2: 0.25, 3: 0.35}
        adm2, tau2, info2 = controller.step(2, scores_r2)
        self.assertEqual(len(adm2), 3)
        self.assertTrue(info2["warmup_active"])

        # Round 3: Online adaptive phase. Client 3 undergoes severe drift (score 0.95)
        scores_r3 = {1: 0.10, 2: 0.15, 3: 0.95}
        adm3, tau3, info3 = controller.step(3, scores_r3)
        self.assertFalse(info3["warmup_active"])
        # Client 3 should be filtered out
        self.assertIn(1, adm3)
        self.assertIn(2, adm3)
        self.assertNotIn(3, adm3)
        self.assertIn(3, info3["excluded_clients"])

    def test_server_drift_controller_client_recovery(self):
        controller = ServerDriftController(
            warmup_rounds=1,
            window_size=5,
            recovery_rounds=2,
            recovery_delta=0.1,
            default_tau_0=0.5,
        )
        # Warmup round
        controller.step(1, {1: 0.2, 2: 0.2})

        # Round 2: Client 2 drifts heavily -> excluded
        controller.step(2, {1: 0.2, 2: 0.9})
        self.assertIn(2, controller.quarantined_clients)

        # Round 3: Client 2 stabilizes (low score 0.1) -> 1st stable round
        adm3, _, info3 = controller.step(3, {1: 0.2, 2: 0.1})
        self.assertEqual(controller.consecutive_stable_rounds[2], 1)

        # Round 4: Client 2 remains stable -> 2nd stable round -> recovered!
        adm4, _, info4 = controller.step(4, {1: 0.2, 2: 0.1})
        self.assertIn(2, adm4)
        self.assertIn(2, info4["recovered_clients"])

    def test_fl_server_drift_filtering_integration(self):
        from federated.server import FLServer
        from federated.aggregators.base import FedAvg

        global_model = nn.Sequential(nn.Linear(10, 2))
        client1_model = copy.deepcopy(global_model)
        client2_model = copy.deepcopy(global_model)

        strat1 = DummyStrategy(client1_model)
        strat2 = DummyStrategy(client2_model)

        c1 = FLClient(client_id=1, model=client1_model, strategy=strat1, device="cpu")
        c2 = FLClient(client_id=2, model=client2_model, strategy=strat2, device="cpu")

        # Mock drift score on clients: client 1 stable (0.1), client 2 drifted (0.95)
        controller = ServerDriftController(warmup_rounds=0, default_tau_0=0.5)
        server = FLServer(
            global_model=global_model,
            aggregator=FedAvg(),
            device="cpu",
            drift_controller=controller,
            enable_drift_filtering=True,
        )
        server.register_clients([c1, c2])

        # Create dummy dataloaders
        X = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=10)
        train_loaders = {1: loader, 2: loader}

        # Override drift scores in mock client metrics
        original_train_1 = c1.local_train
        original_train_2 = c2.local_train

        def mock_train_1(*args, **kwargs):
            m = original_train_1(*args, **kwargs)
            m["drift_score"] = 0.1
            return m

        def mock_train_2(*args, **kwargs):
            m = original_train_2(*args, **kwargs)
            m["drift_score"] = 0.95
            return m

        c1.local_train = mock_train_1
        c2.local_train = mock_train_2

        round_res = server.run_round([1, 2], train_loaders, n_epochs=1)

        self.assertIn("drift_control", round_res)
        self.assertEqual(round_res["n_admitted"], 1)
        self.assertIn(1, round_res["admitted_clients"])
        self.assertNotIn(2, round_res["admitted_clients"])


class TestDriftConfig(unittest.TestCase):
    """Phase 4 integration tests: DriftConfig dataclass and CLI wiring."""

    def test_drift_config_importable_from_config_package(self):
        """DriftConfig must be importable from the top-level config package."""
        from config import DriftConfig
        cfg = DriftConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.detector_type, 'hddm_w')

    def test_drift_config_defaults_match_paper(self):
        """Default hyperparameters should match Algorithm 1 & 2 in Patel et al. 2026."""
        from config import DriftConfig
        cfg = DriftConfig(enabled=True)
        # Server threshold (Algorithm 2)
        self.assertEqual(cfg.warmup_rounds, 3)
        self.assertEqual(cfg.window_size, 10)
        self.assertAlmostEqual(cfg.alpha, 0.8)
        self.assertAlmostEqual(cfg.k, 1.5)
        self.assertAlmostEqual(cfg.p_star, 0.7)
        self.assertAlmostEqual(cfg.eta, 0.05)
        # Client local threshold (Algorithm 1)
        self.assertAlmostEqual(cfg.beta, 0.9)
        self.assertAlmostEqual(cfg.lambda1, 0.1)
        self.assertAlmostEqual(cfg.lambda2, 0.1)
        self.assertAlmostEqual(cfg.theta_base, 0.1)
        # Recovery
        self.assertEqual(cfg.recovery_rounds, 2)
        self.assertAlmostEqual(cfg.recovery_delta, 0.05)

    def test_experiment_config_drift_field_is_none_by_default(self):
        """ExperimentConfig.drift must default to None (backward-compat guarantee)."""
        from config import ExperimentConfig
        exp = ExperimentConfig()
        self.assertIsNone(exp.drift)

    def test_experiment_config_accepts_drift_config(self):
        """ExperimentConfig.drift should accept a DriftConfig instance."""
        from config import ExperimentConfig, DriftConfig
        drift = DriftConfig(enabled=True, detector_type='adwin', k=2.0)
        exp = ExperimentConfig(drift=drift)
        self.assertIsNotNone(exp.drift)
        self.assertTrue(exp.drift.enabled)
        self.assertEqual(exp.drift.detector_type, 'adwin')
        self.assertAlmostEqual(exp.drift.k, 2.0)

    def test_build_configs_from_args_drift_disabled_by_default(self):
        """Without --enable_drift the ExperimentConfig.drift field must remain None."""
        import argparse
        import sys

        # Simulate minimal CLI args (federated but with a compatible method)
        test_argv = [
            "--mode", "federated",
            "--method", "malfsil",
            "--generate_synthetic",
            "--allow_incomplete_benchmark",
        ]
        old_argv = sys.argv
        sys.argv = ["main.py"] + test_argv
        try:
            from main import parse_args, build_configs_from_args
            args = parse_args()
            exp_cfg = build_configs_from_args(args)
        finally:
            sys.argv = old_argv

        self.assertIsNone(exp_cfg.drift)

    def test_build_configs_from_args_enable_drift_flag(self):
        """With --enable_drift the ExperimentConfig.drift must be a DriftConfig(enabled=True)."""
        import sys

        test_argv = [
            "--mode", "federated",
            "--method", "malfsil",
            "--generate_synthetic",
            "--allow_incomplete_benchmark",
            "--enable_drift",
            "--drift_detector", "adwin",
            "--drift_k", "2.5",
            "--drift_warmup", "5",
        ]
        old_argv = sys.argv
        sys.argv = ["main.py"] + test_argv
        try:
            from main import parse_args, build_configs_from_args
            args = parse_args()
            exp_cfg = build_configs_from_args(args)
        finally:
            sys.argv = old_argv

        self.assertIsNotNone(exp_cfg.drift)
        self.assertTrue(exp_cfg.drift.enabled)
        self.assertEqual(exp_cfg.drift.detector_type, 'adwin')
        self.assertAlmostEqual(exp_cfg.drift.k, 2.5)
        self.assertEqual(exp_cfg.drift.warmup_rounds, 5)


class TestPaperFidelity(unittest.TestCase):
    """Validation tests ensuring code matches paper formulas exactly.

    Cross-references:
    - Normalizer discrete state mapping (Patel et al. 2026, Sec. 4.1, page 7)
    - θ_base warmup calibration (Patel et al. 2026, Sec. 4.4, page 12)
    """

    def test_normalizer_drift_state_maps_to_exactly_1(self):
        """Paper Sec. 4.1: discrete drift → 1.0 (exact, not a minimum floor)."""
        norm = DriftScoreNormalizer(window_size=5)
        # Prime the history so z-score path is active
        for z in [0.1, 0.2, 0.3, 0.4, 0.5]:
            norm.record_statistic(z)
        score = norm.normalize(0.0, state="drift")
        self.assertEqual(score, 1.0,
            "Paper specifies drift → 1.0 as an exact discrete mapping, not a floor.")

    def test_normalizer_warning_state_floor_is_0_5(self):
        """Paper Sec. 4.1: discrete warning → 0.5 (floor, not exact override)."""
        norm = DriftScoreNormalizer(window_size=5)
        for z in [0.1, 0.2, 0.3, 0.4, 0.5]:
            norm.record_statistic(z)
        score = norm.normalize(0.0, state="warning")
        self.assertGreaterEqual(score, 0.5,
            "Warning state must produce score >= 0.5 as per paper mapping.")
        self.assertLessEqual(score, 1.0)

    def test_normalizer_stable_state_returns_continuous_score(self):
        """Paper Sec. 4.1: no-alarm → 0 for the discrete mapping baseline."""
        norm = DriftScoreNormalizer(window_size=5)
        # Before any history, stable state with z=0 → 0.0
        score = norm.normalize(0.0, state="stable")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 0.5)

    def test_theta_base_calibrated_from_warmup_median_divergence(self):
        """Paper Sec. 4.4: θ_base initialized from median divergence during warmup."""
        model = nn.Sequential(nn.Linear(10, 4))
        strategy = DummyStrategy(model)
        client = FLClient(client_id=1, model=model, strategy=strategy, device="cpu",
                          theta_base=0.1)

        global_state = {k: v.clone() for k, v in model.state_dict().items()}
        client.set_model_state(global_state)

        # Simulate 3 warmup rounds with increasing perturbations
        divergences = []
        for delta in [0.05, 0.10, 0.20]:
            with torch.no_grad():
                model[0].weight.copy_(
                    global_state["0.weight"] + delta * torch.ones_like(global_state["0.weight"])
                )
            client.record_warmup_divergence()
            divergences.append(client.compute_divergence())

        # Calibrate theta_base from warmup median
        calibrated = client.calibrate_theta_base()
        expected_median = float(np.median(divergences))
        self.assertAlmostEqual(calibrated, expected_median, places=4,
            msg="θ_base must equal the median divergence observed during warmup (paper Sec. 4.4).")
        self.assertAlmostEqual(client.theta_base, expected_median, places=4)
        self.assertTrue(client._theta_base_calibrated)

    def test_theta_base_calibration_is_idempotent(self):
        """Calling calibrate_theta_base() a second time must be a no-op."""
        model = nn.Sequential(nn.Linear(10, 4))
        strategy = DummyStrategy(model)
        client = FLClient(client_id=1, model=model, strategy=strategy, device="cpu",
                          theta_base=0.1)
        global_state = {k: v.clone() for k, v in model.state_dict().items()}
        client.set_model_state(global_state)

        with torch.no_grad():
            model[0].weight.copy_(global_state["0.weight"] + 0.5)
        client.record_warmup_divergence()
        first_value = client.calibrate_theta_base()

        # Modify model again — should have no effect on theta_base
        with torch.no_grad():
            model[0].weight.copy_(global_state["0.weight"] + 5.0)
        client.record_warmup_divergence()
        second_value = client.calibrate_theta_base()

        self.assertEqual(first_value, second_value,
            "calibrate_theta_base() must be idempotent — only first call takes effect.")


class TestFLCLIDS(unittest.TestCase):
    """Unit tests for FL-CL-IDS (Khreasat & Villarrubia Gonzalez, IEEE Access 2026):
    - Page-Hinkley covariate drift detector (Eqs. 10-12)
    - Adaptive Lambda Controller (Eq. 5)
    - Drift-triggered round-skip protocol with minimum-participation floor rho_min=0.5 (Algorithm 1)
    - Online Fisher consolidation with decay gamma=0.9 (Eq. 14)
    """

    def test_page_hinkley_feature_covariate_drift(self):
        from federated.drift.detectors import PageHinkleyDetector

        ph = PageHinkleyDetector(delta=0.005, lambda_ph=50.0, tau_low=0.05, tau_high=0.30)
        rng = np.random.default_rng(42)

        # Reference stationary distribution N(0, 1) across d=100 features
        X_ref = rng.normal(0.0, 1.0, size=(64, 100))
        score0, drift0, warn0 = ph.update_features(X_ref)
        self.assertEqual(score0, 0.0)
        self.assertFalse(drift0)

        # Second stationary batch from same distribution -> low drift score (< tau_low)
        X_stat = rng.normal(0.0, 1.0, size=(64, 100))
        score_stat, drift_stat, _ = ph.update_features(X_stat)
        self.assertLess(score_stat, 0.15)
        self.assertFalse(drift_stat)

        # Covariate shift on 50% of features (mean shift +3.0) -> drift_fraction > tau_high (0.30)
        X_shift = rng.normal(0.0, 1.0, size=(64, 100))
        X_shift[:, :50] += 3.0
        score_shift, drift_shift, warn_shift = ph.update_features(X_shift)
        self.assertGreater(score_shift, 0.30)
        self.assertTrue(drift_shift)
        self.assertTrue(warn_shift)

    def test_adaptive_lambda_controller_equation_5(self):
        from federated.drift.controller import AdaptiveLambdaController

        ctrl = AdaptiveLambdaController(
            lambda_min=10.0,
            lambda_max=1000.0,
            tau_low=0.05,
            tau_high=0.30,
            rho_min=0.50,
            min_active_rounds_per_task=5,
        )
        # Delta < tau_low -> lambda_max
        self.assertAlmostEqual(ctrl.compute_lambda(0.01), 1000.0)
        # Delta > tau_high -> lambda_min
        self.assertAlmostEqual(ctrl.compute_lambda(0.45), 10.0)
        # Midpoint Delta = 0.175 -> midpoint lambda = 505.0
        self.assertAlmostEqual(ctrl.compute_lambda(0.175), 505.0, places=4)

    def test_round_skip_with_minimum_participation_floor(self):
        from federated.drift.controller import AdaptiveLambdaController

        ctrl = AdaptiveLambdaController(
            lambda_min=10.0,
            lambda_max=1000.0,
            tau_low=0.05,
            tau_high=0.30,
            rho_min=0.50,
            min_active_rounds_per_task=5,
        )
        # 6 scheduled clients; 5 have stationary drift < 0.05, only 1 has drift >= 0.05
        scores = {0: 0.20, 1: 0.04, 2: 0.03, 3: 0.02, 4: 0.01, 5: 0.00}
        active, skipped, lambdas, info = ctrl.plan_round(
            client_drift_scores=scores,
            round_in_task=10,
            task_id=1,
        )
        # Minimum participation floor ceil(0.5 * 6) = 3 must recall 2 clients (1 and 2)
        self.assertEqual(len(active), 3)
        self.assertEqual(len(skipped), 3)
        self.assertIn(0, active)
        self.assertIn(1, active)
        self.assertIn(2, active)
        self.assertEqual(info["n_recalled"], 2)


if __name__ == "__main__":
    unittest.main()
