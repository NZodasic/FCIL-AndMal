"""Federated Learning Client.

Local training for federated learning with incremental learning support
and on-device concept drift detection/adaptation (FL-MalDrift).

Reference:
    Patel et al., "FL-MalDrift: a federated learning framework for malware
    detection under local concept drift", Scientific Reports (Nature Portfolio), 2026.
"""

from typing import Dict, List, Optional, Any
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from incremental.base_strategy import IncrementalStrategy
from utils.device import resolve_device
from federated.drift.detectors import BaseDriftDetector, PageHinkleyDetector
from federated.drift.normalizer import DriftScoreNormalizer


class FLClient:
    """Federated Learning Client with FL-MalDrift stability control.

    Handles local training, on-device streaming drift detection, parameter
    divergence measurement, and local stability gating.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        strategy: IncrementalStrategy,
        device: Any = 'auto',
        drift_detector: Optional[BaseDriftDetector] = None,
        drift_normalizer: Optional[DriftScoreNormalizer] = None,
        theta_base: float = 0.1,
        beta: float = 0.9,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
    ):
        """Initialize FL Client.

        Args:
            client_id: Unique client identifier.
            model: Local model.
            strategy: Incremental learning strategy.
            device: Device for training.
            drift_detector: Optional BaseDriftDetector (ADWIN, DDM, EDDM, HDDM).
            drift_normalizer: Optional DriftScoreNormalizer for z-score scaling.
            theta_base: Baseline local stability threshold.
            beta: Exponential decay factor for local threshold updates.
            lambda1: Weighting coefficient for drift score in local threshold.
            lambda2: Weighting coefficient for local validation error.
        """
        self.client_id = client_id
        self.model = model
        self.strategy = strategy
        self.device = resolve_device(device)

        # Training statistics
        self.n_samples = 0
        self.n_steps = 0

        # FL-MalDrift Client-Side Drift & Stability Components
        self.drift_detector = drift_detector
        self.drift_normalizer = drift_normalizer or DriftScoreNormalizer()
        self.theta_base = float(theta_base)
        self.local_theta = float(theta_base)
        self.beta = float(beta)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

        # Paper: θ_base should be calibrated from median divergence during warm-up
        # (Patel et al. 2026, Sec. 4.4 "Hyperparameter configuration").
        # Accumulate divergence observations during warmup rounds, then finalize.
        self._warmup_divergences: List[float] = []
        self._theta_base_calibrated: bool = False

        self.initial_global_state: Optional[Dict[str, torch.Tensor]] = None
        self.last_divergence: float = 0.0
        self.last_drift_score: float = 0.0

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get model state for server aggregation.

        Returns:
            Model state dictionary.
        """
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Set model state from server and cache snapshot for divergence checks.

        Args:
            state_dict: Model state dictionary.
        """
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        # Snapshot broadcasted global parameters
        self.initial_global_state = {k: v.cpu().clone() for k, v in state_dict.items()}

    def compute_divergence(self, global_state: Optional[Dict[str, torch.Tensor]] = None) -> float:
        """Compute relative L2 parameter divergence d_{c,t} from global model:
            d_{c,t} = ||theta_{c,t}^{local} - theta_t||_2 / ||theta_t||_2
        """
        ref_state = global_state if global_state is not None else self.initial_global_state
        if ref_state is None:
            return 0.0

        local_state = self.model.state_dict()
        diff_sq_sum = 0.0
        ref_sq_sum = 0.0

        for k, v_ref in ref_state.items():
            if k in local_state and local_state[k].dtype.is_floating_point:
                v_loc = local_state[k].to(v_ref.device)
                if v_loc.shape == v_ref.shape:
                    diff = (v_loc - v_ref).float()
                    diff_sq_sum += float(torch.sum(diff ** 2).item())
                    ref_sq_sum += float(torch.sum(v_ref.float() ** 2).item())

        denom = math.sqrt(ref_sq_sum) if ref_sq_sum > 0 else 1.0
        divergence = math.sqrt(diff_sq_sum) / max(1e-9, denom)
        self.last_divergence = divergence
        return divergence

    def record_warmup_divergence(self) -> None:
        """Record the current divergence into the warmup accumulator.

        Call this at the end of each warmup round.  Once enough observations
        have been collected, call calibrate_theta_base() to finalize θ_base.

        Paper (Patel et al. 2026, Sec. 4.4): "The baseline stability threshold
        θ_base was initialized from the median divergence observed during the
        warm-up phase of each client, yielding personalized starting points."
        """
        d = self.compute_divergence()
        self._warmup_divergences.append(d)

    def calibrate_theta_base(self) -> float:
        """Finalize θ_base from the median of warmup-phase divergence observations.

        After calling this, theta_base and local_theta are updated with the
        personalized per-client value.  Subsequent calls are no-ops.

        Returns:
            Calibrated theta_base value.
        """
        if self._theta_base_calibrated or not self._warmup_divergences:
            return self.theta_base

        import numpy as _np
        median_d = float(_np.median(self._warmup_divergences))
        self.theta_base = max(1e-4, median_d)  # avoid zero floor
        self.local_theta = self.theta_base  # reset local running estimate
        self._theta_base_calibrated = True
        return self.theta_base

    def update_local_stability_threshold(self, s_ct: float, val_error: float) -> float:
        """Update adaptive local stability threshold theta_{c,t} via EWMA:
            theta_{c,t} = beta * theta_{c,t-1} + (1 - beta) * (theta_base + lambda1 * s_{c,t} + lambda2 * e_{c,t})
        """
        self.local_theta = (
            self.beta * self.local_theta
            + (1.0 - self.beta) * (self.theta_base + self.lambda1 * s_ct + self.lambda2 * val_error)
        )
        return self.local_theta

    def apply_local_adaptation(self, shrink_factor: float = 0.5) -> None:
        """Apply local mitigation when drift or high parameter divergence is detected.

        Shrinks unstable local updates back towards broadcasted global state:
            theta*_{c,t} = (1 - shrink_factor) * theta_{c,t} + shrink_factor * theta_t
        """
        if self.initial_global_state is None:
            return

        current_state = self.model.state_dict()
        adapted_state = {}
        for k, v in current_state.items():
            if k in self.initial_global_state and v.dtype.is_floating_point:
                v_global = self.initial_global_state[k].to(v.device)
                if v.shape == v_global.shape:
                    adapted_state[k] = (1.0 - shrink_factor) * v + shrink_factor * v_global
                else:
                    adapted_state[k] = v
            else:
                adapted_state[k] = v

        self.model.load_state_dict(adapted_state)

    def evaluate_covariate_drift(self, train_loader: DataLoader, max_samples: int = 256) -> float:
        """Compute input-feature Page-Hinkley covariate drift score Delta_i^{(t)} (Eqs. 10-12)."""
        if self.drift_detector is None or not hasattr(self.drift_detector, "update_features"):
            return float(self.last_drift_score)

        batches_x = []
        collected = 0
        for batch in train_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                bx = batch[0]
            else:
                continue
            batches_x.append(bx.detach().cpu().numpy())
            collected += int(bx.size(0))
            if collected >= max_samples:
                break

        if not batches_x:
            return 0.0

        X_np = np.concatenate(batches_x, axis=0)[:max_samples]
        drift_fraction, _, _ = self.drift_detector.update_features(X_np)
        self.last_drift_score = float(drift_fraction)
        return self.last_drift_score

    def set_adaptive_lambda(self, adaptive_lambda: float) -> None:
        """Set per-client adaptive EWC regularization multiplier lambda_i^{(t)} (Eq. 5)."""
        if hasattr(self.strategy, "set_adaptive_lambda"):
            self.strategy.set_adaptive_lambda(adaptive_lambda)
        elif hasattr(self.strategy, "ewc_lambda"):
            self.strategy.ewc_lambda = float(adaptive_lambda)

    def _evaluate_and_adapt_drift(
        self,
        train_loader: DataLoader,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Post-training streaming drift evaluation and local adaptation."""
        self.model.eval()
        divergence = self.compute_divergence()

        # FL-CL-IDS Page-Hinkley path: uses covariate drift fraction Delta_i^{(t)} directly
        # and adapts via AdaptiveLambdaController + Round-Skip rather than parameter shrinkage.
        if isinstance(self.drift_detector, PageHinkleyDetector):
            s_ct = float(self.drift_detector.get_statistic())
            self.last_drift_score = s_ct
            return {
                'drift_score': float(s_ct),
                'raw_drift_statistic': float(s_ct),
                'divergence': float(divergence),
                'local_threshold': float(self.drift_detector.tau_low),
                'drift_detected': bool(self.drift_detector.drift_detected),
                'warning_detected': bool(self.drift_detector.warning_detected),
                'drift_state': str(self.drift_detector.get_state()),
                'drift_adapted': True,
            }

        # Collect sample predictions to update streaming drift detector (FL-MalDrift)
        preds_list: List[int] = []
        targets_list: List[int] = []
        with torch.no_grad():
            batch_count = 0
            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    bx, by = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue
                logits = self.model(bx)
                preds = torch.argmax(logits, dim=-1)
                preds_list.extend(preds.cpu().numpy().tolist())
                targets_list.extend(by.cpu().numpy().tolist())
                batch_count += 1
                if batch_count >= 5:  # Evaluate up to 5 batches to preserve lightweight footprint
                    break

        if preds_list and targets_list:
            drift_detected, warning_detected = self.drift_detector.update_batch(
                np.array(preds_list), np.array(targets_list)
            )
            raw_z = self.drift_detector.get_statistic()
            state = self.drift_detector.get_state()
            acc = float(np.mean(np.array(preds_list) == np.array(targets_list)))
            val_error = 1.0 - acc
        else:
            raw_z = self.drift_detector.get_statistic()
            state = self.drift_detector.get_state()
            drift_detected = self.drift_detector.drift_detected
            warning_detected = self.drift_detector.warning_detected
            val_error = float(1.0 - metrics.get('accuracy', 0.5))

        # Normalize to bounded [0, 1] drift score
        s_ct = self.drift_normalizer.normalize(raw_z, state)
        self.last_drift_score = s_ct

        # Update local stability threshold
        local_thresh = self.update_local_stability_threshold(s_ct, val_error)

        # Local adaptation gating: if divergence or drift score exceeds local threshold
        adapted = False
        if divergence > local_thresh or s_ct > local_thresh or drift_detected:
            self.apply_local_adaptation(shrink_factor=0.5)
            adapted = True

        return {
            'drift_score': float(s_ct),
            'raw_drift_statistic': float(raw_z),
            'divergence': float(divergence),
            'local_threshold': float(local_thresh),
            'drift_detected': bool(drift_detected),
            'warning_detected': bool(warning_detected),
            'drift_state': str(state),
            'drift_adapted': bool(adapted),
        }

    def local_train(
        self,
        train_loader: DataLoader,
        n_epochs: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform local training with optional drift detection and stability gating.

        Args:
            train_loader: DataLoader for local data.
            n_epochs: Number of local epochs.
            **kwargs: Additional arguments for strategy.

        Returns:
            Dictionary of training metrics including drift statistics.
        """
        # Update sample count
        if hasattr(train_loader, 'dataset'):
            self.n_samples = len(train_loader.dataset)

        # Train using strategy
        metrics = self.strategy.train_task(
            train_loader,
            task_id=self.strategy.current_task,
            n_epochs=n_epochs,
            **kwargs
        )

        # Update step count
        self.n_steps = n_epochs * len(train_loader)

        metrics['client_id'] = self.client_id
        metrics['n_samples'] = self.n_samples
        metrics['n_steps'] = self.n_steps

        # FL-MalDrift: Post-training drift detection and local stability evaluation
        if self.drift_detector is not None:
            drift_info = self._evaluate_and_adapt_drift(train_loader, metrics)
            metrics.update(drift_info)
        else:
            metrics['drift_score'] = 0.0
            metrics['divergence'] = self.compute_divergence()
            metrics['drift_detected'] = False
            metrics['warning_detected'] = False
            metrics['drift_state'] = 'stable'
            metrics['local_threshold'] = self.local_theta
            metrics['drift_adapted'] = False

        return metrics

    def before_task(self, task_id: int, n_new_classes: int) -> None:
        """Prepare for new task.

        Args:
            task_id: Current task ID.
            n_new_classes: Number of new classes.
        """
        self.strategy.before_task(task_id, n_new_classes)

    def after_task(self, task_id: int, **kwargs) -> None:
        """Cleanup after task.

        Args:
            task_id: Current task ID.
            **kwargs: Additional arguments.
        """
        self.strategy.after_task(task_id, **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics.

        Returns:
            Dictionary of client statistics.
        """
        return {
            'client_id': self.client_id,
            'n_samples': self.n_samples,
            'n_steps': self.n_steps,
            'current_task': self.strategy.current_task,
            'n_classes_so_far': self.strategy.n_classes_so_far,
            'last_divergence': self.last_divergence,
            'last_drift_score': self.last_drift_score,
            'local_threshold': self.local_theta,
        }
