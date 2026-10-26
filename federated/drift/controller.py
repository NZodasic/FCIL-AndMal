"""Server-Side Dynamic EWMA Threshold Controller (FL-MalDrift Algorithm 2).

Manages:
1. Warm-up baseline calibration (t <= T_0 rounds).
2. Dynamic EWMA threshold updates (mu_t + k * sigma_t) with proportional participation control.
3. Stable client admission filtering C_t = {c | s_{c,t} <= tau_t}.
4. Client recovery mechanism with smoothed drift tracking and consecutive stable rounds.

Reference:
    Patel et al., "FL-MalDrift: a federated learning framework for malware
    detection under local concept drift", Scientific Reports (Nature Portfolio), 2026.
"""

from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np


class AdaptiveLambdaController:
    """FL-CL-IDS Per-Client Adaptive Lambda & Drift-Triggered Round-Skip Controller.

    Implements Section III-C, Equation (5), and Algorithm 1 of:
        Khreasat & Villarrubia Gonzalez, "Drift-Aware Federated Continual Learning
        for Privacy-Preserving Android Malware Intrusion Detection", IEEE Access, 2026.

    1. Adaptive EWC regularization strength (Eq. 5):
        - Delta_i^{(t)} < tau_low               => lambda_i^{(t)} = lambda_max
        - tau_low <= Delta_i^{(t)} <= tau_high  => linear interpolation in [lambda_min, lambda_max]
        - Delta_i^{(t)} > tau_high              => lambda_i^{(t)} = lambda_min
    2. Drift-triggered round-skip protocol with minimum-participation floor (Algorithm 1):
        - Clients with Delta_i^{(t)} < tau_low (after initial task adaptation rounds)
          flag SKIP and omit redundant local training and model upload.
        - If fewer than ceil(rho_min * |S^{(r)}|) clients upload, the server recalls
          skipped clients with the highest Delta_j^{(t)} until the floor is met.
    """

    def __init__(
        self,
        lambda_min: float = 10.0,
        lambda_max: float = 1000.0,
        tau_low: float = 0.05,
        tau_high: float = 0.30,
        rho_min: float = 0.50,
        min_active_rounds_per_task: int = 10,
    ):
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.rho_min = float(rho_min)
        self.min_active_rounds_per_task = int(min_active_rounds_per_task)

        # Cumulative communication accounting across tasks/rounds
        self.total_scheduled_uploads: int = 0
        self.total_actual_uploads: int = 0
        self.total_skipped_uploads: int = 0
        self.total_recalled_uploads: int = 0
        self.client_lambdas: Dict[int, float] = {}
        self.client_drift_scores: Dict[int, float] = {}

    def reset(self) -> None:
        self.total_scheduled_uploads = 0
        self.total_actual_uploads = 0
        self.total_skipped_uploads = 0
        self.total_recalled_uploads = 0
        self.client_lambdas.clear()
        self.client_drift_scores.clear()

    def compute_lambda(self, drift_score: float) -> float:
        """Compute per-client EWC regularization multiplier lambda_i^{(t)} via Eq. (5)."""
        delta = float(np.clip(drift_score, 0.0, 1.0))
        if delta < self.tau_low:
            return self.lambda_max
        if delta > self.tau_high:
            return self.lambda_min
        denom = max(1e-9, self.tau_high - self.tau_low)
        frac = (delta - self.tau_low) / denom
        return float(self.lambda_max - frac * (self.lambda_max - self.lambda_min))

    def plan_round(
        self,
        client_drift_scores: Dict[int, float],
        round_in_task: int = 1,
        task_id: int = 0,
    ) -> Tuple[List[int], List[int], Dict[int, float], Dict[str, Any]]:
        """Execute Algorithm 1 round-skip and minimum-participation floor check.

        Args:
            client_drift_scores: Mapping of client_id -> Delta_i^{(t)} in [0, 1].
            round_in_task: 1-based communication round index within the current task.
            task_id: Current incremental task index (0-based).

        Returns:
            Tuple of:
                - active_client_ids: Clients that perform local training and upload.
                - skipped_client_ids: Clients that skip training and upload this round.
                - lambdas: Mapping of client_id -> adaptive lambda_i^{(t)}.
                - info: Diagnostic dictionary (upload reduction, recalled clients, etc.).
        """
        scheduled_ids = sorted(list(client_drift_scores.keys()))
        n_scheduled = len(scheduled_ids)
        if n_scheduled == 0:
            return [], [], {}, {}

        lambdas: Dict[int, float] = {}
        active_set: List[int] = []
        skipped_candidates: List[int] = []

        # Allow initial rounds of each task for new-class representation acquisition
        allow_skip = (round_in_task > self.min_active_rounds_per_task)

        for cid in scheduled_ids:
            delta_i = float(client_drift_scores[cid])
            lam_i = self.compute_lambda(delta_i)
            lambdas[cid] = lam_i
            self.client_lambdas[cid] = lam_i
            self.client_drift_scores[cid] = delta_i

            if allow_skip and delta_i < self.tau_low:
                skipped_candidates.append(cid)
            else:
                active_set.append(cid)

        # Algorithm 1 (lines 17-21): Enforce minimum participation floor rho_min
        min_required = int(math.ceil(self.rho_min * float(n_scheduled)))
        recalled_clients: List[int] = []
        if len(active_set) < min_required and skipped_candidates:
            needed = min_required - len(active_set)
            # Recall skipped clients with largest Delta_j^{(t)} first ( tie-broken by cyclic round offset )
            sorted_skipped = sorted(
                skipped_candidates,
                key=lambda c: (client_drift_scores[c], (c + round_in_task) % max(1, n_scheduled)),
                reverse=True,
            )
            recalled_clients = sorted_skipped[:needed]
            active_set.extend(recalled_clients)
            skipped_candidates = sorted_skipped[needed:]

        active_client_ids = sorted(active_set)
        skipped_client_ids = sorted(skipped_candidates)

        self.total_scheduled_uploads += n_scheduled
        self.total_actual_uploads += len(active_client_ids)
        self.total_skipped_uploads += len(skipped_client_ids)
        self.total_recalled_uploads += len(recalled_clients)

        upload_reduction_pct = (
            100.0 * float(self.total_skipped_uploads) / float(max(1, self.total_scheduled_uploads))
        )

        info = {
            "task_id": int(task_id),
            "round_in_task": int(round_in_task),
            "n_scheduled": n_scheduled,
            "n_active_uploads": len(active_client_ids),
            "n_skipped_uploads": len(skipped_client_ids),
            "n_recalled": len(recalled_clients),
            "active_clients": active_client_ids,
            "skipped_clients": skipped_client_ids,
            "recalled_clients": recalled_clients,
            "mean_drift_score": float(np.mean(list(client_drift_scores.values()))),
            "mean_lambda": float(np.mean(list(lambdas.values()))),
            "cumulative_upload_reduction_pct": float(upload_reduction_pct),
        }
        return active_client_ids, skipped_client_ids, lambdas, info


class ServerDriftController:
    """Dynamic EWMA Participation Threshold Controller (Algorithm 2)."""

    def __init__(
        self,
        warmup_rounds: int = 3,
        window_size: int = 10,
        alpha: float = 0.8,
        k: float = 1.5,
        target_participation_budget: float = 0.7,
        gain_eta: float = 0.05,
        tau_min: float = 0.1,
        tau_max: float = 0.95,
        default_tau_0: float = 0.5,
        quantile_q: float = 0.90,
        recovery_rounds: int = 2,
        recovery_delta: float = 0.05,
        recovery_gamma: float = 0.8,
    ):
        """Initialize server drift controller.

        Args:
            warmup_rounds: Warm-up phase duration (T_0, default 3).
            window_size: Sliding window size (W, default 10).
            alpha: EWMA smoothing factor in [0, 1) (default 0.8).
            k: Statistical control chart limit factor (default 1.5).
            target_participation_budget: Target active fraction p* in (0, 1) (default 0.7).
            gain_eta: Proportional feedback gain for participation control (default 0.05).
            tau_min: Minimum allowable threshold value (default 0.1).
            tau_max: Maximum allowable threshold value (default 0.95).
            default_tau_0: Fallback initial threshold before empirical calibration.
            quantile_q: Quantile used to calibrate baseline tau_0 from warm-up scores (default 0.90).
            recovery_rounds: Consecutive stable rounds (R) required for client reactivation (default 2).
            recovery_delta: Tolerance margin delta for re-entry threshold tau_re = tau_t + delta (default 0.05).
            recovery_gamma: Exponential smoothing factor for client recovery drift tracking (default 0.8).
        """
        self.warmup_rounds = int(warmup_rounds)
        self.window_size = int(window_size)
        self.alpha = float(alpha)
        self.k = float(k)
        self.p_star = float(target_participation_budget)
        self.eta = float(gain_eta)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.tau_0 = float(default_tau_0)
        self.current_tau = float(default_tau_0)
        self.quantile_q = float(quantile_q)

        # Recovery parameters
        self.recovery_rounds = int(recovery_rounds)
        self.recovery_delta = float(recovery_delta)
        self.recovery_gamma = float(recovery_gamma)

        # State tracking
        self.current_round: int = 0
        self.score_history: List[float] = []
        self.warmup_scores: List[float] = []
        self.smoothed_client_scores: Dict[int, float] = {}
        self.consecutive_stable_rounds: Dict[int, int] = {}
        self.quarantined_clients: set = set()

    def reset(self) -> None:
        """Reset controller state."""
        self.current_round = 0
        self.score_history.clear()
        self.warmup_scores.clear()
        self.smoothed_client_scores.clear()
        self.consecutive_stable_rounds.clear()
        self.quarantined_clients.clear()
        self.current_tau = self.tau_0

    def compute_window_statistics(self) -> Tuple[float, float]:
        """Compute empirical mean mu_t and std sigma_t over sliding window W."""
        if not self.score_history:
            return 0.5, 0.1
        recent = self.score_history[-max(len(self.score_history), self.window_size):]
        mu = float(np.mean(recent))
        sigma = float(np.std(recent))
        return mu, sigma

    def step(
        self,
        round_idx: int,
        client_scores: Dict[int, float],
    ) -> Tuple[List[int], float, Dict[str, Any]]:
        """Execute one round of Algorithm 2: Dynamic threshold update and client selection.

        Args:
            round_idx: Current communication round index (1-based).
            client_scores: Dict mapping client_id to normalized drift score s_{c,t} in [0, 1].

        Returns:
            Tuple of:
                - admitted_client_ids: List of client IDs admitted for global aggregation.
                - current_tau: Updated threshold tau_t.
                - info: Diagnostic metrics dictionary.
        """
        self.current_round = round_idx
        client_ids = sorted(list(client_scores.keys()))
        n_total = len(client_ids)
        if n_total == 0:
            return [], self.current_tau, {}

        # Record scores into history
        current_scores = [float(client_scores[cid]) for cid in client_ids]
        self.score_history.extend(current_scores)
        if len(self.score_history) > self.window_size * 50:
            self.score_history = self.score_history[-self.window_size * 20:]

        # Update smoothed scores for recovery tracking: s_bar = gamma * s_bar + (1 - gamma) * s
        for cid in client_ids:
            s = float(client_scores[cid])
            if cid not in self.smoothed_client_scores:
                self.smoothed_client_scores[cid] = s
            else:
                self.smoothed_client_scores[cid] = (
                    self.recovery_gamma * self.smoothed_client_scores[cid]
                    + (1.0 - self.recovery_gamma) * s
                )

        tau_hat = self.current_tau
        mu_t, sigma_t = 0.5, 0.1
        p_t = 1.0
        recovered_clients: List[int] = []

        if round_idx <= self.warmup_rounds:
            # Phase 1: Warm-up (t <= T_0)
            self.warmup_scores.extend(current_scores)
            # Calibrate initial threshold tau_0 from empirical distribution
            if len(self.warmup_scores) >= 3:
                self.tau_0 = float(np.quantile(self.warmup_scores, self.quantile_q))
                self.tau_0 = float(np.clip(self.tau_0, self.tau_min, self.tau_max))
            self.current_tau = self.tau_0
            # During warm-up, all clients are admitted to collect baseline dynamics
            admitted_clients = list(client_ids)
        else:
            # Phase 2: Online Adaptive Updates (t > T_0)
            mu_t, sigma_t = self.compute_window_statistics()
            # Statistical control limit: tau_hat = mu_t + k * sigma_t
            tau_hat = mu_t + self.k * sigma_t

            # Realized fraction of active clients under previous threshold
            active_count = sum(1 for cid in client_ids if client_scores[cid] <= self.current_tau)
            p_t = float(active_count) / float(n_total)

            # Dynamic EWMA update with proportional participation feedback:
            # tau_t = clip[tau_min, tau_max]( alpha * tau_{t-1} + (1 - alpha) * tau_hat + eta * (p* - p_t) )
            updated_tau = (
                self.alpha * self.current_tau
                + (1.0 - self.alpha) * tau_hat
                + self.eta * (self.p_star - p_t)
            )
            self.current_tau = float(np.clip(updated_tau, self.tau_min, self.tau_max))

            # Track existing quarantine set prior to current round evaluation
            previously_quarantined = set(self.quarantined_clients)

            # Base selection: unquarantined clients with score <= tau_t
            admitted_set = set()
            for cid in client_ids:
                if cid not in previously_quarantined:
                    if client_scores[cid] <= self.current_tau:
                        admitted_set.add(cid)
                    else:
                        self.quarantined_clients.add(cid)
                        self.consecutive_stable_rounds[cid] = 0

            # Recovery mechanism for previously quarantined clients:
            # Must maintain smoothed drift score below tau_re for R consecutive rounds
            tau_re = self.current_tau + self.recovery_delta
            for cid in previously_quarantined:
                if cid in client_ids:
                    s_bar = self.smoothed_client_scores[cid]
                    if s_bar <= tau_re:
                        self.consecutive_stable_rounds[cid] = (
                            self.consecutive_stable_rounds.get(cid, 0) + 1
                        )
                        if self.consecutive_stable_rounds[cid] >= self.recovery_rounds:
                            # Reintegrate recovered client into federation
                            admitted_set.add(cid)
                            recovered_clients.append(cid)
                            self.consecutive_stable_rounds[cid] = 0
                            self.quarantined_clients.discard(cid)
                    else:
                        self.consecutive_stable_rounds[cid] = 0

            # Safety fallback: Ensure at least one client is admitted to prevent federation deadlock
            if not admitted_set:
                best_cid = min(client_ids, key=lambda c: client_scores[c])
                admitted_set.add(best_cid)

            admitted_clients = sorted(list(admitted_set))

        excluded_clients = [cid for cid in client_ids if cid not in admitted_clients]

        info = {
            "round": round_idx,
            "tau_t": float(self.current_tau),
            "tau_hat": float(tau_hat),
            "mu_t": float(mu_t),
            "sigma_t": float(sigma_t),
            "participation_fraction_p_t": float(p_t),
            "target_budget_p_star": float(self.p_star),
            "n_total": n_total,
            "n_admitted": len(admitted_clients),
            "admitted_clients": admitted_clients,
            "excluded_clients": excluded_clients,
            "recovered_clients": recovered_clients,
            "warmup_active": (round_idx <= self.warmup_rounds),
        }

        return admitted_clients, self.current_tau, info
