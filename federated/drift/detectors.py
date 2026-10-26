"""Concept Drift Detectors for Federated Android Malware Detection (FL-MalDrift).

Implements lightweight on-device streaming drift detectors:
- ADWIN (Adaptive Windowing)
- DDM (Drift Detection Method)
- EDDM (Early Drift Detection Method)
- HDDM (Hoeffding Drift Detection Method: HDDM-A and HDDM-W)

Reference:
    Patel et al., "FL-MalDrift: a federated learning framework for malware
    detection under local concept drift", Scientific Reports (Nature Portfolio), 2026.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import math
import numpy as np


class BaseDriftDetector(ABC):
    """Abstract Base Class for On-Device Streaming Drift Detectors."""

    def __init__(self, name: str = "BaseDriftDetector"):
        self.name = name
        self.drift_detected: bool = False
        self.warning_detected: bool = False
        self.total_samples: int = 0
        self.raw_statistic: float = 0.0

    def reset(self) -> None:
        """Reset internal detector statistics and flags."""
        self.drift_detected = False
        self.warning_detected = False
        self.total_samples = 0
        self.raw_statistic = 0.0
        self.reset_statistics()

    @abstractmethod
    def reset_statistics(self) -> None:
        """Reset algorithm-specific state variables."""
        pass

    @abstractmethod
    def update(self, error: float) -> Tuple[bool, bool]:
        """Update detector with a single prediction error/loss.

        Args:
            error: 0.0 for correct prediction, 1.0 for misclassification (or loss value).

        Returns:
            Tuple of (drift_detected, warning_detected).
        """
        pass

    def update_batch(self, preds: np.ndarray, targets: np.ndarray) -> Tuple[bool, bool]:
        """Update detector with a batch of predictions and ground truths.

        Args:
            preds: Array of predicted labels or binary predictions.
            targets: Array of target labels.

        Returns:
            Tuple of (drift_detected, warning_detected).
        """
        preds = np.asarray(preds)
        targets = np.asarray(targets)
        errors = (preds != targets).astype(np.float64)

        drift = False
        warning = False
        for err in errors:
            d, w = self.update(float(err))
            drift = drift or d
            warning = warning or w
        return drift, warning

    def get_statistic(self) -> float:
        """Return the raw detector-specific statistic (z_{c,t})."""
        return float(self.raw_statistic)

    def get_state(self) -> str:
        """Return symbolic state: 'drift', 'warning', or 'stable'."""
        if self.drift_detected:
            return "drift"
        elif self.warning_detected:
            return "warning"
        return "stable"


class ADWINDetector(BaseDriftDetector):
    """Adaptive Windowing (ADWIN) Drift Detector.

    Monitors a dynamic sliding window of error rates and splits it into two
    sub-windows W1 and W2. Drift is detected when the difference between their
    means |mu_1 - mu_2| exceeds a Hoeffding-derived bound:
        z_{c,t}^{ADWIN} = |mu_1 - mu_2|
        drift if |mu_1 - mu_2| > epsilon_cut
    """

    def __init__(self, delta: float = 0.002, max_window_size: int = 2000, min_subwindow_size: int = 5):
        super().__init__(name="ADWIN")
        self.delta = delta
        self.max_window_size = max_window_size
        self.min_subwindow_size = min_subwindow_size
        self.window: List[float] = []
        self.reset()

    def reset_statistics(self) -> None:
        self.window = []

    def update(self, error: float) -> Tuple[bool, bool]:
        self.total_samples += 1
        self.window.append(float(error))
        if len(self.window) > self.max_window_size:
            self.window.pop(0)

        n = len(self.window)
        self.drift_detected = False
        self.warning_detected = False
        max_diff = 0.0

        if n >= 2 * self.min_subwindow_size:
            step = max(1, n // 20)
            for split_idx in range(self.min_subwindow_size, n - self.min_subwindow_size + 1, step):
                w1 = self.window[:split_idx]
                w2 = self.window[split_idx:]
                n1 = len(w1)
                n2 = len(w2)
                mu1 = float(np.mean(w1))
                mu2 = float(np.mean(w2))
                diff = abs(mu1 - mu2)
                if diff > max_diff:
                    max_diff = diff

                m = 1.0 / (1.0 / n1 + 1.0 / n2)
                delta_prime = self.delta / math.log(max(2.0, float(n)))
                eps_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / max(1e-9, delta_prime)))

                if diff > 0.7 * eps_cut:
                    self.warning_detected = True

                if diff > eps_cut:
                    self.drift_detected = True
                    self.window = self.window[split_idx:]
                    break

        self.raw_statistic = max_diff
        return self.drift_detected, self.warning_detected


class DDMDetector(BaseDriftDetector):
    """Drift Detection Method (DDM).

    Monitors error rate p_i and standard deviation s_i = sqrt(p_i * (1 - p_i) / i).
    Tracks the minimum observed (p + s)_min.
        z_{c,t}^{DDM} = (p_i + s_i) - (p + s)_min
        warning if p_i + s_i >= (p + s)_min + warning_level * s_min
        drift if p_i + s_i >= (p + s)_min + drift_level * s_min
    """

    def __init__(self, min_samples: int = 30, warning_level: float = 2.0, drift_level: float = 3.0):
        super().__init__(name="DDM")
        self.min_samples = min_samples
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()

    def reset_statistics(self) -> None:
        self.p: float = 1.0
        self.s: float = 0.0
        self.p_min: float = float("inf")
        self.s_min: float = float("inf")
        self.ps_min: float = float("inf")
        self.sample_idx: int = 0
        self.error_count: int = 0

    def update(self, error: float) -> Tuple[bool, bool]:
        self.total_samples += 1
        self.sample_idx += 1
        if error > 0.5:
            self.error_count += 1

        self.drift_detected = False
        self.warning_detected = False

        if self.sample_idx >= self.min_samples:
            self.p = self.error_count / float(self.sample_idx)
            self.s = math.sqrt(max(0.0, self.p * (1.0 - self.p) / float(self.sample_idx)))
            ps = self.p + self.s

            if ps < self.ps_min:
                self.p_min = self.p
                self.s_min = self.s
                self.ps_min = ps

            diff = max(0.0, ps - self.ps_min)
            self.raw_statistic = diff

            drift_thresh = self.ps_min + self.drift_level * max(1e-5, self.s_min)
            warn_thresh = self.ps_min + self.warning_level * max(1e-5, self.s_min)

            if ps >= drift_thresh:
                self.drift_detected = True
                self.reset_statistics()
            elif ps >= warn_thresh:
                self.warning_detected = True
        else:
            self.raw_statistic = 0.0

        return self.drift_detected, self.warning_detected


class EDDMDetector(BaseDriftDetector):
    """Early Drift Detection Method (EDDM).

    Focuses on the average distance d_i between consecutive classification errors
    and its standard deviation s_d. Tracks maximum distance bound (d + s_d)_max:
        z_{c,t}^{EDDM} = 1 - (d_i + s_d) / (d + s_d)_max
        warning if (d_i + s_d) / (d + s_d)_max < warning_threshold (0.95)
        drift if (d_i + s_d) / (d + s_d)_max < drift_threshold (0.90)
    """

    def __init__(self, min_errors: int = 30, warning_threshold: float = 0.95, drift_threshold: float = 0.90):
        super().__init__(name="EDDM")
        self.min_errors = min_errors
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.reset()

    def reset_statistics(self) -> None:
        self.num_errors: int = 0
        self.last_error_sample: int = 0
        self.distances: List[float] = []
        self.d_mean: float = 0.0
        self.d_std: float = 0.0
        self.max_ps: float = 0.0

    def update(self, error: float) -> Tuple[bool, bool]:
        self.total_samples += 1
        self.drift_detected = False
        self.warning_detected = False

        if error > 0.5:
            self.num_errors += 1
            dist = float(self.total_samples - self.last_error_sample)
            self.last_error_sample = self.total_samples
            self.distances.append(dist)

            if len(self.distances) > 200:
                self.distances.pop(0)

            if self.num_errors >= self.min_errors:
                self.d_mean = float(np.mean(self.distances))
                self.d_std = float(np.std(self.distances))
                curr_ps = self.d_mean + 2.0 * self.d_std

                if curr_ps > self.max_ps:
                    self.max_ps = curr_ps

                if self.max_ps > 1e-6:
                    ratio = curr_ps / self.max_ps
                    self.raw_statistic = max(0.0, 1.0 - ratio)

                    if ratio < self.drift_threshold:
                        self.drift_detected = True
                        self.reset_statistics()
                    elif ratio < self.warning_threshold:
                        self.warning_detected = True
                else:
                    self.raw_statistic = 0.0

        return self.drift_detected, self.warning_detected


class HDDMDetector(BaseDriftDetector):
    """Hoeffding Drift Detection Method (HDDM).

    Uses Hoeffding's inequality to monitor shifts in error rate:
        z_{c,t}^{HDDM} = |mu_t - mu_0|
        drift if |mu_t - mu_0| > epsilon

    Variants:
        - 'HDDM_A': Arithmetic moving average
        - 'HDDM_W': Exponentially Weighted Moving Average (EWMA, default in FL-MalDrift)
    """

    def __init__(
        self,
        variant: str = "HDDM_W",
        alpha: float = 0.001,
        warning_alpha: float = 0.005,
        decay: float = 0.85,
        min_samples: int = 15,
    ):
        super().__init__(name=f"HDDM ({variant})")
        self.variant = variant.upper()
        self.alpha = alpha
        self.warning_alpha = warning_alpha
        self.decay = decay
        self.min_samples = min_samples
        self.reset()

    def reset_statistics(self) -> None:
        self.n_samples: int = 0
        self.mu_curr: float = 0.0
        self.mu_min: float = float("inf")
        self.weight_sum: float = 0.0
        self.weighted_error: float = 0.0

    def update(self, error: float) -> Tuple[bool, bool]:
        self.total_samples += 1
        self.n_samples += 1
        err = float(error)

        self.drift_detected = False
        self.warning_detected = False

        if self.variant == "HDDM_W":
            self.weighted_error = self.decay * self.weighted_error + err
            self.weight_sum = self.decay * self.weight_sum + 1.0
            self.mu_curr = self.weighted_error / max(1e-9, self.weight_sum)
        else:
            self.mu_curr += (err - self.mu_curr) / float(self.n_samples)

        if self.n_samples >= self.min_samples:
            if self.mu_curr < self.mu_min:
                self.mu_min = self.mu_curr

            diff = max(0.0, self.mu_curr - self.mu_min)
            self.raw_statistic = diff

            eff_n = float(self.n_samples)
            eps_drift = math.sqrt(math.log(1.0 / max(1e-12, self.alpha)) / (2.0 * eff_n))
            eps_warning = math.sqrt(math.log(1.0 / max(1e-12, self.warning_alpha)) / (2.0 * eff_n))

            if diff > eps_drift:
                self.drift_detected = True
                self.reset_statistics()
            elif diff > eps_warning:
                self.warning_detected = True
        else:
            self.raw_statistic = 0.0

        return self.drift_detected, self.warning_detected


class KLDriftDetector(BaseDriftDetector):
    """
    Sliding-Window Data Drift Detector based on Kullback-Leibler (KL) Divergence.
    
    Implements Algorithm 2 from Falcon (Yang et al., IEEE TCE 2026):
    - Maintains a reference window D_ref and a sliding current window D_cur.
    - Constructs histograms with B bins over streaming inputs/errors.
    - Applies Laplace smoothing: P'_i = (P_i + eps) / (1 + eps * B).
    - Measures divergence: D_{KL}(P' || Q') = sum_i P'_i * log(P'_i / Q'_i).
    - Evaluates backup Jensen-Shannon divergence for robustness check.
    - Triggers drift when KL > tau and refreshes reference distribution.
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.15,
        warning_ratio: float = 0.7,
        num_bins: int = 10,
        eps: float = 1e-4,
    ):
        super().__init__(name="KLDrift")
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.warning_threshold = float(threshold * warning_ratio)
        self.num_bins = int(num_bins)
        self.eps = float(eps)

        self.ref_window: List[float] = []
        self.cur_window: List[float] = []
        self.ref_hist: Optional[np.ndarray] = None
        self.js_divergence: float = 0.0

    def reset_statistics(self) -> None:
        self.ref_window = []
        self.cur_window = []
        self.ref_hist = None
        self.js_divergence = 0.0

    def _compute_distribution(self, data: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Compute normalized probability distribution with Laplace smoothing."""
        counts, _ = np.histogram(data, bins=bins)
        total = max(1, len(data))
        probs = counts / total
        smoothed = (probs + self.eps) / (1.0 + self.eps * self.num_bins)
        return smoothed / np.sum(smoothed)

    def _compute_kl(self, p: np.ndarray, q: np.ndarray) -> float:
        """Kullback-Leibler divergence D_KL(P || Q)."""
        return float(np.sum(p * np.log(p / q)))

    def _compute_js(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)."""
        m = 0.5 * (p + q)
        return float(0.5 * self._compute_kl(p, m) + 0.5 * self._compute_kl(q, m))

    def update(self, error: float) -> Tuple[bool, bool]:
        """Update detector with a single value."""
        self.total_samples += 1
        self.drift_detected = False
        self.warning_detected = False

        # Initialize reference window
        if len(self.ref_window) < self.window_size:
            self.ref_window.append(float(error))
            if len(self.ref_window) == self.window_size:
                # Reference window is fully initialized
                ref_arr = np.array(self.ref_window, dtype=np.float64)
                bins = np.linspace(0.0, 1.0, self.num_bins + 1)
                self.ref_hist = self._compute_distribution(ref_arr, bins)
            return False, False

        # Add to current sliding window
        self.cur_window.append(float(error))
        if len(self.cur_window) > self.window_size:
            self.cur_window.pop(0)

        if len(self.cur_window) >= max(10, self.window_size // 2):
            ref_arr = np.array(self.ref_window, dtype=np.float64)
            cur_arr = np.array(self.cur_window, dtype=np.float64)
            bins = np.linspace(0.0, 1.0, self.num_bins + 1)

            p_ref = self._compute_distribution(ref_arr, bins)
            q_cur = self._compute_distribution(cur_arr, bins)

            kl_div = self._compute_kl(p_ref, q_cur)
            js_div = self._compute_js(p_ref, q_cur)
            self.raw_statistic = float(kl_div)
            self.js_divergence = float(js_div)

            if kl_div > self.threshold:
                self.drift_detected = True
                # Algorithm 2, lines 14-15: update reference window to current window
                self.ref_window = list(self.cur_window)
                self.cur_window = []
                self.ref_hist = q_cur
            elif kl_div > self.warning_threshold:
                self.warning_detected = True

        return self.drift_detected, self.warning_detected

    def update_batch(self, preds: np.ndarray, targets: np.ndarray) -> Tuple[bool, bool]:
        """Update detector with batch predictions and targets."""
        preds = np.asarray(preds)
        targets = np.asarray(targets)
        errors = (preds != targets).astype(np.float64)
        for err in errors:
            d, w = self.update(float(err))
            if d:
                break
        return self.drift_detected, self.warning_detected


class PageHinkleyDetector(BaseDriftDetector):
    """Page-Hinkley (PH) Covariate & Stream Drift Detector for FL-CL-IDS.

    Implements Section III-E, Equations (10)-(12) of:
        Khreasat & Villarrubia Gonzalez, "Drift-Aware Federated Continual Learning
        for Privacy-Preserving Android Malware Intrusion Detection", IEEE Access, 2026.

    For each feature k in {1, ..., d}, maintains running mean x_bar_k, cumulative
    deviation m_k^{(n)}, and extremum M_k^{(n)}:
        m_k^{(n)} = m_k^{(n-1)} + (x_k^{(n)} - x_bar_k - delta)
        M_k^{(n)} = max_{1 <= j <= n} m_k^{(j)}
        PH_k^{(n)} = M_k^{(n)} - m_k^{(n)}
    The normalized client drift score is the fraction of features exceeding lambda_PH:
        Delta_i^{(t)} = (1 / d) * sum_{k=1}^d I[PH_k > lambda_PH] in [0, 1].
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_ph: float = 50.0,
        alpha: float = 0.90,
        tau_low: float = 0.05,
        tau_high: float = 0.30,
        min_instances: int = 10,
    ):
        super().__init__(name="PageHinkley")
        self.delta = float(delta)
        self.lambda_ph = float(lambda_ph)
        self.alpha = float(alpha)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.min_instances = int(min_instances)
        self.reset()

    def reset_statistics(self) -> None:
        # 1D scalar stream statistics (for BaseDriftDetector compatibility)
        self.x_mean_1d: float = 0.0
        self.m_down_1d: float = 0.0
        self.M_down_1d: float = 0.0
        self.m_up_1d: float = 0.0
        self.M_up_1d: float = 0.0
        self.n_1d: int = 0

        # d-dimensional feature covariate statistics (Eqs. 10-12)
        self.ref_mean: Optional[np.ndarray] = None
        self.ref_std: Optional[np.ndarray] = None
        self.m_down: Optional[np.ndarray] = None
        self.M_down: Optional[np.ndarray] = None
        self.m_up: Optional[np.ndarray] = None
        self.M_up: Optional[np.ndarray] = None
        self.per_feature_ph: Optional[np.ndarray] = None
        self.feature_drift_fraction: float = 0.0
        self.rounds_in_current_regime: int = 0

    def update(self, error: float) -> Tuple[bool, bool]:
        """Update 1D Page-Hinkley detector with a single scalar observation."""
        self.total_samples += 1
        self.n_1d += 1
        x = float(error)

        self.x_mean_1d += (x - self.x_mean_1d) / float(self.n_1d)
        self.m_down_1d = self.alpha * self.m_down_1d + (x - self.x_mean_1d - self.delta)
        self.M_down_1d = max(self.M_down_1d, self.m_down_1d)

        self.m_up_1d = self.alpha * self.m_up_1d + (x - self.x_mean_1d + self.delta)
        self.M_up_1d = min(self.M_up_1d, self.m_up_1d)

        ph_stat = max(self.M_down_1d - self.m_down_1d, self.m_up_1d - self.M_up_1d)
        eff_thresh = min(self.lambda_ph, max(1.5, 0.15 * self.n_1d))
        norm_score = float(np.clip(ph_stat / max(1e-6, eff_thresh), 0.0, 1.0))

        self.raw_statistic = norm_score
        self.drift_detected = False
        self.warning_detected = False

        if self.n_1d >= self.min_instances:
            if ph_stat > eff_thresh or norm_score >= self.tau_high:
                self.drift_detected = True
                self.m_down_1d = 0.0
                self.M_down_1d = 0.0
                self.m_up_1d = 0.0
                self.M_up_1d = 0.0
            elif norm_score >= self.tau_low:
                self.warning_detected = True

        return self.drift_detected, self.warning_detected

    def update_features(self, features: np.ndarray) -> Tuple[float, bool, bool]:
        """Evaluate Page-Hinkley covariate drift across d input features (Eqs. 10-12).

        Args:
            features: 2D array of shape (N, d) containing local input feature vectors.

        Returns:
            Tuple of (drift_fraction Delta_i^{(t)} in [0, 1], drift_detected, warning_detected).
        """
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples, d = X.shape
        if n_samples == 0 or d == 0:
            return 0.0, False, False

        self.total_samples += n_samples
        self.rounds_in_current_regime += 1

        batch_mean = np.mean(X, axis=0)
        batch_std = np.std(X, axis=0) + 1e-6

        if self.ref_mean is None or self.ref_mean.shape[0] != d:
            # First observation window initializes reference feature statistics
            self.ref_mean = batch_mean.copy()
            self.ref_std = batch_std.copy()
            self.m_down = np.zeros(d, dtype=np.float64)
            self.M_down = np.zeros(d, dtype=np.float64)
            self.m_up = np.zeros(d, dtype=np.float64)
            self.M_up = np.zeros(d, dtype=np.float64)
            self.per_feature_ph = np.zeros(d, dtype=np.float64)
            self.feature_drift_fraction = 0.0
            self.raw_statistic = 0.0
            self.drift_detected = False
            self.warning_detected = False
            return 0.0, False, False

        # Standardized deviations relative to running reference distribution
        z_X = (X - self.ref_mean) / self.ref_std
        centered_down = z_X - self.delta
        centered_up = z_X + self.delta

        # Cumulative Page-Hinkley sums along the sample stream n = 1..N (Eqs. 10-11)
        cum_down = self.alpha * self.m_down + np.cumsum(centered_down, axis=0)
        cum_up = self.alpha * self.m_up + np.cumsum(centered_up, axis=0)

        max_cum_down = np.maximum(self.M_down, np.max(cum_down, axis=0))
        min_cum_up = np.minimum(self.M_up, np.min(cum_up, axis=0))

        self.m_down = cum_down[-1]
        self.M_down = max_cum_down
        self.m_up = cum_up[-1]
        self.M_up = min_cum_up

        ph_down = self.M_down - self.m_down
        ph_up = self.m_up - self.M_up
        # Also include total cumulative excursion from baseline for sustained mean shift
        ph_shift = np.maximum(np.abs(self.m_down), np.abs(self.m_up))
        self.per_feature_ph = np.maximum(np.maximum(ph_down, ph_up), ph_shift)

        # Effective threshold scales with shard size up to lambda_ph = 50.0 (Table 2)
        eff_lambda_ph = min(self.lambda_ph, max(3.0, 0.35 * float(n_samples)))

        # Eq. (12): Delta_i^{(t)} = (1 / d) * sum_{k=1}^d I[PH_k > lambda_PH]
        drifted_mask = self.per_feature_ph > eff_lambda_ph
        drift_fraction = float(np.mean(drifted_mask))

        self.feature_drift_fraction = drift_fraction
        self.raw_statistic = drift_fraction
        self.drift_detected = bool(drift_fraction > self.tau_high)
        self.warning_detected = bool(drift_fraction >= self.tau_low)

        # Adapt reference mean gradually so stationary rounds within a task decay PH
        adapt_rate = 0.35 if self.warning_detected else 0.20
        self.ref_mean = (1.0 - adapt_rate) * self.ref_mean + adapt_rate * batch_mean
        self.ref_std = (1.0 - adapt_rate) * self.ref_std + adapt_rate * batch_std

        # Reset cumulative extrema on drifted features after detection to avoid permanent saturation
        if np.any(drifted_mask):
            self.m_down[drifted_mask] *= 0.5
            self.M_down[drifted_mask] = self.m_down[drifted_mask]
            self.m_up[drifted_mask] *= 0.5
            self.M_up[drifted_mask] = self.m_up[drifted_mask]

        return drift_fraction, self.drift_detected, self.warning_detected


def build_drift_detector(detector_type: str = "hddm_w", **kwargs) -> BaseDriftDetector:
    """Factory function to instantiate specified drift detector.

    Args:
        detector_type: One of ('page_hinkley', 'hddm_w', 'hddm_a', 'adwin', 'ddm', 'eddm', 'kl_drift').
        **kwargs: Optional hyperparameter overrides.

    Returns:
        Instance of BaseDriftDetector.
    """
    dtype = detector_type.lower()
    if dtype in ("page_hinkley", "page-hinkley", "ph", "fl_cl_ids", "ewcdr"):
        return PageHinkleyDetector(**kwargs)
    elif dtype in ("hddm_w", "hddm-w", "hddm"):
        return HDDMDetector(variant="HDDM_W", **kwargs)
    elif dtype in ("hddm_a", "hddm-a"):
        return HDDMDetector(variant="HDDM_A", **kwargs)
    elif dtype == "adwin":
        return ADWINDetector(**kwargs)
    elif dtype == "ddm":
        return DDMDetector(**kwargs)
    elif dtype == "eddm":
        return EDDMDetector(**kwargs)
    elif dtype in ("kl_drift", "kl", "falcon_drift", "falcon"):
        return KLDriftDetector(**kwargs)
    else:
        raise ValueError(
            f"Unsupported drift detector '{detector_type}'. Supported: "
            f"['page_hinkley', 'hddm_w', 'hddm_a', 'adwin', 'ddm', 'eddm', 'kl_drift']"
        )

