"""
Hidden Markov Model Regime Detection — Gaps B1, B2.

Replaces the rule-based regime detector with a probabilistic
3-state Gaussian HMM based on Hamilton (1989) regime-switching.

States:
  S₀ = BULL:     μ > 0, σ low   (trending up, low volatility)
  S₁ = BEAR:     μ < 0, σ high  (trending down, high volatility)
  S₂ = SIDEWAYS: μ ≈ 0, σ med   (range-bound, moderate volatility)

Observations (4-dimensional):
  O₁ = NIFTY daily log-returns
  O₂ = India VIX level (normalized)
  O₃ = Market breadth (advance/decline ratio)
  O₄ = Delivery volume % (NSE smart money proxy)

Key methods:
  - fit():      Train HMM on 5 years of daily data (Baum-Welch / EM)
  - filter():   P(state_t | observations_1:t) — real-time regime probability
  - predict():  P(state_{t+h} | O_1:t) using transition matrix
  - expected_duration(): 1 / (1 - A[s,s]) days per state

Academic references:
  - Hamilton (1989) "A New Approach to Economic Analysis of Nonstationary Time Series"
  - Rabiner (1989) "A Tutorial on Hidden Markov Models"
  - Calvet & Fisher (2004) "Markov Switching Multifractal"
  - Kritzman et al. (2012) "Regime Shifts: Implications for Dynamic Strategies"

Integration:
  - regime_detector.py: HMM → fallback to rule-based if confidence < 0.6
  - forecast_combiner.py: Probabilistic weight blending using P(state) vector
  - position_sizer.py: Regime-conditioned volatility target
  - carver_pipeline.py: Transition warnings for stop tightening
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "hmm_model.pkl"
_REGIME_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "hmm_regime_log.jsonl"


# ═══════════════════════════════════════════════════════════════
# State Mapping
# ═══════════════════════════════════════════════════════════════

HMM_STATE_MAP = {
    0: "TRENDING_BULL",
    1: "TRENDING_BEAR",
    2: "RANGE_BOUND",
}

HMM_STATE_NAMES = ["BULL", "BEAR", "SIDEWAYS"]


@dataclass
class HMMRegimeSnapshot:
    """HMM-based regime assessment with probability distribution."""
    regime: str                    # Most likely regime name
    state_index: int               # 0=BULL, 1=BEAR, 2=SIDEWAYS
    probabilities: List[float]     # [P(BULL), P(BEAR), P(SIDEWAYS)]
    confidence: float              # max(probabilities)
    transition_matrix: List[List[float]] = field(default_factory=list)
    expected_durations: Dict[str, float] = field(default_factory=dict)
    predicted_5d: List[float] = field(default_factory=list)  # 5-day ahead probs
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "state_index": self.state_index,
            "probabilities": {
                "BULL": round(self.probabilities[0], 4),
                "BEAR": round(self.probabilities[1], 4),
                "SIDEWAYS": round(self.probabilities[2], 4),
            },
            "confidence": round(self.confidence, 4),
            "expected_durations": self.expected_durations,
            "predicted_5d": {
                "BULL": round(self.predicted_5d[0], 4) if self.predicted_5d else 0,
                "BEAR": round(self.predicted_5d[1], 4) if self.predicted_5d else 0,
                "SIDEWAYS": round(self.predicted_5d[2], 4) if self.predicted_5d else 0,
            },
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# Markov Regime Model
# ═══════════════════════════════════════════════════════════════

class MarkovRegimeModel:
    """3-state Gaussian HMM for NSE market regime detection.

    Uses hmmlearn.GaussianHMM or a lightweight built-in implementation
    when hmmlearn is not available.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 4,
        n_iter: int = 100,
        tol: float = 1e-4,
        min_confidence: float = 0.6,
    ):
        self.n_states = n_states
        self.n_features = n_features
        self.n_iter = n_iter
        self.tol = tol
        self.min_confidence = min_confidence
        self._model = None
        self._fitted = False
        self._means = None
        self._covars = None
        self._transmat = None
        self._startprob = None

    def fit(self, observations: np.ndarray) -> "MarkovRegimeModel":
        """Fit HMM on historical observations via EM (Baum-Welch).

        Parameters
        ----------
        observations : np.ndarray, shape (T, n_features)
            Columns: [log_returns, vix_normalized, breadth, delivery_pct]
            T should be ≥ 500 (2+ years of daily data).

        Returns
        -------
        self
        """
        T, n_feat = observations.shape
        if T < 100:
            raise ValueError(f"Need ≥100 observations, got {T}")

        try:
            from hmmlearn.hmm import GaussianHMM
            self._model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=42,
            )
            self._model.fit(observations)
            self._transmat = self._model.transmat_.copy()
            self._means = self._model.means_.copy()
            self._covars = self._model.covars_.copy()
            self._startprob = self._model.startprob_.copy()
            self._fitted = True

            # Sort states so 0=BULL (highest return mean), 1=BEAR (lowest), 2=SIDEWAYS
            self._sort_states()

            logger.info(
                "HMM fit complete: %d observations, %d states, means=%s",
                T, self.n_states, [round(m[0], 6) for m in self._means],
            )
        except ImportError:
            logger.warning("hmmlearn not installed — using built-in EM fallback")
            self._fit_builtin(observations)

        return self

    def _fit_builtin(self, observations: np.ndarray) -> None:
        """Lightweight built-in EM training when hmmlearn unavailable.

        Implements simplified Baum-Welch for Gaussian emissions.
        """
        T, D = observations.shape
        K = self.n_states

        # Initialize with K-means clustering
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(observations)

        # Initialize parameters
        self._means = np.array([observations[labels == k].mean(axis=0) for k in range(K)])
        self._covars = np.array([
            np.cov(observations[labels == k].T) + np.eye(D) * 1e-6
            if np.sum(labels == k) > D else np.eye(D) * 0.01
            for k in range(K)
        ])
        self._transmat = np.full((K, K), 1.0 / K)
        np.fill_diagonal(self._transmat, 0.9)
        self._transmat /= self._transmat.sum(axis=1, keepdims=True)
        self._startprob = np.ones(K) / K

        # EM iterations
        for iteration in range(self.n_iter):
            # E-step: compute responsibilities
            log_likelihoods = np.zeros((T, K))
            for k in range(K):
                log_likelihoods[:, k] = self._log_gaussian(
                    observations, self._means[k], self._covars[k]
                )

            # Forward-backward
            alpha, beta, scale = self._forward_backward(log_likelihoods)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # Xi computation (transition posteriors)
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi_sum[i, j] += (
                            alpha[t, i] *
                            self._transmat[i, j] *
                            np.exp(log_likelihoods[t + 1, j]) *
                            beta[t + 1, j]
                        )
            xi_sum /= xi_sum.sum(axis=1, keepdims=True) + 1e-300

            # M-step: update parameters
            self._startprob = gamma[0] / (gamma[0].sum() + 1e-300)
            self._transmat = xi_sum

            for k in range(K):
                g_k = gamma[:, k]
                total_g = g_k.sum() + 1e-300
                self._means[k] = (g_k[:, None] * observations).sum(axis=0) / total_g
                diff = observations - self._means[k]
                self._covars[k] = (
                    (g_k[:, None, None] * (diff[:, :, None] @ diff[:, None, :])).sum(axis=0)
                    / total_g + np.eye(D) * 1e-6
                )

        self._sort_states()
        self._fitted = True
        logger.info("Built-in HMM fit complete: %d iterations", self.n_iter)

    def _log_gaussian(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute log probability of X under multivariate Gaussian."""
        D = X.shape[1]
        diff = X - mean
        try:
            cov_inv = np.linalg.inv(cov)
            log_det = np.log(np.linalg.det(cov) + 1e-300)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(D)
            log_det = 0.0
        mahal = np.sum(diff @ cov_inv * diff, axis=1)
        return -0.5 * (D * np.log(2 * np.pi) + log_det + mahal)

    def _forward_backward(self, log_likes: np.ndarray) -> Tuple:
        """Forward-backward algorithm with scaling."""
        T, K = log_likes.shape
        likes = np.exp(log_likes - log_likes.max(axis=1, keepdims=True))

        alpha = np.zeros((T, K))
        beta = np.zeros((T, K))
        scale = np.zeros(T)

        # Forward
        alpha[0] = self._startprob * likes[0]
        scale[0] = alpha[0].sum() + 1e-300
        alpha[0] /= scale[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self._transmat) * likes[t]
            scale[t] = alpha[t].sum() + 1e-300
            alpha[t] /= scale[t]

        # Backward
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self._transmat @ (likes[t + 1] * beta[t + 1]))
            beta[t] /= scale[t + 1] + 1e-300

        return alpha, beta, scale

    def _sort_states(self) -> None:
        """Sort states so 0=BULL (highest mean return), 1=BEAR (lowest), 2=SIDEWAYS."""
        if self._means is None:
            return
        return_means = self._means[:, 0]  # First feature is log-returns
        order = np.argsort(return_means)[::-1]  # Descending by return mean

        self._means = self._means[order]
        self._covars = self._covars[order]
        self._transmat = self._transmat[order][:, order]
        self._startprob = self._startprob[order]

        if self._model is not None:
            try:
                self._model.means_ = self._means
                self._model.covars_ = self._covars
                self._model.transmat_ = self._transmat
                self._model.startprob_ = self._startprob
            except Exception:
                pass

    def filter(self, observations: np.ndarray) -> np.ndarray:
        """Return P(state_t | O_1:t) for each timestep.

        Parameters
        ----------
        observations : np.ndarray, shape (T, n_features)

        Returns
        -------
        np.ndarray, shape (T, n_states) — posterior probabilities
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._model is not None:
            return self._model.predict_proba(observations)

        # Built-in forward pass
        T = observations.shape[0]
        log_likes = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            log_likes[:, k] = self._log_gaussian(observations, self._means[k], self._covars[k])

        alpha, _, _ = self._forward_backward(log_likes)
        return alpha

    def get_current_regime(self, observations: np.ndarray) -> HMMRegimeSnapshot:
        """Get current regime with full probability vector.

        Parameters
        ----------
        observations : np.ndarray, shape (T, n_features)
            Recent observations (at least 10 days).

        Returns
        -------
        HMMRegimeSnapshot
        """
        probs = self.filter(observations)
        current_probs = probs[-1]
        state_idx = int(np.argmax(current_probs))
        regime_name = HMM_STATE_MAP[state_idx]
        confidence = float(current_probs[state_idx])

        # 5-day ahead prediction
        predicted_5d = self.predict_regime(current_probs, horizon=5)

        # Expected durations
        durations = {}
        for i, name in enumerate(HMM_STATE_NAMES):
            durations[name] = round(self.expected_duration(i), 1)

        # Transition matrix
        trans = []
        if self._transmat is not None:
            trans = self._transmat.tolist()

        snap = HMMRegimeSnapshot(
            regime=regime_name,
            state_index=state_idx,
            probabilities=current_probs.tolist(),
            confidence=confidence,
            transition_matrix=trans,
            expected_durations=durations,
            predicted_5d=predicted_5d.tolist(),
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(
            "HMM regime: %s (conf=%.0f%%), P=[%.2f, %.2f, %.2f], 5d_pred=[%.2f, %.2f, %.2f]",
            regime_name, confidence * 100,
            *current_probs, *predicted_5d,
        )

        return snap

    def predict_regime(self, current_probs: np.ndarray, horizon: int = 5) -> np.ndarray:
        """Predict regime probabilities h days ahead using transition matrix.

        P(S_{t+h}) = P(S_t) × A^h
        """
        if self._transmat is None:
            return current_probs.copy()

        A = self._transmat
        future_probs = current_probs.copy()
        for _ in range(horizon):
            future_probs = future_probs @ A
        return future_probs

    def expected_duration(self, state: int) -> float:
        """Expected duration in state (days) = 1 / (1 - A[s,s])."""
        if self._transmat is None:
            return 20.0
        diag = self._transmat[state, state]
        if diag >= 1.0:
            return 999.0
        return 1.0 / (1.0 - diag + 1e-10)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Return the transition matrix A."""
        if self._transmat is not None:
            return self._transmat
        return np.eye(self.n_states)

    def save(self, path: Optional[Path] = None) -> None:
        """Persist trained model to disk."""
        path = path or _MODEL_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "fitted": self._fitted,
            "n_states": self.n_states,
            "n_features": self.n_features,
            "means": self._means.tolist() if self._means is not None else None,
            "covars": [c.tolist() for c in self._covars] if self._covars is not None else None,
            "transmat": self._transmat.tolist() if self._transmat is not None else None,
            "startprob": self._startprob.tolist() if self._startprob is not None else None,
            "trained_at": datetime.utcnow().isoformat(),
        }
        path.write_text(json.dumps(state, indent=2))
        logger.info("HMM model saved to %s", path)

    def load(self, path: Optional[Path] = None) -> bool:
        """Load a previously trained model from disk.

        Returns True on success, False if no model found.
        """
        path = path or _MODEL_CACHE_PATH
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text())
            self.n_states = state["n_states"]
            self.n_features = state["n_features"]
            self._means = np.array(state["means"]) if state["means"] else None
            self._covars = np.array(state["covars"]) if state["covars"] else None
            self._transmat = np.array(state["transmat"]) if state["transmat"] else None
            self._startprob = np.array(state["startprob"]) if state["startprob"] else None
            self._fitted = state.get("fitted", False)
            logger.info("HMM model loaded from %s (trained at %s)", path, state.get("trained_at"))
            return True
        except Exception as exc:
            logger.warning("Failed to load HMM model: %s", exc)
            return False


# ═══════════════════════════════════════════════════════════════
# Markov Signal Filter (transition-aware signal quality)
# ═══════════════════════════════════════════════════════════════

def markov_signal_filter(
    forecast: float,
    current_regime_prob: np.ndarray,
    transition_matrix: np.ndarray,
) -> float:
    """Apply Markov-informed signal quality adjustment.

    If regime is likely transitioning to bear:
      - Dampen BUY signals
      - For long-only: strengthen exit signals

    Parameters
    ----------
    forecast : float
        Raw Carver-scale forecast.
    current_regime_prob : np.ndarray
        [P(BULL), P(BEAR), P(SIDEWAYS)]
    transition_matrix : np.ndarray
        3×3 transition matrix A.

    Returns
    -------
    float
        Adjusted forecast.
    """
    p_bull, p_bear, p_side = current_regime_prob

    # Expected next-period probabilities
    next_prob = current_regime_prob @ transition_matrix
    p_bear_next = next_prob[1]
    p_bull_next = next_prob[0]

    if forecast > 0:  # BUY signal
        # Dampen if transition to bear is elevated
        if p_bear_next > 0.20:
            bearing_risk = max(0.3, 1.0 - (p_bear_next - 0.10) * 2.0)
            return forecast * bearing_risk
        # Amplify in confident bull
        if p_bull > 0.75 and p_bull_next > 0.65:
            return min(20.0, forecast * 1.15)
    else:
        # Amplify sell signal in confident bear
        if p_bear > 0.7:
            return max(-20.0, forecast * 1.2)

    return forecast


def get_hmm_blended_weights(
    prob_vector: np.ndarray,
    regime_strategy_weights: Dict[str, Dict[str, float]],
    all_sources: List[str],
) -> Dict[str, float]:
    """Blend forecast weights using HMM regime probability distribution.

    Instead of binary regime → weight lookup, this computes:
        weights = P(bull) × W_bull + P(bear) × W_bear + P(side) × W_side

    Parameters
    ----------
    prob_vector : np.ndarray
        [P(BULL), P(BEAR), P(SIDEWAYS)]
    regime_strategy_weights : dict
        {regime_name: {source: weight}} from regime_strategy_mix.py
    all_sources : list[str]
        All forecast source names.

    Returns
    -------
    dict[str, float]
        Blended weights summing to ~1.0.
    """
    regime_names = ["TRENDING_BULL", "TRENDING_BEAR", "RANGE_BOUND"]
    blended: Dict[str, float] = {}

    for source in all_sources:
        w = 0.0
        for i, reg_name in enumerate(regime_names):
            reg_w = regime_strategy_weights.get(reg_name, {}).get(source, 0.0)
            w += prob_vector[i] * reg_w
        blended[source] = w

    # Normalize
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}

    return blended


# ═══════════════════════════════════════════════════════════════
# Data Preparation Helpers
# ═══════════════════════════════════════════════════════════════

def prepare_hmm_observations(
    nifty_df: "pd.DataFrame",
    vix_df: Optional["pd.DataFrame"] = None,
    breadth_values: Optional[list] = None,
    delivery_values: Optional[list] = None,
) -> np.ndarray:
    """Prepare observation matrix from market data.

    Parameters
    ----------
    nifty_df : pd.DataFrame
        NIFTY 50 OHLCV data.
    vix_df : pd.DataFrame | None
        India VIX OHLCV data (Close column used).
    breadth_values : list | None
        Daily advance/decline ratio (0-1). If None, estimated from NIFTY.
    delivery_values : list | None
        Daily delivery volume % (0-1). If None, uses constant 0.5.

    Returns
    -------
    np.ndarray, shape (T, 4)
    """
    import pandas as pd

    close = nifty_df["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()

    # Feature 1: Log returns
    log_returns = np.log(close / close.shift(1)).dropna()
    T = len(log_returns)

    # Feature 2: VIX (normalized 0-1 range)
    if vix_df is not None and not vix_df.empty:
        vix_close = vix_df["Close"]
        if hasattr(vix_close, "squeeze"):
            vix_close = vix_close.squeeze()
        vix_aligned = vix_close.reindex(log_returns.index, method="ffill").fillna(16.0)
        vix_norm = (vix_aligned - 10) / 30  # Normalize: 10=0, 40=1
        vix_norm = vix_norm.clip(0, 1)
    else:
        vix_norm = pd.Series(0.3, index=log_returns.index)  # Default moderate VIX

    # Feature 3: Breadth
    if breadth_values and len(breadth_values) >= T:
        breadth = pd.Series(breadth_values[-T:], index=log_returns.index)
    else:
        # Estimate breadth from NIFTY return sign (proxy)
        breadth = pd.Series(0.5, index=log_returns.index)
        breadth[log_returns > 0.003] = 0.65
        breadth[log_returns < -0.003] = 0.35

    # Feature 4: Delivery volume
    if delivery_values and len(delivery_values) >= T:
        delivery = pd.Series(delivery_values[-T:], index=log_returns.index)
    else:
        delivery = pd.Series(0.5, index=log_returns.index)

    observations = np.column_stack([
        log_returns.values,
        vix_norm.values[-T:],
        breadth.values[-T:],
        delivery.values[-T:],
    ])

    # Remove any NaN rows
    mask = ~np.isnan(observations).any(axis=1)
    observations = observations[mask]

    return observations


# ═══════════════════════════════════════════════════════════════
# Singleton convenience
# ═══════════════════════════════════════════════════════════════

_hmm_model: Optional[MarkovRegimeModel] = None


def get_hmm_model() -> MarkovRegimeModel:
    """Get or create the global HMM model instance."""
    global _hmm_model
    if _hmm_model is None:
        _hmm_model = MarkovRegimeModel()
        _hmm_model.load()  # Try loading cached model
    return _hmm_model
