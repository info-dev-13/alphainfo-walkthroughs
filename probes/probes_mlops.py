"""
recipes/probes_mlops.py — ML / MLOps drift probe library.

Fourth vertical, alongside finance, biomedical, and industrial.
ML model monitoring is one of the cleanest commercial fits for
structural detection because:

  * Ground-truth labels arrive late (fraud confirmations 30-90 days
    after the prediction; churn after the next billing cycle), so
    accuracy-based drift triggers are too slow.
  * What you HAVE in real time: the prediction stream and the input
    features. Both can drift independently.
  * What changes when drift starts is structural, not labelable: the
    distribution of predictions shifts, a feature column moves, or
    the joint distribution warps.

Where you'd reach for this library
----------------------------------

Typical signals fed to ``auto_diagnose`` with these probes:

  * The model's prediction stream (scores 0-1, probabilities, logits)
    over a recent window vs a launch-time baseline window.
  * A single feature column over a recent window vs baseline.
  * Per-class prediction count time series.
  * The error/loss stream from a regression model.

The result is "what KIND of drift is this?" — pinpointing the failure
mode lets MLOps teams escalate appropriately (re-train vs investigate
input pipeline vs flag a single feature for the data team).

Probes calibrated to ML-drift events
------------------------------------

  * prediction_collapse_to_uniform — confident bimodal becomes flat /
    unimodal around 0.5 (model is hedging).
  * prediction_concentration_increase — predictions cluster tighter
    around their modes (model is overconfident, possibly memorised
    a degenerate input).
  * class_imbalance_shift — for binary classifiers, proportion of
    positive predictions shifts (e.g. 50/50 → 70/30).
  * bimodality_collapse — clear two-peak distribution becomes one peak
    (the classifier stopped distinguishing classes).
  * outlier_rate_increase — fraction of >3σ predictions grows (model
    encountering new input regimes).
  * feature_mean_shift — covariate drift on a single feature column.
  * feature_variance_collapse — feature variance shrinks (e.g. an
    upstream filter started clipping).
  * latency_creep — slow degradation in inference latency (model or
    infra issue).
  * temporal_pattern_change — predictions develop a periodic rhythm
    that wasn't there at launch (cron-job feed corruption?).
  * loss_baseline_shift — for regression: error magnitude shifts.

Benign controls — "natural production noise":

  * within_session_jitter — small numeric jitter on predictions
    (rounding, hashing).
  * minor_class_proportion_jitter — ±2% class proportion drift over
    the day (real human-traffic patterns do this).
  * single_outlier_burst — one short anomaly window (5 weird inputs
    in 1000) — should NOT trigger alarm.

Important caveat
----------------

These probes are STRUCTURAL FINGERPRINT MATCHES, not retraining
verdicts. A diagnosis of "prediction_collapse_to_uniform" doesn't
say "retrain now" — it says "the structural fingerprint matches a
known degradation pattern". Real production decisions should pair
this with business-metric monitoring (the slow-but-true accuracy
signal) and operator review.

The intended use is to SHORTEN time-to-detect. If accuracy-based
monitoring catches drift in 7-30 days, structural detection might
flag the same drift in hours.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from recipes.auto_diagnose import Probe


# ─────────────────────────────────────────────────────────────────────────────
# Prediction-distribution probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_prediction_collapse_to_uniform(predictions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pull modes toward 0.5 (model losing confidence)."""
    p = predictions.copy().astype(np.float64)
    n = len(p)
    mid = n // 2
    # In the second half, shrink the distance from 0.5
    p[mid:] = 0.5 + 0.5 * (p[mid:] - 0.5) * 0.4
    return p


def probe_prediction_concentration_increase(predictions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Predictions cluster tighter around their modes (overconfidence)."""
    p = predictions.copy().astype(np.float64)
    n = len(p)
    mid = n // 2
    # Pull values toward the nearest mode (0 or 1)
    second = p[mid:]
    second = np.where(second > 0.5, 0.95 + 0.02 * (second - 0.5), 0.05 + 0.02 * (second - 0.5))
    p[mid:] = np.clip(second, 0.0, 1.0)
    return p


def probe_class_imbalance_shift(predictions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift proportion of positive predictions in the second half."""
    p = predictions.copy().astype(np.float64)
    n = len(p)
    mid = n // 2
    second = p[mid:]
    # Push values up by 0.2 (more "positives"), then clip
    p[mid:] = np.clip(second + 0.2, 0.0, 1.0)
    return p


def probe_bimodality_collapse(predictions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace the second half with a unimodal Gaussian centered at 0.5
    — both class modes have collapsed."""
    p = predictions.copy().astype(np.float64)
    n = len(p)
    mid = n // 2
    p[mid:] = np.clip(0.5 + 0.10 * rng.standard_normal(n - mid), 0.0, 1.0)
    return p


def probe_outlier_rate_increase(predictions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sprinkle 5-8% of the second half with extreme values (0.0 or 1.0)."""
    p = predictions.copy().astype(np.float64)
    n = len(p)
    mid = n // 2
    n_outliers = max(5, int(0.06 * (n - mid)))
    idx = rng.choice(np.arange(mid, n), size=n_outliers, replace=False)
    for i in idx:
        p[i] = 0.0 if rng.random() < 0.5 else 1.0
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Feature / covariate-drift probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_feature_mean_shift(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift the mean of the second half by 1σ — classic covariate drift."""
    v = values.copy().astype(np.float64)
    n = len(v)
    mid = n // 2
    sigma = float(np.std(v[:mid]))
    v[mid:] = v[mid:] + sigma
    return v


def probe_feature_variance_collapse(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shrink the second half's variance by 70% — upstream filter issue."""
    v = values.copy().astype(np.float64)
    n = len(v)
    mid = n // 2
    mu = float(np.mean(v[mid:]))
    v[mid:] = mu + (v[mid:] - mu) * 0.3
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Behavioural / temporal probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_latency_creep(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slow upward drift — models slow inference latency degradation."""
    n = len(values)
    sigma = float(np.std(values))
    drift = np.linspace(0.0, 1.5 * sigma, n)
    return values.astype(np.float64) + drift


def probe_temporal_pattern_change(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a periodic component into the second half — pattern that
    didn't exist at launch (cron-corrupted feed?)."""
    v = values.copy().astype(np.float64)
    n = len(v)
    mid = n // 2
    sigma = float(np.std(v))
    cycles = max(8, n // 60)
    t = np.arange(n - mid)
    v[mid:] = v[mid:] + 0.6 * sigma * np.sin(2 * np.pi * cycles * t / (n - mid))
    return v


def probe_loss_baseline_shift(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For regression models — overall error magnitude grows."""
    v = values.copy().astype(np.float64)
    n = len(v)
    mid = n // 2
    sigma = float(np.std(v))
    v[mid:] = v[mid:] * 1.5 + 0.5 * sigma
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Benign controls — "production noise that should NOT page anyone"
# ─────────────────────────────────────────────────────────────────────────────


def control_within_session_jitter(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tiny per-sample jitter (rounding-level)."""
    sigma = float(np.std(values))
    return values + 0.02 * sigma * rng.standard_normal(len(values))


def control_minor_class_proportion_jitter(predictions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """±2% class proportion shift — natural human-traffic variation."""
    p = predictions.copy().astype(np.float64)
    n = len(p)
    mid = n // 2
    p[mid:] = np.clip(p[mid:] + 0.02, 0.0, 1.0)
    return p


def control_single_outlier_burst(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """5 outliers in a short window — one weird user batch, not drift."""
    v = values.copy().astype(np.float64)
    n = len(v)
    sigma = float(np.std(values))
    burst_start = int(rng.integers(n // 4, 3 * n // 4))
    n_burst = min(5, n - burst_start)
    v[burst_start:burst_start + n_burst] = (
        v[burst_start:burst_start + n_burst] + 3.0 * sigma * rng.standard_normal(n_burst)
    )
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Public registries
# ─────────────────────────────────────────────────────────────────────────────


MLOPS_PROBES: Dict[str, Probe] = {
    "prediction_collapse_to_uniform":   probe_prediction_collapse_to_uniform,
    "prediction_concentration_increase": probe_prediction_concentration_increase,
    "class_imbalance_shift":            probe_class_imbalance_shift,
    "bimodality_collapse":              probe_bimodality_collapse,
    "outlier_rate_increase":            probe_outlier_rate_increase,
    "feature_mean_shift":               probe_feature_mean_shift,
    "feature_variance_collapse":        probe_feature_variance_collapse,
    "latency_creep":                    probe_latency_creep,
    "temporal_pattern_change":          probe_temporal_pattern_change,
    "loss_baseline_shift":              probe_loss_baseline_shift,
}


MLOPS_BENIGN_CONTROLS: Dict[str, Probe] = {
    "within_session_jitter":           control_within_session_jitter,
    "minor_class_proportion_jitter":   control_minor_class_proportion_jitter,
    "single_outlier_burst":            control_single_outlier_burst,
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic prediction-stream generator
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_predictions(
    n: int = 1000,
    confidence: float = 0.85,
    positive_rate: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic binary-classifier prediction stream.

    confidence: how tight the modes are (0.5 = uniform, 1.0 = perfect).
    positive_rate: fraction of samples predicted as the positive class.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n * positive_rate)
    n_neg = n - n_pos
    spread = max(0.001, 1 - confidence) * 0.15
    pos = confidence + spread * rng.standard_normal(n_pos)
    neg = (1 - confidence) + spread * rng.standard_normal(n_neg)
    return np.clip(np.concatenate([pos, neg]), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def _demo():
    """Apply 4 known transformations to a baseline prediction stream and
    verify auto_diagnose names the right probe."""
    import os
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long")
    os.environ.setdefault("MASTER_API_KEY", "ai_master_test")

    from recipes.auto_diagnose import auto_diagnose

    rng = np.random.default_rng(0)
    baseline = synthetic_predictions(n=1000, confidence=0.85, positive_rate=0.5, seed=0)

    test_cases = [
        ("predictions collapsing to 0.5",  probe_prediction_collapse_to_uniform(baseline, rng)),
        ("class imbalance shifted",        probe_class_imbalance_shift(baseline, rng)),
        ("feature_mean_shift (1σ)",        probe_feature_mean_shift(baseline, rng)),
        ("just session jitter (benign)",  control_within_session_jitter(baseline, rng)),
    ]

    class _LocalClient:
        def __init__(self):
            from fastapi.testclient import TestClient
            from api.app import app
            self.tc = TestClient(app)
            self.key = os.environ["MASTER_API_KEY"]

        def analyze_vector(self, channels, sampling_rate=1.0, baselines=None,
                           domain="generic", use_multiscale=True,
                           include_semantic=False):
            from types import SimpleNamespace
            payload = {
                "channels": channels, "sampling_rate": sampling_rate,
                "domain": domain, "use_multiscale": use_multiscale,
                "include_semantic": include_semantic,
            }
            if baselines is not None:
                payload["baselines"] = baselines
            r = self.tc.post(
                "/v1/analyze/vector",
                headers={"X-API-Key": self.key},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            ch_raw = data.get("channels", {}) or {}
            if isinstance(ch_raw, dict):
                ch = {k: SimpleNamespace(**v) for k, v in ch_raw.items()}
            else:
                ch = [SimpleNamespace(**c) for c in ch_raw]
            return SimpleNamespace(
                structural_score=data.get("structural_score"),
                change_detected=data.get("change_detected"),
                confidence_band=data.get("confidence_band"),
                channels=ch,
            )

    client = _LocalClient()
    for label, current in test_cases:
        out = auto_diagnose(
            client,
            signal=current.tolist(),
            baseline=baseline.tolist(),
            sampling_rate=1.0,
            domain="ai_ml",
            probes=MLOPS_PROBES,
            benign_controls=MLOPS_BENIGN_CONTROLS,
        )
        print(f"\nground truth: {label}")
        print(f"  diagnosis  : {out['diagnosis']:35}  confidence={out['confidence']:.3f}")
        print(f"  benign sim : {out['benign_similarity']:.3f}")
        print(f"  top-3 probes:")
        for name, score in out["ranked"][:3]:
            print(f"    {name:35}  {score:.3f}")


if __name__ == "__main__":
    _demo()
