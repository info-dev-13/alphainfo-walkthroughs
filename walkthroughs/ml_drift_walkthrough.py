"""
recipes/notebooks/ml_drift_walkthrough.py — Detect ML model drift
without ground-truth labels.

ML drift detection in production is one of the cleanest commercial
fits for structural detection. The setup:

  - You deployed a model.
  - It scored well at launch.
  - Weeks later, performance is degrading but you don't know exactly
    when, why, or which segment.
  - Ground-truth labels arrive WAY after predictions (fraud
    confirmations 30-90 days later, churn after the next billing
    cycle), so retraining triggers based on accuracy are too slow.

What you have at inference time, in real time:
  - The MODEL'S PREDICTIONS (scores / probabilities).
  - The INPUT FEATURES.

What changes when drift starts:
  - Distribution of predictions shifts (covariate drift on features
    propagates to outputs).
  - Distribution of feature columns shifts.
  - Sometimes a single feature column shifts; sometimes the whole
    joint distribution.

structural detection is the right shape for this:
  - Feed the prediction stream as a 1-D signal vs a baseline window
    from launch. Run the `distribution_shift` intent. The
    `histogram` channel reads distribution drift; `z_norm` reads
    shape; `spectrum` reads cyclical pattern shifts.
  - For multi-feature drift, build a `analyze_vector` ensemble where
    each channel is a feature column. The per-channel score
    pinpoints which feature drifted.

This walkthrough simulates a simple drift scenario:

  Day 0 (launch):     model predicts a stable bimodal distribution
                       around 0.2 / 0.8 (typical binary classifier).
  Day 30 (drift):     covariate shift moves the population — predictions
                       collapse toward 0.5 (model less confident) and
                       the bimodality flattens.
  Day 31 (catastrophic): one feature drifts by 3σ — single-channel
                          analysis isolates which feature caused it.

Everything runs in-process; no real ML model needed for the demo.
The point is to show the SHAPE of the analysis a real MLOps team
would set up.

Usage
-----

    cd alphainfo/
    .venv/bin/python -m recipes.notebooks.ml_drift_walkthrough
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long")
os.environ.setdefault("MASTER_API_KEY", "ai_master_test")


class _LocalClient:
    def __init__(self):
        from fastapi.testclient import TestClient
        from api.app import app
        self.tc = TestClient(app)
        self.key = os.environ["MASTER_API_KEY"]

    def _post(self, path, payload):
        r = self.tc.post(path, headers={"X-API-Key": self.key}, json=payload)
        r.raise_for_status()
        return r.json()

    def analyze(self, signal, baseline=None, sampling_rate=1.0,
                domain="generic", include_semantic=False, use_multiscale=True,
                metadata=None):
        from types import SimpleNamespace
        payload = {
            "signal": list(signal), "sampling_rate": sampling_rate,
            "domain": domain, "include_semantic": include_semantic,
            "use_multiscale": use_multiscale,
        }
        if baseline is not None:
            payload["baseline"] = list(baseline)
        if metadata is not None:
            payload["metadata"] = metadata
        return SimpleNamespace(**self._post("/v1/analyze/stream", payload))

    def analyze_batch(self, signals, sampling_rate=1.0, baselines=None,
                      domain="generic", use_multiscale=True,
                      include_semantic=False):
        from types import SimpleNamespace
        payload = {
            "signals": [list(s) for s in signals],
            "sampling_rate": sampling_rate, "domain": domain,
            "use_multiscale": use_multiscale,
            "include_semantic": include_semantic,
        }
        if baselines is not None:
            payload["baselines"] = [list(b) for b in baselines]
        d = self._post("/v1/analyze/batch", payload)
        return SimpleNamespace(
            results=[SimpleNamespace(**x) for x in d.get("results", [])],
        )

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
        d = self._post("/v1/analyze/vector", payload)
        ch_raw = d.get("channels", {}) or {}
        if isinstance(ch_raw, dict):
            ch = {k: SimpleNamespace(**v) for k, v in ch_raw.items()}
        else:
            ch = [SimpleNamespace(**c) for c in ch_raw]
        return SimpleNamespace(
            structural_score=d.get("structural_score"),
            change_detected=d.get("change_detected"),
            confidence_band=d.get("confidence_band"),
            channels=ch,
        )


def section_header(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def simulate_predictions(n_samples: int, regime: str, seed: int) -> np.ndarray:
    """Simulate a binary classifier's prediction stream under three regimes.

    'launch'      — confident bimodal (0.2 / 0.8 with low spread)
    'drift'       — collapsing toward 0.5 (less confident, flatter peaks)
    'catastrophic'— even more uniform, with significant 0.4-0.6 mass
    """
    rng = np.random.default_rng(seed)
    if regime == "launch":
        # 50% predicted as positive class with prob ~0.8, 50% as negative ~0.2
        pos = 0.8 + 0.05 * rng.standard_normal(n_samples // 2)
        neg = 0.2 + 0.05 * rng.standard_normal(n_samples - n_samples // 2)
        return np.clip(np.concatenate([pos, neg]), 0.0, 1.0)
    elif regime == "drift":
        # Modes pulling toward 0.5 (model losing confidence)
        pos = 0.65 + 0.10 * rng.standard_normal(n_samples // 2)
        neg = 0.35 + 0.10 * rng.standard_normal(n_samples - n_samples // 2)
        return np.clip(np.concatenate([pos, neg]), 0.0, 1.0)
    elif regime == "catastrophic":
        # Predictions almost uniform — model is calling 50/50 on most cases
        pos = 0.55 + 0.15 * rng.standard_normal(n_samples // 2)
        neg = 0.45 + 0.15 * rng.standard_normal(n_samples - n_samples // 2)
        return np.clip(np.concatenate([pos, neg]), 0.0, 1.0)
    raise ValueError(regime)


def simulate_features(n_samples: int, drift_feature_idx: int | None,
                       seed: int) -> dict[str, np.ndarray]:
    """Simulate 5 input features. If drift_feature_idx is set, that
    feature has a 3σ shift; the others are stationary."""
    rng = np.random.default_rng(seed)
    feats = {}
    for i in range(5):
        col = rng.standard_normal(n_samples)
        if i == drift_feature_idx:
            col = col + 3.0  # shifted by 3σ
        feats[f"feature_{i}"] = col
    return feats


def main():
    print()
    print("┌" + "─" * 68 + "┐")
    print("│  alphainfo end-to-end walkthrough — ML drift detection           │")
    print("│  Detect drift in classifier predictions + isolate culpable feat. │")
    print("└" + "─" * 68 + "┘")

    client = _LocalClient()
    n_per_window = 500

    # ─────────────────────────────────────────────────────────────────────────
    # Section 1 — Baseline window (launch)
    # ─────────────────────────────────────────────────────────────────────────
    section_header("1. Baseline window (model at launch, day 0)")
    baseline_preds = simulate_predictions(n_per_window, regime="launch", seed=0)
    print(f"\n  predictions: n={len(baseline_preds)}, mean={baseline_preds.mean():.3f}, "
          f"std={baseline_preds.std():.3f}")
    print("  bimodal — confident classifier (peaks at ~0.2 and ~0.8)")

    # ─────────────────────────────────────────────────────────────────────────
    # Section 2 — Drift detection on prediction stream
    # ─────────────────────────────────────────────────────────────────────────
    section_header("2. Detect drift on prediction stream (intent='distribution_shift')")
    drift_preds = simulate_predictions(n_per_window, regime="drift", seed=1)
    print(f"\n  drift window: n={len(drift_preds)}, mean={drift_preds.mean():.3f}, "
          f"std={drift_preds.std():.3f}  (model less confident)")

    from recipes.intents import dispatch
    out = dispatch(
        client, intent="distribution_shift",
        signal=drift_preds.tolist(), baseline=baseline_preds.tolist(),
        sampling_rate=1.0, domain="generic",
    )
    print(f"\n  aggregate score: {out['aggregate_score']:.4f}  ({out['confidence_band']})")
    print("  responsible top-3 channels:")
    for name, score in out["channel_ranking"][:3]:
        print(f"    • {name:14}  {score:.4f}")

    # Validate that distribution-sensitive channels lit up
    dist_channels = {"histogram", "z_norm", "minmax", "spectrum"}
    responsible = set(out["responsible"])
    if dist_channels & responsible:
        print("\n  ✓ Distribution-sensitive channel detected drift (expected behaviour)")
    else:
        print("\n  ! No distribution channel in top-3 — review encoder set")

    # ─────────────────────────────────────────────────────────────────────────
    # Section 3 — Catastrophic drift with one shifted feature
    # ─────────────────────────────────────────────────────────────────────────
    section_header("3. Multi-feature analysis: WHICH feature drifted?")
    print("\n  Scenario: 5 input features, baseline all stationary.")
    print("  In the new window, feature_2 has shifted by 3σ. Detect + isolate.\n")

    baseline_feats = simulate_features(n_per_window, drift_feature_idx=None, seed=10)
    drifted_feats = simulate_features(n_per_window, drift_feature_idx=2, seed=11)

    print("  baseline:")
    for name, col in baseline_feats.items():
        print(f"    {name:12}  mean={col.mean():+.3f}  std={col.std():.3f}")
    print("\n  drifted (feature_2 shifted by 3σ):")
    for name, col in drifted_feats.items():
        print(f"    {name:12}  mean={col.mean():+.3f}  std={col.std():.3f}")

    out2 = client.analyze_vector(
        channels={k: v.tolist() for k, v in drifted_feats.items()},
        baselines={k: v.tolist() for k, v in baseline_feats.items()},
        sampling_rate=1.0, domain="generic",
    )
    print(f"\n  Aggregate structural score: {out2.structural_score:.4f}  ({out2.confidence_band})")
    print("  Per-feature ranking (lower = more drifted):")
    feat_scores = sorted(
        ((name, ch.structural_score) for name, ch in out2.channels.items()),
        key=lambda kv: kv[1],
    )
    for name, score in feat_scores:
        marker = " ← drifted" if name == "feature_2" else ""
        print(f"    {name:12}  {score:.4f}{marker}")

    if feat_scores[0][0] == "feature_2":
        print("\n  ✓ The drifted feature was correctly identified as the most-changed channel")
    else:
        print(f"\n  ! Top-ranked drift channel was {feat_scores[0][0]!r}, not feature_2")

    # ─────────────────────────────────────────────────────────────────────────
    # Section 4 — Operational use
    # ─────────────────────────────────────────────────────────────────────────
    section_header("4. Operational pipeline shape")
    print("""
  In production, the same shape works as a continuous monitor:

    1. At launch: store a 'reference window' of predictions + features
       (just numpy arrays — typically 5-10K samples).

    2. Every N hours (or after K predictions): assemble a new
       'current window' of the same size.

    3. Run dispatch(intent='distribution_shift', ...) on the
       prediction stream. If aggregate_score < threshold (e.g. 0.5)
       AND a distribution channel is in the top-3 responsible:
         → page the on-call MLOps engineer.

    4. Run analyze_vector with feature columns as channels to
       isolate WHICH feature drifted. Score per feature is the
       prioritised investigation list.

    5. Optional: chain `regime_change` to BOTH locate and label
       the drift kind via auto_diagnose. (Default probes work;
       tighter probes can be domain-specific.)

  Cost: ~2 analyses per check (one prediction stream + one vector
  call). At hourly cadence, that's 1,440 analyses/month — fits in
  the Starter plan with room for retries.

  ⚠ This is a STRUCTURAL drift signal, not a verdict. Fold it
    into your existing canary / shadow / A-B framework. Use it to
    SHORTEN the time-to-detect, not to replace business-metric
    monitoring (which still needs the labels you're waiting for).
""")


if __name__ == "__main__":
    main()
