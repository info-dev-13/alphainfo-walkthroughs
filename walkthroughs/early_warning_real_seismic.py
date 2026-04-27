"""
recipes/notebooks/early_warning_real_seismic.py — REAL-DATA validation
of the multivariate-encoder + structural-fingerprint early-warning
pattern.

Why a sibling notebook
----------------------

`multivariate_early_warning_walkthrough.py` shows the technique on
synthetic data. The natural follow-up question is whether the
pattern survives real data with a real precursor. This notebook
is the answer.

We validate the SAME pattern on a real catastrophic-event signal
that ships with the repo:

  data/real_datasets/seismic/seismic_background_noise.npy
  data/real_datasets/seismic/seismic_p_wave_event.npy

A seismic signal containing a P-wave precursor + main earthquake event
is the canonical real-world "stable phase → catastrophic amplitude
event" problem: a slow regime change followed by a high-amplitude
event. If the multivariate-encoder + structural-fingerprint pattern
wins lead time on this real signal, the technique generalises
beyond the synthetic demo.

What it shows
-------------

  * Same encode_window() trick (means, stds, deriv energy, pairwise
    correlations) but the channels are now derived views of a single
    real seismic trace: raw, absolute envelope, short-time energy,
    rectified, gradient.
  * Same structural-fingerprint vs stable-phase baseline.
  * Same comparison: structural alert vs naive amplitude threshold.
  * Same metric: how many samples earlier did structure flag the
    event compared with raw amplitude?

Honest framing
--------------

  * Single-channel seismic with derived "channels" is not the same
    as a truly multivariate process. We're demonstrating that the
    technique survives real data with a real precursor; we are NOT
    claiming AlphaInfo predicts earthquakes.
  * The lead time depends on the encoder. The seismic encoder here is
    deliberately simple — production seismic monitoring uses STA/LTA,
    cross-correlation arrays, machine learning, etc. AlphaInfo measures
    the structural change between encoded windows; the encoder is
    where domain knowledge lives.
  * One real shot is not a proof. The contribution of this notebook
    is reproducibility on a public, identifiable dataset — readers
    can try alternative encoders or run their own seismic data
    through the same pipeline.

Run
---

    python -m recipes.notebooks.early_warning_real_seismic

Quota: ~50 calls (in-process TestClient — no real key needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Multi-channel adapter from a single seismic trace
# ─────────────────────────────────────────────────────────────────────────────


def derived_channels(trace: np.ndarray, smooth_win: int = 50) -> np.ndarray:
    """Build a (5, n_samples) "multi-channel" view from a single
    seismic trace. Each derived channel is a different structural
    view of the same physical signal:

      0. raw                           — the trace itself.
      1. absolute envelope             — short-time |x|.
      2. short-time energy             — running x^2 mean.
      3. rectified high-pass           — |diff|, captures velocity.
      4. running kurtosis (windowed)   — distribution shape proxy.

    These are the kinds of features a seismologist computes by hand;
    the structural encoder downstream then summarises their joint
    behaviour per window.
    """
    n = len(trace)
    # 1. Absolute envelope (smoothed)
    env = np.abs(trace)
    env_smooth = np.convolve(env, np.ones(smooth_win) / smooth_win, mode="same")

    # 2. Short-time energy
    energy = trace ** 2
    energy_smooth = np.convolve(energy, np.ones(smooth_win) / smooth_win, mode="same")

    # 3. Rectified high-pass = |diff|
    rect_hp = np.zeros(n)
    rect_hp[1:] = np.abs(np.diff(trace))

    # 4. Running kurtosis (proxy via 4th moment / 2nd moment^2)
    kurt = np.zeros(n)
    half = smooth_win // 2
    for i in range(half, n - half):
        win = trace[i - half:i + half + 1]
        m = win.mean()
        var = ((win - m) ** 2).mean() + 1e-12
        m4 = ((win - m) ** 4).mean()
        kurt[i] = m4 / (var ** 2)

    return np.stack([trace, env_smooth, energy_smooth, rect_hp, kurt])


def encode_window(window: np.ndarray) -> np.ndarray:
    """Same encoder as multivariate_early_warning_walkthrough.encode_window —
    means + stds + derivative energy + pairwise correlations."""
    n_ch = window.shape[0]
    means = window.mean(axis=1)
    stds = window.std(axis=1)
    derivatives = np.diff(window, axis=1)
    deriv_energy = (derivatives ** 2).sum(axis=1)
    corrs = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            si, sj = stds[i] + 1e-12, stds[j] + 1e-12
            corrs.append(
                float(((window[i] - means[i]) * (window[j] - means[j])).mean()
                      / (si * sj))
            )
    return np.concatenate([means, stds, deriv_energy, np.asarray(corrs)])


def slide_windows(channels: np.ndarray, window_size: int, step: int):
    """Yield (end_idx, window) pairs over a multi-channel array."""
    n = channels.shape[1]
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        yield end, channels[:, start:end]


def find_event_onset(trace: np.ndarray, threshold_ratio: float = 5.0,
                     stable_window: int = 2000) -> int:
    """Find the sample index at which absolute amplitude first exceeds
    `threshold_ratio` × the stable-phase std. Used as the ground-truth
    'event onset' for the seismic trace."""
    stable_std = float(np.std(trace[:stable_window]))
    threshold = threshold_ratio * stable_std
    above = np.where(np.abs(trace) > threshold)[0]
    if len(above) == 0:
        return -1
    return int(above[0])


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """End-to-end real-data validation."""
    import os
    os.environ.setdefault(
        "JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long",
    )
    os.environ.setdefault("MASTER_API_KEY", "ai_master_test")
    os.environ.setdefault("SKIP_DB", "1")

    from fastapi.testclient import TestClient
    from api.app import app
    tc = TestClient(app)
    key = os.environ["MASTER_API_KEY"]

    print("┌────────────────────────────────────────────────────────────────┐")
    print("│  alphainfo end-to-end walkthrough — REAL-DATA early warning     │")
    print("│  Multivariate structural fingerprint vs amplitude threshold     │")
    print("│  Source: data/real_datasets/seismic/ (single-trace seismogram)  │")
    print("└────────────────────────────────────────────────────────────────┘\n")

    # 1. Load real seismic trace
    print("══════════════════════════════════════════════════════════════════════")
    print("  1. Load real seismic trace")
    print("══════════════════════════════════════════════════════════════════════\n")
    repo_root = Path(__file__).parent.parent.parent
    bg_path = repo_root / "data" / "real_datasets" / "seismic" / "seismic_background_noise.npy"
    pw_path = repo_root / "data" / "real_datasets" / "seismic" / "seismic_p_wave_event.npy"
    if not pw_path.exists():
        print(f"  ✗ data not found at {pw_path}")
        print("  This walkthrough expects the public seismic dataset shipped in")
        print("  data/real_datasets/seismic/. Skipping.")
        return

    trace = np.load(pw_path)
    bg = np.load(bg_path)
    print(f"  trace shape:        {trace.shape}")
    print(f"  background shape:   {bg.shape}")
    print(f"  trace amplitude:    [{trace.min():.3f}, {trace.max():.3f}]")
    print(f"  bg amplitude:       [{bg.min():.3f}, {bg.max():.3f}]")

    event_onset = find_event_onset(trace, threshold_ratio=5.0, stable_window=2000)
    print(f"\n  ground-truth event onset (5σ amplitude rule): sample {event_onset}")

    # 2. Build derived channels
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  2. Build 5 derived channels")
    print("══════════════════════════════════════════════════════════════════════\n")
    channels = derived_channels(trace, smooth_win=50)
    print(f"  channels shape:     {channels.shape}")
    print("  channels:           [raw, env_smooth, energy_smooth, rect_hp, kurt]")

    # 3. Slide structural feature windows
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  3. Slide windows + structural encoding")
    print("══════════════════════════════════════════════════════════════════════\n")
    window_size = 500
    step = 100
    windows = list(slide_windows(channels, window_size=window_size, step=step))
    feature_dim = len(encode_window(windows[0][1]))
    print(f"  windows:            {len(windows)}")
    print(f"  window_size:        {window_size}")
    print(f"  step:               {step}")
    print(f"  feature_dim:        {feature_dim}")

    # Stable baseline = mean of feature vectors from the FIRST 30% of the
    # trace (well before the event). Using event_onset would leak.
    stable_cutoff = int(0.30 * trace.shape[0])
    stable_features = [
        encode_window(w) for end, w in windows if end < stable_cutoff
    ]
    if not stable_features:
        print("  ✗ no stable windows; widen baseline period")
        return
    baseline_feature = np.mean(stable_features, axis=0)
    print(f"  stable baseline:    {len(stable_features)} windows from samples 0..{stable_cutoff}")

    # 4. analyze_batch — compare every window's features to the baseline
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  4. structural fingerprint per window")
    print("══════════════════════════════════════════════════════════════════════\n")
    feature_vectors = [encode_window(w).tolist() for end, w in windows]
    payload = {
        "signals": feature_vectors,
        "baselines": [baseline_feature.tolist()] * len(feature_vectors),
        "sampling_rate": 1.0,
        "domain": "generic",
        "use_multiscale": True,
        "include_semantic": False,
    }
    r = tc.post(
        "/v1/analyze/batch",
        headers={"X-API-Key": key},
        json=payload,
    )
    r.raise_for_status()
    batch = r.json()
    print(f"  quota_consumed:     {len(feature_vectors)}")
    print(f"  results:            {len(batch.get('results', []))}")

    scores = []
    for i, (end, _) in enumerate(windows):
        item = batch["results"][i]
        scores.append((end, item.get("structural_score"), item.get("confidence_band")))

    # 5. Calibrate threshold from stable-phase scores
    stable_scores = [s for end, s, _ in scores
                     if end < stable_cutoff and s is not None]
    threshold = float(np.percentile(stable_scores, 5)) if stable_scores else 0.5
    print(f"\n  stable-phase 5th percentile score: {threshold:.3f}")
    print(f"  → using as alert threshold (with 3-window persistence)")

    # 6. Find first structural alert
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  5. Structural alert vs naive amplitude alert")
    print("══════════════════════════════════════════════════════════════════════\n")
    structural_alert = None
    consec = 0
    for end, score, _ in scores:
        if score is not None and score < threshold:
            consec += 1
            if consec >= 3:
                structural_alert = end
                break
        else:
            consec = 0

    # Naive amplitude alert: first time |x| > 5σ over 3 windows
    stable_std = float(np.std(trace[:2000]))
    amp_threshold = 5.0 * stable_std
    amp_alert = None
    consec = 0
    for end, w in windows:
        # raw channel is row 0
        amp = float(np.abs(w[0]).max())
        if amp > amp_threshold:
            consec += 1
            if consec >= 3:
                amp_alert = end
                break
        else:
            consec = 0

    print(f"  event onset (5σ on individual sample):           sample {event_onset}")
    print(f"  structural alert (3 cons. windows < {threshold:.3f}):  end={structural_alert}")
    print(f"  amplitude  alert (3 cons. windows |x| > 5σ):     end={amp_alert}")
    print()
    print("  Both window-based alerts fire after the per-sample 5σ")
    print("  event onset — they need persistence to be meaningful. The")
    print("  fair comparison is structural vs amplitude on the same")
    print("  3-window persistence rule:")
    if structural_alert and amp_alert:
        diff = amp_alert - structural_alert
        print(f"    structural beats 3-window amplitude by:        {diff:+d} samples")

    # 7. Worst windows
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  6. Worst (lowest-score) windows")
    print("══════════════════════════════════════════════════════════════════════\n")
    valid = [(end, s, b) for end, s, b in scores if s is not None]
    valid.sort(key=lambda x: x[1])
    for end, score, band in valid[:5]:
        phase = ("stable" if end < stable_cutoff
                 else "event" if event_onset > 0 and end > event_onset
                 else "transition")
        print(f"  end={end:5d}  score={score:.3f}  band={band:11}  phase={phase}")

    # 8. Summary
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  Summary")
    print("══════════════════════════════════════════════════════════════════════\n")
    print("  This walkthrough used:")
    print(f"    • Real seismic trace ({trace.shape[0]} samples) with a P-wave")
    print(f"      precursor + main event in data/real_datasets/seismic/")
    print("    • 5 derived channels [raw, env, energy, |diff|, kurt]")
    print("    • Multivariate structural encoder (means, stds, deriv energy,")
    print("      pairwise correlations) → feature vector per window")
    print("    • analyze_batch comparing each window against a stable baseline")
    print("    • 5th-percentile-of-stable threshold + 3-window persistence")
    print()
    print("  Same encoder pattern as multivariate_early_warning_walkthrough.py")
    print("  — applied here to a real catastrophic-event signal. This is a")
    print("  generalisation test, not a proof: success here does NOT mean")
    print("  AlphaInfo predicts earthquakes. It means the multivariate-")
    print("  feature-encoding + structural-fingerprint pattern survives real")
    print("  data with a real precursor.")


if __name__ == "__main__":
    main()
