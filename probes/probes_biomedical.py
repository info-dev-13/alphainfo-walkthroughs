"""
recipes/probes_biomedical.py — Biomedical-domain probe library.

Companion to ``recipes.probes_finance``. Where finance probes model
market regime changes (vol_regime_2x, crash_event), biomedical probes
model **clinically-meaningful events** in cardiac and electrophysiological
signals — the kind of regime change that, if missed, can mean a missed
arrhythmia, a missed seizure, or a sensor failure that gets logged as a
patient event.

Domain focus
------------

The default probes here target ECG-like signals: a periodic baseline
with sharp R-wave peaks, optionally with T-waves and slower P-waves.
The same probes work on EEG, EMG, SpO2, and PPG with adjusted scale —
biomedical signals share the structure that the engine cares about
(periodic shape + transient events + slow drifts).

Probes calibrated to clinical events
------------------------------------

  * tachycardia        — heart rate increases (period shortens).
  * bradycardia        — heart rate decreases (period lengthens).
  * arrhythmia         — cycle-to-cycle interval becomes irregular.
  * pvc_injection      — a single premature ventricular contraction.
  * amplitude_attenuation — gradual signal-amplitude loss (electrode
                            dry, contact degradation).
  * baseline_wander    — slow drift from breathing or motion.
  * powerline_60hz     — 50/60 Hz noise injection (mains interference).
  * lead_disconnect    — abrupt drop to near-zero (lead came off).
  * st_segment_shift   — slow elevation/depression of the segment
                          between QRS and T (myocardial ischemia).
  * emg_contamination  — short bursts of high-frequency noise (muscle
                          twitch / movement artifact).
  * seizure_burst      — high-amplitude rhythmic spike-wave (EEG-style).

Benign controls — what "patient at rest, sensor working" looks like:

  * within_hr_variability   — natural beat-to-beat jitter at 5%.
  * minor_breathing_artifact — tiny periodic baseline (~0.3 Hz).
  * single_motion_glitch     — one sub-σ blip (touched the lead).

Important caveat
----------------

These probes are **engineering primitives**, not clinical diagnostics.
A diagnosis of "tachycardia" from auto_diagnose means "the structural
fingerprint of this signal vs the resting baseline matches the
fingerprint of a sped-up baseline." It does NOT mean "this patient is
in tachycardia and needs intervention." Real clinical use must combine
this with rhythm-classification algorithms, vital-sign context, and a
licensed clinician.

The intended consumer is signal-processing tooling: pipeline gating
("which captures look like artifact?"), QA dashboards, retrospective
labeling, and research. The structural fingerprint is fast and
domain-agnostic; clinical interpretation belongs upstream.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from recipes.auto_diagnose import Probe


# ─────────────────────────────────────────────────────────────────────────────
# Heart-rate / period probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_tachycardia(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compress the time axis of the second half by 1.5×. Models a
    sudden HR increase from ~70 to ~105 bpm."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    second = s[mid:]
    # Resample by 1.5× — interpolate onto a shorter grid then tile back.
    new_len = int(len(second) / 1.5)
    if new_len < 10:
        return s
    idx = np.linspace(0, len(second) - 1, new_len)
    compressed = np.interp(idx, np.arange(len(second)), second)
    # Tile to fill remaining length
    tiled = np.tile(compressed, (len(second) // new_len) + 2)[:len(second)]
    s[mid:] = tiled
    return s


def probe_bradycardia(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stretch the time axis of the second half by 1.5×. Models a HR
    decrease from ~70 to ~47 bpm — bradycardia onset."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    second = s[mid:]
    # Resample by stretching — same length out but slower content.
    target_input = int(len(second) / 1.5)
    if target_input < 10:
        return s
    src = second[:target_input]
    idx = np.linspace(0, len(src) - 1, len(second))
    s[mid:] = np.interp(idx, np.arange(len(src)), src)
    return s


def probe_arrhythmia(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject random per-sample time jitter in the second half — models
    the loss of regular rhythm. The signal still has the same beats but
    the inter-beat intervals scatter."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    second = s[mid:]
    # Build a non-monotonic resample index with up to 8% jitter
    jitter = 0.08 * len(second) * rng.standard_normal(len(second))
    idx = np.clip(np.arange(len(second)) + jitter, 0, len(second) - 1)
    s[mid:] = np.interp(idx, np.arange(len(second)), second)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Single-event probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_pvc_injection(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a single premature beat: a sharp anomalous spike followed
    by a compensatory pause. Models a PVC (premature ventricular
    contraction)."""
    s = signal.copy()
    n = len(s)
    pos = int(rng.integers(n // 3, 3 * n // 4))
    sigma = float(np.std(signal))
    # Premature spike — wide, opposite-polarity from typical R-wave
    width = max(5, n // 200)
    half = width // 2
    lo = max(0, pos - half)
    hi = min(n, pos + half)
    s[lo:hi] = -3.0 * sigma  # negative-going, ~3σ
    # Compensatory pause — flat for a beat-and-a-half
    pause_end = min(n, pos + 3 * width)
    s[hi:pause_end] = float(np.mean(signal))
    return s


def probe_lead_disconnect(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Abrupt amplitude collapse to near-zero in the last quarter —
    models a lead/electrode disconnection. Common false-positive for
    real arrhythmia detectors."""
    s = signal.copy()
    n = len(s)
    cutoff = 3 * n // 4
    s[cutoff:] = 0.05 * float(np.std(signal)) * rng.standard_normal(n - cutoff)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Slow / drift probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_amplitude_attenuation(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a linear amplitude decay from 1.0 to 0.4 over the signal.
    Models gradual loss of contact / dry electrode / sensor degradation."""
    n = len(signal)
    decay = np.linspace(1.0, 0.4, n)
    return signal * decay


def probe_baseline_wander(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a slow sinusoidal drift at ~0.25 Hz — the typical baseline
    wander frequency from breathing artifact."""
    n = len(signal)
    sigma = float(np.std(signal))
    # Pseudo-frequency assuming the signal's index roughly maps to time.
    # 0.25 Hz drift over the full signal = roughly 1/4 cycle visible.
    drift = 0.5 * sigma * np.sin(np.linspace(0, 0.5 * np.pi, n))
    return signal + drift


def probe_st_segment_shift(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a constant offset to the second half — models slow ST-segment
    elevation (one of the ECG markers of myocardial ischemia)."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    sigma = float(np.std(signal))
    s[mid:] = s[mid:] + 0.5 * sigma
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Noise probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_powerline_60hz(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a 60 Hz sinusoid at ~30% of signal std (the typical
    mains-interference level after a faulty notch filter). Assumes
    sampling rate around 250-500 Hz so 60 Hz fits in the band."""
    n = len(signal)
    sigma = float(np.std(signal))
    # Use index/n as a proxy for time; 60 cycles per signal = clear interference
    interference = 0.3 * sigma * np.sin(np.linspace(0, 2 * np.pi * 60, n))
    return signal + interference


def probe_emg_contamination(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a short burst of high-frequency noise — models EMG
    contamination from a muscle twitch overlapping ECG/EEG capture."""
    s = signal.copy()
    n = len(s)
    burst_start = int(rng.integers(n // 3, 2 * n // 3))
    burst_len = max(5, n // 30)
    burst_end = min(n, burst_start + burst_len)
    sigma = float(np.std(signal))
    s[burst_start:burst_end] = s[burst_start:burst_end] + 2.0 * sigma * rng.standard_normal(burst_end - burst_start)
    return s


def probe_seizure_burst(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject high-amplitude rhythmic spike-wave activity — models the
    ictal phase of a seizure as it would appear on EEG. Replaces the
    middle 25% of the signal."""
    s = signal.copy()
    n = len(s)
    sz_start = 3 * n // 8
    sz_end = 5 * n // 8
    sigma = float(np.std(signal))
    span = sz_end - sz_start
    # 3 Hz spike-wave is the classic absence-seizure signature
    spike_wave = 4.0 * sigma * np.sin(np.linspace(0, 2 * np.pi * (span / 80.0), span))
    # Sharpen to look like spikes via clipping
    spike_wave = np.sign(spike_wave) * np.abs(spike_wave) ** 0.5 * 4.0 * sigma
    s[sz_start:sz_end] = spike_wave
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Benign controls — what "normal patient on a working sensor" looks like
# ─────────────────────────────────────────────────────────────────────────────


def control_within_hr_variability(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a tiny natural beat-to-beat amplitude jitter (5% of std).
    Healthy hearts vary slightly; this should NOT trigger an alarm."""
    sigma = float(np.std(signal))
    return signal + 0.05 * sigma * rng.standard_normal(len(signal))


def control_minor_breathing_artifact(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a subtle ~0.3 Hz baseline wobble at 10% of std. The kind of
    thing that's always present on a chest-lead ECG and is filtered out
    upstream — should NOT register as a regime change."""
    n = len(signal)
    sigma = float(np.std(signal))
    wobble = 0.1 * sigma * np.sin(np.linspace(0, 2 * np.pi * 0.3 * (n / 250.0), n))
    return signal + wobble


def control_single_motion_glitch(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One sub-1σ blip — patient touched the lead momentarily. Common
    enough that it shouldn't fire a regime alarm."""
    s = signal.copy()
    n = len(s)
    pos = int(rng.integers(n // 4, 3 * n // 4))
    sigma = float(np.std(signal))
    s[pos] = s[pos] + 0.7 * sigma
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Public registries
# ─────────────────────────────────────────────────────────────────────────────


BIOMEDICAL_PROBES: Dict[str, Probe] = {
    "tachycardia":           probe_tachycardia,
    "bradycardia":           probe_bradycardia,
    "arrhythmia":            probe_arrhythmia,
    "pvc_injection":         probe_pvc_injection,
    "lead_disconnect":       probe_lead_disconnect,
    "amplitude_attenuation": probe_amplitude_attenuation,
    "baseline_wander":       probe_baseline_wander,
    "st_segment_shift":      probe_st_segment_shift,
    "powerline_60hz":        probe_powerline_60hz,
    "emg_contamination":     probe_emg_contamination,
    "seizure_burst":         probe_seizure_burst,
}


BIOMEDICAL_BENIGN_CONTROLS: Dict[str, Probe] = {
    "within_hr_variability":     control_within_hr_variability,
    "minor_breathing_artifact":  control_minor_breathing_artifact,
    "single_motion_glitch":      control_single_motion_glitch,
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic ECG signal — for the demo and for tests
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_ecg(
    duration_seconds: float = 10.0,
    sampling_rate: float = 250.0,
    hr_bpm: float = 70.0,
    noise_level: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic ECG-like signal.

    Models each beat as the superposition of:
      * R-wave: sharp Gaussian (sigma ~ 5 ms).
      * T-wave: broader Gaussian (sigma ~ 40 ms) offset by 300 ms.
      * P-wave: small Gaussian (sigma ~ 15 ms) offset by -150 ms.
    Plus a small additive white-noise component.

    NOT a clinically-realistic ECG. Adequate for exercising the engine
    and the probe library on biomedical-shaped signals.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_seconds * sampling_rate)
    t = np.arange(n) / sampling_rate
    period = 60.0 / hr_bpm
    signal = np.zeros(n)
    beat_times = np.arange(period, t[-1], period)
    for bt in beat_times:
        signal += 1.0 * np.exp(-((t - bt) ** 2) / (2 * 0.005 ** 2))            # R-wave
        signal += 0.3 * np.exp(-((t - bt - 0.3) ** 2) / (2 * 0.04 ** 2))       # T-wave
        signal += 0.15 * np.exp(-((t - bt + 0.15) ** 2) / (2 * 0.015 ** 2))    # P-wave
    signal += noise_level * rng.standard_normal(n)
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def _demo():
    """Exercise the biomedical probes against a synthetic ECG signal.
    Each test case is the baseline ECG transformed by a known probe;
    the diagnosis should name that probe (or a close mirror)."""
    import os
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long")
    os.environ.setdefault("MASTER_API_KEY", "ai_master_test")

    from recipes.auto_diagnose import auto_diagnose

    rng = np.random.default_rng(0)
    baseline = synthetic_ecg(
        duration_seconds=10.0, sampling_rate=250.0, hr_bpm=70.0,
        noise_level=0.02, seed=0,
    )

    test_cases = [
        ("tachycardia onset",   probe_tachycardia(baseline, rng)),
        ("PVC injected",        probe_pvc_injection(baseline, rng)),
        ("lead disconnected",   probe_lead_disconnect(baseline, rng)),
        ("baseline wander",     probe_baseline_wander(baseline, rng)),
        ("just breathing",      control_minor_breathing_artifact(baseline, rng)),
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
                "channels": channels,
                "sampling_rate": sampling_rate,
                "domain": domain,
                "use_multiscale": use_multiscale,
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
            sampling_rate=250.0,
            domain="biomedical",
            probes=BIOMEDICAL_PROBES,
            benign_controls=BIOMEDICAL_BENIGN_CONTROLS,
        )
        print(f"\nground truth: {label}")
        print(f"  diagnosis  : {out['diagnosis']:25}  confidence={out['confidence']:.3f}")
        print(f"  benign sim : {out['benign_similarity']:.3f}")
        print("  top-3 probes:")
        for name, score in out["ranked"][:3]:
            print(f"    {name:25}  {score:.3f}")


if __name__ == "__main__":
    _demo()
