"""
recipes/probes_industrial.py — Industrial / vibration / SCADA probe library.

Third vertical, alongside finance and biomedical. Industrial monitoring
is one of the most natural commercial fits for structural detection:
predictive maintenance has a clear ROI (avoid unscheduled downtime),
data is plentiful and synchronous (every spinning machine generates
a vibration time series), and the regime changes are physically
interpretable (bearing wear, imbalance, cavitation, looseness).

Domain focus
------------

Probes here target rotating-machinery vibration captured from
accelerometers (typical sample rates 1-25 kHz, typical signal shape
= sum of harmonics of the rotational frequency + noise + occasional
impulses). The same probes apply to:

  * Pump cavitation monitoring.
  * HVAC blower diagnostics.
  * Compressor health.
  * Wind-turbine drivetrain monitoring.
  * General SCADA process variables (with the slow-drift probes).

Probes calibrated to industrial events
--------------------------------------

  * motor_imbalance       — fundamental rotational-freq amplitude grows
                            (mass imbalance on shaft).
  * bearing_wear_emergence — non-harmonic frequencies appear (early
                              bearing-defect signature).
  * harmonic_emergence    — 2× or 3× harmonic amplifies (looseness or
                            misalignment).
  * cavitation_burst      — broadband high-frequency noise burst (pump
                            cavitation onset).
  * shock_event           — sudden large impulse (impact / object strike).
  * gradual_amplitude_growth — slow amplitude trend (progressive wear).
  * process_drift         — slow mean shift (calibration drift / process
                            change).
  * sensor_saturation     — clipping at +/-Vmax (over-range fault).
  * sensor_dropout        — abrupt amplitude collapse (cable / sensor
                            failure).
  * speed_change          — frequency shift (load change, VFD command).
  * looseness_spikes      — random impulse train (loose mounting / chips).

Benign controls — what "machine running normally" looks like:

  * within_load_variation     — small periodic load swings.
  * minor_speed_variation     — VFD-controlled speed wander (~1%).
  * single_passing_object     — one spike from a passing person/cart.

Important caveat
----------------

Diagnoses here are STRUCTURAL FINGERPRINT MATCHES, not engineering
verdicts. A diagnosis of "bearing_wear_emergence" does NOT mean
"replace the bearing." It means "the structural fingerprint of this
capture vs the known-good baseline matches the fingerprint of a
synthetic bearing-wear injection." Real predictive-maintenance use
should pair this with vibration-spectrum analysis, kurtosis tracking,
and a maintenance engineer.

The intended consumer is monitoring tooling: anomaly gating (which
captures need a closer look?), trend dashboards, retrospective
event labeling, A/B comparison after maintenance, and research.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from recipes.auto_diagnose import Probe


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_motor_imbalance(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a synchronous (1× rotation) sinusoid in the second half.
    Models mass imbalance — the most common rotating-machinery fault."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    sigma = float(np.std(signal))
    # Assume the baseline already has rotation; we add a strong 1× component.
    # Use a frequency proportional to a fraction of the index range —
    # comparable to the dominant freq in the synthetic baseline below.
    freq_cycles = max(20, n // 100)  # ~20 cycles in the second half
    t = np.arange(n - mid)
    s[mid:] = s[mid:] + 1.5 * sigma * np.sin(2 * np.pi * freq_cycles * t / (n - mid))
    return s


def probe_bearing_wear_emergence(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject high-frequency content (~5× rotational) into the second half.
    Bearing-defect signatures (BPFO, BPFI, BSF) are non-integer multiples
    of rotation that ride on top of the carrier. We approximate with a
    5× pseudo-frequency at advanced-defect amplitude — early-stage
    bearing wear is below this and is best detected via long-term
    trending, not single-capture diagnosis."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    sigma = float(np.std(signal))
    freq_cycles = max(80, n // 25)
    t = np.arange(n - mid)
    s[mid:] = s[mid:] + 1.2 * sigma * np.sin(2 * np.pi * freq_cycles * t / (n - mid))
    # Modulate slightly to break perfect periodicity (real defects do this)
    s[mid:] = s[mid:] + 0.25 * sigma * rng.standard_normal(n - mid)
    return s


def probe_harmonic_emergence(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Amplify the 2× harmonic in the second half. Models looseness or
    shaft misalignment, both of which preferentially raise even harmonics."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    sigma = float(np.std(signal))
    freq_cycles = max(40, n // 50)  # 2× the assumed baseline rotation
    t = np.arange(n - mid)
    s[mid:] = s[mid:] + 0.8 * sigma * np.sin(2 * np.pi * freq_cycles * t / (n - mid))
    return s


def probe_speed_change(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compress the time axis of the second half by 1.3× — models a
    speed increase from a VFD command or load change."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    second = s[mid:]
    new_len = int(len(second) / 1.3)
    if new_len < 10:
        return s
    idx = np.linspace(0, len(second) - 1, new_len)
    compressed = np.interp(idx, np.arange(len(second)), second)
    tiled = np.tile(compressed, (len(second) // new_len) + 2)[:len(second)]
    s[mid:] = tiled
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Single-event probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_cavitation_burst(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a 1-second broadband high-frequency noise burst — the
    classic pump-cavitation signature on accelerometer data."""
    s = signal.copy()
    n = len(s)
    burst_start = int(rng.integers(n // 3, 2 * n // 3))
    burst_len = max(30, n // 20)
    burst_end = min(n, burst_start + burst_len)
    sigma = float(np.std(signal))
    s[burst_start:burst_end] = s[burst_start:burst_end] + 3.0 * sigma * rng.standard_normal(burst_end - burst_start)
    return s


def probe_shock_event(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a single ~6σ impulse — models an impact event (object
    strike, dropped tool, sudden load engagement)."""
    s = signal.copy()
    n = len(s)
    pos = int(rng.integers(n // 4, 3 * n // 4))
    sigma = float(np.std(signal))
    width = max(3, n // 500)
    s[pos:pos + width] = s[pos:pos + width] + 6.0 * sigma
    return s


def probe_looseness_spikes(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a train of random impulses — models loose mounting where
    the machine periodically chatters against its housing."""
    s = signal.copy()
    n = len(s)
    sigma = float(np.std(signal))
    # 5-15 random spikes scattered through the second half
    second_half = np.arange(n // 2, n)
    n_spikes = int(rng.integers(5, 16))
    positions = rng.choice(second_half, size=n_spikes, replace=False)
    for p in positions:
        s[p] = s[p] + 4.0 * sigma * (1.0 if rng.random() > 0.5 else -1.0)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Slow / drift probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_gradual_amplitude_growth(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a 1.0 → 2.0× amplitude ramp. Models progressive wear or
    fault evolution where vibration grows slowly."""
    n = len(signal)
    growth = np.linspace(1.0, 2.0, n)
    return signal * growth


def probe_process_drift(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a slow linear drift in the mean. Models calibration drift
    or a process variable creeping out of spec."""
    n = len(signal)
    sigma = float(np.std(signal))
    drift = np.linspace(0.0, 1.5 * sigma, n)
    return signal + drift


# ─────────────────────────────────────────────────────────────────────────────
# Sensor / capture probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_sensor_saturation(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Clip the second half at ±2σ — models a sensor / ADC operating
    over its dynamic range (saturation fault)."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    sigma = float(np.std(signal))
    s[mid:] = np.clip(s[mid:], -2.0 * sigma, 2.0 * sigma)
    return s


def probe_sensor_dropout(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Abrupt amplitude collapse to near-zero in the last quarter —
    models a cable or sensor failure mid-capture."""
    s = signal.copy()
    n = len(s)
    cutoff = 3 * n // 4
    s[cutoff:] = 0.05 * float(np.std(signal)) * rng.standard_normal(n - cutoff)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Benign controls — "machine running normally"
# ─────────────────────────────────────────────────────────────────────────────


def control_within_load_variation(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a slow ~0.5 Hz amplitude modulation — natural load variation
    on a steady-running machine. Should NOT trigger an alarm."""
    n = len(signal)
    mod = 1.0 + 0.05 * np.sin(np.linspace(0, 4 * np.pi, n))
    return signal * mod


def control_minor_speed_variation(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a tiny ±1% phase jitter — VFD speed dithering. Real machines
    never run at perfectly constant speed."""
    n = len(signal)
    if n < 20:
        return signal.copy()
    jitter = 0.01 * n * rng.standard_normal(n)
    idx = np.clip(np.arange(n) + jitter, 0, n - 1)
    return np.interp(idx, np.arange(n), signal)


def control_single_passing_object(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One sub-2σ blip — a person walking past, a forklift in the next
    aisle. Common enough that it shouldn't fire a fault alarm."""
    s = signal.copy()
    n = len(s)
    pos = int(rng.integers(n // 4, 3 * n // 4))
    sigma = float(np.std(signal))
    s[pos] = s[pos] + 1.5 * sigma
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Public registries
# ─────────────────────────────────────────────────────────────────────────────


INDUSTRIAL_PROBES: Dict[str, Probe] = {
    "motor_imbalance":           probe_motor_imbalance,
    "bearing_wear_emergence":    probe_bearing_wear_emergence,
    "harmonic_emergence":        probe_harmonic_emergence,
    "speed_change":              probe_speed_change,
    "cavitation_burst":          probe_cavitation_burst,
    "shock_event":               probe_shock_event,
    "looseness_spikes":          probe_looseness_spikes,
    "gradual_amplitude_growth":  probe_gradual_amplitude_growth,
    "process_drift":             probe_process_drift,
    "sensor_saturation":         probe_sensor_saturation,
    "sensor_dropout":            probe_sensor_dropout,
}


INDUSTRIAL_BENIGN_CONTROLS: Dict[str, Probe] = {
    "within_load_variation":  control_within_load_variation,
    "minor_speed_variation":  control_minor_speed_variation,
    "single_passing_object":  control_single_passing_object,
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic vibration signal — for the demo and tests
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_vibration(
    duration_seconds: float = 10.0,
    sampling_rate: float = 5000.0,
    rotation_hz: float = 30.0,
    n_harmonics: int = 4,
    noise_level: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic rotating-machinery vibration signal.

    Models the signal as a sum of harmonics of `rotation_hz` with
    decreasing amplitude, plus white noise. Adequate for exercising
    the engine and probe library on industrial-shaped signals.

    Default params model a ~1800 RPM motor (30 Hz) sampled at 5 kHz —
    typical for a vibration-monitoring CMS.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_seconds * sampling_rate)
    t = np.arange(n) / sampling_rate
    signal = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        amp = 1.0 / k  # decreasing amplitude per harmonic
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * k * rotation_hz * t + phase)
    signal += noise_level * rng.standard_normal(n)
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def _demo():
    """Apply each of three known transformations to a baseline vibration
    signal and verify auto_diagnose names the right probe."""
    import os
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long")
    os.environ.setdefault("MASTER_API_KEY", "ai_master_test")

    from recipes.auto_diagnose import auto_diagnose

    rng = np.random.default_rng(0)
    baseline = synthetic_vibration(
        duration_seconds=2.0, sampling_rate=5000.0,
        rotation_hz=30.0, n_harmonics=4, noise_level=0.1, seed=0,
    )

    test_cases = [
        ("bearing wear emerging",   probe_bearing_wear_emergence(baseline, rng)),
        ("shock event",             probe_shock_event(baseline, rng)),
        ("sensor dropout",          probe_sensor_dropout(baseline, rng)),
        ("gradual wear",            probe_gradual_amplitude_growth(baseline, rng)),
        ("just load variation",    control_within_load_variation(baseline, rng)),
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
            sampling_rate=5000.0,
            domain="sensors",  # alphainfo's canonical domain for industrial
            probes=INDUSTRIAL_PROBES,
            benign_controls=INDUSTRIAL_BENIGN_CONTROLS,
        )
        print(f"\nground truth: {label}")
        print(f"  diagnosis  : {out['diagnosis']:30}  confidence={out['confidence']:.3f}")
        print(f"  benign sim : {out['benign_similarity']:.3f}")
        print("  top-3 probes:")
        for name, score in out["ranked"][:3]:
            print(f"    {name:30}  {score:.3f}")


if __name__ == "__main__":
    _demo()
