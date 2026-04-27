"""
recipes/probes_energy_grid.py — Power / energy-grid probe library.

Seventh vertical. Power-system signals share structural similarities
with industrial vibration (rotating-mass dynamics) but introduce
distinct fault families: frequency excursions, voltage events,
harmonic distortion, generation trips, and renewable intermittency.

Domain focus
------------

These probes target signals captured at the substation, feeder, or
asset level:

  * System frequency (50/60 Hz nominal) — sub-second to multi-second.
  * Bus voltage RMS — sag/swell/flicker fingerprints.
  * Active load profiles — minute-scale demand curves.
  * Generation output — solar/wind farm aggregate.
  * Tie-line flow — interconnection schedules.
  * Power-quality (THD, harmonic spectra) — feeder health.

Calibrated probes
-----------------

  * frequency_excursion       — sustained frequency offset (UFLS / OFLS
                                threshold approach).
  * frequency_oscillation     — undamped inter-area oscillation
                                (low-inertia regime symptom).
  * voltage_sag               — short voltage dip (fault on adjacent feeder).
  * voltage_swell             — short voltage rise (load rejection).
  * harmonic_distortion       — 5th/7th harmonic emerges (large
                                non-linear load energised).
  * load_spike                — sudden demand jump (industrial start-up).
  * load_drop                 — sudden demand collapse (load shedding).
  * solar_intermittency       — fast PV cloud-edge oscillation.
  * unit_trip                 — abrupt generation loss (governor /
                                breaker action).
  * peak_shift                — peak demand timing moves
                                (DR programme, behavioural change).
  * inertia_loss              — frequency response to disturbance
                                degrades (synthetic-inertia loss,
                                renewable mix increase).

Benign controls — what "normal grid operation" looks like:

  * temperature_demand_drift     — slow seasonal load creep.
  * dispatch_following           — gentle ramp around setpoint.
  * weekend_baseload_difference  — different but stable.

Important caveat
----------------

A diagnosis here is a structural-fingerprint match, not a protection
relay. Use these alongside RMS thresholds, breaker telemetry, and
DFR/PMU streams. The intended consumers are:

  * SCADA anomaly gating (which captures need an engineer's eye?).
  * Post-event forensics (compress 1 hour of grid telemetry into
    one paragraph).
  * Asset-health regression detection.
  * Renewable-integration regime analysis.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from recipes.auto_diagnose import Probe


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic baselines
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_grid_frequency(
    duration_seconds: float = 60.0,
    sampling_rate: float = 50.0,
    nominal_hz: float = 60.0,
    drift_amplitude: float = 0.025,
    noise_level: float = 0.005,
    seed: int = 0,
) -> np.ndarray:
    """Build a deterministic system-frequency baseline.

    Returns an array of frequency-deviation samples around nominal_hz.
    Includes slow drift (load-following) and small white noise.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_seconds * sampling_rate)
    t = np.arange(n) / sampling_rate
    drift = drift_amplitude * np.sin(2 * np.pi * t / 30.0)
    noise = noise_level * rng.standard_normal(n)
    return nominal_hz + drift + noise


def synthetic_load_profile(
    duration_hours: int = 96,
    base_mw: float = 1200.0,
    peak_amp: float = 0.35,
    noise_level: float = 0.03,
    seed: int = 0,
) -> np.ndarray:
    """Build a 96-hour load-profile baseline (15-min resolution would be
    typical; here we use 1-hour for compactness).

    Components: daily peak around 18:00, weekday/weekend variation,
    multiplicative noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(duration_hours, dtype=float)
    hour_of_day = t % 24
    daily = 1.0 + peak_amp * np.exp(
        -((hour_of_day - 18) ** 2) / (2 * 5.0 ** 2)
    )
    day_of_week = (t // 24) % 7
    weekday_factor = np.where(day_of_week < 5, 1.0, 0.92)
    series = base_mw * daily * weekday_factor
    series = series * (1.0 + noise_level * rng.standard_normal(duration_hours))
    return series


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_frequency_excursion(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sustained frequency offset of -0.4 Hz over the second half.
    Models a generation/load imbalance that the AGC has not yet
    recovered — approaches under-frequency load shedding territory."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    s[mid:] = s[mid:] - 0.4
    return s


def probe_frequency_oscillation(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Undamped 0.5 Hz oscillation overlaid on the second half — the
    signature of inter-area or local-mode oscillation in a low-inertia
    system."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    sigma = max(float(np.std(signal)), 1e-6)
    t = np.arange(n - mid) / max(1.0, n - mid)
    osc = 8.0 * sigma * np.sin(2 * np.pi * 0.5 * t * (n - mid) / 50.0)
    s[mid:] = s[mid:] + osc
    return s


def probe_inertia_loss(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Increase the magnitude of fast fluctuations in the second half
    (3× std of high-frequency component). Loss of synthetic inertia
    means small disturbances cause larger frequency deviations."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    sigma_hf = float(np.std(np.diff(signal)))
    s[mid:] = s[mid:] + 3.0 * sigma_hf * rng.standard_normal(n - mid)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Voltage-event probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_voltage_sag(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Drop the second half by ~12% for a short window (~5% of length).
    Models a remote fault that pulls voltage down briefly."""
    s = signal.copy()
    n = len(s)
    sag_start = int(n * 0.6)
    sag_len = max(3, int(n * 0.05))
    sag_end = min(sag_start + sag_len, n)
    s[sag_start:sag_end] = s[sag_start:sag_end] * 0.88
    return s


def probe_voltage_swell(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lift the second half by ~10% for a short window. Models load
    rejection or capacitor switching causing a momentary swell."""
    s = signal.copy()
    n = len(s)
    swell_start = int(n * 0.62)
    swell_len = max(3, int(n * 0.05))
    swell_end = min(swell_start + swell_len, n)
    s[swell_start:swell_end] = s[swell_start:swell_end] * 1.10
    return s


def probe_harmonic_distortion(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a 5th-harmonic-like ripple to the second half. Models a
    large non-linear load (VFD, arc furnace) energising and pushing
    THD above acceptable limits."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    sigma = max(float(np.std(signal)), 1e-6)
    t = np.arange(n - mid)
    ripple = 0.6 * sigma * np.sin(2 * np.pi * 5.0 * t / max(1.0, n - mid))
    s[mid:] = s[mid:] + ripple
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Load / generation probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_load_spike(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sudden 1.6× demand burst over a short window. Industrial load
    energising, large pump start, or cold-load pickup after an outage."""
    s = signal.copy()
    n = len(s)
    spike_start = int(n * 0.65)
    spike_len = max(3, int(n * 0.05))
    spike_end = min(spike_start + spike_len, n)
    s[spike_start:spike_end] = s[spike_start:spike_end] * 1.6
    return s


def probe_load_drop(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Demand drops sharply (~50%) over a short window. Models load
    shedding, large industrial trip, or feeder isolation."""
    s = signal.copy()
    n = len(s)
    drop_start = int(n * 0.65)
    drop_len = max(3, int(n * 0.06))
    drop_end = min(drop_start + drop_len, n)
    s[drop_start:drop_end] = s[drop_start:drop_end] * 0.5
    return s


def probe_unit_trip(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Step-down of ~8% sustained over the full second half. Models
    losing a generator and the resulting reduction in delivered MW
    while neighbours pick up the slack."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    s[mid:] = s[mid:] * 0.92
    return s


def probe_solar_intermittency(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject fast (cloud-edge-scale) 8% oscillations into the second
    half. Renewable-rich feeder seeing intermittent PV output."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    base = max(float(np.mean(np.abs(signal))), 1e-6)
    t = np.arange(n - mid)
    osc = 0.08 * base * np.sin(2 * np.pi * t / max(2.0, (n - mid) / 6))
    s[mid:] = s[mid:] + osc + 0.02 * base * rng.standard_normal(n - mid)
    return s


def probe_peak_shift(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-shift the second half by 3 hours — peak demand moves
    earlier or later. Models DR programme adoption or behavioural
    change (WFH timing)."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    shift = 3
    if n - mid > shift:
        s[mid:] = np.concatenate([s[mid + shift:], s[mid:mid + shift]])
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Benign controls
# ─────────────────────────────────────────────────────────────────────────────


def control_temperature_demand_drift(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slow linear drift of 2% over the full series — gradual seasonal
    load creep. Within normal envelope."""
    n = len(signal)
    base = float(np.mean(np.abs(signal)))
    drift = np.linspace(0.0, 0.02 * base, n)
    return signal + drift


def control_dispatch_following(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gentle 1% oscillation around setpoint — normal AGC / dispatch
    activity."""
    n = len(signal)
    base = float(np.mean(np.abs(signal)))
    t = np.arange(n)
    return signal + 0.01 * base * np.sin(2 * np.pi * t / max(20.0, n / 6))


def control_weekend_baseload_difference(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tiny multiplicative noise (~1%) — the noise floor of normal grid
    measurement. Should not trip a regime alarm."""
    return signal * (1.0 + 0.01 * rng.standard_normal(len(signal)))


# ─────────────────────────────────────────────────────────────────────────────
# Public registry
# ─────────────────────────────────────────────────────────────────────────────


ENERGY_GRID_PROBES: Dict[str, Probe] = {
    "frequency_excursion":     probe_frequency_excursion,
    "frequency_oscillation":   probe_frequency_oscillation,
    "inertia_loss":            probe_inertia_loss,
    "voltage_sag":             probe_voltage_sag,
    "voltage_swell":           probe_voltage_swell,
    "harmonic_distortion":     probe_harmonic_distortion,
    "load_spike":              probe_load_spike,
    "load_drop":               probe_load_drop,
    "unit_trip":               probe_unit_trip,
    "solar_intermittency":     probe_solar_intermittency,
    "peak_shift":              probe_peak_shift,
}


ENERGY_GRID_BENIGN_CONTROLS: Dict[str, Probe] = {
    "temperature_demand_drift":     control_temperature_demand_drift,
    "dispatch_following":           control_dispatch_following,
    "weekend_baseload_difference":  control_weekend_baseload_difference,
}


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def _demo():
    import os
    os.environ.setdefault(
        "JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long",
    )
    os.environ.setdefault("MASTER_API_KEY", "ai_master_test")
    os.environ.setdefault("SKIP_DB", "1")

    from recipes.auto_diagnose import auto_diagnose

    rng = np.random.default_rng(0)
    baseline = synthetic_load_profile(seed=0)

    test_cases = [
        ("load_spike",        probe_load_spike(baseline, rng)),
        ("unit_trip",         probe_unit_trip(baseline, rng)),
        ("voltage_sag",       probe_voltage_sag(baseline, rng)),
        ("benign drift",      control_temperature_demand_drift(baseline, rng)),
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
            payload = {
                "channels": channels, "sampling_rate": sampling_rate,
                "domain": domain, "use_multiscale": use_multiscale,
                "include_semantic": include_semantic,
            }
            if baselines is not None:
                payload["baselines"] = baselines
            r = self.tc.post(
                "/v1/analyze/vector",
                headers={"X-API-Key": self.key}, json=payload,
            )
            r.raise_for_status()
            from types import SimpleNamespace
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
            domain="generic",
            probes=ENERGY_GRID_PROBES,
            benign_controls=ENERGY_GRID_BENIGN_CONTROLS,
        )
        print(f"\nground truth: {label}")
        print(f"  diagnosis: {out['diagnosis']:25}  confidence={out['confidence']:.3f}")
        print(f"  benign sim: {out['benign_similarity']:.3f}")
        print("  top-3 probes:")
        for name, score in out["ranked"][:3]:
            print(f"    {name:25}  {score:.3f}")


if __name__ == "__main__":
    _demo()
