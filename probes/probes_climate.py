"""
recipes/probes_climate.py — Climate / environmental probe library.

Eighth vertical. Climate signals (temperature, precipitation, sea
level, atmospheric pressure, river discharge, ice extent) carry
structural fingerprints distinct from finance or industrial: long
seasonal cycles, low-frequency drifts, and slow regime shifts.

Domain focus
------------

Probes here target signals over months to decades:

  * Daily / monthly mean temperature.
  * Precipitation totals.
  * Sea-surface temperature anomalies.
  * Sea-level rise series.
  * River discharge.
  * Atmospheric CO₂ at a station.
  * Ice extent / albedo proxies.

Probes calibrated to climate events
-----------------------------------

  * regime_shift_warming         — slow trend up.
  * regime_shift_cooling         — slow trend down (post-volcanic
                                   forcing, regional cooling pattern).
  * trend_acceleration           — second derivative grows (super-linear
                                   warming).
  * variance_increase            — climate variability widens (more
                                   weather "swings").
  * extreme_events_burst         — cluster of high-amplitude events
                                   (heatwave / cold-snap window).
  * seasonal_amplitude_compress  — milder seasons (high-latitude warming).
  * seasonal_amplitude_amplify   — sharper seasons (continentality
                                   intensification).
  * seasonal_phase_shift         — peaks earlier/later (phenology shift).
  * tipping_point                — abrupt level change after slow
                                   approach (regime hysteresis).
  * low_freq_drift               — multidecadal oscillation emerges.
  * oscillation_emergence        — ENSO-like cycle appears where there
                                   was none.

Benign controls — what "ordinary climate variability" looks like:

  * weather_noise              — daily fluctuations within envelope.
  * enso_normal_phase          — typical El Niño / La Niña cycle.
  * volcanic_minor_aerosol     — small dip and recovery.

Important caveat
----------------

Climate detection is a domain with strong scientific consensus on
some signals and active debate on others. These probes are NOT a
substitute for IPCC-quality attribution analysis — they are a fast
structural-fingerprint layer for:

  * Anomaly gating across many station / grid-cell time series.
  * Retrospective regime-shift discovery for regional studies.
  * Comparison of CMIP model output vs observations on structural
    grounds.
  * Educational and exploratory work.

Pair with proper trend-attribution methods (block bootstrap, change-
point statistics, or Mann-Kendall tests) before drawing conclusions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from recipes.auto_diagnose import Probe


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic baselines
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_temperature_anomaly(
    duration_days: int = 1825,        # 5 years
    mean_temp: float = 15.0,
    seasonal_amp: float = 8.0,
    noise_level: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
    """Build a deterministic daily-mean-temperature baseline with strong
    annual seasonality and weather-scale noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(duration_days, dtype=float)
    seasonal = seasonal_amp * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
    weather = noise_level * rng.standard_normal(duration_days)
    return mean_temp + seasonal + weather


def synthetic_sst_anomaly(
    duration_months: int = 240,        # 20 years
    drift_amp: float = 0.05,
    noise_level: float = 0.18,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic sea-surface-temperature anomaly time series in °C
    relative to a 30-year reference, monthly resolution."""
    rng = np.random.default_rng(seed)
    t = np.arange(duration_months, dtype=float)
    enso_like = 0.3 * np.sin(2 * np.pi * t / 42.0)  # ~3.5-year period
    drift = drift_amp * t / duration_months
    noise = noise_level * rng.standard_normal(duration_months)
    return drift + enso_like + noise


# ─────────────────────────────────────────────────────────────────────────────
# Trend / drift probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_regime_shift_warming(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a slow linear warming trend over the full series — global-
    warming-like fingerprint at decadal scale."""
    s = signal.copy()
    n = len(s)
    sigma = max(float(np.std(signal)), 1e-6)
    trend = np.linspace(0.0, 1.5 * sigma, n)
    return s + trend


def probe_regime_shift_cooling(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slow cooling trend — regional cooling, prolonged volcanic
    forcing, or a multidecadal cool phase."""
    s = signal.copy()
    n = len(s)
    sigma = max(float(np.std(signal)), 1e-6)
    trend = np.linspace(0.0, -1.5 * sigma, n)
    return s + trend


def probe_trend_acceleration(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Quadratic trend — second derivative is positive. Models a
    departure from linear warming into a super-linear regime."""
    s = signal.copy()
    n = len(s)
    sigma = max(float(np.std(signal)), 1e-6)
    t = np.linspace(0.0, 1.0, n)
    accel = 2.0 * sigma * (t ** 2)
    return s + accel


def probe_low_freq_drift(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a single multidecadal-scale oscillation across the series —
    a slow up-then-down (or down-then-up) drift covering one full
    half-cycle."""
    s = signal.copy()
    n = len(s)
    sigma = max(float(np.std(signal)), 1e-6)
    t = np.arange(n)
    drift = 1.2 * sigma * np.sin(np.pi * t / n)
    return s + drift


# ─────────────────────────────────────────────────────────────────────────────
# Variance / extremes probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_variance_increase(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiply the residuals (signal minus its long-term smooth) by 2×
    in the second half. Climate variability widens without changing
    the mean."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    win = max(7, n // 50)
    smooth = np.convolve(s, np.ones(win) / win, mode="same")
    resid = s - smooth
    s[mid:] = smooth[mid:] + 2.0 * resid[mid:]
    return s


def probe_extreme_events_burst(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a cluster of high-amplitude events (heatwave / cold-snap
    window) in the second half — short series of large positive
    anomalies."""
    s = signal.copy()
    n = len(s)
    sigma = max(float(np.std(signal)), 1e-6)
    cluster_start = int(n * 0.62)
    cluster_len = max(5, int(n * 0.04))
    cluster_end = min(cluster_start + cluster_len, n)
    extras = 3.5 * sigma * np.abs(rng.standard_normal(cluster_end - cluster_start))
    s[cluster_start:cluster_end] = s[cluster_start:cluster_end] + extras
    return s


def probe_tipping_point(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """After a slow approach, the second-half mean steps up by 1.2σ —
    abrupt regime hysteresis after critical slowing-down. Distinct
    from a smooth trend because the change is concentrated."""
    s = signal.copy()
    n = len(s)
    sigma = max(float(np.std(signal)), 1e-6)
    mid = int(n * 0.55)
    s[mid:] = s[mid:] + 1.2 * sigma
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Seasonality probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_seasonal_amplitude_compress(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiply the seasonal anomaly by 0.65 in the second half — high-
    latitude warming compresses winter-summer range."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    mu_hf = float(np.mean(s[mid:]))
    s[mid:] = mu_hf + (s[mid:] - mu_hf) * 0.65
    return s


def probe_seasonal_amplitude_amplify(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiply the seasonal anomaly by 1.5 in the second half — sharper
    seasons (continentality intensification, weakening jet stream)."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    mu_hf = float(np.mean(s[mid:]))
    s[mid:] = mu_hf + (s[mid:] - mu_hf) * 1.5
    return s


def probe_seasonal_phase_shift(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-shift the second half by ~10% of the seasonal period
    (≈37 days for daily data), peaks moving earlier/later. Phenology /
    spring-onset signal."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    shift = 37
    if n - mid > shift:
        s[mid:] = np.concatenate([s[mid + shift:], s[mid:mid + shift]])
    return s


def probe_oscillation_emergence(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a low-frequency oscillation (ENSO-like, ~3.5-year period
    if daily data) that emerges in the second half."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    sigma = max(float(np.std(signal)), 1e-6)
    t = np.arange(n - mid)
    osc = 0.6 * sigma * np.sin(2 * np.pi * t / max(120.0, (n - mid) / 4))
    s[mid:] = s[mid:] + osc
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Benign controls
# ─────────────────────────────────────────────────────────────────────────────


def control_weather_noise(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add small day-to-day noise (~10% of std) — ordinary weather
    fluctuations within the climate envelope."""
    sigma = float(np.std(signal))
    return signal + 0.1 * sigma * rng.standard_normal(len(signal))


def control_enso_normal_phase(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A typical ENSO cycle of moderate amplitude — slowly varies the
    series by ~0.3σ. Within historical climate variability."""
    n = len(signal)
    sigma = max(float(np.std(signal)), 1e-6)
    t = np.arange(n)
    cycle = 0.3 * sigma * np.sin(2 * np.pi * t / max(180.0, n / 3))
    return signal + cycle


def control_volcanic_minor_aerosol(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small dip then recovery (~0.4σ over ~10% of series) — a minor
    volcanic-aerosol forcing event that decays. Localised but
    structurally benign."""
    s = signal.copy()
    n = len(s)
    sigma = max(float(np.std(signal)), 1e-6)
    start = int(n * 0.4)
    win = max(5, int(n * 0.10))
    end = min(start + win, n)
    t = np.linspace(0.0, np.pi, end - start)
    dip = -0.4 * sigma * np.sin(t)
    s[start:end] = s[start:end] + dip
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Public registry
# ─────────────────────────────────────────────────────────────────────────────


CLIMATE_PROBES: Dict[str, Probe] = {
    "regime_shift_warming":         probe_regime_shift_warming,
    "regime_shift_cooling":         probe_regime_shift_cooling,
    "trend_acceleration":           probe_trend_acceleration,
    "low_freq_drift":               probe_low_freq_drift,
    "variance_increase":            probe_variance_increase,
    "extreme_events_burst":         probe_extreme_events_burst,
    "tipping_point":                probe_tipping_point,
    "seasonal_amplitude_compress":  probe_seasonal_amplitude_compress,
    "seasonal_amplitude_amplify":   probe_seasonal_amplitude_amplify,
    "seasonal_phase_shift":         probe_seasonal_phase_shift,
    "oscillation_emergence":        probe_oscillation_emergence,
}


CLIMATE_BENIGN_CONTROLS: Dict[str, Probe] = {
    "weather_noise":           control_weather_noise,
    "enso_normal_phase":       control_enso_normal_phase,
    "volcanic_minor_aerosol":  control_volcanic_minor_aerosol,
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
    baseline = synthetic_temperature_anomaly(seed=0)

    test_cases = [
        ("regime_shift_warming",   probe_regime_shift_warming(baseline, rng)),
        ("variance_increase",      probe_variance_increase(baseline, rng)),
        ("tipping_point",          probe_tipping_point(baseline, rng)),
        ("benign weather noise",   control_weather_noise(baseline, rng)),
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
            probes=CLIMATE_PROBES,
            benign_controls=CLIMATE_BENIGN_CONTROLS,
        )
        print(f"\nground truth: {label}")
        print(f"  diagnosis: {out['diagnosis']:25}  confidence={out['confidence']:.3f}")
        print(f"  benign sim: {out['benign_similarity']:.3f}")
        print("  top-3 probes:")
        for name, score in out["ranked"][:3]:
            print(f"    {name:25}  {score:.3f}")


if __name__ == "__main__":
    _demo()
