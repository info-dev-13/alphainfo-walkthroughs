"""
recipes/probes_logistics.py — Logistics / supply-chain probe library.

Sixth vertical, alongside finance/biomedical/industrial/mlops/security.
Logistics time series come from order systems, fulfillment platforms,
delivery telematics, warehouse WMS, and last-mile dispatchers. Typical
shapes are hourly or daily counts, lead-time distributions, queue
depths, and inventory levels.

Domain focus
------------

These probes target operations-layer signals where structural changes
have direct P&L impact:

  * E-commerce demand series (hourly orders, daily GMV).
  * Fulfillment lead times (order → ship time).
  * Warehouse queue depths / pick rates.
  * Delivery success / on-time rates.
  * Inventory level series (stockouts manifest as flat-zero floors).

The same fingerprints map to ride-hailing demand, food-delivery flow,
and call-center queue depth — wherever a discrete demand process
meets a finite-capacity service process.

Probes calibrated to logistics events
-------------------------------------

  * demand_spike            — sudden burst (Black Friday onset, viral
                              product, channel surge).
  * demand_collapse         — order rate drops (outage, payment fail,
                              competitor undercut).
  * inventory_stockout      — series clips at zero (out-of-stock).
  * lead_time_creep         — gradual fulfillment-time increase.
  * lead_time_burst         — episode of high latencies (carrier delay,
                              warehouse incident).
  * queue_oscillation       — amplitude swings grow (control-loop
                              instability between dispatch tiers).
  * weekly_pattern_break    — usual weekly seasonality flattens
                              (holiday window, system change).
  * weekly_pattern_shift    — peak hour moves (channel mix change).
  * bullwhip_amplification  — variance amplifies downstream
                              (forecast over-reaction).
  * sustained_demand_shift  — new normal level (price change, TV ad).

Benign controls — "operations are running normally":

  * routine_promotion          — +5% lift, no structural change.
  * staff_shift_handover       — small periodic noise.
  * weather_minor_perturbation — tiny wobble in delivery times.

Important caveat
----------------

These probes operate on the SHAPE of an operational time series, not
on per-order metadata. They are an alarm-gating layer, not a root-cause
analyser — pair them with order-level logs and cohort analysis when
debugging a specific event.

Use them for:

  * Daily ops anomaly gating ("yesterday's demand pattern looks
    structurally different from the prior 30 days").
  * Post-incident retrospective fingerprinting.
  * Channel A/B comparison.
  * Capacity-planning regime detection (when did demand structure
    actually shift?).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from recipes.auto_diagnose import Probe


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic baseline — hourly orders for a steady e-commerce shop
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_orders_hourly(
    duration_hours: int = 720,        # 30 days
    base_rate: float = 80.0,
    weekly_amp: float = 0.20,
    daily_amp: float = 0.45,
    noise_level: float = 0.08,
    seed: int = 0,
) -> np.ndarray:
    """Build a deterministic hourly-orders baseline.

    Components:
      * baseline rate (orders/hour)
      * 7-day weekly cycle (weekdays > weekend, light dip Mon)
      * 24-hour daily cycle (peak around 8pm local)
      * Poisson-like multiplicative noise (gaussian approx)

    Returns a non-negative float array of length ``duration_hours``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(duration_hours, dtype=float)

    # Daily cycle — peak ~20:00, trough ~04:00
    hour_of_day = t % 24
    daily = 1.0 + daily_amp * np.cos(2 * np.pi * (hour_of_day - 20) / 24)

    # Weekly cycle — Sat/Sun a bit lower, Mon dip
    day_of_week = (t // 24) % 7
    weekly_pat = np.array([0.95, 0.98, 1.00, 1.02, 1.04, 0.92, 0.88])
    weekly = 1.0 + weekly_amp * (weekly_pat[day_of_week.astype(int)] - 1.0)

    series = base_rate * daily * weekly
    series = series * (1.0 + noise_level * rng.standard_normal(duration_hours))
    return np.maximum(series, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Demand-shape probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_demand_spike(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a sharp 4× demand burst spanning ~4% of the series in the
    second half. Models Black Friday onset, viral product launch, or
    a channel-mix surge. Distinct from ``lead_time_burst`` because it
    affects volume not latency."""
    s = signal.copy()
    n = len(s)
    burst_start = int(n * 0.62)
    burst_len = max(4, int(n * 0.04))
    burst_end = min(burst_start + burst_len, n)
    s[burst_start:burst_end] = s[burst_start:burst_end] * 4.0
    return s


def probe_demand_collapse(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order rate drops to ~25% of baseline in the second half. Models
    payment outage, channel ban, or critical competitor event."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    s[mid:] = s[mid:] * 0.25
    return s


def probe_inventory_stockout(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hard-clip the second half at zero for ~10% of samples — the
    structural fingerprint of an item going out-of-stock (orders no
    longer flow on that SKU). Different from collapse because the
    floor is exactly zero, not a fraction of baseline."""
    s = signal.copy()
    n = len(s)
    stockout_start = int(n * 0.6)
    stockout_len = max(4, int(n * 0.10))
    stockout_end = min(stockout_start + stockout_len, n)
    s[stockout_start:stockout_end] = 0.0
    return s


def probe_sustained_demand_shift(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift the second half up by 1.4× — a new normal level. Models
    the structural lift after a TV ad, price drop, or major SEO win.
    Differs from ``demand_spike`` because the elevated level persists."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    s[mid:] = s[mid:] * 1.4
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Lead-time / latency probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_lead_time_creep(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a slow linear ramp on top of the second half — fulfillment
    times creeping up. Models warehouse capacity strain or labour
    shortage. Slow, not bursty."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    base = float(np.mean(np.abs(signal)))
    ramp = np.linspace(0.0, 0.6 * base, n - mid)
    s[mid:] = s[mid:] + ramp
    return s


def probe_lead_time_burst(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject an episode of 3× elevated values for a ~6% window. Models
    a carrier disruption (port strike, weather, cyber-incident at
    a 3PL) — sudden onset, sudden recovery."""
    s = signal.copy()
    n = len(s)
    burst_start = int(n * 0.68)
    burst_len = max(4, int(n * 0.06))
    burst_end = min(burst_start + burst_len, n)
    s[burst_start:burst_end] = s[burst_start:burst_end] * 3.0
    return s


def probe_queue_oscillation(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a low-frequency oscillation that grows in amplitude over the
    second half — signature of a control loop going unstable (under-
    /over-correction between forecasting and capacity tiers)."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    base = float(np.std(signal))
    t = np.arange(n - mid)
    growth = np.linspace(0.3, 1.2, n - mid)
    osc = growth * base * np.sin(2 * np.pi * t / max(20, (n - mid) // 4))
    s[mid:] = s[mid:] + osc
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Pattern-shape probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_weekly_pattern_break(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flatten the weekly cycle in the second half by replacing values
    with their local mean — preserves volume but loses structure."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    win = 24
    for i in range(mid, n, win):
        end = min(i + win, n)
        s[i:end] = float(np.mean(s[i:end]))
    return s


def probe_weekly_pattern_shift(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-shift the daily/weekly cycle by 6 hours in the second half —
    peak hour moves. Models a channel-mix change (mobile vs desktop,
    or geo expansion to a different timezone)."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    shift = 6
    if n - mid > shift:
        s[mid:] = np.concatenate([s[mid + shift:], s[mid:mid + shift]])
    return s


def probe_bullwhip_amplification(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Amplify the variance of the second half by 2.2× while keeping
    the mean. Models forecast over-reaction propagating through a
    multi-tier supply chain (the bullwhip effect)."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    mu = float(np.mean(s[mid:]))
    s[mid:] = mu + (s[mid:] - mu) * 2.2
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Benign controls — what "ops normal" looks like
# ─────────────────────────────────────────────────────────────────────────────


def control_routine_promotion(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """+5% lift on the second half — small marketing pulse. Should not
    trigger a regime alarm."""
    s = signal.copy()
    n = len(s)
    mid = n // 2
    s[mid:] = s[mid:] * 1.05
    return s


def control_staff_shift_handover(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add small periodic noise (~3% of std) — between-shift handover
    micro-pauses that don't change overall structure."""
    sigma = float(np.std(signal))
    return signal + 0.03 * sigma * rng.standard_normal(len(signal))


def control_weather_minor_perturbation(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Tiny multiplicative wobble (~2% of value). Models routine weather
    or seasonality already inside the normal envelope."""
    return signal * (1.0 + 0.02 * rng.standard_normal(len(signal)))


# ─────────────────────────────────────────────────────────────────────────────
# Public registry
# ─────────────────────────────────────────────────────────────────────────────


LOGISTICS_PROBES: Dict[str, Probe] = {
    "demand_spike":              probe_demand_spike,
    "demand_collapse":           probe_demand_collapse,
    "inventory_stockout":        probe_inventory_stockout,
    "sustained_demand_shift":    probe_sustained_demand_shift,
    "lead_time_creep":           probe_lead_time_creep,
    "lead_time_burst":           probe_lead_time_burst,
    "queue_oscillation":         probe_queue_oscillation,
    "weekly_pattern_break":      probe_weekly_pattern_break,
    "weekly_pattern_shift":      probe_weekly_pattern_shift,
    "bullwhip_amplification":    probe_bullwhip_amplification,
}


LOGISTICS_BENIGN_CONTROLS: Dict[str, Probe] = {
    "routine_promotion":          control_routine_promotion,
    "staff_shift_handover":       control_staff_shift_handover,
    "weather_minor_perturbation": control_weather_minor_perturbation,
}


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def _demo():
    """Show that auto_diagnose recovers the right logistics fingerprint
    for several synthetic transformations of a normal hourly-order
    baseline."""
    import os
    os.environ.setdefault(
        "JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long",
    )
    os.environ.setdefault("MASTER_API_KEY", "ai_master_test")
    os.environ.setdefault("SKIP_DB", "1")

    from recipes.auto_diagnose import auto_diagnose

    rng = np.random.default_rng(0)
    baseline = synthetic_orders_hourly(seed=0)

    test_cases = [
        ("demand_spike",         probe_demand_spike(baseline, rng)),
        ("inventory_stockout",   probe_inventory_stockout(baseline, rng)),
        ("queue_oscillation",    probe_queue_oscillation(baseline, rng)),
        ("benign: promotion",    control_routine_promotion(baseline, rng)),
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
            probes=LOGISTICS_PROBES,
            benign_controls=LOGISTICS_BENIGN_CONTROLS,
        )
        print(f"\nground truth: {label}")
        print(f"  diagnosis: {out['diagnosis']:25}  confidence={out['confidence']:.3f}")
        print(f"  benign sim: {out['benign_similarity']:.3f}")
        print("  top-3 probes:")
        for name, score in out["ranked"][:3]:
            print(f"    {name:25}  {score:.3f}")


if __name__ == "__main__":
    _demo()
