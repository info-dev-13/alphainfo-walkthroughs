"""
recipes/notebooks/encoding_walkthrough.py — Universal encoding discovery.

Most users hit the same wall when starting with alphainfo: "I have this
signal, what encoder do I use?". The 3 verticals (finance, biomedical,
industrial) cover the well-known cases, but the long tail of users —
weather, network telemetry, urban sensors, geophysical signals,
process variables, anything custom — still needs to figure it out.

This walkthrough shows the meta-recipe in action. ``discover_encoding``
inspects an unknown signal, profiles its statistics + structure, and
recommends a ranked encoder set with reasoning. ``auto_encode`` then
applies those recommendations and runs ``feature_ensemble`` end-to-end.

Four scenarios — one from each "unfamiliar" category:

  1. **Weather** — Temperature time series. Periodic + low noise.
  2. **Network** — Packet-rate telemetry with traffic spike.
  3. **Power-grid frequency** — Stationary at 60 Hz with a brief
     under-frequency event.
  4. **Geophone / seismic** — Quiet baseline interrupted by an event burst.

For each, we run discover_encoding on the raw signal, see what it
recommends, then call auto_encode and inspect which channels actually
lit up. The rationale alignment between recommendation and result is
the whole point of this walkthrough.

Usage
-----

    cd alphainfo/
    .venv/bin/python -m recipes.notebooks.encoding_walkthrough

In-process FastAPI TestClient — no API key, no network beyond the
synthetic signal generation.
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
    """Same shim as the other walkthroughs."""

    def __init__(self):
        from fastapi.testclient import TestClient
        from api.app import app
        self.tc = TestClient(app)
        self.key = os.environ["MASTER_API_KEY"]

    def _post(self, path, payload):
        r = self.tc.post(path, headers={"X-API-Key": self.key}, json=payload)
        r.raise_for_status()
        return r.json()

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


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic signal generators — 4 unfamiliar shapes
# ─────────────────────────────────────────────────────────────────────────────


def make_weather_temp(rng):
    """Hourly temperature for a week — strong daily cycle + light noise.
    Compare two weeks: identical pattern vs week with a heatwave anomaly."""
    n = 168  # 7 days * 24 h
    t = np.arange(n) / 24.0
    # Daily cycle + weekly drift
    base = 20 + 8 * np.sin(2 * np.pi * t - np.pi / 2) + 0.5 * t
    baseline = base + 0.5 * rng.standard_normal(n)
    # Heatwave: +5°C in days 4-5
    test = base.copy() + 0.5 * rng.standard_normal(n)
    heatwave_start = 24 * 3
    heatwave_end = 24 * 5
    test[heatwave_start:heatwave_end] += 5.0
    return baseline, test, "weather (hourly temperature, 1 week)"


def make_network_packetrate(rng):
    """Packets/sec from a network monitor — mostly steady, occasional
    bursts. Test signal has a sustained DDoS-like surge."""
    n = 1000
    base = 100 + 5 * rng.standard_normal(n)
    # Add 5-10 random bursts in baseline (normal traffic)
    for _ in range(rng.integers(5, 11)):
        idx = rng.integers(0, n)
        base[idx] += rng.uniform(50, 100)
    baseline = np.clip(base, 0, None)
    # Test: same pattern + a sustained surge in middle 20%
    test = 100 + 5 * rng.standard_normal(n)
    for _ in range(rng.integers(5, 11)):
        idx = rng.integers(0, n)
        test[idx] += rng.uniform(50, 100)
    surge_start, surge_end = 400, 600
    test[surge_start:surge_end] += rng.uniform(80, 150, size=200)
    return baseline, np.clip(test, 0, None), "network (packets/sec, surge in middle)"


def make_power_grid_freq(rng):
    """Power grid frequency at 60 Hz mains. Tightly bounded; brief
    under-frequency event = grid imbalance."""
    n = 600  # 10 minutes at 1 Hz
    baseline = 60.0 + 0.05 * rng.standard_normal(n)
    test = 60.0 + 0.05 * rng.standard_normal(n)
    # Under-frequency event at sample 400-450 (0.5 Hz dip)
    test[400:450] -= 0.5
    return baseline, test, "power-grid frequency (1 Hz cadence)"


def make_seismic_burst(rng):
    """Geophone — quiet baseline interrupted by a P-wave-like burst."""
    n = 5000
    baseline = 0.01 * rng.standard_normal(n)
    test = 0.01 * rng.standard_normal(n)
    # Decaying oscillation in samples 2000-2500
    burst_start, burst_end = 2000, 2500
    tt = np.linspace(0, 4 * np.pi, burst_end - burst_start)
    burst = 1.5 * np.sin(tt) * np.exp(-tt / 4)
    test[burst_start:burst_end] += burst
    return baseline, test, "geophone / seismic (P-wave burst)"


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough
# ─────────────────────────────────────────────────────────────────────────────


def section_header(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def main():
    print()
    print("┌" + "─" * 68 + "┐")
    print("│  alphainfo end-to-end walkthrough — universal encoding discovery │")
    print("│  Meta-recipe for signals outside the 3 verticals                 │")
    print("└" + "─" * 68 + "┘")

    from recipes.encoding_guide import auto_encode, discover_encoding

    rng = np.random.default_rng(42)
    cases = [
        make_weather_temp(rng),
        make_network_packetrate(rng),
        make_power_grid_freq(rng),
        make_seismic_burst(rng),
    ]

    client = _LocalClient()

    for baseline, test, label in cases:
        section_header(label)
        # Step 1: discover (no API call)
        disc = discover_encoding(test, baseline=baseline)
        print("\n  [discover_encoding]")
        print(f"    summary:   {disc['summary']}")
        print(f"    intent:    {disc['suggested_intent']}")
        print(f"    primary:   {[e['encoder'] for e in disc['primary_encoders']]}")
        print(f"    secondary: {[e['encoder'] for e in disc['secondary_encoders']]}")
        if disc["encoders_to_avoid"]:
            print(f"    avoid:     {disc['encoders_to_avoid']}")
        if disc["warnings"]:
            print("    warnings:")
            for w in disc["warnings"]:
                print(f"      ⚠ {w[:100]}{'...' if len(w) > 100 else ''}")

        # Step 2: auto_encode applies the recommendations + runs feature_ensemble
        print("\n  [auto_encode → feature_ensemble]")
        result = auto_encode(
            client, signal=test, baseline=baseline,
            sampling_rate=1.0, domain="generic",
        )
        if result["ensemble"] is None:
            print("    skipped — discover suggested grammar_change (use event_grammar)")
            continue
        out = result["ensemble"]
        print(f"    aggregate score: {out['aggregate_score']:.3f}  ({out['confidence_band']})")
        print("    responsible top-3 channels:")
        for name, score in out["channel_ranking"][:3]:
            print(f"      • {name:14}  {score:.3f}")

        # Validate the recommendation worked
        responsible_names = set(out["responsible"])
        primary_names = {e["encoder"] for e in disc["primary_encoders"]}
        intersection = responsible_names & primary_names
        if intersection:
            print(f"\n    ✓ {len(intersection)}/{len(out['responsible'])} of the responsible channels")
            print(f"      were primary recommendations: {sorted(intersection)}")
        else:
            print("\n    ! none of the responsible channels were primary recommendations.")
            print(f"      responsible: {sorted(responsible_names)}")
            print(f"      recommended: {sorted(primary_names)}")
            print("      → discovery heuristic missed something — review.")

    section_header("Summary")
    print()
    print("  Pipeline: discover_encoding (local diagnosis) → auto_encode")
    print("  (feature_ensemble with the recommended encoders) → see which")
    print("  channels lit up vs which were recommended.")
    print()
    print("  This is meta-recipe — when you have a signal and don't know")
    print("  where to start, this is the entry point. It teaches you the")
    print("  encoding choice by showing reasoning, then applies it for you.")
    print()
    print("  For domain-specific cases that are well-understood (finance,")
    print("  biomedical, industrial), the pre-built probe libraries +")
    print("  intent dispatcher are still the better entry point. discover_encoding")
    print("  is for the long tail.")
    print()


if __name__ == "__main__":
    main()
