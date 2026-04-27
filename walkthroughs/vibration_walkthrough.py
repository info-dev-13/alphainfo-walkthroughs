"""
recipes/notebooks/vibration_walkthrough.py — Predictive maintenance demo.

Third walkthrough — finance, biomedical, industrial. The scenario:
a vibration-monitoring system on a 1800-RPM motor captures 5 seconds
at 5 kHz. Three condition states are evaluated against a known-good
baseline (machine commissioned, running normally):

  1. **Healthy** — same machine 1 month later, just minor load swings.
     Should diagnose as **benign** (no maintenance action needed).
  2. **Bearing wear emerging** — high-frequency content appears, the
     classic early-defect signature. Should diagnose as
     **bearing_wear_emergence** (schedule inspection).
  3. **Shock event** — single ~6σ impulse during the capture, e.g.
     forklift bumped the housing. Should diagnose as **shock_event**
     (review camera, may be benign one-off).

Why predictive maintenance is the strongest commercial fit
----------------------------------------------------------

  * Clear ROI: catching a bearing fault 2 weeks before failure
    avoids unscheduled downtime — typically 10-100× the cost of the
    bearing itself.
  * Synchronous data: every spinning machine produces a continuous
    vibration time series. Always-on, well-sampled, no labelling
    needed for structural detection.
  * Interpretable diagnoses: "bearing wear" / "imbalance" /
    "looseness" are terms maintenance engineers already use.
  * Existing CMS pipelines: condition-monitoring systems are mature;
    structural fingerprinting plugs in upstream of vibration-spectrum
    analysis as a fast pre-screen.

Important caveat
----------------

This is signal-processing tooling, NOT a maintenance verdict. A
diagnosis of "bearing_wear_emergence" means "structural fingerprint
matches the synthetic bearing-wear injection" — NOT "replace the
bearing." Real predictive-maintenance pipelines pair this with
vibration-spectrum analysis (FFT peaks at BPFO/BPFI), kurtosis
trending, and a maintenance engineer's review.

Usage
-----

    cd alphainfo/
    .venv/bin/python -m recipes.notebooks.vibration_walkthrough

In-process FastAPI TestClient. No API key, no external sensor data,
no industrial dependency. The vibration signal is generated
synthetically (recipes.probes_industrial.synthetic_vibration) — adequate
for showing the pipeline shape. Real CMS data drops in directly.
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


# ─────────────────────────────────────────────────────────────────────────────
# In-process SDK shim
# ─────────────────────────────────────────────────────────────────────────────


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
            "signal": list(signal),
            "sampling_rate": sampling_rate,
            "domain": domain,
            "include_semantic": include_semantic,
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
            "sampling_rate": sampling_rate,
            "domain": domain,
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


def section_header(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print()
    print("┌" + "─" * 68 + "┐")
    print("│  alphainfo end-to-end walkthrough — industrial / vibration       │")
    print("│  Diagnose 3 motor-condition snapshots vs commissioning baseline  │")
    print("└" + "─" * 68 + "┘")

    from recipes.probes_industrial import (
        synthetic_vibration,
        probe_bearing_wear_emergence, probe_shock_event,
        control_within_load_variation,
    )
    from recipes.intents import dispatch

    # Commissioning baseline: motor running normally
    fs = 5000.0  # Hz
    duration = 2.0  # 2-second capture window — typical CMS snapshot
    rotation_hz = 30.0  # 1800 RPM

    section_header("1. Commissioning baseline — motor at 1800 RPM, 5 kHz capture")
    baseline = synthetic_vibration(
        duration_seconds=duration, sampling_rate=fs,
        rotation_hz=rotation_hz, n_harmonics=4, noise_level=0.1, seed=0,
    )
    print(f"\n  duration: {duration:.1f} s ({len(baseline)} samples)")
    print(f"  fs:       {fs:g} Hz")
    print(f"  rotation: {rotation_hz:.1f} Hz ({rotation_hz*60:.0f} RPM)")
    print(f"  RMS:      {np.sqrt(np.mean(baseline**2)):.3f}")

    rng = np.random.default_rng(42)

    # Three condition snapshots — all start from the SAME underlying baseline
    # so the only structural difference is the probe (or control) applied.
    # In real CMS data each snapshot would be from the same machine running
    # the same operating point, so this matches reality more closely than
    # using different noise realizations.
    snapshots = [
        ("1 month later — healthy",       control_within_load_variation(
            baseline.copy(), rng,
        ), {"benign"}),
        ("Bearing wear emerging",         probe_bearing_wear_emergence(
            baseline.copy(), rng,
        ), {"bearing_wear_emergence", "harmonic_emergence"}),
        ("Forklift bump (shock event)",   probe_shock_event(
            baseline.copy(), rng,
        ), {"shock_event", "looseness_spikes"}),
    ]

    client = _LocalClient()

    section_header("2. Diagnose each snapshot via intent='what_kind_of_change'")
    print("\n  Each snapshot is compared structurally to the commissioning")
    print("  baseline. The intent layer auto-wires INDUSTRIAL_PROBES because")
    print("  we pass domain='sensors' (alphainfo's canonical industrial domain).\n")

    for label, snapshot, expected in snapshots:
        out = dispatch(
            client, intent="what_kind_of_change",
            signal=snapshot.tolist(), baseline=baseline.tolist(),
            sampling_rate=fs, domain="sensors",
        )
        diagnosis = out["diagnosis"]
        ok = diagnosis in expected
        marker = "✓" if ok else "?"
        print(f"  {label}")
        print(f"    diagnosis     = {diagnosis:30}  {marker}")
        print(f"    confidence    = {out['confidence']:.3f}")
        print(f"    benign sim    = {out['benign_similarity']:.3f}")
        print(f"    expected one of: {sorted(expected)}")
        print("    top-3 probes:")
        for name, score in out["ranked"][:3]:
            print(f"      {name:30}  {score:.3f}")
        print()

    section_header("3. Suggested action mapping (engineering, not verdict)")
    print()
    print("  Diagnosis                  Maintenance action")
    print("  -------------------------  ------------------------------------")
    print("  benign                     None — log and continue trending.")
    print("  motor_imbalance            Schedule shaft balancing.")
    print("  bearing_wear_emergence     Inspect bearing; schedule replacement window.")
    print("  harmonic_emergence         Check shaft alignment / mounting torque.")
    print("  shock_event                Review camera + log; usually one-off.")
    print("  looseness_spikes           Inspect mounting / housing fasteners.")
    print("  cavitation_burst           For pumps: check inlet pressure / NPSH.")
    print("  gradual_amplitude_growth   Trend over time; replace when sustained.")
    print("  process_drift              Calibration check.")
    print("  sensor_saturation          Check sensor range / mounting.")
    print("  sensor_dropout             Replace sensor / cable.")
    print("  speed_change               Verify VFD command vs operating point.")

    section_header("Summary")
    print()
    print("  Pipeline: 2s vibration capture → auto_diagnose with INDUSTRIAL_PROBES")
    print("  → action mapping for the maintenance team.")
    print()
    print("  Same pattern works for any rotating machinery: pumps, compressors,")
    print("  fans, turbines, gearboxes. Adjust rotation_hz and harmonics for the")
    print("  specific asset.")
    print()
    print("  ⚠ This is screening tooling, NOT a maintenance verdict. Pair with")
    print("    vibration-spectrum analysis (FFT peaks at BPFO/BPFI) and a")
    print("    maintenance engineer's review.")
    print()


if __name__ == "__main__":
    main()
