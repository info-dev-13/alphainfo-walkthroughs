"""
recipes/notebooks/ecg_physionet_walkthrough.py — Real ECG via PhysioNet.

Companion to ``ecg_walkthrough.py`` which uses synthetic ECG. This
notebook pulls a real arrhythmia record from PhysioNet's MIT-BIH
database via the ``wfdb`` library (already in requirements.txt) and
runs the same pipeline (windowed → auto_diagnose with BIOMEDICAL_PROBES)
on actual cardiac signals.

Records used
------------

  * 100 — normal sinus rhythm, no significant arrhythmia.
  * 200 — frequent PVCs (premature ventricular contractions) and a
            ventricular bigeminy episode. Both are well-annotated.

We compare a window from record 200 (with PVCs) against a window from
record 100 (clean) as the baseline. The structural diagnosis should
flag the difference.

Usage
-----

    cd alphainfo/
    .venv/bin/python -m recipes.notebooks.ecg_physionet_walkthrough

Network access required for the first run. ``wfdb`` caches records
under ``~/.wfdb`` so subsequent runs are offline.

Important caveat
----------------

This is engineering tooling, NOT clinical diagnosis. The MIT-BIH
records have rigorous expert annotations precisely because rhythm
classification is hard and specialised. Use this as a demonstration
that the structural pipeline detects the difference between healthy
and arrhythmic tracings, not as a diagnostic claim.
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
# In-process SDK shim (same as the synthetic walkthrough)
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
# Data fetcher
# ─────────────────────────────────────────────────────────────────────────────


def fetch_mitbih_record(
    record_name: str,
    duration_seconds: float = 30.0,
    channel: int = 0,
):
    """Pull `duration_seconds` of an MIT-BIH record from PhysioNet.

    Args:
        record_name: e.g. "100", "200", "234".
        duration_seconds: How much to read.
        channel: 0 (MLII) or 1 (V1/V2/V5 depending on record).

    Returns:
        (signal_array, sampling_rate_hz)
    """
    try:
        import wfdb
    except ImportError as e:
        raise RuntimeError(
            "wfdb not installed. Run: pip install wfdb"
        ) from e

    # PhysioNet hosts MIT-BIH at https://physionet.org/files/mitdb/1.0.0/
    # wfdb.rdrecord with pn_dir downloads + caches in ~/.wfdb.
    record = wfdb.rdrecord(
        record_name,
        pn_dir="mitdb",
        sampto=int(duration_seconds * 360),  # MIT-BIH is at 360 Hz
        channels=[channel],
    )
    return record.p_signal[:, 0], float(record.fs)


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print()
    print("┌" + "─" * 68 + "┐")
    print("│  alphainfo end-to-end walkthrough — biomedical / real ECG        │")
    print("│  MIT-BIH record 100 (clean) vs record 200 (frequent PVCs)        │")
    print("└" + "─" * 68 + "┘")

    section_header("1. Fetch real ECG record from PhysioNet")
    print("\nPulling MIT-BIH record 100 (normal sinus rhythm) — clean control...")
    try:
        record_100, fs = fetch_mitbih_record("100", duration_seconds=60.0)
        print(f"  ✓ record 100: {len(record_100)} samples at {fs:g} Hz")
        print("\nPulling MIT-BIH record 200 (frequent PVCs / bigeminy)...")
        record_200, fs2 = fetch_mitbih_record("200", duration_seconds=60.0)
        print(f"  ✓ record 200: {len(record_200)} samples at {fs2:g} Hz")
        assert fs == fs2, "Both records must be at the same sampling rate"
    except Exception as e:
        print(f"\n⚠️  PhysioNet fetch failed: {e}")
        print("\nThis walkthrough requires network access on first run; wfdb")
        print("caches downloads under ~/.wfdb so subsequent runs are offline.")
        print("\nFalling back to the synthetic walkthrough (run that instead):")
        print("    .venv/bin/python -m recipes.notebooks.ecg_walkthrough")
        return

    client = _LocalClient()

    # Compare WITHIN a single record — first 30s vs second 30s of record 200.
    # This isolates within-patient temporal differences (PVCs come and go) from
    # between-patient differences (different lead placement, electrode noise,
    # signal scale). Cross-patient comparison is a different problem and
    # would require z-normalised signals first.
    section_header("2. Diagnose record 200: second 30s vs first 30s (within-patient)")
    print("\n  Comparing the SAME patient's signal across two halves of the")
    print("  capture isolates temporal/event differences from cross-patient")
    print("  factors (lead placement, electrode noise, signal scale).\n")
    half = len(record_200) // 2
    baseline_half = record_200[:half]
    test_half = record_200[half:]

    from recipes.intents import dispatch
    out = dispatch(
        client, intent="what_kind_of_change",
        signal=test_half.tolist(), baseline=baseline_half.tolist(),
        sampling_rate=fs, domain="biomedical",
    )
    print(f"  diagnosis    = {out['diagnosis']}")
    print(f"  confidence   = {out['confidence']:.3f}")
    print(f"  benign sim   = {out['benign_similarity']:.3f}")
    print("\n  top-5 probes (descending similarity):")
    for name, score in out["ranked"][:5]:
        print(f"    {name:25}  {score:.3f}")

    section_header("3. Localise — sliding window over record 200")
    print("\n  Run a 1s sliding window over the full 60s of record 200,")
    print("  using the first 5s of record 100 (clean reference) as the")
    print("  baseline. Flag windows whose structural_score drops below 0.5.\n")

    from recipes.windowed import analyze_windowed
    win_out = analyze_windowed(
        client, record_200.tolist(),
        baseline=record_100[:int(5 * fs)].tolist(),
        sampling_rate=fs, window_size=int(fs), step=int(fs / 4),
        domain="biomedical",
    )
    timeline = np.asarray(win_out["timeline"])
    starts = win_out["starts"]
    flagged = [(starts[i], timeline[i]) for i in range(len(timeline)) if timeline[i] < 0.5]
    print(f"  windows scanned: {win_out['n_windows']}")
    print(f"  worst window:    sample {win_out['worst_at_t']} "
          f"(t = {win_out['worst_at_t']/fs:.1f}s, score={win_out['worst_score']:.3f})")
    print(f"  flagged windows (score < 0.5): {len(flagged)}")
    for s, score in flagged[:10]:
        print(f"    sample {s:6d}  (t = {s/fs:5.1f}s)  score = {score:.3f}")

    section_header("Summary")
    print()
    print("  Pipeline tested on REAL clinical data — MIT-BIH records 100 & 200")
    print("  from PhysioNet, no synthetic injection. The SAME recipe code")
    print("  (recipes.intents.dispatch + recipes.windowed) runs on synthetic")
    print("  and real data with no changes.")
    print()
    print("  ⚠ Engineering screening tooling, not clinical diagnosis.")
    print("    Pair with rhythm classifier + clinician review.")
    print()


if __name__ == "__main__":
    main()
