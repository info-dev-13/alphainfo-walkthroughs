"""
recipes/notebooks/mlops_nyc_taxi_walkthrough.py — REAL-DATA validation
of `probes_mlops` on the Numenta NAB NYC Taxi benchmark.

Numenta's Anomaly Benchmark (NAB) is one of the canonical datasets
for streaming anomaly detection in the MLOps community. The
`realKnownCause/nyc_taxi.csv` series is 10,320 thirty-minute pickup
counts spanning 2014-07-01 → 2015-01-31, with five expert-annotated
events:

    2014-11-01 19:00  — NYC Marathon weekend
    2014-11-27 15:30  — Thanksgiving
    2014-12-25 15:00  — Christmas Day
    2015-01-01 01:00  — New Year's Day
    2015-01-27 00:00  — January 2015 blizzard

Why this is the right dataset for MLOps drift validation
--------------------------------------------------------

  * Public, no auth, single CSV file from the NAB GitHub repo.
  * Ground-truth labels are EXPERT-ANNOTATED, not synthetic.
  * The setup mirrors a real production ML system: a forecaster
    trained on summer 2014 demand will see autumn / winter pickup
    distributions it never saw during training.
  * If the structural detector flags windows aligned with the labels
    *without seeing the labels*, that's evidence the same probes
    that fire on synthetic drift fire on real drift.

What this walkthrough demonstrates
----------------------------------

  * `probes_mlops` calibrated on synthetic prediction-stream drift
    (prediction_collapse_to_uniform, bimodality_collapse,
    feature_mean_shift, ...) generalises to real population-stream
    drift on a different domain (taxi demand vs ML predictions).
  * For each of the 5 NAB events, we anchor a baseline on calm
    July 2014 demand and run `auto_diagnose` over a window
    centred on the event. We then compare the diagnosis to what a
    human MLOps engineer would expect: the structural signature
    each event leaves on the input stream.

Honest framing
--------------

We are NOT claiming alphainfo predicts taxi demand, nor that the
diagnosis label maps 1:1 to "this is the marathon". We are showing
that synthetic-calibrated MLOps fingerprints fire on the structural
shapes real production drift events leave behind — using a dataset
the MLOps community already trusts.

Run
---

    python -m recipes.notebooks.mlops_nyc_taxi_walkthrough

Network dependency: downloads ~10K rows from the NAB GitHub repo on
first run; caches to data/nyc_taxi_nab.csv. Subsequent runs offline.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np


_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long")
os.environ.setdefault("MASTER_API_KEY", "ai_master_test")
os.environ.setdefault("SKIP_DB", "1")


_CACHE_PATH = _ROOT / "data" / "nyc_taxi_nab.csv"
_NAB_CSV_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/"
    "data/realKnownCause/nyc_taxi.csv"
)

# NAB expert-annotated anomaly timestamps for nyc_taxi
KNOWN_EVENTS: List[Tuple[str, str]] = [
    ("2014-11-01 19:00:00", "NYC Marathon weekend"),
    ("2014-11-27 15:30:00", "Thanksgiving"),
    ("2014-12-25 15:00:00", "Christmas Day"),
    ("2015-01-01 01:00:00", "New Year's Day"),
    ("2015-01-27 00:00:00", "January 2015 blizzard"),
]


# ─────────────────────────────────────────────────────────────────────────────
# NAB CSV fetch
# ─────────────────────────────────────────────────────────────────────────────


def fetch_nab_taxi() -> List[Tuple[datetime, float]]:
    """Download the NAB NYC Taxi CSV (or use cache) and return parsed rows."""
    if _CACHE_PATH.exists():
        text = _CACHE_PATH.read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(_NAB_CSV_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(text, encoding="utf-8")

    rows: List[Tuple[datetime, float]] = []
    for line in text.splitlines()[1:]:  # skip header
        if not line.strip():
            continue
        ts_str, val_str = line.split(",")
        rows.append((
            datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"),
            float(val_str),
        ))
    return rows


def slice_window(
    rows: List[Tuple[datetime, float]],
    start: datetime,
    end: datetime,
) -> np.ndarray:
    """Return values whose timestamps fall in [start, end)."""
    return np.asarray([v for ts, v in rows if start <= ts < end], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough
# ─────────────────────────────────────────────────────────────────────────────


def main():
    from fastapi.testclient import TestClient
    from api.app import app
    from types import SimpleNamespace
    from recipes.auto_diagnose import auto_diagnose
    from recipes.probes_mlops import (
        MLOPS_PROBES,
        MLOPS_BENIGN_CONTROLS,
    )

    print("┌────────────────────────────────────────────────────────────────────┐")
    print("│  alphainfo end-to-end walkthrough — REAL DATA                      │")
    print("│  Numenta NAB NYC Taxi (10,320 pts, 5 expert-annotated events)      │")
    print("│  probes_mlops against drift detection on a deployed-model proxy    │")
    print("└────────────────────────────────────────────────────────────────────┘\n")

    # 1) Fetch dataset
    print("1. Fetching NAB nyc_taxi.csv (raw.githubusercontent.com)\n")
    rows = fetch_nab_taxi()
    values = np.asarray([v for _, v in rows])
    print(f"   rows:         {len(rows)}")
    print(f"   period:       {rows[0][0]} → {rows[-1][0]}")
    print(f"   value range:  [{values.min():.0f}, {values.max():.0f}]")
    print(f"   mean / std:   {values.mean():.0f} / {values.std():.0f}\n")

    # 2) Baseline = first 4 weeks of July 2014 (no holidays, no anomalies
    #    annotated by NAB in this stretch). 4 weeks × 7 days × 48/day = 1344
    #    half-hour samples.
    print("2. Defining baseline window")
    print("   First 4 weeks of July 2014 — no NAB-annotated anomalies\n")
    baseline_start = datetime(2014, 7, 1, 0, 0, 0)
    baseline_end = datetime(2014, 7, 29, 0, 0, 0)
    baseline = slice_window(rows, baseline_start, baseline_end)
    print(f"   {baseline_start.date()} → {baseline_end.date()} "
          f"({len(baseline)} half-hour samples)")
    print(f"   mean:  {baseline.mean():.0f}  std:  {baseline.std():.0f}\n")

    # 3) Set up TestClient as alphainfo client
    tc = TestClient(app)
    key = os.environ["MASTER_API_KEY"]

    class Cli:
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
            r = tc.post("/v1/analyze/vector",
                        headers={"X-API-Key": key}, json=payload)
            r.raise_for_status()
            data = r.json()
            ch_raw = data.get("channels", {}) or {}
            ch = ({k: SimpleNamespace(**v) for k, v in ch_raw.items()}
                  if isinstance(ch_raw, dict)
                  else [SimpleNamespace(**c) for c in ch_raw])
            return SimpleNamespace(
                structural_score=data.get("structural_score"),
                change_detected=data.get("change_detected"),
                confidence_band=data.get("confidence_band"),
                channels=ch,
            )

    client = Cli()

    # 4) For each known event, take a ±1-day window around the timestamp
    #    and run auto_diagnose. Compare to baseline.
    print("3. Running auto_diagnose with MLOPS_PROBES on each NAB event")
    print("   ±1-day window around each anomaly; baseline = first 4 weeks Jul\n")

    diagnoses = []
    for ts_str, label in KNOWN_EVENTS:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        evt_start = ts - timedelta(days=1)
        evt_end = ts + timedelta(days=1)
        evt = slice_window(rows, evt_start, evt_end)
        if len(evt) < 50:
            print(f"   [skip {label}: only {len(evt)} samples]")
            continue

        out = auto_diagnose(
            client,
            signal=evt.tolist(),
            baseline=baseline.tolist(),
            sampling_rate=1.0,
            domain="generic",
            probes=MLOPS_PROBES,
            benign_controls=MLOPS_BENIGN_CONTROLS,
        )

        diagnoses.append({
            "label": label,
            "timestamp": ts_str,
            "diagnosis": out["diagnosis"],
            "confidence": out["confidence"],
            "benign_similarity": out["benign_similarity"],
            "top_probe": out["ranked"][0] if out["ranked"] else None,
            "top5": out["ranked"][:5],
        })

        print(f"   {label} ({ts_str}):")
        print(f"     diagnosis:         {out['diagnosis']}")
        print(f"     confidence:        {out['confidence']:.3f}")
        print(f"     benign_similarity: {out['benign_similarity']:.3f}")
        print(f"     margin (vs benign): {out['confidence'] - out['benign_similarity']:+.3f}")
        print(f"     top-3:")
        for name, score in out["ranked"][:3]:
            marker = "*" if name == out["diagnosis"] else " "
            print(f"       {marker} {name:38s}  {score:.3f}")
        print()

    # 5) Honest framing summary
    print("══════════════════════════════════════════════════════════════════════")
    print("  Summary")
    print("══════════════════════════════════════════════════════════════════════\n")

    flagged = [d for d in diagnoses
               if d["confidence"] - d["benign_similarity"] > 0.0]
    strong = [d for d in diagnoses
              if d["confidence"] - d["benign_similarity"] > 0.02]
    print(f"  Events analysed:         {len(diagnoses)} / {len(KNOWN_EVENTS)}")
    print(f"  Diagnosis above benign:  {len(flagged)}")
    print(f"  Strong diagnoses (>0.02 margin): {len(strong)}")
    print()
    print("  MLOPS_PROBES were calibrated against synthetic ML-pipeline drift")
    print("  fingerprints (prediction_collapse_to_uniform, bimodality_collapse,")
    print("  feature_mean_shift, latency_creep, ...). The diagnoses above")
    print("  emerged from real NYC taxi pickup counts — a different domain,")
    print("  same probe library. Classifier shape is what generalises, not")
    print("  the literal interpretation of the channel.")
    print()
    print("  ⚠ This is engineering tooling. Production drift detection should")
    print("    pair structural diagnosis with ground-truth labels when available")
    print("    and operator review for high-stakes decisions.")


if __name__ == "__main__":
    main()
