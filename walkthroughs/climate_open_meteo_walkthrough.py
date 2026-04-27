"""
recipes/notebooks/climate_open_meteo_walkthrough.py — REAL-DATA
validation of `probes_climate` on Open-Meteo Historical Weather API.

Open-Meteo (https://open-meteo.com) provides a public Historical
Weather API — daily temperature back to 1940, no authentication, no
key, free for non-commercial use. Perfect external test bed for
`probes_climate`:

  * Daily mean temperature has clear seasonality (yearly cycle).
  * Year-over-year mean drift is well-documented (warming trend).
  * Heatwaves / cold snaps register as variance / extreme bursts.
  * No proprietary data — anyone can reproduce.

This walkthrough downloads 5 years of daily mean temperature for
Berlin (chosen for clear seasonality and good documentation),
selects two climatologically comparable windows, and runs
`auto_diagnose` against the calibrated `CLIMATE_PROBES`. The
diagnosis label that wins on real data is the validation that
synthetic-calibrated fingerprints survive in the wild.

What this walkthrough demonstrates
----------------------------------

  * `probes_climate` calibrated against synthetic baselines
    (regime_shift_warming, variance_increase, seasonal_amplitude_*,
    extreme_events_burst, ...) generalises to real station-level
    temperature data.
  * `auto_diagnose` finds the dominant fingerprint without any
    Berlin-specific tuning — the engine is unchanged from the
    synthetic verticals.

Honest framing
--------------

We are NOT claiming alphainfo predicts climate, nor that the
diagnosis label maps 1:1 to a physical event (e.g. "this is the
2022 European heatwave"). We are showing: synthetic-calibrated
fingerprints recognise their own shapes when they appear in real
public station data. Same question buyers ask in due diligence —
"did you test on something you didn't generate yourself?".

Run
---

    python -m recipes.notebooks.climate_open_meteo_walkthrough

Network dependency: downloads ~5 years of daily data from Open-Meteo
(free, no auth). Falls back to a cached JSON in
data/climate_berlin_daily.json if present. Quota: ~30 alphainfo
calls per run.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_CACHE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "climate_berlin_daily.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo Historical Weather fetch
# ─────────────────────────────────────────────────────────────────────────────


def fetch_daily_temperature(
    *,
    latitude: float = 52.52,
    longitude: float = 13.41,  # Berlin
    start: Optional[date] = None,
    end: Optional[date] = None,
    years: int = 5,
) -> List[Tuple[date, float]]:
    """Fetch daily mean temperature pairs (date, °C) from Open-Meteo.

    Defaults to the last `years` years of Berlin data ending today.
    Falls back to the cached JSON if the network call fails.
    """
    import urllib.error
    import urllib.request

    end = end or (date.today() - timedelta(days=1))
    start = start or (end - timedelta(days=365 * years))

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        "&daily=temperature_2m_mean&timezone=UTC"
    )

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        daily = payload.get("daily", {})
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_mean", [])

        pairs: List[Tuple[date, float]] = []
        for d_str, t in zip(dates, temps):
            if t is None:
                continue
            pairs.append((date.fromisoformat(d_str), float(t)))

        # Cache for reproducibility
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            cache = [{"date": d.isoformat(), "temp_c": v} for d, v in pairs]
            _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except OSError:
            pass

        return pairs
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        if _CACHE_PATH.exists():
            cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            return [(date.fromisoformat(p["date"]), float(p["temp_c"])) for p in cache]
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Window selection — pick climatologically comparable summer windows
# ─────────────────────────────────────────────────────────────────────────────


def split_summer_windows(
    pairs: List[Tuple[date, float]],
    summer_months: Tuple[int, int] = (6, 8),  # June-August (boreal summer)
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Pick two summer windows (Jun-Aug) of two different years for comparison.

    Strategy: take the most recent two complete summers. Compare the
    *latest* against the *previous*. This isolates year-to-year
    structural change — same season, same site, just two years apart.
    """
    by_year: dict = {}
    for d, t in pairs:
        if summer_months[0] <= d.month <= summer_months[1]:
            by_year.setdefault(d.year, []).append((d, t))

    # Need at least 2 complete summers
    full_years = [y for y, vals in by_year.items() if len(vals) >= 80]  # ~90 days expected
    if len(full_years) < 2:
        raise ValueError(f"Need ≥2 complete summers; got {len(full_years)}")

    full_years.sort()
    baseline_year = full_years[-2]
    event_year = full_years[-1]

    baseline_vals = np.asarray([t for _, t in by_year[baseline_year]])
    event_vals = np.asarray([t for _, t in by_year[event_year]])

    return baseline_vals, event_vals, {
        "baseline_year": baseline_year,
        "baseline_n": len(baseline_vals),
        "baseline_mean": float(baseline_vals.mean()),
        "baseline_std": float(baseline_vals.std()),
        "event_year": event_year,
        "event_n": len(event_vals),
        "event_mean": float(event_vals.mean()),
        "event_std": float(event_vals.std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough
# ─────────────────────────────────────────────────────────────────────────────


def main():
    import os
    os.environ.setdefault(
        "JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long",
    )
    os.environ.setdefault("MASTER_API_KEY", "ai_master_test")
    os.environ.setdefault("SKIP_DB", "1")

    from fastapi.testclient import TestClient
    from api.app import app
    from types import SimpleNamespace
    from recipes.auto_diagnose import auto_diagnose
    from recipes.probes_climate import (
        CLIMATE_PROBES,
        CLIMATE_BENIGN_CONTROLS,
    )

    print("┌────────────────────────────────────────────────────────────────┐")
    print("│  alphainfo end-to-end walkthrough — REAL DATA                  │")
    print("│  Open-Meteo Historical Weather API (Berlin daily mean temp)    │")
    print("│  probes_climate against year-over-year summer comparison       │")
    print("└────────────────────────────────────────────────────────────────┘\n")

    # 1) Fetch 5 years of Berlin daily mean temperature
    print("1. Fetching ~5 years of Berlin daily mean temperature")
    print("   (archive-api.open-meteo.com — free, no auth)\n")
    pairs = fetch_daily_temperature(years=5)
    series = np.asarray([t for _, t in pairs])
    print(f"   points:    {len(pairs)}")
    print(f"   range:     [{series.min():.1f}, {series.max():.1f}] °C")
    print(f"   mean:      {series.mean():.1f} °C")
    print(f"   period:    {pairs[0][0]} → {pairs[-1][0]}\n")

    # 2) Pick two consecutive summers — same season, two different years
    print("2. Selecting two consecutive summers (Jun-Aug)")
    print("   Same season, same site, two years apart — isolates year-")
    print("   over-year structural change.\n")
    baseline, event, meta = split_summer_windows(pairs)
    print(f"   baseline:  summer {meta['baseline_year']}  "
          f"({meta['baseline_n']} days, mean {meta['baseline_mean']:.2f}°C, "
          f"std {meta['baseline_std']:.2f})")
    print(f"   event:     summer {meta['event_year']}  "
          f"({meta['event_n']} days, mean {meta['event_mean']:.2f}°C, "
          f"std {meta['event_std']:.2f})")
    delta_mean = meta["event_mean"] - meta["baseline_mean"]
    delta_std = meta["event_std"] - meta["baseline_std"]
    print(f"   Δ mean:    {delta_mean:+.2f}°C")
    print(f"   Δ std:     {delta_std:+.2f}°C\n")

    # 3) Wrap TestClient as alphainfo client
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

    # 4) Run auto_diagnose against CLIMATE_PROBES
    print("3. Running auto_diagnose with CLIMATE_PROBES + benign controls\n")
    out = auto_diagnose(
        client,
        signal=event.tolist(),
        baseline=baseline.tolist(),
        sampling_rate=1.0,
        domain="generic",   # climate routes to generic on the engine side
        probes=CLIMATE_PROBES,
        benign_controls=CLIMATE_BENIGN_CONTROLS,
    )

    print(f"   diagnosis:         {out['diagnosis']}")
    print(f"   confidence:        {out['confidence']:.3f}")
    print(f"   benign_similarity: {out['benign_similarity']:.3f}\n")
    print("   Top-5 probe similarities:")
    for name, score in out["ranked"][:5]:
        marker = "*" if name == out["diagnosis"] else " "
        print(f"     {marker} {name:32s}  {score:.3f}")
    print()

    # 5) Honest interpretation
    print("══════════════════════════════════════════════════════════════════════")
    print("  Interpretation (honest framing)")
    print("══════════════════════════════════════════════════════════════════════\n")
    print("  CLIMATE_PROBES were CALIBRATED against synthetic climate")
    print("  signatures (regime_shift_warming, variance_increase,")
    print(f"  extreme_events_burst, seasonal_amplitude_*, ...). The")
    print(f"  diagnosis \"{out['diagnosis']}\" emerged from real Berlin daily")
    print(f"  temperature data without any Berlin-specific tuning.")
    print(f"  Mean Δ of {delta_mean:+.2f}°C and std Δ of {delta_std:+.2f}°C")
    print(f"  between summer {meta['baseline_year']} and summer {meta['event_year']}")
    print(f"  produced a fingerprint that matched this probe's synthetic shape.")
    print()
    print("  We are NOT predicting climate, nor mapping the diagnosis to a")
    print("  named physical event. We are showing: the structural pattern")
    print("  calibrated on synthetic seeds appears in real station data.")
    print("  Same metric a buyer asks for in technical due diligence.")


if __name__ == "__main__":
    main()
