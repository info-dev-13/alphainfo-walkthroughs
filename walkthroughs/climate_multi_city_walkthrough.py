"""
recipes/notebooks/climate_multi_city_walkthrough.py — Multi-city extension of
the climate validation, reinforcing the geography-agnostic claim.

Companion to ``climate_open_meteo_walkthrough.py`` (Berlin). The original
question — "does the detector know when nothing structural changed?" —
is answered specifically for one site. The natural follow-up: does it
hold across hemispheres, climate zones, and seasonal patterns?

Cities chosen, each on a different climate regime:

    Berlin       (52.52  N,  13.41 E)  — temperate continental, NH
    Tokyo        (35.68  N, 139.76 E)  — humid subtropical, NH
    São Paulo    (-23.55 S, -46.63 W)  — subtropical highland, SH
    Reykjavík    (64.13  N, -21.82 W)  — subarctic oceanic, NH

Same probe library calibrated against synthetic. Same auto_diagnose.
Each city gets two consecutive summers compared (summer is NH Jun-Aug
for the three northern cities; SH Dec-Feb for São Paulo). Expected
result: a benign verdict for each — none of these year-on-year summer
comparisons should carry structural change. Together they extend
the specificity claim from "the detector doesn't false-positive in
Berlin" to "the detector doesn't false-positive across four distinct
climate regimes".

Usage
-----

    python -m recipes.notebooks.climate_multi_city_walkthrough

Network dependency: downloads ~5 years × 4 cities of daily data from
Open-Meteo. Caches per city under ``data/climate_<city>_daily.json``.
Subsequent runs are offline.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long")
os.environ.setdefault("MASTER_API_KEY", "ai_master_test")
os.environ.setdefault("SKIP_DB", "1")


# ─────────────────────────────────────────────────────────────────────────────
# Cities — each gets its own cache file and summer-month convention
# ─────────────────────────────────────────────────────────────────────────────

CITIES = [
    {
        "slug": "berlin",
        "name": "Berlin",
        "country": "Germany",
        "regime": "temperate continental",
        "hemisphere": "N",
        "lat": 52.52,
        "lon": 13.41,
        "summer_months": (6, 8),  # Jun, Jul, Aug
    },
    {
        "slug": "tokyo",
        "name": "Tokyo",
        "country": "Japan",
        "regime": "humid subtropical",
        "hemisphere": "N",
        "lat": 35.68,
        "lon": 139.76,
        "summer_months": (6, 8),
    },
    {
        "slug": "saopaulo",
        "name": "São Paulo",
        "country": "Brazil",
        "regime": "subtropical highland",
        "hemisphere": "S",
        "lat": -23.55,
        "lon": -46.63,
        "summer_months": (12, 2),  # Dec, Jan, Feb (wraps year)
    },
    {
        "slug": "reykjavik",
        "name": "Reykjavík",
        "country": "Iceland",
        "regime": "subarctic oceanic",
        "hemisphere": "N",
        "lat": 64.13,
        "lon": -21.82,
        "summer_months": (6, 8),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo fetch (per-city cache)
# ─────────────────────────────────────────────────────────────────────────────


def fetch_daily_temperature(
    *,
    slug: str,
    latitude: float,
    longitude: float,
    years: int = 5,
) -> List[Tuple[date, float]]:
    cache = _ROOT / "data" / f"climate_{slug}_daily.json"

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=365 * years)

    if cache.exists():
        try:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            return [
                (date.fromisoformat(d), float(t))
                for d, t in payload
                if t is not None
            ]
        except Exception:
            pass

    import urllib.error
    import urllib.request

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        "&daily=temperature_2m_mean&timezone=UTC"
    )

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    daily = payload["daily"]
    pairs = []
    for d_str, t in zip(daily["time"], daily["temperature_2m_mean"]):
        if t is None:
            continue
        pairs.append((date.fromisoformat(d_str), float(t)))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps([[d.isoformat(), t] for d, t in pairs]),
        encoding="utf-8",
    )
    return pairs


def split_summer_windows(
    pairs: List[Tuple[date, float]],
    summer_months: Tuple[int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    """Pick last two complete summers and return (baseline, event, meta).

    summer_months is (start_month, end_month). If start > end (e.g. (12, 2))
    the window wraps the year boundary — treated as "summer of the year that
    contains the start month".
    """
    s_start, s_end = summer_months

    def in_summer(d: date) -> Optional[int]:
        """Return the year a given date 'belongs to' as part of summer.
        For NH (Jun-Aug): year is the calendar year.
        For SH (Dec-Feb): year is the year of the December (so Jan/Feb 2025
        belong to the 2024 summer that started in Dec 2024).
        """
        m = d.month
        if s_start <= s_end:
            return d.year if s_start <= m <= s_end else None
        # wrap around: months s_start..12 OR 1..s_end
        if m >= s_start:
            return d.year
        if m <= s_end:
            return d.year - 1
        return None

    by_year: dict = {}
    for d, t in pairs:
        y = in_summer(d)
        if y is not None:
            by_year.setdefault(y, []).append((d, t))

    # Need at least two complete summers
    complete = [y for y, items in sorted(by_year.items()) if len(items) >= 70]
    if len(complete) < 2:
        return None
    baseline_year = complete[-2]
    event_year = complete[-1]

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
    from fastapi.testclient import TestClient
    from api.app import app
    from types import SimpleNamespace
    from recipes.auto_diagnose import auto_diagnose
    from recipes.probes_climate import (
        CLIMATE_PROBES,
        CLIMATE_BENIGN_CONTROLS,
    )

    print("┌────────────────────────────────────────────────────────────────────┐")
    print("│  alphainfo end-to-end walkthrough — REAL DATA                      │")
    print("│  Multi-city climate (4 cities, 4 climate regimes, 4 hemispheres)   │")
    print("│  probes_climate against year-over-year summer comparison          │")
    print("└────────────────────────────────────────────────────────────────────┘\n")

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

    results = []
    for city in CITIES:
        print(f"━━━ {city['name']}, {city['country']} "
              f"({city['regime']}, hemisphere {city['hemisphere']}) ━━━")
        print(f"  fetching {city['lat']}, {city['lon']}...")
        pairs = fetch_daily_temperature(
            slug=city["slug"],
            latitude=city["lat"],
            longitude=city["lon"],
        )
        series = np.asarray([t for _, t in pairs])
        print(f"  points: {len(pairs)}  range: [{series.min():.1f}, "
              f"{series.max():.1f}] °C  mean: {series.mean():.1f}")

        split = split_summer_windows(pairs, city["summer_months"])
        if split is None:
            print(f"  [skip {city['name']}: insufficient data]")
            print()
            continue
        baseline, event, meta = split
        print(f"  baseline summer {meta['baseline_year']}: n={meta['baseline_n']}, "
              f"mean={meta['baseline_mean']:.2f}°C, σ={meta['baseline_std']:.2f}")
        print(f"  event    summer {meta['event_year']}: n={meta['event_n']}, "
              f"mean={meta['event_mean']:.2f}°C, σ={meta['event_std']:.2f}")
        delta_mean = meta["event_mean"] - meta["baseline_mean"]
        print(f"  Δ mean: {delta_mean:+.2f}°C, "
              f"Δ σ: {meta['event_std'] - meta['baseline_std']:+.2f}")

        out = auto_diagnose(
            client,
            signal=event.tolist(),
            baseline=baseline.tolist(),
            sampling_rate=1.0,
            domain="generic",
            probes=CLIMATE_PROBES,
            benign_controls=CLIMATE_BENIGN_CONTROLS,
        )
        margin = out["confidence"] - out["benign_similarity"]
        results.append({
            "city": city["name"],
            "regime": city["regime"],
            "hemisphere": city["hemisphere"],
            "delta_mean": delta_mean,
            "delta_std": meta["event_std"] - meta["baseline_std"],
            "diagnosis": out["diagnosis"],
            "confidence": out["confidence"],
            "benign": out["benign_similarity"],
            "margin": margin,
            "top3": out["ranked"][:3],
        })

        print(f"  diagnosis: {out['diagnosis']}")
        print(f"  confidence: {out['confidence']:.3f}, "
              f"benign: {out['benign_similarity']:.3f}, "
              f"margin: {margin:+.3f}")
        print(f"  top-3:")
        for name, score in out["ranked"][:3]:
            m = "*" if name == out["diagnosis"] else " "
            print(f"    {m} {name:32s}  {score:.3f}")
        print()

    print("══════════════════════════════════════════════════════════════════════")
    print("  Summary: benign-specificity across 4 climate regimes")
    print("══════════════════════════════════════════════════════════════════════\n")

    weak_diagnoses = [r for r in results if r["margin"] < 0.02]
    print(f"  Cities analysed: {len(results)} / {len(CITIES)}")
    print(f"  Weak / benign diagnoses (margin < 0.02): {len(weak_diagnoses)} / {len(results)}")
    print(f"  (Expected: high — year-over-year summer comparisons should be benign)")
    print()
    print(f"  {'City':16s} {'regime':24s} {'Δ mean':>8s} {'margin':>8s}  diagnosis")
    print(f"  {'-'*16:16s} {'-'*24:24s} {'-'*8:>8s} {'-'*8:>8s}  {'-'*30}")
    for r in results:
        print(f"  {r['city']:16s} {r['regime']:24s} "
              f"{r['delta_mean']:>+7.2f}°C {r['margin']:>+8.3f}  {r['diagnosis']}")
    print()
    print("  CLIMATE_PROBES were calibrated on synthetic climate fingerprints.")
    print("  Across 4 climate regimes (continental, subtropical, highland, subarctic)")
    print("  and 2 hemispheres, the detector is expected to return WEAK diagnoses")
    print("  for year-over-year summer comparisons — none of these comparisons")
    print("  should carry true structural change. Specificity that holds across")
    print("  geographies is the property a buyer actually relies on in production.")


if __name__ == "__main__":
    main()
