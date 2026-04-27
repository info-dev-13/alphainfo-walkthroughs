"""
recipes/notebooks/finance_walkthrough.py — End-to-end SPY analysis.

Walks through a real-world structural-detection workflow on SPY data
pulled live from Yahoo Finance. The scenario: detect the COVID-19
crash regime change without prior knowledge of when it happened.

Pipeline:

  1. Fetch ~6 years of daily SPY closes (2018-01 → 2024-01).
  2. Define a "calm" baseline window (Q4 2019 — 60 trading days).
  3. Define a "test" window (Q1 2020 — 60 trading days, contains the
     COVID crash).
  4. Run the `regime_change` intent → expect:
       WHERE  = late Feb / early Mar 2020
       WHAT   = vol_regime_2x or crash_event
  5. Run `volatility_shift` intent on the same windows → expect
     `rms_ratio` and `absdiff` channels lighting up (the volatility
     ratio is the dominant signal in real crashes).
  6. Run `motif_search` over the full 6-year history with the COVID
     crash window as the motif → expect the 2018 Vol-mageddon (Feb
     2018) and the 2022 inflation drawdown to appear in the top-K.

This is what the eval report asked for: a public-data, no-API-key
notebook that walks 3-4 recipes end-to-end so a developer evaluating
the product sees the workflow on data they already understand.

Usage
-----

    cd alphainfo/
    .venv/bin/python -m recipes.notebooks.finance_walkthrough

This calls the in-process FastAPI TestClient — no external API key
required. To run against live alphainfo.io, swap `_LocalClient` for
`alphainfo.AlphaInfo(api_key="ai_...")`.

Dependencies: `yfinance` (already in requirements.txt) for the data
fetch. Network access is needed for the Yahoo Finance call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────


_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long")
os.environ.setdefault("MASTER_API_KEY", "ai_master_test")


# ─────────────────────────────────────────────────────────────────────────────
# Data fetcher
# ─────────────────────────────────────────────────────────────────────────────


def fetch_spy_returns(start: str = "2018-01-01", end: str = "2024-01-01"):
    """Pull SPY adjusted closes from Yahoo Finance and return log-returns
    plus the date index. Uses yfinance — no API key required."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError(
            "yfinance not installed. Run: pip install yfinance"
        )

    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(
            "yfinance returned no data — check your network connection. "
            "Falling back to synthetic data for the demo."
        )
    # yfinance >=0.2 returns multi-column with ticker level; flatten.
    if isinstance(df.columns, type(df.columns)) and df.columns.nlevels == 2:
        close = df["Close"]["SPY"]
    else:
        close = df["Close"]
    log_returns = np.log(close / close.shift(1)).dropna()
    return log_returns.values, [d.date().isoformat() for d in log_returns.index]


# ─────────────────────────────────────────────────────────────────────────────
# In-process SDK shim — same shape as the real Python SDK
# ─────────────────────────────────────────────────────────────────────────────


class _LocalClient:
    """Drop-in for `alphainfo.AlphaInfo(api_key=...)` that drives the
    in-process FastAPI app — no network round-trips, no API key beyond
    the test default."""

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


# ─────────────────────────────────────────────────────────────────────────────
# Walkthrough sections
# ─────────────────────────────────────────────────────────────────────────────


def section_header(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def section_1_fetch_data():
    section_header("1. Fetch SPY daily log-returns (2018-2024)")
    print("\nPulling from Yahoo Finance via yfinance...")
    try:
        returns, dates = fetch_spy_returns("2018-01-01", "2024-01-01")
    except Exception as e:
        print(f"\n⚠️  Network fetch failed: {e}")
        print("   Falling back to synthetic SPY-like data for the demo.")
        # Generate ISO-formatted business-day dates starting 2018-01-01 so
        # downstream date-based slicing (e.g. idx_of('2019-10-01')) works
        # the same way as with real yfinance data. Without this, dates like
        # 'day-0' fail string-comparison against ISO targets and every
        # slice collapses to length 0.
        from datetime import date, timedelta
        rng = np.random.default_rng(2020)
        n = 1500
        returns = 0.0004 + 0.012 * rng.standard_normal(n)
        # Inject a synthetic crash period at sample 540-580 (Feb-Mar 2020
        # equivalent on a ~1500-day calendar starting 2018-01-01).
        returns[540:580] = returns[540:580] * 4.0 - 0.02
        # Build ISO dates skipping weekends.
        dates = []
        cur = date(2018, 1, 1)
        while len(dates) < n:
            if cur.weekday() < 5:  # Mon-Fri
                dates.append(cur.isoformat())
            cur += timedelta(days=1)
    print(f"\n  trading days fetched: {len(returns)}")
    print(f"  range:                {dates[0]} → {dates[-1]}")
    print(f"  mean daily return:    {returns.mean():+.5f}")
    print(f"  daily vol (std):      {returns.std():.5f}")
    return returns, dates


def section_2_find_baseline_and_test_windows(returns, dates):
    """Pick a calm period (Q4 2019) and a stressed period (Q1 2020)."""
    section_header("2. Define baseline (calm) and test (stressed) windows")
    # Find indices closest to our target dates
    def idx_of(target):
        for i, d in enumerate(dates):
            if d >= target:
                return i
        return len(dates) - 1

    base_start = idx_of("2019-10-01")
    base_end   = idx_of("2019-12-31")
    test_start = idx_of("2020-01-01")
    test_end   = idx_of("2020-04-01")

    baseline = returns[base_start:base_end]
    test     = returns[test_start:test_end]

    print(f"\n  baseline (calm Q4'19):    {dates[base_start]} → {dates[base_end-1]}")
    print(f"    n={len(baseline)}, mean={baseline.mean():+.5f}, vol={baseline.std():.5f}")
    print(f"  test (stressed Q1'20):    {dates[test_start]} → {dates[test_end-1]}")
    print(f"    n={len(test)}, mean={test.mean():+.5f}, vol={test.std():.5f}")
    print(f"\n  vol ratio (test/baseline): {test.std()/baseline.std():.2f}×")
    return baseline, test, (base_start, base_end, test_start, test_end)


def section_3_regime_change(client, baseline, test):
    section_header("3. Run intent='regime_change' — find WHERE + WHAT KIND")
    from recipes.intents import dispatch

    out = dispatch(
        client, intent="regime_change",
        signal=test.tolist(), baseline=baseline.tolist(),
        sampling_rate=1.0, domain="finance",
        window_size=20, step=5,
    )
    where = out["where"]
    what = out["what_kind"]

    print("\n  WHERE — windowed search:")
    print(f"    n_windows         = {where['n_windows']}")
    print(f"    worst window starts at sample {where['worst_at_t']}")
    print(f"    worst structural_score = {where['worst_score']:.3f}")

    print("\n  WHAT KIND — auto_diagnose with finance probes:")
    print(f"    diagnosis    = {what['diagnosis']}")
    print(f"    confidence   = {what['confidence']:.3f}")
    print("    top-5 probes:")
    for name, score in what["ranked"][:5]:
        print(f"      {name:25}  {score:.3f}")
    return out


def section_4_volatility_shift(client, baseline, test):
    section_header("4. Run intent='volatility_shift' — vol-sensitive ensemble")
    from recipes.intents import dispatch

    out = dispatch(
        client, intent="volatility_shift",
        signal=test.tolist(), baseline=baseline.tolist(),
        sampling_rate=1.0, domain="finance",
    )
    print(f"\n  aggregate score: {out['aggregate_score']:.3f}")
    print(f"  band:            {out['confidence_band']}")
    print(f"  responsible:     {out['responsible']}")
    print("\n  Per-channel scores (lower = changed more):")
    for name, score in out["channel_ranking"]:
        print(f"    {name:14}  {score:.3f}")
    return out


def section_5_motif_search(client, returns, dates, test_window_idx):
    """Use the COVID crash window as a motif and search for similar crash
    patterns across the full 6-year history."""
    section_header("5. motif_search — find SIMILAR crash patterns in 2018-2024")
    from recipes.motif_search import motif_search

    test_start, test_end = test_window_idx
    motif = returns[test_start:test_end]
    print(f"\n  motif: {dates[test_start]} → {dates[test_end-1]}  ({len(motif)} days)")
    print(f"  searching across {len(returns)} trading days...")

    hits = motif_search(
        client, host_signal=returns.tolist(), motif=motif.tolist(),
        sampling_rate=1.0, top_k=5, step=15,
        domain="finance",
    )

    print(f"\n  Top {len(hits)} matches (most similar to COVID crash):")
    for h in hits:
        if h["start"] < len(dates):
            window_date = dates[h["start"]]
        else:
            window_date = "out-of-range"
        print(f"    rank {h['rank']}  {window_date}  score={h['score']:.3f}")
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print()
    print("┌" + "─" * 68 + "┐")
    print("│  alphainfo end-to-end walkthrough — finance / SPY                 │")
    print("│  Detects the COVID-19 crash regime change with no prior knowledge │")
    print("└" + "─" * 68 + "┘")

    returns, dates = section_1_fetch_data()
    baseline, test, idx = section_2_find_baseline_and_test_windows(returns, dates)

    client = _LocalClient()

    section_3_regime_change(client, baseline, test)
    section_4_volatility_shift(client, baseline, test)
    section_5_motif_search(client, returns, dates, (idx[2], idx[3]))

    section_header("Summary")
    print("\n  This walkthrough used 3 recipes via the intent layer:")
    print("    • regime_change    → WHERE + WHAT KIND (chains windowed + auto_diagnose)")
    print("    • volatility_shift → rms-emphasised feature ensemble")
    print("    • motif_search     → find similar patterns in history")
    print()
    print("  No API key required — runs against the in-process app.")
    print("  To run against alphainfo.io, swap _LocalClient for AlphaInfo(api_key=...).")
    print()


if __name__ == "__main__":
    main()
