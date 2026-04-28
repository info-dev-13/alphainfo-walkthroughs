"""
finance_simple_compare.py — A self-contained walkthrough.

Most walkthroughs in this repo use private engine modules
(`api.app`, `recipes.intents`, `recipes.auto_diagnose`...) so they
aren't directly runnable by external readers — they're reference
implementations that document methodology, not standalone demos.

This script is the exception: it depends ONLY on the public alphainfo
Python SDK and `yfinance`. Clone, install, set an API key, run.

What it does
------------

Pulls SPY daily closes for the last ~6 years, picks a calm period as
baseline (Q4 2019), picks a turbulent period as signal (Q1 2020 —
contains the COVID-19 crash), and asks the public alphainfo API:
*"does the signal still behave like the baseline?"*

Expected output: a `confidence_band` of `unstable` or `transition`
and a low `structural_score` (closer to 0 = more changed).

Setup
-----

    pip install alphainfo>=1.5.27 yfinance numpy
    export ALPHAINFO_API_KEY=ai_your_key   # https://www.alphainfo.io/register
    python walkthroughs/finance_simple_compare.py

If you don't have an API key yet, the free tier (50 calls/month, no
credit card) is more than enough — sign up at
<https://www.alphainfo.io/register>.

Why this matters
----------------

Demonstrates the canonical use case in 30 lines: you have a known-good
period, you have a current period, you ask the engine if the structure
is preserved. No engine source needed, no recipes needed, no
TestClient — just the SDK + a key.
"""

from __future__ import annotations

import os
import sys

import numpy as np


def fetch_spy_closes(start: str = "2018-01-01", end: str = "2024-01-01"):
    """Pull SPY adjusted closes from Yahoo Finance — no API key needed."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance not installed. Run: pip install yfinance"
        )

    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise SystemExit(
            "yfinance returned no data — check your network connection."
        )
    if df.columns.nlevels == 2:
        close = df["Close"]["SPY"]
    else:
        close = df["Close"]
    log_returns = np.log(close / close.shift(1)).dropna()
    return log_returns.values, [d.date().isoformat() for d in log_returns.index]


def main() -> int:
    api_key = os.environ.get("ALPHAINFO_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: set ALPHAINFO_API_KEY first.\n"
            "Get a free key (50 calls/month, no credit card) at "
            "https://www.alphainfo.io/register",
            file=sys.stderr,
        )
        return 2

    try:
        from alphainfo import AlphaInfo
    except ImportError:
        raise SystemExit(
            "alphainfo SDK not installed. Run: pip install alphainfo"
        )

    print("Fetching SPY daily log-returns from Yahoo Finance...")
    returns, dates = fetch_spy_closes()
    print(f"  loaded {len(returns)} trading days "
          f"({dates[0]} → {dates[-1]})")

    # Baseline: a calm period (Q4 2019, ~63 sessions before the crash).
    baseline_start = next(i for i, d in enumerate(dates) if d >= "2019-10-01")
    baseline_end   = next(i for i, d in enumerate(dates) if d >= "2020-01-01")
    baseline = returns[baseline_start:baseline_end].tolist()

    # Signal: Q1 2020, contains the COVID crash.
    signal_start = baseline_end
    signal_end   = next(i for i, d in enumerate(dates) if d >= "2020-04-01")
    signal = returns[signal_start:signal_end].tolist()

    print(f"\nBaseline (calm reference): "
          f"{dates[baseline_start]} → {dates[baseline_end - 1]} "
          f"({len(baseline)} sessions)")
    print(f"Signal   (window analysed): "
          f"{dates[signal_start]} → {dates[signal_end - 1]} "
          f"({len(signal)} sessions)")

    print("\nCalling client.compare(...) on the public alphainfo API...")
    client = AlphaInfo(api_key=api_key)
    result = client.compare(
        signal=signal,
        baseline=baseline,
        sampling_rate=1.0,        # 1 sample per trading day
        domain="finance",         # vertical calibration for fat-tailed returns
    )

    print(f"\n{'─' * 60}")
    print(f"  structural_score:  {result.structural_score:.4f}")
    print(f"  change_score:      {result.change_score:.4f}")
    print(f"  confidence_band:   {result.confidence_band}")
    print(f"  change_detected:   {result.change_detected}")
    print(f"  domain_applied:    {result.domain_applied}")
    print(f"  analysis_id:       {result.analysis_id}")
    print(f"{'─' * 60}\n")

    if result.confidence_band == "unstable":
        print("✓ The engine reports the Q1 2020 window as STRUCTURALLY DIVERGENT")
        print("  from the calm Q4 2019 baseline — consistent with the COVID-19")
        print("  regime change.")
    elif result.confidence_band == "transition":
        print("⚠ The engine reports a TRANSITION — the signal is moving away")
        print("  from baseline structure but hasn't fully diverged.")
    else:
        print("Note: result reports STABLE — unexpected for this window. "
              "Worth re-checking the date ranges your network returned.")

    print(
        "\nReplay this exact analysis any time with:\n"
        f"    client.audit_replay({result.analysis_id!r})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
