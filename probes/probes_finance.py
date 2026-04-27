"""
recipes/probes_finance.py — Finance-domain probe library.

The default probes in `recipes.auto_diagnose.DEFAULT_PROBES` are
general-purpose: amplitude_2x, regime_break, drift, etc. They work
for synthetic benchmarks but their fingerprints don't always map
cleanly to *what actually happens* in market data. A "regime break"
in pricing series is more specifically a volatility-regime shift,
or a trend reversal, or a fat-tail event.

This module defines probes calibrated to real market regime changes,
documented from observed structural shifts in SPY / BTC / FX / VIX
during known events (COVID crash, 2008 crisis, dot-com, taper
tantrum). Each probe is a deterministic transform of a baseline
returns series — given the same input + seed, it produces the
same perturbed output.

When to use these instead of DEFAULT_PROBES
-------------------------------------------

Pass `probes=FINANCE_PROBES` (or a subset) to
``recipes.auto_diagnose.auto_diagnose`` when:

  * Your data is returns / log-returns / price changes.
  * You want diagnoses that map to interpretable market events.
  * You want benign-controls calibrated to "ordinary trading-day
    noise" rather than generic gaussian noise.

Probe design notes
------------------

These probes operate on RETURNS, not on prices. Convert prices to
returns first via `np.diff(np.log(prices))` or `prices.pct_change()`.
The structural engine compares structure, not level — running the
probe library directly on prices triggers `probe_offset_shift` for
every steady-uptrend asset, which is uninformative.

For the same reason, several probes (skew_inversion, fat_tail) work
on *the distribution shape of the returns*, not on the path. They're
most useful when paired with a `feature_ensemble` configured to
include `histogram` and `spectrum` channels.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from recipes.auto_diagnose import Probe


# ─────────────────────────────────────────────────────────────────────────────
# Volatility-regime probes — most common finance regime change
# ─────────────────────────────────────────────────────────────────────────────


def probe_vol_regime_2x(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Realized-vol doubles in the second half. Models a transition
    from a calm market to a stressed one (e.g. Q4 2019 → Q1 2020 SPY)."""
    r = returns.copy()
    n = len(r)
    mid = n // 2
    r[mid:] = r[mid:] * 2.0
    return r


def probe_vol_regime_05x(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Realized-vol halves in the second half. Models a vol-crush — Fed
    intervention, post-crisis stabilisation, or a low-VIX regime onset."""
    r = returns.copy()
    n = len(r)
    mid = n // 2
    r[mid:] = r[mid:] * 0.5
    return r


def probe_vol_clustering(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject GARCH-like volatility clustering into the second half — a
    short burst of high-vol days followed by mean-reversion. Differs
    from vol_regime_2x in the burstiness."""
    r = returns.copy()
    n = len(r)
    mid = n // 2
    sigma = float(np.std(returns[:mid]))
    burst_len = max(3, n // 20)
    burst_start = mid + n // 8
    burst_end = min(burst_start + burst_len, n)
    r[burst_start:burst_end] = r[burst_start:burst_end] + 3.0 * sigma * rng.standard_normal(burst_end - burst_start)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Trend / drift probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_trend_reversal(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip the sign of returns in the second half. An uptrend becomes
    a downtrend without changing volatility. Models a market top or
    sentiment flip."""
    r = returns.copy()
    n = len(r)
    mid = n // 2
    r[mid:] = -r[mid:]
    return r


def probe_drift_to_zero(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mean drift collapses to zero in the second half — a trending
    asset becomes range-bound. Volatility unchanged."""
    r = returns.copy()
    n = len(r)
    mid = n // 2
    second_half_mean = float(np.mean(r[mid:]))
    r[mid:] = r[mid:] - second_half_mean
    return r


def probe_drift_amplified(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mean drift triples in the second half — momentum acceleration.
    Volatility unchanged."""
    r = returns.copy()
    n = len(r)
    mid = n // 2
    extra_drift = 2.0 * float(np.mean(r[:mid]))
    r[mid:] = r[mid:] + extra_drift
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Tail / event probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_crash_event(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a single -5σ down-move. Models a flash crash, single-day
    capitulation, or a circuit-breaker event."""
    r = returns.copy()
    n = len(r)
    pos = int(rng.integers(n // 3, 3 * n // 4))
    sigma = float(np.std(returns))
    r[pos] = r[pos] - 5.0 * sigma
    return r


def probe_recovery_pop(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject a single +5σ up-move. Models a sharp relief rally,
    short-squeeze, or stimulus announcement bounce."""
    r = returns.copy()
    n = len(r)
    pos = int(rng.integers(n // 3, 3 * n // 4))
    sigma = float(np.std(returns))
    r[pos] = r[pos] + 5.0 * sigma
    return r


def probe_fat_tail(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace the second half with t-distribution noise (df=3) at
    matched mean and a slightly inflated scale. Captures the
    'kurtosis spike' that real crashes show — distribution shape
    changes without much path drama on average days."""
    r = returns.copy()
    n = len(r)
    mid = n // 2
    sigma = float(np.std(returns[:mid]))
    mu = float(np.mean(returns[:mid]))
    # Student-t with df=3: heavy tails. Scale to matched-ish std.
    t_samples = rng.standard_t(df=3.0, size=n - mid) * sigma * 0.6
    r[mid:] = mu + t_samples
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Microstructure / autocorrelation probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_momentum_emerge(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject lag-1 positive autocorrelation (rho=0.5) in the second
    half. Models a transition from random walk to momentum regime
    — common during prolonged trends."""
    r = returns.copy()
    n = len(r)
    mid = n // 2
    rho = 0.5
    for i in range(mid + 1, n):
        r[i] = rho * r[i - 1] + np.sqrt(1 - rho * rho) * r[i]
    return r


def probe_mean_reversion_emerge(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inject lag-1 NEGATIVE autocorrelation (rho=-0.5) in the second
    half. Models a transition to mean-reverting regime — common after
    a vol shock or in choppy sideways markets."""
    r = returns.copy()
    n = len(r)
    mid = n // 2
    rho = -0.5
    for i in range(mid + 1, n):
        r[i] = rho * r[i - 1] + np.sqrt(1 - rho * rho) * r[i]
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Benign controls — what "no real regime change" looks like
# ─────────────────────────────────────────────────────────────────────────────


def control_within_session_noise(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add tiny independent noise at 5% of historical std. The kind of
    day-to-day variation that should NOT trigger a regime alarm."""
    sigma = float(np.std(returns))
    return returns + 0.05 * sigma * rng.standard_normal(len(returns))


def control_resample_half(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample (with replacement) from the empirical distribution of
    the FIRST half — same regime statistics, different sample path.
    Should look indistinguishable structurally."""
    n = len(returns)
    mid = n // 2
    out = returns.copy()
    out[mid:] = rng.choice(returns[:mid], size=n - mid, replace=True)
    return out


def control_minor_drift(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a barely-perceptible drift of 0.1σ over the full series."""
    n = len(returns)
    sigma = float(np.std(returns))
    drift = np.linspace(0.0, 0.1 * sigma, n)
    return returns + drift


# ─────────────────────────────────────────────────────────────────────────────
# Public registry
# ─────────────────────────────────────────────────────────────────────────────


FINANCE_PROBES: Dict[str, Probe] = {
    "vol_regime_2x":         probe_vol_regime_2x,
    "vol_regime_05x":        probe_vol_regime_05x,
    "vol_clustering":        probe_vol_clustering,
    "trend_reversal":        probe_trend_reversal,
    "drift_to_zero":         probe_drift_to_zero,
    "drift_amplified":       probe_drift_amplified,
    "crash_event":           probe_crash_event,
    "recovery_pop":          probe_recovery_pop,
    "fat_tail":              probe_fat_tail,
    "momentum_emerge":       probe_momentum_emerge,
    "mean_reversion_emerge": probe_mean_reversion_emerge,
}


FINANCE_BENIGN_CONTROLS: Dict[str, Probe] = {
    "within_session_noise": control_within_session_noise,
    "resample_half":        control_resample_half,
    "minor_drift":          control_minor_drift,
}


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def _demo():
    """Run auto_diagnose with the finance probe set against three
    synthetic-but-realistic transformations of an SPY-like return
    series. Each diagnosis should map to the transformation we applied."""
    import os
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long")
    os.environ.setdefault("MASTER_API_KEY", "ai_master_test")

    from recipes.auto_diagnose import auto_diagnose

    rng = np.random.default_rng(0)
    n = 500
    # Baseline: gaussian returns at 1% daily vol, slight positive drift.
    baseline = 0.0005 + 0.01 * rng.standard_normal(n)

    # Three test cases, each is the baseline transformed by ONE finance probe.
    test_cases = [
        ("vol doubled, second half",  probe_vol_regime_2x(baseline, rng)),
        ("trend reversed",            probe_trend_reversal(baseline, rng)),
        ("crash event injected",      probe_crash_event(baseline, rng)),
        ("just resampled (benign)",   control_resample_half(baseline, rng)),
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
                "channels": channels,
                "sampling_rate": sampling_rate,
                "domain": domain,
                "use_multiscale": use_multiscale,
                "include_semantic": include_semantic,
            }
            if baselines is not None:
                payload["baselines"] = baselines
            r = self.tc.post(
                "/v1/analyze/vector",
                headers={"X-API-Key": self.key},
                json=payload,
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
            domain="finance",
            probes=FINANCE_PROBES,
            benign_controls=FINANCE_BENIGN_CONTROLS,
        )
        print(f"\nground truth: {label}")
        print(f"  diagnosis: {out['diagnosis']:25}  confidence={out['confidence']:.3f}")
        print(f"  benign sim: {out['benign_similarity']:.3f}")
        print("  top-3 probes:")
        for name, score in out["ranked"][:3]:
            print(f"    {name:25}  {score:.3f}")


if __name__ == "__main__":
    _demo()
