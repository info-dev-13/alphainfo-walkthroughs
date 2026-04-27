"""
recipes/probes_security.py — Defender-side anomaly probes.

Fifth vertical of the calibrated probe library. Designed for blue-team
use cases: SOC dashboards, on-call paging, observability platforms.
Names describe **what the defender's metric looks like**, not what the
adversary intends. We're a structural-anomaly detector for the
defender's telemetry, not a model of attack tradecraft.

Concrete defender-side telemetry these probes operate on:

  * Per-minute request count (edge / load balancer / WAF).
  * Per-minute failed-auth count (auth service success / failure split).
  * Latency time series (P50 or P99 over a rolling window).
  * Error-rate (5xx fraction) over a rolling window.
  * Distinct-path-cardinality per source (WAF / API gateway).
  * Any SIEM aggregation column (e.g. requests-per-IP, session length).

Probes (defender perspective — what the metric looks like)
----------------------------------------------------------

  * traffic_volume_spike       — sudden request-rate burst.
  * traffic_volume_collapse    — sudden drop (origin failure / blackhole).
  * sustained_baseline_shift   — overall rate moves to a new baseline.
  * distribution_widening      — variance grows (cascading instability).
  * auth_failure_burst         — narrow, dense burst on the failed-auth
                                  metric (the SOC dashboard "wall").
  * slow_auth_failure_pattern  — low-rate sustained elevation on the
                                  failed-auth metric (rate-limit-evasion
                                  shape — easy to miss without a
                                  structural detector).
  * path_enumeration_burst     — ramping request count or distinct-path-
                                  cardinality over a contained window.
  * latency_spike_event        — single window of high latency
                                  (downstream hiccup, slowloris-shaped).
  * latency_creep              — latency growing slowly (memory leak,
                                  thread-pool exhaustion).
  * error_rate_burst           — 5xx rate jumping (deploy regression,
                                  dependency failure).
  * temporal_offhours          — traffic at hours that should be quiet.

Benign controls — "production noise SOC analysts already filter":

  * peak_hours_natural      — normal business-hours load increase.
  * gradual_growth          — slow secular traffic growth (week over week).
  * single_user_mistake     — one user mistyping password 3 times.

Fingerprint disambiguation
--------------------------

The "elevated burst on quiet baseline" probes (``auth_failure_burst``,
``error_rate_burst``, ``temporal_offhours``, ``path_enumeration_burst``)
used to overlap heavily because they were all smooth bumps differing
only by width × amplitude. The library was retuned on 2026-04-25 to
give each a distinct structural signature:

  * ``auth_failure_burst``      — narrow (~2.5%), very high (8σ + jitter),
                                  noisy spike-train character
  * ``temporal_offhours``       — wide  (~10%),    low   (2σ),  smooth
  * ``error_rate_burst``        — wide  (~10%),    mid   (4σ),  smooth
  * ``path_enumeration_burst``  — wide  (~6%),     ramp  + jitter

The width × amplitude × jitter combination produces distinguishable
fingerprints. Real SOC use should still pair the structural diagnosis
with rule-based context (auth-server logs, source IPs, request paths)
before firing automated blocks.

Renamed in 1.5.27 (defender-perspective rename)
-----------------------------------------------

  bruteforce_burst        → auth_failure_burst
  slow_credential_stuff   → slow_auth_failure_pattern
  scan_burst              → path_enumeration_burst

The old function names and the old SECURITY_PROBES_LEGACY dict are
kept as backward-compatible aliases for callers that referenced them
by string key. New code should use the new names.

Important caveat
----------------

These probes are STRUCTURAL FINGERPRINT MATCHES, not security verdicts.
A diagnosis of "bruteforce_burst" doesn't say "block this IP" — it
says "the structural fingerprint matches the synthetic bruteforce
injection". Real SOC use should pair this with:

  * Rule-based triggers (failed-login thresholds, SIEM rules).
  * Reputation feeds (known-bad IPs, ASN reputation).
  * Operator review for novel patterns.

The intended use is to FLAG candidate windows for investigation, not
to fire automated blocks. Used right, it can collapse the signal-to-
noise ratio that SOC teams complain about.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from recipes.auto_diagnose import Probe


# ─────────────────────────────────────────────────────────────────────────────
# Volume probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_traffic_volume_spike(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """5× volume spike sustained over ~15% of the window — DDoS / scraper.
    Distinct from `traffic_volume_collapse` (which scales DOWN over half
    the signal) by both direction and width — the sustained 5× burst
    creates a unique structural fingerprint."""
    v = values.copy().astype(np.float64)
    n = len(v)
    burst_start = int(rng.integers(n // 3, n // 2))
    burst_len = max(30, n // 7)  # ~15% of window
    burst_end = min(n, burst_start + burst_len)
    v[burst_start:burst_end] = v[burst_start:burst_end] * 5.0
    return v


def probe_traffic_volume_collapse(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Volume drops to ~10% in the second half — origin blackhole."""
    v = values.copy().astype(np.float64)
    n = len(v)
    mid = n // 2
    v[mid:] = v[mid:] * 0.10
    return v


def probe_sustained_baseline_shift(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overall rate steps up by 50% in the second half — could be
    a campaign launch or rerouted traffic."""
    v = values.copy().astype(np.float64)
    n = len(v)
    mid = n // 2
    v[mid:] = v[mid:] * 1.5
    return v


def probe_distribution_widening(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Variance widens significantly — destabilising system."""
    v = values.copy().astype(np.float64)
    n = len(v)
    mid = n // 2
    mu = float(np.mean(v[mid:]))
    v[mid:] = mu + (v[mid:] - mu) * 2.5
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Auth / bruteforce probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_auth_failure_burst(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Narrow, very-high, NOISY burst on a failed-auth series.

    SOC-side framing: the defender is looking at a per-minute count of
    failed-login events. A "wall" of authentication failures lands in
    a tight time window — dozens of attempts, dense, with jitter (not
    a smooth elevation). The probe's job is to give that observable
    signal a structural fingerprint:

      width:     ~2% of window (tight time window)
      amplitude: 8σ DC floor PLUS 2σ random jitter (spike-train
                  character, not smooth elevation)

    Distinct from the other "elevated burst" probes:

      auth_failure_burst         →  2% × 8σ + jitter  (this probe)
      temporal_offhours          → 10% × 2σ smooth    (wide, low, smooth)
      error_rate_burst           → 10% × 4σ smooth    (wide, mid, smooth)
      path_enumeration_burst     →  6% × ramp+noise   (wide, ramping)

    Tuned 2026-04-25 after auto_diagnose was matching this against
    temporal_offhours in the demo (different shapes overlapped in
    fingerprint space). The narrower-and-noisier signature gives a
    richer spectral / variance fingerprint that disambiguates.

    What the SOC does with this diagnosis: page the on-call, correlate
    source IPs in the auth log, check rate-limit policy, optionally
    trigger temporary block on the offending CIDR. AlphaInfo names
    the structural anomaly; the SOC playbook decides the response.
    """
    v = values.copy().astype(np.float64)
    n = len(v)
    sigma = float(np.std(values))
    burst_start = int(rng.integers(n // 4, 3 * n // 4))
    burst_len = max(8, n // 40)  # ~2.5% — clearly narrower than the
                                  # 5-10% siblings
    burst_end = min(n, burst_start + burst_len)
    # 8σ DC floor + 2σ noise — looks like a tight "wall" of failed
    # auth events on the defender's metric, not a smooth elevation.
    width = burst_end - burst_start
    v[burst_start:burst_end] = (
        v[burst_start:burst_end]
        + 8.0 * sigma
        + rng.normal(0.0, 2.0 * sigma, width)
    )
    return v


def probe_slow_auth_failure_pattern(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Low-rate sustained elevation on a failed-auth series.

    SOC-side framing: failed-auth count nudges up and stays up — not
    enough to trip a per-minute rate-limit, but enough that the
    cumulative count diverges from baseline over the second half of
    the window. The defender sees a slowly-elevated mean with normal-
    looking variance — easy to miss without a structural detector.

    What SOC does with this: review distinct-source-IP cardinality on
    the failed-auth stream, tighten lockout policy, look for
    coordinated patterns across many IPs.
    """
    v = values.copy().astype(np.float64)
    n = len(v)
    mid = n // 2
    sigma = float(np.std(values))
    v[mid:] = v[mid:] + 0.5 * sigma + 0.2 * sigma * rng.standard_normal(n - mid)
    return v


def probe_path_enumeration_burst(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Methodical traffic burst on a per-second request count or
    distinct-path-cardinality series — ramping with high-frequency
    jitter, characteristic of automated endpoint enumeration.

    SOC-side framing: the defender's metric (requests/sec on the
    edge, or distinct-paths-per-source) ramps up over a contained
    window and the variance is unusually structured. Defender
    response: review WAF / load-balancer logs, look for unusual
    user-agent / referer patterns, optionally rate-limit the source.
    """
    v = values.copy().astype(np.float64)
    n = len(v)
    sigma = float(np.std(values))
    burst_start = int(rng.integers(n // 3, 2 * n // 3))
    burst_len = max(20, n // 25)
    burst_end = min(n, burst_start + burst_len)
    ramp = np.linspace(0, 4 * sigma, burst_end - burst_start)
    v[burst_start:burst_end] = v[burst_start:burst_end] + ramp + 0.6 * sigma * rng.standard_normal(burst_end - burst_start)
    return v


# Backward-compat aliases (old names preserved for SDK 1.5.27- callers).
# Will be removed in a future major bump; meanwhile the new defender-
# centric names are the preferred form.
probe_bruteforce_burst = probe_auth_failure_burst
probe_slow_credential_stuff = probe_slow_auth_failure_pattern
probe_scan_burst = probe_path_enumeration_burst


# ─────────────────────────────────────────────────────────────────────────────
# Latency / error-rate probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_latency_spike_event(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Single window of 3× normal latency."""
    v = values.copy().astype(np.float64)
    n = len(v)
    spike_start = int(rng.integers(n // 4, 3 * n // 4))
    spike_len = max(5, n // 50)
    spike_end = min(n, spike_start + spike_len)
    v[spike_start:spike_end] = v[spike_start:spike_end] * 3.0
    return v


def probe_latency_creep(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slow upward drift — latency creeping over time (mem leak)."""
    n = len(values)
    sigma = float(np.std(values))
    drift = np.linspace(0.0, 1.5 * sigma, n)
    return values.astype(np.float64) + drift


def probe_error_rate_burst(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Error rate jumps in a short window — deploy regression / dependency
    failure. Lasts ~10% of the signal."""
    v = values.copy().astype(np.float64)
    n = len(v)
    sigma = float(np.std(values))
    burst_start = int(rng.integers(n // 4, 3 * n // 4))
    burst_len = max(20, n // 10)
    burst_end = min(n, burst_start + burst_len)
    v[burst_start:burst_end] = v[burst_start:burst_end] + 4.0 * sigma
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Temporal probes
# ─────────────────────────────────────────────────────────────────────────────


def probe_temporal_offhours(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sustained moderate traffic during a quiet window — models
    'someone is logging in at 3 AM, every minute, for an hour'.
    Distinct from bruteforce_burst by being wider (~10% of window)
    and lower amplitude (2σ vs 8σ)."""
    v = values.copy().astype(np.float64)
    n = len(v)
    sigma = float(np.std(values))
    offhours_start = 3 * n // 4
    offhours_len = n // 10  # ~10% — wider than bruteforce
    offhours_end = min(n, offhours_start + offhours_len)
    v[offhours_start:offhours_end] = v[offhours_start:offhours_end] + 2.0 * sigma
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Benign controls — what SOC analysts already filter
# ─────────────────────────────────────────────────────────────────────────────


def control_peak_hours_natural(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Natural business-hours load increase — sinusoidal envelope at
    25% amplitude. Should NOT trigger alarm."""
    v = values.copy().astype(np.float64)
    n = len(v)
    envelope = 1.0 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, n))
    return v * envelope


def control_gradual_growth(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slow secular traffic growth (10% across the period)."""
    n = len(values)
    growth = np.linspace(1.0, 1.10, n)
    return values.astype(np.float64) * growth


def control_single_user_mistake(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One user mistyping password — 3 consecutive failed attempts at
    one timestamp. Should NOT register as a brute force."""
    v = values.copy().astype(np.float64)
    n = len(v)
    sigma = float(np.std(values))
    pos = int(rng.integers(n // 4, 3 * n // 4))
    v[pos:min(n, pos + 3)] = v[pos:min(n, pos + 3)] + 1.0 * sigma
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Public registries
# ─────────────────────────────────────────────────────────────────────────────


SECURITY_PROBES: Dict[str, Probe] = {
    # Volume / distribution anomalies (no auth context)
    "traffic_volume_spike":      probe_traffic_volume_spike,
    "traffic_volume_collapse":   probe_traffic_volume_collapse,
    "sustained_baseline_shift":  probe_sustained_baseline_shift,
    "distribution_widening":     probe_distribution_widening,
    # Defender-side observables on auth/path streams
    # (renamed in 1.5.27 from bruteforce_burst / slow_credential_stuff /
    # scan_burst → defender perspective: what the SOC's metric looks
    # like, not what the adversary intends).
    "auth_failure_burst":        probe_auth_failure_burst,
    "slow_auth_failure_pattern": probe_slow_auth_failure_pattern,
    "path_enumeration_burst":    probe_path_enumeration_burst,
    # Service-side anomalies
    "latency_spike_event":       probe_latency_spike_event,
    "latency_creep":              probe_latency_creep,
    "error_rate_burst":           probe_error_rate_burst,
    "temporal_offhours":          probe_temporal_offhours,
}


# Legacy registry — preserves the pre-1.5.27 names for any caller still
# importing by string key. New code should use SECURITY_PROBES.
SECURITY_PROBES_LEGACY: Dict[str, Probe] = {
    **{k: v for k, v in SECURITY_PROBES.items()
       if k not in ("auth_failure_burst", "slow_auth_failure_pattern", "path_enumeration_burst")},
    "bruteforce_burst":      probe_auth_failure_burst,         # deprecated alias
    "slow_credential_stuff": probe_slow_auth_failure_pattern,  # deprecated alias
    "scan_burst":            probe_path_enumeration_burst,     # deprecated alias
}


SECURITY_BENIGN_CONTROLS: Dict[str, Probe] = {
    "peak_hours_natural":  control_peak_hours_natural,
    "gradual_growth":      control_gradual_growth,
    "single_user_mistake": control_single_user_mistake,
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic security-telemetry generator
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_request_rate(
    n: int = 1000,
    baseline_rate: float = 100.0,
    noise_pct: float = 0.05,
    business_hours_envelope: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic per-minute request-rate time series.

    Defaults: 1000 minutes (~16h) at 100 req/min baseline with 5% noise
    and a sinusoidal business-hours envelope.
    """
    rng = np.random.default_rng(seed)
    base = baseline_rate * (1.0 + noise_pct * rng.standard_normal(n))
    if business_hours_envelope:
        env = 1.0 + 0.20 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, n))
        base = base * env
    return np.clip(base, 0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def _demo():
    """Apply 4 known security events to a baseline traffic stream and
    verify auto_diagnose names the right probe."""
    import os
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret-that-is-at-least-32-chars-long")
    os.environ.setdefault("MASTER_API_KEY", "ai_master_test")

    from recipes.auto_diagnose import auto_diagnose

    rng = np.random.default_rng(0)
    baseline = synthetic_request_rate(n=1000, baseline_rate=100.0, seed=0)

    test_cases = [
        ("DDoS-like volume spike",        probe_traffic_volume_spike(baseline, rng)),
        ("brute-force burst",              probe_bruteforce_burst(baseline, rng)),
        ("origin blackhole",               probe_traffic_volume_collapse(baseline, rng)),
        ("just business-hours peak",      control_peak_hours_natural(baseline, rng)),
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
            from types import SimpleNamespace
            payload = {
                "channels": channels, "sampling_rate": sampling_rate,
                "domain": domain, "use_multiscale": use_multiscale,
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
            domain="security",
            probes=SECURITY_PROBES,
            benign_controls=SECURITY_BENIGN_CONTROLS,
        )
        print(f"\nground truth: {label}")
        print(f"  diagnosis  : {out['diagnosis']:30}  confidence={out['confidence']:.3f}")
        print(f"  benign sim : {out['benign_similarity']:.3f}")
        print(f"  top-3 probes:")
        for name, score in out["ranked"][:3]:
            print(f"    {name:30}  {score:.3f}")


if __name__ == "__main__":
    _demo()
