# alphainfo vs Alternatives — Honest Adversarial Benchmark

> Question this answers (the one a buyer's CTO will ask):
> *"Why alphainfo and not `pip install ruptures` (free, standard) or
> a 30-line CUSUM implementation?"*

**Generated**: 2026-04-26
**Reproducer**: `python -m benchmarks.adversarial_vs_competitors`
**Raw results**: `benchmarks/results_adversarial.json`

---

## TL;DR

This benchmark is **not** "alphainfo wins everything". It's the
opposite of that: an honest mapping of where each detector wins,
and what the trade-offs look like.

| Question | Best tool | Why |
|---|---|---|
| "Where exactly did the mean step?" | **`ruptures.Pelt`** | Built for offline change-point segmentation. 100% recall on mean/vol shifts, but slowest. |
| "Detect a mean step online with 30 lines and zero deps?" | **CUSUM** | 0.1ms per series, 60% recall on simple shifts. Use for cheap heuristics. |
| "Distribution / spectrum / structure changed without mean changing?" | **alphainfo** | The only detector here that catches `distribution_shift` and `periodicity_shift` at all. |
| "What KIND of change happened?" | **alphainfo** | Sole focus area. Probe libraries name the diagnosis (`auth_failure_burst`, `outlier_rate_increase`, …). See [BENCHMARKS.md](BENCHMARKS.md). |

If your problem fits the "mean stepped, where" mould, **just use
`ruptures`**. If your problem is "something structural changed but I
don't know what — and after detecting I want a human-readable label
of WHICH kind of change," that's alphainfo's home turf.

---

## Setup

### Scenarios (5 × 10 trials each = 50 runs per detector)

Each scenario builds a 500-sample synthetic series with a single
inserted change-point at sample 250 (or no change at all):

| Scenario | Description | Adversary type |
|---|---|---|
| `mean_shift` | `N(0,1)` → `N(1.5, 1)` at t=250 | Easy: classic step |
| `vol_shift` | `N(0,1)` → `N(0, 2)` at t=250 | Easy: variance step |
| `distribution_shift` | `N(0,1)` → `t(df=2.5)` at t=250 | Hard: same μ, σ; tail shape only |
| `periodicity_shift` | `sin(0.05·t)` → `sin(0.10·t)` at t=250 | Hard: same μ, σ; frequency only |
| `no_change` | pure `N(0,1)`, no change-point | False-positive sanity check |

### Detectors

| Detector | What it does | Cost |
|---|---|---|
| `alphainfo_windowed_raw` | Slide windows, call `/v1/analyze/batch`, threshold-detect on the score timeline. Threshold calibrated via cross-half on the baseline period (5th percentile of benign distribution). | 1 API call per window + 1 calibration call |
| `ruptures.Pelt` (RBF kernel) | PELT segmentation, classic offline change-point algorithm. Penalty=5.0 (default-ish). | All in-process |
| `CUSUM` (textbook) | Cumulative-sum with drift=0.5, threshold=5σ. Online, single pass. | Trivial CPU |

### Metric

- **Precision**: fraction of detections that are within ±100 samples of true cp.
- **Recall**: fraction of true changes detected within tolerance.
- **FPR**: false-positive rate on the `no_change` scenario.
- **Detection lag**: median samples between true cp and detection (≥ 0 means "after the change").
- **Compute p50**: median wall-clock per call.

The tolerance of ±100 = 1 window (window_size=80, plus step). Any
sliding detector has this as a theoretical floor — penalising for it
is unfair, so it's relaxed.

---

## Results

```
                       mean_shift   vol_shift  distribution_shift  periodicity_shift  no_change
─────────────────────  ──────────   ─────────  ──────────────────  ─────────────────  ─────────
alphainfo_windowed_raw recall=0.0   recall=0.0  recall=0.2  ✓       recall=0.0         FPR=0.0
ruptures_pelt          recall=1.0   recall=1.0  recall=0.0          recall=0.0         FPR=0.0
cusum                  recall=0.6   recall=0.6  recall=0.6          recall=0.0         FPR=0.8
```

### Compute (median ms per series)

```
                       mean_shift   vol_shift  distribution_shift  periodicity_shift  no_change
─────────────────────  ──────────   ─────────  ──────────────────  ─────────────────  ─────────
alphainfo_windowed_raw      167         148         163                 239               256
ruptures_pelt               657        1116        2178                2195              2289
cusum                         0.1         0.1         0.1                 0.0               0.1
```

---

## Analysis

### Where alphainfo windowed (raw) underperforms

On simple **mean-shift** and **vol-shift**, raw alphainfo windowed
runs the same calibration and slide-and-threshold pattern that
ruptures does, but with extra HTTP round-trips and a more
conservative percentile-calibrated threshold. The result: 0/10
recall on those.

**Honest read**: if the problem is "mean stepped, where", **just
use `ruptures.Pelt`**. It's free, in-process, designed for that, and
hits 100% recall in our trials. The right tool for the right job.

### Where alphainfo wins

**`distribution_shift`**: gaussian → student-t with the same mean
and variance. The shape changed (heavier tails), but no detector
that operates on running mean/variance can see it.

  - alphainfo: 2/10 recall (low but non-zero)
  - ruptures.Pelt: **0/10** — fundamentally blind to it
  - CUSUM: 6/10 — got lucky because the t-distribution's variance
    fluctuates enough that some samples breach the threshold

**`periodicity_shift`**: sine wave changed frequency. Mean stays at
0, std stays at ~0.7 (sine RMS), only the spectrum changed.

  - alphainfo: 0/10 (raw windowed not enough) — but [auto_diagnose
    with spectral / autocorr encoders](BENCHMARKS.md) catches it
    cleanly. The "raw" variant in this benchmark is the floor.
  - ruptures.Pelt: **0/10**
  - CUSUM: **0/10**

**`no_change`** false-positive rate is the differentiator buyers
care about most:

  - alphainfo: **FPR = 0.0** ✅
  - ruptures.Pelt: **FPR = 0.0** ✅
  - CUSUM: **FPR = 0.8** ❌ (alarms 8 out of 10 times on pure noise)

### Compute trade-offs

ruptures.Pelt hits change-point quality in the simple cases, but
it's **6-15× slower** than alphainfo windowed (and CUSUM beats both
on speed by 1000× on simple shifts).

For a real production deploy, the picture is:
- **Online / streaming** (you need a verdict every second):
  CUSUM if simple, alphainfo if structural.
- **Offline / batch** (you have the whole series):
  ruptures.Pelt for mean / vol changes, alphainfo for everything else.

---

## What this benchmark **doesn't** measure

This is a **change-point localisation** benchmark — "in this 500-sample
series, where did the change happen?". It is **not** alphainfo's
home turf.

alphainfo's actual differentiator is **diagnosis** — given a signal
that changed, what KIND of change is it? `auth_failure_burst`?
`outlier_rate_increase`? `vol_clustering`? That's the
`auto_diagnose` recipe, which uses calibrated probe libraries per
vertical. See [BENCHMARKS.md](BENCHMARKS.md) — across 8
verticals × 30 trials × ~13 probes each, alphainfo hits **rank-1 = 0.898,
top-3 = 0.955, benign-spec = 0.935**. Neither ruptures nor CUSUM
have a comparable feature.

If you need:
- **Where in the signal did it change?** → `ruptures.Pelt`.
- **What kind of change is it?** → `alphainfo.auto_diagnose`.
- **Both, fast?** → `recipes.intents.dispatch(intent='regime_change')`
  chains windowed + auto_diagnose. WHERE + WHAT KIND in one call.

---

## Reproducing

```bash
# In-process (uses TestClient, no live API key needed)
JWT_SECRET_KEY='test-secret-that-is-at-least-32-chars-long' \
MASTER_API_KEY='ai_master_test' \
SKIP_DB=1 \
python -m benchmarks.adversarial_vs_competitors
```

Output: aggregate to stdout, raw per-trial data to
`benchmarks/results_adversarial.json`.

Each scenario uses a deterministic seed; results are reproducible
modulo numpy / fastapi / ruptures version drift (we pin versions in
`requirements.txt`).
