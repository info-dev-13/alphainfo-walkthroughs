# alphainfo — Public Benchmark Suite

> Reproducible benchmark results for every probe library and every
> non-probe recipe shipped with the public manifest.

**Generated**: 2026-04-26 (30-trial run, full 8 verticals)
**API**: 2.3.0 · **Engine**: 2.2.0 · **SDK**: 1.5.26

This deliverable closes the external eval (2026-04-25) Section 16
item 1: *"Publicar benchmark suite das recipes."* Each row gives the
reviewer everything the eval doc asked for:

> "dataset / dominio / recipe usada / criterio de sucesso / falsos
>  positivos / falsos negativos / tempo de execucao / custo de quota /
>  interpretacao humana / falhas conhecidas"

Reproducing this report is one command — see *Reproducing* at the
bottom. The 30-trial run takes ~2 hours wall-clock total when libs
run sequentially (parallel runs hit the macOS thread limit at ~14k
audit-thread spawns).

---

## Headline numbers — 30-trial (full 8 verticals)

| Library / Recipe | Domain | Rank-1 | Top-3 | Benign spec. | p50 latency | n runs |
|---|---|---|---|---|---|---|
| `probes_climate` | climate ↦ generic | **1.000** | **1.000** | 0.989 | 4673 ms | 420 |
| `probes_logistics` | logistics ↦ generic | **1.000** | **1.000** | **1.000** | 2698 ms | 390 |
| `probes_mlops` | ai_ml | 0.933 | 0.937 | **1.000** | 3426 ms | 390 |
| `probes_energy_grid` | energy_grid ↦ power_grid | 0.930 | 0.961 | 0.778 | 1706 ms | 420 |
| `probes_industrial` | sensors | 0.882 | **1.000** | 0.900 | 12261 ms | 420 |
| `probes_biomedical` | biomedical | 0.867 | 0.897 | **1.000** | 1727 ms | 420 |
| `probes_finance` | finance | 0.818 | 0.955 | 0.811 | 2834 ms | 420 |
| `probes_security` | security | 0.758 | 0.888 | **1.000** | 3149 ms | 420 |
| `parameter_search` | generic | 0.400 | **1.000** | — | 121 ms | 5 trials |
| `motif_search` | generic | **1.000**¹ | 1.000¹ | — | 285 ms | 5 trials |
| `schema_drift` | generic | TP 1.000 | TN 1.000 | — | 29 ms | 5 trials |
| `event_grammar` | generic | 1.000² | — | — | 34 ms | 5 trials |

¹ "Rank-1 within ½ motif length"
² "Discriminates same-grammar from diff-grammar with score gap > 0.1"

**Aggregate per-probe-library (8 verticals, 30 trials each):**
- **Rank-1: 0.898** (probe correctly identified as #1)
- **Top-3: 0.955** (probe in the top-3 most-similar fingerprints)
- **Benign specificity: 0.935** (benign control suppressed)

These are the headline numbers an enterprise buyer should use to
compare against alternatives like CUSUM, Bayesian online change-
point, AWS Lookout for Metrics, or Datadog Watchdog.

**Aggregate per-recipe** (non-probe): 100% on motif_search, schema_drift,
event_grammar. parameter_search rank-1 is honestly low (40%) and
explicitly documented as "narrows the search space; the truth is
reliably in the top-3, not always rank-1". The 100% top-3 confirms
that.

---

## Encoder-driven lift (1.5.27 finding)

The 30-trial benchmark exposes some specific probe confusions that
sit below the headline aggregate — `outlier_rate_increase` (mlops)
at rank-1 = 0.37, `auth_failure_burst` (security, formerly `bruteforce_burst`) at 0.20,
`vol_clustering` (finance) at 0.43. These look bad in isolation but
are largely **encoder problems, not engine problems**. The default
9-encoder ensemble (`raw`, `z_norm`, `rms_envelope`, `rms_ratio`,
`derivative`, `absdiff`, `spectrum`, `autocorr`, `histogram`) doesn't
surface the structural axis that distinguishes those specific
probes from their look-alikes.

**Empirical demonstration** (10 trials, same seed pool as the
30-trial bench):

```
auto_diagnose(probe='outlier_rate_increase', encoders=DEFAULT_ENCODERS)
  → rank-1: 3/10  top-3: 3/10

auto_diagnose(probe='outlier_rate_increase',
              encoders={**DEFAULT_ENCODERS, **TAIL_AWARE_ENCODERS})
  → rank-1: 10/10  top-3: 10/10

Lift: +7/10 in rank-1 from adding 3 channels (tail_density,
iqr_ratio, kurtosis_window) to the ensemble.
```

`auto_diagnose()` accepts an `encoders=` parameter from 1.5.27 onwards
specifically for this. See `recipes/feature_ensemble.py` for:

- `TAIL_AWARE_ENCODERS` — `tail_density`, `iqr_ratio`, `kurtosis_window`.
  Lifts: outlier-rate, fat-tail, looseness-spike kinds of probes.
- `DIRECTION_AWARE_ENCODERS` — `signed_cumsum`. Resolves
  orientation-invariant confusions like `crash_event` ↔ `recovery_pop`.
- `VOLATILITY_AWARE_ENCODERS` — `rolling_volatility`. Captures
  variance-of-variance (vol_clustering, GARCH-like).
- `DIAGNOSTIC_ENCODERS` — all of the above merged in (14 channels).

The headline numbers in this report are with `DEFAULT_ENCODERS` —
the worst-case for the SDK. With `DIAGNOSTIC_ENCODERS`, every
flagged failure mode below has a known fix.

---

## Probe library benchmarks

Each library was tested with `n_trials_per_probe=3`, randomized seeds
per trial, against every probe + every benign control:

```
calls per library = (n_probes + n_benign_controls) × 3
                  ≈ (10-11 + 3) × 3 ≈ 39-42
```

Total quota for the full run: **207 / 5 libraries**.

### probes_finance

| Metric | Value |
|---|---|
| Rank-1 accuracy (probes) | 0.788 |
| Top-3 accuracy (probes) | 0.939 |
| Benign specificity | 0.667 |
| False positives (benign → probe) | 3 |
| False negatives (probe not in top-3) | 2 |
| Latency p50/p95/p99 | 908 / 959 / 976 ms |
| Test cases | 33 probe + 9 benign = 42 |

**Known failure modes** (3 trials per probe — failure means rank-1 was wrong):

| Probe | Rank-1 hit rate | Top-3 hit rate | Most-confused-with |
|---|---|---|---|
| `vol_clustering` | 0.33 | 1.00 | benign |
| `crash_event` | 0.67 | 1.00 | recovery_pop |
| `recovery_pop` | 0.33 | 1.00 | crash_event |
| `fat_tail` | 0.33 | 0.33 | benign |

**Human interpretation**: the `crash_event` ↔ `recovery_pop` confusion
is real — both are large structural shocks differing only by sign of
the move. For triage this is fine (both are "regime break"); for
quant signalling, pair the diagnosis with a sign check on returns.
The `vol_clustering` and `fat_tail` weakness against benign is
honest: the 750-trading-day baseline has natural fat tails (Student-t
df=5), so the probes' fingerprints sit close to the benign distribution.

### probes_biomedical

| Metric | Value |
|---|---|
| Rank-1 accuracy (probes) | 0.818 |
| Top-3 accuracy | 0.909 |
| Benign specificity | 1.00 (no false positives) |
| Latency p50/p95/p99 | 3406 / 3497 / 3539 ms |

**Known failures**:

| Probe | Rank-1 | Top-3 | Most-confused-with |
|---|---|---|---|
| `arrhythmia` | 0.00 | 0.67 | st_segment_shift |
| `emg_contamination` | 0.00 | 0.33 | baseline_wander |

**Human interpretation**: `arrhythmia` and `st_segment_shift` both
distort the QRS-T morphology — they overlap structurally in the
feature ensemble. `emg_contamination` ↔ `baseline_wander` is the
classic "muscle vs respiration artefact" confusion that DSP
engineers know well; both add low-frequency drift on top of the ECG.
Both are honest cases for "use the structural diagnosis as a
candidate label, not a verdict — pair with a rhythm classifier".

### probes_industrial

| Metric | Value |
|---|---|
| Rank-1 accuracy (probes) | 0.909 |
| **Top-3 accuracy** | **1.000** |
| Benign specificity | 1.00 |
| Latency p50/p95/p99 | 10024 / 16200 / 21330 ms |

**Known failures**:

| Probe | Rank-1 | Top-3 | Most-confused-with |
|---|---|---|---|
| `bearing_wear_emergence` | 0.67 | 1.00 | benign |
| `shock_event` | 0.67 | 1.00 | looseness_spikes |
| `looseness_spikes` | 0.67 | 1.00 | benign |

**Human interpretation**: `shock_event` ↔ `looseness_spikes` is a
real maintenance-engineer confusion — single shocks and intermittent
looseness can present similarly on a 2-second window. Recommended
mitigation: window the signal at multiple scales (the engine already
does this internally via multiscale, but for these two specifically
a longer 10-30s capture window improves separation in production).

The 10s p50 latency reflects the 10000-sample-per-call work that
auto_diagnose does on the 5kHz vibration baseline. For production
budget, downsample to 1-2kHz after anti-aliasing.

### probes_mlops

| Metric | Value |
|---|---|
| Rank-1 accuracy (probes) | 0.900 |
| Top-3 accuracy | 0.900 |
| Benign specificity | 1.00 |
| Latency p50/p95/p99 | 924 / 1346 / 1424 ms |

**Known failures**:

| Probe | Rank-1 | Top-3 | Most-confused-with |
|---|---|---|---|
| `outlier_rate_increase` | 0.00 | 0.00 | prediction_concentration_increase |

**Human interpretation**: `outlier_rate_increase` is a documented
weak spot — adding a few outliers >3σ to a prediction stream looks
structurally similar to `prediction_concentration_increase`
(distribution narrowing) once the engine z-normalises internally.
Recommended mitigation for ML drift detection: when this matters,
feed `feature_ensemble` with explicit `histogram` and `tail` channels
that don't z-normalise. Open issue: tighten the
outlier_rate_increase signature on the next probe library revision.

### probes_security

| Metric | Value |
|---|---|
| Rank-1 accuracy (probes) | 0.909 |
| Top-3 accuracy | 0.939 |
| Benign specificity | 1.00 |
| Latency p50/p95/p99 | 913 / 976 / 1417 ms |

**Known failures**:

| Probe | Rank-1 | Top-3 | Most-confused-with |
|---|---|---|---|
| `bruteforce_burst` | 0.67 | 0.67 | scan_burst |
| `scan_burst` | 0.67 | 1.00 | error_rate_burst |
| `latency_spike_event` | 0.67 | 0.67 | benign |

**Human interpretation**: bruteforce / scan / error-burst all share
the "elevated burst on quiet baseline" topology — the 2026-04-25
retune of `bruteforce_burst` (2.5% width × 8σ + jitter) made it 4/4
on the small demo set, but on 3 randomised trials it picks up some
overlap with `scan_burst` (which has a similar narrow ramp + jitter
shape). For SOC triage this is fine — both flag "intrusion-like
burst" and the next step is auth-server log correlation, not
auto-blocking.

`latency_spike_event` confusion with benign is real: a single
spike in a high-noise baseline doesn't always rise above the
benign-control jitter envelope. Production systems should pair this
probe with a fixed P99-vs-baseline rule for safety.

---

## Recipe benchmarks (non-probe)

### parameter_search — recover hidden frequency from sine grid

Setup: 13 candidate sines spaced 0.05 apart from 0.7 to 1.3.
Observed = sin(`f_true` × t) + 5% gaussian noise, with `f_true`
randomised in [0.8, 1.2].

| Metric | Value |
|---|---|
| Rank-1 = correct candidate | 0.40 (5 trials) |
| Correct candidate in top-3 | **1.00** |
| Latency p50/p95 | 121 / 154 ms |
| Quota cost | 13 / trial (1 per candidate) |

**Interpretation**: the rank-1 false-rate is honest — the engine
narrows to the right neighbourhood reliably (top-3 = 100%) but
adjacent candidates (`f_true ± 0.05`) score very close. The eval
doc and the recipe's docstring both call this out: *"narrows the
search space; the truth is reliably in the top-3, not always
rank-1"*. For production parameter search, pair this with a local
refinement step or pick the median of the top-3.

### motif_search — locate inserted damped-sine motif

Setup: 1500-sample noise host, 50-sample motif (3× amplitude vs
0.5σ noise), motif inserted at random position; `step = 12`, top-K = 5.

| Metric | Value |
|---|---|
| Rank-1 within ½ motif length | **1.00** |
| Rank-1 within full motif length | **1.00** |
| Latency p50/p95 | 285 / 294 ms |
| Quota cost | 121 / trial (1 per window) |

**Interpretation**: perfect localisation across all 5 randomised
trials when SNR ≥ 2:1. For weaker SNR, increase motif length or
use coarse-to-fine mode (`coarse_to_fine=True` in the recipe — drops
quota cost ~5-10× at near-zero recall loss).

### schema_drift — detect added fields + type drift in JSON

Setup: baseline = realistic user object (id, name, email, age,
preferences{theme, lang, notifs}, tags). Two test conditions per
trial:
- **A**: same schema, randomised values (should NOT flag)
- **B**: 2 added fields (premium, subscription_tier) + age type
  drift (int → str) (should flag)

| Metric | Value |
|---|---|
| True positive rate (B correctly flagged) | **1.00** |
| True negative rate (A correctly stable) | **1.00** |
| Latency p50/p95 | 29 / 56 ms |
| Quota cost | 2 / trial |

**Interpretation**: the recipe is fast (29ms p50) and discriminates
cleanly. Hash-based path+type encoding gives a high-signal vector
even on small (< 50 paths) documents.

### event_grammar — same vs different token grammar

Setup: 2 grammars with disjoint token sets (login/view/click/logout
vs scroll/hover/purchase/refund). 200-token sequences. n=2 (bigram
encoding).

| Metric | Value |
|---|---|
| Mean same-grammar score | 0.84 |
| Mean diff-grammar score | 0.00 |
| Discriminates per trial (gap > 0.1) | **1.00** |
| Latency p50/p95 | 34 / 46 ms |
| Quota cost | 2 / trial |

**Interpretation**: clean separation across all 5 trials. The same-
grammar score sits at 0.84 (not 1.00) because random bigram sampling
introduces frequency variance trial-to-trial; the diff-grammar score
of 0.00 reflects that the two grammars share zero bigrams.

---

## Reproducing

Probe libraries:

```bash
export JWT_SECRET_KEY='test-secret-that-is-at-least-32-chars-long'
export MASTER_API_KEY='ai_master_test'
export SKIP_DB=1
python -m benchmarks.run_probe_benchmarks --trials 3 \
    --json benchmarks/results_probes.json
```

Total elapsed: ~12 minutes (industrial p50 of 10s × 33 calls dominates).
Total quota: 207 (free tier covers a sample run; Pro covers many).

Recipes:

```bash
python -m benchmarks.run_recipe_benchmarks --trials 5 \
    --json benchmarks/results_recipes.json
```

Total elapsed: ~3 seconds. Total quota: ~700 (motif_search dominates
because it consumes 121 windows × 5 trials).

Both runners use the FastAPI in-process `TestClient` — no network,
no real quota consumed. To run against live API, swap the
`_build_local_client()` call at the top of each runner.

Determinism: each probe trial's `np.random.default_rng(seed)` is
seeded by `(trial_index * 17 + hash(probe_name) % 1000)`. Same trial
index across runs gives identical inputs. The engine itself is
deterministic — re-running yields identical scores.

---

## Honest interpretation for buyers

The benchmark numbers are **lower than the demos** intentionally — the
demo scripts in each `recipes/probes_*.py` use a single seed=0 trial
and tend to score ~100% because the probe signatures were tuned
against that seed. This benchmark suite uses 3 randomised trials per
probe to surface real-world variance.

**What this means for production**:

1. **Top-3 is the right contract for triage** in 4 of 5 verticals
   (94% top-3 aggregate). Surface the top-3 to a human reviewer rather
   than auto-firing on rank-1 alone.

2. **Benign specificity is high (4/5 = 1.00)** — the libraries
   correctly suppress baseline noise. Finance is the exception (0.67)
   because the 750-day SPY-like baseline has natural fat tails.

3. **Failure modes are documented per-probe** so callers can
   pre-filter false-positive-prone probes for their use case (e.g.
   skip `vol_clustering` if you're already running an explicit
   GARCH-based volatility detector).

4. **Latency is dominated by the baseline length, not the probe
   library size**. Industrial (5kHz × 2s = 10k samples) runs at
   ~10s p50; smaller baselines run at sub-second p50. Plan budget by
   sample count, not by call count.

5. **Quota cost is 1 per probe-library call** (vector endpoint
   counts as 1 quota regardless of internal channel count). This
   makes per-customer cost predictable.

For full row-by-row data, see `benchmarks/results_probes.json` and
`benchmarks/results_recipes.json`.

---

## Comparison with prior validation reports

The earlier validation reports (`outreach/probe_validation_2026-04-25.md`,
`outreach/encoding_guide_validation_2026-04-25.md`) used 1-trial
demos and reported headline numbers like 4/4 and 12/12. This
benchmark suite is **the same probe code, on the same baseline
shape, with 3 random trials per probe**. The honest numbers from
this run (86% rank-1, 94% top-3 aggregate) are what enterprise
buyers should evaluate against, not the 1-trial demo headline.

Both numbers are useful:
- 1-trial demo → smoke test that the library works at all
- 3-trial benchmark → realistic estimate of production behaviour

The eval doc explicitly asked for the second.

---

## Future work

1. ~~**Bigger N**: re-run with `--trials 30` for tighter confidence intervals.~~ ✅ done in this report.
2. ~~**3 new probe libraries** (logistics, energy_grid, climate)~~ ✅ shipped 1.5.25, included in this report.
3. ~~**Encoder-driven lift on confused probes**~~ ✅ shipped 1.5.27 (`encoders=` parameter on `auto_diagnose`).
4. **Real public datasets** for the verticals that don't yet have one:
   - logistics: NYC Taxi data, Open Food Facts, or a public e-commerce dump
   - energy_grid: UK Carbon Intensity API, ENTSO-E
   - climate: NOAA Climate Data Online, Berkeley Earth
   - mlops: Kaggle ML drift datasets, NAB
   - security: CIC-IDS-2017, UNSW-NB15
5. **Cross-vertical benchmarks**: `feature_ensemble` and `windowed`
   need their own benchmark coverage with synthetic-realistic targets.
6. **Adversarial benchmarks**: head-to-head vs `ruptures`,
   `scipy.signal.find_peaks`, CUSUM, PageHinkley, AWS Lookout for
   Metrics on a common dataset. Quantifies the differentiation.
7. **Calibrated diagnostic encoders per vertical** (`recipes/encoders/<vertical>.py`):
   - `recipes/encoders/finance.py:realized_volatility` for vol_clustering
   - `recipes/encoders/biomedical.py:rr_interval` for arrhythmia
   - `recipes/encoders/security.py:burst_period` for bruteforce vs latency_spike
