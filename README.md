# alphainfo walkthroughs

Reference source for the real-data walkthroughs that accompany
[alphainfo's engineering blog](https://www.alphainfo.io/blog).

This repository exists to make the methodology auditable: when a blog
post says "we ran our calibrated probes against UK carbon intensity
and got `peak_shift` confidence 0.966", the corresponding script lives
here for inspection.

## What's in here

```
walkthroughs/   Standalone scripts demonstrating each validation —
                one per blog post (UK National Grid, Berlin climate,
                MIT-BIH ECG, seismic, market-regime, etc.)

probes/         The 8 calibrated probe libraries used by the engine
                (finance, biomedical, industrial, mlops, security,
                logistics, energy_grid, climate). Each documents the
                fingerprint shapes the engine matches against.

data/           Cached real-world signals so anyone can reproduce
                the published numbers offline.

BENCHMARKS.md                The 30-trial comparative benchmark —
                             rank-1 / top-3 / benign-specificity per
                             vertical, no cherry-picking.

COMPARISON_VS_ALTERNATIVES.md  Honest head-to-head against `ruptures`,
                               CUSUM, and z-score thresholds.
```

## Reproducing the published numbers

The walkthrough scripts in this repo call the alphainfo engine
directly via an in-process `TestClient`. They depend on private
modules (`api.app`, `recipes.intents`, `recipes.auto_diagnose`, ...)
that ship with the engine repository — **not in this public repo**.

That means: **most scripts in `walkthroughs/` are reference
implementations, not turnkey demos.** They document methodology so
the published numbers can be audited line by line, but they will
fail with `ModuleNotFoundError` if you try to run them without
the engine.

To actually run an analysis end-to-end as an external reader, use
one of these public surfaces:

1. **`walkthroughs/finance_simple_compare.py`** — the only script in
   this repo that's fully standalone. Depends only on the public SDK
   (`pip install alphainfo`) and `yfinance`. Pulls real SPY data,
   compares Q1 2020 (COVID crash) against Q4 2019 (calm baseline)
   via the public API, prints scores. Set up:

   ```bash
   pip install -r requirements.txt yfinance
   export ALPHAINFO_API_KEY=ai_your_key   # https://www.alphainfo.io/register
   python walkthroughs/finance_simple_compare.py
   ```

2. **alphainfo SDK directly** — `pip install alphainfo`. Free tier:
   50 analyses/month, no credit card, at
   <https://www.alphainfo.io/register>. The SDK's three canonical
   verbs cover most use cases:

   ```python
   from alphainfo import AlphaInfo
   client = AlphaInfo(api_key="ai_...")

   # Most common: "is the signal still like the baseline?"
   result = client.compare(signal=current, baseline=normal, sampling_rate=1.0)

   # No baseline? Triage:
   result = client.detect_internal_change(signal=data, sampling_rate=1.0)

   # Multi-channel:
   result = client.analyze_vector(channels={"a": ..., "b": ...}, sampling_rate=1.0)
   ```

3. **REST API directly** — POST to `/v1/analyze` etc. Documented at
   <https://www.alphainfo.io/quickstart>. Curl-friendly, language-
   agnostic.

The data sources used by every walkthrough are themselves public:

| Walkthrough | Data source |
|---|---|
| `energy_grid_uk_carbon_walkthrough` | <https://api.carbonintensity.org.uk> (UK National Grid, no auth) |
| `climate_open_meteo_walkthrough`    | <https://archive-api.open-meteo.com> (no auth) |
| `ecg_physionet_walkthrough`         | <https://physionet.org> (MIT-BIH Arrhythmia Database via `wfdb`) |
| `early_warning_real_seismic`        | Public seismogram bundled in `data/` |

So even if you can't run the scripts verbatim, the input is reproducible
and the methodology is on display.

## Why this repo exists separately

The full alphainfo codebase (the FastAPI app, the structural engine,
the auth layer, the Stripe integration) lives in a private repository.
We do not want a public mirror of every line of production code — both
for security and for keeping the surface area of what's reviewed manageable.

But anyone evaluating alphainfo's claims should be able to read the
exact script that produced the numbers in a blog post. That's the
purpose of this repository: a small, focused, auditable artifact —
nothing more, nothing less.

## License

The walkthrough source and probe definitions are released under the
[Apache License 2.0](LICENSE). Feel free to read, copy, fork, and
adapt for your own evaluations.

The cached `data/*.json` files were retrieved from public APIs without
authentication; their original terms (Open-Meteo CC-BY 4.0, UK
National Grid CC-BY 4.0, PhysioNet ODC-BY 1.0) apply.

## Contact

- Engineering: <https://www.alphainfo.io/contact>
- Issues with a walkthrough script: open one in this repository.
- Sales / enterprise: contato@alphainfo.io
