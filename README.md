# delphes-pipeline

Delphes fast-simulation sample preparation and a **card-validation gate** for the
`HH → bb̄ τ⁺τ⁻` NSBI phenomenology study.

This repository implements the pipeline architecture of the working note
*"Delphes Sample Preparation and a Sustainable Fast-Simulation Analysis Pipeline
for NSBI Phenomenology Studies"* (J. Pan, KIT). Its immediate job is to let you
**validate the CMS Delphes card before spending full-production compute**: point
it at the signal Delphes output, run the gate, get a machine-readable go/no-go.

## What this is for, concretely

You have finished the signal `HH → bb̄ τ⁺τ⁻` Delphes runs (on Perlmutter at
`/pscratch/sd/j/jing/dihiggs/raw`). Before launching the full background
production, you want confidence that the card produces sane, non-broken output
and that the object response sits where the tuning targets expect. This package
runs two things on the existing signal sample:

1. **Pilot Gate** — the six card-header sanity checks (weight sign &
   negative-weight fraction, `Jet.Flavor` filled, gen taus present, `m_bb`
   width, MET resolution, kB/event). A nonzero exit code blocks production.
2. **Level-0 object response** — measures b-tag / c-mistag / τ_h efficiency,
   jet→τ_h fake rate, MET resolution, and e/μ efficiency vs `(pT, η)` from the
   Delphes output, and overlays each against the card's transcribed
   parametrisation (a *closure* check: do the modules behave as configured?),
   with a documented hook to drop digitised POG curves once tuning set **v0**
   lands.

Levels 1–4 of the validation ladder (candles, signal A×ε, binned-limit,
κ_λ-critical `m_HH` response) are scaffolded as stubs — they need background
samples and v0 tuning you do not have yet.

## Quick start

```bash
pixi run -e nsbi-env-gpu \
  python -m delphes_pipeline.validation.run_validation \
  --config delphes_pipeline/validation/config.example.yml
```

Outputs land in `outputs/validation/`:

- `report.json` — every check, machine-readable, with provenance metadata.
- `summary.md` — human-readable pass/fail table.
- `plots/*.png` — CMS-style overlay plots.

The process exit code is `0` iff every **gate-severity** check passes.

## Self-test (no Delphes file needed)

```bash
pytest -q
```

The test suite generates a tiny synthetic Delphes-like ROOT file with *injected*
object efficiencies and asserts the measurement recovers them and the gate logic
fires correctly. This is how the scaffold is verified without a real sample.

## Layout

See [`DESIGN.md`](DESIGN.md) for the architecture and the directory map.

## Three lenses on one measurement (validation · tuning · plots)

Object response is measured once in `delphes_pipeline/core/observables.py` and
consumed three ways:

```bash
# 1. validation gate (pass/fail vs the card formula) — exit code is the gate
bash runners/validate_all.sh                       # all six kappa_lambda points

# 2. tuning report (residual vs POG/anchor target -> which card knob, note Sec.3-4)
bash runners/run_tuning.sh                          # -> outputs/.../tuning_report.md

# 3. validation + signal-baseline figures (object spectra, m_bb, gen m_HH vs kappa_lambda)
bash runners/make_validation_plots.sh              # -> outputs/.../figures/
```

The card-patch justifications (cited to CMS AN-25-103) are in
[`docs/card_patch_validation.md`](docs/card_patch_validation.md). To tune an
object response, drop a digitised POG/anchor curve into
`delphes_pipeline/validation/references/data/<observable>.json`, re-run the
tuning report, adjust the card knob it names, and repeat.
