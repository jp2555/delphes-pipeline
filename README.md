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
