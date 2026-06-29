# DESIGN — delphes-pipeline card-validation gate

Status: approved 2026-06-21. Implements the working note (J. Pan, KIT),
§5–§8 (sample plan, validation ladder, pipeline architecture).

## Goal

Validate the CMS Delphes card `cards/cms_card_v0.tcl` on the already-produced
signal `HH → bb̄ τ⁺τ⁻` sample **before** committing full-production compute.
The deliverable is a headless, rerun-on-every-card-revision (CI-like) gate that
emits a machine-readable verdict.

Key fact shaping scope: **the attached card is the pre-tuning baseline.** Its
`BTagging`/`TauTagging` blocks are still the stock Delphes formulas (note §3
tuning set v0 not yet applied). So the meaningful work *now* is the Pilot Gate
plus the Level-0 measurement *infrastructure* (closure against the card's own
transcribed formulas), with a hook for digitised POG curves once v0 lands.
A full Level-0 POG-overlay match is not yet meaningful on this card.

## Architecture: config-driven level runner, pluggable checks

`run_validation.py` reads a YAML config, opens the Delphes file once, builds a
`ValidationContext`, and dispatches to the enabled levels. Each check is an
isolated function returning a `CheckResult` (name, measured, target, tolerance,
pass/fail, severity, plot path). A reporter aggregates → `report.json` +
`summary.md` + `plots/`, and the process exit code is the gate.

Rejected alternatives: notebook-per-level (cannot run headless as a gate; no
CI rerun) and a monolithic script (does not scale to Levels 1–4 × 3 channels).

## Contracts (the spine — authored as one coherent unit)

- `core/result.py` — `CheckResult` dataclass + `Severity{GATE,WARN,INFO}` +
  builder helpers (`gate_max`, `gate_within`, `info`, ...). A `GATE` failure
  sets the nonzero exit code; `WARN` is advisory; `INFO` never fails.
- `core/io.py` — `DelphesEvents(path, treename="Delphes")`: lazy uproot reader
  exposing collections as awkward record arrays with lower-cased fields:
  `.jets (pt,eta,phi,mass,flavor,btag,tautag)`, `.electrons`, `.muons`,
  `.photons`, `.gen (pid,status,pt,eta,phi,mass,m1,m2,d1,d2,charge)`,
  `.genjets`, `.fatjets`, `.met (met,phi)`, `.genmet (met,phi)`,
  `.scalar_ht`, `.weights` (first Event.Weight per event), `.n` (entries),
  `.bytes_per_event`. Also `load_ntuple(path)` for the flat ntuple.
- `core/context.py` — `ValidationContext`: bundles `config`, `events`
  (raw `DelphesEvents`), `ntuple` (built lazily, may be None), `references`
  (`ReferenceStore`), output/plot dirs, `provenance` dict, and
  `tol(level, key, default)` for config tolerance lookup.
- `core/references.py` — `ReferenceStore(dir)`: `expected(quantity, pt, eta)`
  returns the card's transcribed parametrisation (via `references/card_formulas`)
  unless a digitised POG JSON for `quantity` exists in the references dir, in
  which case that is returned. `card_formulas.expected(quantity, pt, eta)` is
  the contract the leaf module implements.
- `core/provenance.py` — `collect(card_path, input_path, n_events, config)`:
  card SHA-256, git commit, generator/container versions (from `versions.json`
  if present), input file + N events; stamped into `report.json`.
- `core/plotting.py` — mplhep CMS style (degrades to plain matplotlib if mplhep
  is absent); `efficiency_overlay(...)`, `resolution_plot(...)`.
- `core/report.py` — `Report`: collects `CheckResult`s, writes `report.json` /
  `summary.md`, computes the gate exit code.
- `ntuplizer/schema.py` — `FLAT_SCHEMA` (NanoAOD-compatible flat columns, note
  Table 1) + `DELPHES_TO_FLAT` mapping. Contract for `convert.py`.

### Uniform level contract

Each level exposes `run(ctx: ValidationContext) -> list[CheckResult]`:
- `validation/pilot_gate/checks.py : run(ctx)`
- `validation/level0_objects/__init__.py : run(ctx)` aggregates
  `btag.run / tau.run / met.run / leptons.run`, each `run(ctx) -> list[CheckResult]`.

## What reads what

- **Pilot Gate** and **Level 0** read the *raw* Delphes tree (`ctx.events`).
  Delphes already provides `Jet.Flavor` (parton flavour), `Jet.BTag`,
  `Jet.TauTag`, gen `Particle`, and `GenMissingET`, so object efficiencies are
  measured directly without rebuilding gen-matching from scratch (τ_h and
  lepton efficiencies do match reco→gen by ΔR; b-tag uses `Jet.Flavor`).
- The **ntuplizer** (`Delphes ROOT → flat NanoAOD-like ntuple`) is exercised by
  a dedicated test and by an optional schema round-trip check in the gate; it is
  the bridge to Levels 1–4 (downstream analysis), not a Level-0 dependency.

## Fleshed vs. stubbed

- **Fleshed:** `core/*`, `ntuplizer/{schema,objects,convert}.py`,
  `pilot_gate/checks.py`, `level0_objects/{btag,tau,met,leptons,reference}.py`,
  `references/card_formulas.py`, `tests/*`.
- **Stubbed (README/schema only):** `generation/`, `cards/tuning/*.json`,
  `validation/level{1,2,3,4}_*/` (need background samples / v0 tuning).

## Object & measurement definitions (Level 0)

- **b-tag / c-mistag efficiency:** per (pT, η) bin, fraction of jets with
  `Jet.Flavor == 5` (resp. 4, and `0`/light for mistag) passing the `Jet.BTag`
  bit. Closure target: the card `BTagging` `EfficiencyFormula` for that flavour.
- **τ_h efficiency:** reco jets with `Jet.TauTag` matched (ΔR<0.4) to a gen
  hadronic τ, over all gen hadronic τ in acceptance, vs (pT, η). jet→τ_h fake:
  `TauTag` jets *not* matched to a gen τ over all jets. Closure target: card
  `TauTagging` (0.6 genuine, 0.01 mistag).
- **MET resolution:** width of `(MET_x − GenMET_x)` and `(MET_y − GenMET_y)`
  vs ΣE_T / qT. Closure: monotonic, finite, below the Pilot-Gate ceiling.
- **e/μ efficiency:** reco e/μ matched to gen e/μ vs (pT, η). Closure target:
  card `ElectronEfficiency` / `MuonEfficiency` plateaus and floors.

## Pilot Gate checks (note: card header)

1. `Event.Weight` sign present; negative-weight fraction ≤ tolerance.
2. `Jet.Flavor` branch present and not all-zero (flavour association ran).
3. Gen hadronic/leptonic taus present in `Particle` (signal must contain τ).
4. `m_bb` peak width (two highest-BTag jets) finite and ≤ ceiling.
5. MET resolution (vs GenMET) finite and ≤ ceiling.
6. kB/event = file size / N entries ≤ storage-projection ceiling.

All thresholds live in `config.example.yml` under `tolerances.pilot_gate`.

## Provenance & reproducibility (note §8.3)

Every `report.json` carries: card SHA-256, tuning-set version, generator/
container versions, git commit, input path, N events, timestamp (stamped by the
caller, not inside the workflow). Tuning JSONs in `cards/tuning/` follow
`_schema.json`: source figure/DP id, date, WP definition, parametrisation.

## Verification

`pytest` builds a synthetic Delphes-like ROOT fixture with injected efficiencies
and asserts (a) the measurement recovers the injected truth within statistics,
(b) the gate passes on a good fixture and fails on a deliberately broken one,
(c) the quantitative gates (MET scale offset, smeared m_bb) actually fail on
those defects, and (d) the ntuplizer round-trips the schema with the contract
dtypes. Runs locally (Python 3.13 + uproot) and in the pixi linux env on
Perlmutter.

## Implementation notes (as built)

- **Statistics-aware closure** (`core/closure.py`, shared by b-tag/tau/lepton):
  a per-pT bin fails only if it misses the card by more than the relative
  tolerance (`closure_rel_tol`, 5%) **and** more than `closure_nsigma` (2) of the
  *expected*-rate binomial error. The expected-rate error is non-zero even at a
  measured rate of 0/1, so a low-rate quantity (light mistag ~1%) is not failed
  for Poisson scatter. This keeps the 5% systematic floor while not flagging bins
  statistically consistent with the card — a pure 5%-relative per-bin rule is
  unreachable for low-rate quantities at any realistic statistics.
- **τ_h is reco-jet based** (`level0_objects/tau.py`): efficiency = fraction of
  the *unique nearest* reco jet to each acceptance gen τ that is `TauTag==1`
  (mistag = `TauTag` rate among jets not near any gen τ). This is real-data safe
  — leptonic-τ decays produce no nearby jet and drop out naturally, and gen
  multi-copy τ records do not inflate the count. *Known approximation:* the
  efficiency target is the hadronic-τ efficiency; refining the gen-τ hadronic
  selection by decay mode is deferred (it does not affect the closure on the
  hadronic fixture). ΔR matching lives in `core/matching.py`.
- **Lepton efficiency** uses unique nearest-neighbour gen→reco matching (each
  reco claimed once) so double counting cannot bias the efficiency high; only the
  plateau (`pt > pt_bins[0]`, barrel `|η|≤1.5`) is closure-tested, the turn-on
  region is deferred.
- **m_bb pilot check** gates on the core-window (100–150 GeV) std **and** the
  fraction of leading pairs in that core: a window-bounded std saturates, so a
  grossly smeared/shifted b-response is caught by the collapse of the core
  fraction.
- **MET pilot check** uses RMS-about-zero (not variance-about-the-mean) so a MET
  scale/offset error is caught, not just the spread.
- **Ntuplizer is wired into the gate** as the `ntuplizer` level: it converts a
  slice and validates schema coverage + counts, so a broken ntuplizer fails the
  gate (it is also round-tripped in `tests/`). Output is a *nested* parquet
  (top-level columns are the collections `Jet, Tau, …` plus scalars), read back
  via `core.io.load_ntuple` as `ntuple.Jet.pt` etc.

## Sustainable validation + tuning: one measurement, three lenses

The measurement of object response is centralised in `core/observables.py` and
consumed by three lenses, so a retuned selection changes one place:

- **validation** (`validation/`, the gate) — `closure.closure_from_profile`
  compares a measured `Profile` to the **card-formula** target → pass/fail
  `CheckResult`. The Level-0 leaves are thin wrappers over `observables`.
- **tuning** (`tuning/`) — compares the same `Profile` to a **POG/anchor target**
  (digitised curve / unity response / anchor mass peak) → residual + the card
  knob to turn (`cards/tuning/diagnostic_map.json`, note §3–4) → `tuning_report.{md,json}`
  + measured-vs-target overlays. Run `runners/run_tuning.sh`; iterate after each
  card edit. This is the lens for the note's tuning set v0.
- **plots** (`plots/`) — validation + signal-baseline figures
  (`runners/make_validation_plots.sh`): jet/τ_h/lepton spectra justifying the
  patches, m_bb, visible m_ττ, and gen m_HH across the six κ_λ points. See
  `docs/card_patch_validation.md` for the AN-25-103-cited justifications.

`core/observables.py` adds, beyond the gate's efficiencies, the tuning
observables the note calls for: **τ-jet and b-jet energy response** (reco/GenJet,
GenJets being neutrino-filtered = visible reference; §3.2, §4.3) and the
**m_bb peak** position/width (§4.3). MET resolution vs ΣE_T is shared.

**Extension points** (`extensions/`, documented stubs): trigger emulation
(`trigger.py`, the AN-25-103 Table-2 2024 menu encoded; §4.1) and the m_ττ
estimator (`mtautau.py`, decision D1: FastMTT / covariance-free; §3.3). The
Level-1 candles (Z→ττ, tt̄) remain stubs under `validation/level1_candles`.

### NanoAOD anchor target mode (note §3, §6.4)

The note's primary tuning anchor is the private CMS NanoAOD: tune Delphes object
response until it matches the *same* response measured on the NanoAOD. With
`anchor.enabled: true` in the config, `core/nanoaod.py` (`NanoAODEvents`,
duck-typed to `DelphesEvents` — `jets.btag` is the tagger discriminant
thresholded at the working point) lets the shared `observables` measure the
b-tag and lepton targets directly; `tuning/anchor.py` adds the τ_h efficiency
(NanoAOD `GenVisTau` matched to a DeepTau-Medium `Tau`) and the overall MET
resolution. The anchor target supersedes the digitised-curve/card-formula target,
so the efficiency/MET observables become actionable (residual + knob) without any
hand-digitisation. Branch names and WPs are config (`anchor.branches`,
`anchor.wp`), so a different NanoAOD era is a config edit. This is the recommended
way to tune against CMS 2024 MC.

**b-tag WP from CVMFS.** With `correctionlib`, `anchor.wp.btag_medium` is resolved
from the official BTV `jsonpog-integration` JSON (the `<tagger>_wp_values`
correction) rather than hand-entered — `tuning/correctionlib_wp.py`, auto-finding
the UParT `*_wp_values` name (override per the config). `python -m
delphes_pipeline.tuning.correctionlib_wp <json>` lists the available corrections.

## Level-1 candles (note §6.2)

Two candles run on the **background** Delphes samples (`config.candles.{ttbar,ztautau}`),
aggregated by `validation/level1_candles/run(ctx)`:

- **tt̄ dilepton** (`TTto2L2Nu`) — eμ-OS selection; the GATE is the in-situ
  tag-counting closure ε_b = 2N₂/(N₁+2N₂) (note Eq. 2) vs the tuned input
  (card formula / anchor); plus A×ε and the pᵀᵐⁱˢˢ tail.
- **Z→ττ** (`DYto2Tau`) — ℓτ_h/τ_hτ_h with a b-veto; visible m_ττ peak/width,
  channel ratio, and the low-mass jet→τ_h fake sideband. The model-independent
  "peak at m_Z" check is hooked to the m_ττ estimator (decision D1,
  `extensions/mtautau.py`) and reports *pending* until that is built.

The estimator-dependent checks and the POG/anchor yield targets are the
documented remainder; the tt̄ ε_b closure is the one GATE today.
