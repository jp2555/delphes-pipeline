# Validating the `cms_card_v0` patches against CMS AN-25-103

**Goal.** Reproduce the CMS HH→bb̄τ⁺τ⁻ analysis as a Delphes baseline. The card
`cards/cms_card_v0.tcl` is the stock CMS Delphes card (3.5.1pre09, no pileup)
plus five patches (`grep PATCH-`). This note justifies each patch against the
Run-3 reference analysis **CMS AN-25-103** (*Search for non-resonant HH→bb̄ττ
with Run-3 data*, 172.3 fb⁻¹ at 13.6 TeV, 2022–2024 MC) and lists the validation
figure that demonstrates it. Section/figure citations are to AN-25-103 unless
noted (the Run-2 analysis AN-18-121 corroborates the object definitions).

> **How the figures are produced.** The object/baseline figures read the Delphes
> samples on Perlmutter, so they are generated there:
> `bash runners/make_validation_plots.sh` → `outputs/validation/figures/`.
> The lepton-floor figure is pure card formula and is committed at
> [`figures/lepton_eff_floor_curves.png`](figures/lepton_eff_floor_curves.png).

## Patch → CMS reference → consequence

| Patch | Change | AN-25-103 reference | Why (consequence if unpatched) |
|---|---|---|---|
| **1** | jet `ParameterR` 0.5 → **0.4** (GenJetFinder, FastJetFinder; FatJet kept 0.8) | §4.7: jets are anti-kₜ, **AK4 = R 0.4** / AK8 = R 0.8; the resolved categories use AK4 | Stock CMS Delphes uses R=0.5. The analysis b-jets and τ_h candidates are AK4; an R=0.5 jet over-clusters, shifting m_bb and the jet–τ separation away from the CMS object definition. |
| **2** | `JetPTMin` 20 → **15 GeV** (GenJetFinder, FastJetFinder) | analysis cuts AK4 jets and b-jets at pT>20 (§5.1.3) and τ_h candidates at pT>20, \|η\|<2.3 (§4.6); jet resolution is 15–20% at pT~30 (§4.7) | τ_h candidates *are* jets in Delphes. A generation floor **at** the 20 GeV analysis cut biases the turn-on: JES/JER smearing (~15–20%) migrates events across 20 GeV, and events that should migrate up are absent. A 15 GeV floor leaves the 20 GeV cut on a fully-populated spectrum. |
| **3** | electron ID floor 10 → **7 GeV** | §5.1.2 third-lepton veto: reject events with an extra electron at **pT>10** | The veto must reconstruct leptons *below* its threshold to apply. With the stock 10 GeV floor (≡ the veto threshold), a 7–10 GeV electron is invisible and cannot be vetoed → extra-lepton backgrounds leak in. Floor 7 gives margin. |
| **4** | muon ID floor 10 → **5 GeV** | §5.1.2 third-lepton veto: reject events with an extra muon at **pT>6** | The stock 10 GeV floor sits *above* the 6 GeV veto, so muons in 6–10 GeV are never reconstructed and the veto is under-applied (tt̄/DY with a soft third muon leak in). Floor 5 makes the 6 GeV veto emulatable with margin. |
| **5** | TreeWriter `full`/`lite` selectable | — (bookkeeping) | No kinematic effect; storage only (lite drops constituent branches for fake-source bulk). Not validated with a plot. |
| **6** *(v1 only)* | BTagging formulas → **UParT-AK4 Medium** fits (b: `0.904−3.53/pT`, c: `0.094+2.22/pT`, light: `0.019+0.32·e^(−pT/30)`) | §4.3 b-jets use UParT-AK4; WP Medium = 0.1272 (BTV 2024_Summer24); anchor: 2024 NanoAODv15 kl=1 `derive_maps`; published WP defs 10%/1%/0.1% light mistag (arXiv:2412.05863) | Stock formulas are Run-1 CSV-era (arXiv:1211.4462): b-eff ~0.63–0.71 (vs 0.76–0.89 measured), c mistag *rises* with pT (measured falls 0.18→0.10), light ~1–2% flat (measured 16→2% falling). A card baseline near the 2024 anchor makes the downstream re-tag a small correction, not a rescue. Max \|fit − anchor bin\| ≤ 0.018. |
| **7** *(v1 only)* | TauTagging → **DeepTau v2p5 VSjet Medium** (τ: `0.776−9.7/pT`, fake 0.004) + `DeltaR` 0.5 → **0.4** | §4.6 τ_h use DeepTau v2p5, Medium VSjet (NanoAODv15 WP 5); anchor: GenVisTau-matched unique-nearest on the same 2024 NanoAODv15; published Medium plateau ≈0.70, fake ≈0.5% (arXiv:2601.19431) | Stock (flat 0.60 / 1% fake) is pre-DeepTau vintage: it *overshoots* the real 20–40 GeV τ_h efficiency (0.39–0.58) and overstates the fake rate ~2–5×, distorting the τ_hτ_h acceptance and the low-m_ττ fake sideband. ΔR 0.4 matches AK4 and the downstream re-tag matching. Caveat: fake rates are environment-dependent (anchor = b-rich signal); background fake maps must be re-derived on QCD/W+jets. |

**Card versioning.** `cms_card_v0.tcl` (PATCH-1…5) is the provenance of the
already-produced `/ceph` samples and stays frozen. `cms_card_v1.tcl` = v0 +
PATCH-6/7 — use it for new productions. The Level-0 closure targets follow the
config's `card:` path automatically (`card_formulas.for_card`), and
`tests/test_card_v1.py` pins the tcl ↔ python transcription and the anchor fit.

## Per-patch validation figures

**PATCH-1 (AK4 R=0.4).** `jet_eta_spectrum.png` (AK4 η with the b-tag \|η\|<2.4
acceptance) and `mbb_peak.png` (the two highest-b-tag AK4 jets reconstruct the
visible H→bb̄ peak). *Expected:* a clean m_bb peak near ~110–125 GeV (visible,
no b-energy regression per §4.3) — i.e. AK4 jets reconstruct the bb̄ resonance
the analysis relies on.

**PATCH-2 (jet floor 15).** `jet_pt_spectrum.png` (AK4 jet pT, log-y, with the
15 GeV gen floor and 20 GeV analysis cut marked) and `tauh_pt_spectrum.png`
(τ-candidate pT). *Expected:* the spectrum is smooth and fully populated through
20 GeV — no generation cliff at the analysis cut — and the τ_h-candidate yield is
flat/sane at 20–25 GeV (the card-header pilot concern).

**PATCH-3/4 (lepton floors).** Committed formula figure
[`lepton_eff_floor_curves.png`](figures/lepton_eff_floor_curves.png): stock vs
patched ID-efficiency floors with the veto thresholds marked — the patched floors
sit below the vetoes so the third-lepton veto is emulatable. Data cross-check
`lepton_pt_spectra.png`: reco e/μ populate below 10 GeV down to the 7/5 floors.

## Signal-baseline figures (is the CMS baseline reproducible?)

- **`mhh_klambda.png`** — gen-level m_HH overlaid across the six κ_λ points
  (−1, 0, 1, 2.45, 3, 5). *Expected:* the κ_λ-dependent shape — threshold
  enhancement near 2m_H at κ_λ≈0, the SM dip, the high-mass tail at κ_λ=5 — which
  is exactly the shape the analysis DNN/limit exploits (§5.5.3). This is the
  headline "the samples carry the right di-Higgs physics" plot.
- **`mtautau_visible.png`** — visible di-τ mass peak (below m_H; neutrinos
  missing), the input to the τ_h energy-scale tuning (§3.2).
- **`multiplicities.png`** — n jets / b-tag / τ-candidate / lepton sanity.

## Caveats (stated, not hidden)

- The card is the **pre-v0-tuning baseline**: BTagging/TauTagging are stock
  formulas, so object-response *closure* against the card is not the same as
  matching CMS performance — that is what the **tuning loop**
  (`docs`/`tuning_report.md`, `runners/run_tuning.sh`) drives, comparing measured
  response to the POG/anchor targets (UParT-2024, DeepTau v2p5, PUPPI).
- The Run-3 third-lepton-veto thresholds (e>10, μ>6) are transcribed in the card
  header from AN-25-103 §5.1.2; confirm against that section when finalising.
- gen m_HH uses the two leading-pT gen Higgs (heuristic); verify on the Powheg
  record that it selects the two physical Higgs, not two copies of one.
