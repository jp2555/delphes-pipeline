# Delphes tuning procedure — specification for an NSBI-consistency audit

**Purpose of this document.** It describes, in enough detail to audit, every correction the
Delphes tuning applies to the HH→bb̄ττ samples that feed an unbinned NSBI measurement of
κ_λ. The question being asked is *not* "does Delphes match CMS" (it largely does, see
`docs/notes/tauh_cutflow_note.pdf`) but **"does the way we make it match break an assumption
NSBI relies on?"** Section 7 lists the specific concerns; sections 1–6 are the facts needed
to judge them. Nothing here requires access to the session that produced it.

Code referenced: `delphes_pipeline/tuning/{anchor,maps}.py`,
`delphes_pipeline/core/observables.py`, `delphes_pipeline/ntuplizer/convert.py`,
`runners/production.sh`.

---

## 1. What NSBI is being used for here

Three signal samples generated at κ_λ ∈ {0, 1, 5}, 5M events each, plus tt̄ (299M events)
and (planned) DY. NSBI learns per-event likelihood ratios between κ_λ hypotheses, and
between signal and background, from a 10-feature vector:

```
mHH, cosThetaStar, pHH_T, mbb, dR_bb, mtautau, dR_tautau, dphi_HH, pH1_T, pH2_T
```

The ratio is learned from the *joint* density, so anything that distorts correlations
between these features — not just their marginals — is in scope. Two ratios matter and they
have different exposure to the issues below:

- **κ_λ vs κ_λ.** All three κ_λ samples are corrected with the *same* map file, so a
  common distortion cancels to first order in this ratio.
- **signal vs background.** Signal and tt̄ are corrected with *different* map files
  (§6). A distortion that differs between them does **not** cancel here.

## 2. Architecture

Delphes samples are produced once from the CMS card (`cards/cms_card_v1.tcl`) and **never
re-made**. All tuning is *downstream*: a set of pT-binned maps is measured on a real CMS
NanoAODv15 anchor, serialised to JSON, and applied when the Delphes ROOT files are converted
to parquet ntuples. "Tuned" therefore means "measured on CMS and transferred", not "the card
was refit".

```
CMS NanoAOD anchor ──derive_maps──> maps_*.json ──convert.py──> tuned parquet ntuples ──> NSBI
Delphes ROOT ───────────────────────────────────┘
```

The anchor is the same `PowhegBugFix` reco campaign the Delphes signal samples were
generated from. Maps are derived from κ_λ = 1 only and applied to all three κ_λ points.

## 3. The maps: what each one is

Binning is `DEFAULT_PT_BINS` = `[20, 30, 40, 50, 70, 100, 150, 200, 300]` GeV, i.e. **8 bins
in object pT and nothing else** unless stated. Values outside the range are **flat
extrapolated** from the edge bin.

### 3.1 Measured on the CMS anchor only

| map | what it is | binned in |
|---|---|---|
| `btag_eff_b`, `btag_eff_c`, `btag_mistag_light` | P(UParT-AK4 Medium tag) for true b / c / light jets | pT, per flavour |
| `tau_eff` | P(DeepTau v2.5 Medium) for a jet on a `GenVisTau` | pT |
| `tau_mistag` | P(DeepTau v2.5 Medium) for a jet with no gen τ | pT |
| `tau_mass` | τ_h visible mass — **median + 21 quantiles per bin** | pT |
| `tau_energy_response` (= `tau_response`) | CMS τ_h reco/gen pT — **median + 101 quantiles per bin** | gen pT |
| `tau_fake_response` | same, for fake τ (tagged, no gen τ), over the matched GenJet | pT |
| `electron_eff`, `muon_eff` | prompt barrel lepton reco efficiency, unique gen→reco match at ΔR 0.2 | pT |
| `met_resolution` | CMS pT_miss resolution per component | **jet H_T** (7 bins) |

### 3.2 Requiring the Delphes response as well

| map | formula | clipping |
|---|---|---|
| `bjet_escale` | 1 / (Delphes b-jet response vs its **own** `GenJet`) | clipped to [0.5, 2.0] |
| `tau_escale` | anchor τ response / Delphes τ response | clipped to [0.5, 2.0] |
| `met_smear` | √(σ²_anchor − σ²_Delphes) per H_T bin | ≥ 0 by construction |
| `electron_sf`, `muon_sf` | anchor lepton eff / Delphes lepton eff | clipped to [0.5, 2.0] |

Two asymmetries here are deliberate and are relevant to the audit:

- **b-jets are self-anchored.** `bjet_escale` corrects the Delphes b-jet response to *its
  own* GenJet, never to CMS. It therefore fixes the energy **scale** and does **not** touch
  the **resolution**. The card also runs without pileup, so part of the real CMS b-jet
  resolution has no counterpart to correct.
- **τ are anchored to CMS, not to gen.** A Delphes τ_h *is* an AK4 jet (jet-level pT,
  underlying event inside R = 0.4) while a CMS `Tau` is an HPS visible-τ object. Inverting
  to unity against a GenJet would preserve that definitional gap, so the τ maps target the
  CMS τ energy scale directly.

### 3.3 Statistics guards

- `MIN_QUANTILE_COUNT = 200`: a quantile bin with fewer entries is **redirected to the
  nearest bin that has the statistics**, rather than sampled from noise. In the current
  `tau_fake_response`, bins 5–7 (the highest pT) fall below this and are borrowed.
- The quantile grid is 101 points on levels [0.005, 0.995]. An earlier 21-point grid
  inflated q99 by 7.2% because the top segment interpolated to the sample maximum.
- τ visible mass is capped just below m_τ (1.70 GeV) or FastMTT's hadronic prior has no
  solution.

## 4. How the corrections are applied

`retag_jets()` — fixed order, and the order is load-bearing because every map was *measured*
against a CMS-scale quantity and so must be *applied* at a CMS-scale pT:

```
1. escale          b-jets: pt *= bjet_escale, mass *= bjet_escale     [deterministic]
                   (τ branch SKIPPED when the resampling map exists)
2. resample real τ gen-matched τ: pt <- draw from anchor quantiles     [STOCHASTIC]
                   at the GEN pT, gated on pt > 20 and |eta| <= 2.5
3. retag btag      Bernoulli(btag_eff_{b,c,light}[pt]) from Jet.Flavor [STOCHASTIC]
4. retag tautag    Bernoulli(tau_eff[pt]) if a gen τ matches,          [STOCHASTIC]
                   else Bernoulli(tau_mistag[pt])
5. resample fake τ jets newly tagged τ with no gen τ: pt <- draw       [STOCHASTIC]
                   from tau_fake_response quantiles
6. tau_mass        τ_h mass <- draw from anchor per-pT quantiles       [STOCHASTIC]
---
7. met_smear       pT_miss += N(0, σ[jet H_T]) per component           [STOCHASTIC]
```

Step 2 must precede steps 3–4 because `tau_eff` is read at the jet pT. Steps 2 and 5 are
split because *which* jets are fakes is an outcome of the draw in step 4 — an earlier
version gated resampling on the incoming Delphes `TauTag` while step 4 draws a fresh
independent Bernoulli, so only about half the τ were corrected.

**Randomness.** All stochastic steps draw from one `numpy.random.Generator`, seeded **per
shard** (each of the 1394 HTCondor jobs has its own seed recorded in the production plan).
The order is fixed so a given seed reproduces identical output.

**Per-event weight.** `lepton_sf` (from `electron_sf`/`muon_sf`) is written to the ntuple as
a **weight column**, not applied as a selection. Delphes leptons are neither dropped nor
added. `genWeight` is Delphes' `Event.Weight[0]` and can be negative.

## 5. What is NOT corrected

- **b-jet energy resolution** (only the scale — §3.2).
- **Pileup**, and therefore pileup jets, which cannot exist in the samples at all.
- **pT_miss anisotropy.** `met_smear` is isotropic; the real resolution is not
  (measured 1.24 vs 1.05 major/minor axis ratio).
- **DY**, which has no CMS anchor yet, so no DY maps exist.
- `tau_mistag` **outside its derivation sample** — fake rates are strongly jet-flavour
  dependent, so the signal-derived map is not valid for fake-dominated backgrounds
  (QCD, W+jets).

## 6. Per-process maps — the central structural fact

Maps are derived **separately per physics process** and applied per process:

| sample | map file | anchor |
|---|---|---|
| signal, all κ_λ | `cards/tuning/maps_v1.json` | HH→bb̄ττ κ_λ=1 NanoAOD (`PowhegBugFix`) |
| tt̄ | `cards/tuning/maps_ttbar_v1.json` | TT NanoAOD |
| DY | *(does not exist)* | *(none downloaded)* |

The stated justification (`runners/production.sh` header) is that the lepton scale factors
differ between signal and tt̄ by up to 20%, and that the fake-τ response is a jet-substructure
property that does not transfer from a quark-rich signal to a gluon-rich background.

Note the split within this: `btag_eff_b/c/light` are resolved **by true jet flavour**, so
they are object-level quantities that ought to be process-independent. `tau_eff`,
`tau_mistag`, `tau_response`, `tau_fake_response`, `tau_mass`, `electron_sf`, `muon_sf` and
`met_smear` are **not** resolved by anything that distinguishes the processes, so a
per-process map absorbs composition differences into what is nominally a detector model.

## 7. Questions for the audit

Ordered by our estimate of severity. We have not resolved any of these.

**Q1 — Process-dependent detector model.** In nature there is one detector. Signal and tt̄
are corrected with different functions (§6). The learned ratio p_S(x)/p_B(x) therefore
contains a detector-modelling difference that does not exist in data, and a sufficiently
flexible network can exploit "which map was applied" as if it were physics. Is this a bias
on κ_λ, and if so in which direction? Does the argument change for maps that *are*
flavour-resolved (b-tag) versus those that are not (all τ maps, lepton SFs, `met_smear`)?
Would parameterising the τ maps in an object-level variable that separates the processes
(e.g. jet flavour or prong multiplicity) make them transferable, and is that the fix?

**Q2 — The event no longer balances.** Step 2/5 changes τ pT but MET propagation is **off by
default**: `propagate_met=False`. The rationale is that the AK4-vs-visible-τ difference is
largely a cone *definition* offset rather than mismeasured energy, so moving all of it into
the recoil would fabricate missing energy (measured: it hands back ~⅓ of the m_ττ
correction). The consequence is that after tuning, Σp_T of the visible objects and pT_miss
are no longer consistent with each other. FastMTT solves the τ energy fractions x₁, x₂
*against* pT_miss, so `mtautau` is computed from a resampled τ momentum and an unresampled
recoil. Both `mtautau` and (implicitly) `met` are NSBI inputs. How much does this distort the
joint (m_ττ, MET, m_HH) density, and is the distortion common to signal and background?

**Q3 — Independent draws destroy correlations.** Every stochastic step draws independently:
the two τ legs' energy responses are independent of each other; the τ visible mass is drawn
independently of the τ energy and of the partner leg; the b-tag and τ-tag Bernoullis are
independent per jet and independent of each other. In reality these are correlated — decay
mode ties visible mass to prong multiplicity and to energy sharing, a τ-tagged jet is
unlikely to be b-tagged, and per-event pileup correlates all of them. NSBI learns the joint
density, so absent correlations are a modelling error in precisely the quantity being
learned. Which of these matters most, and is any of it estimable from the anchor?

**Q4 — Stochastic corrections as part of the density.** The detector is a random function of
the event: the same Delphes event with a different seed gives different tags, τ energies and
τ masses. This is an unbiased sample of the CMS response *if* the maps are right, but it adds
variance and it means the tuned sample is one realisation. Does NSBI training treat this
correctly, or should the correction be applied multiple times per event (an ensemble) to
represent the response distribution rather than one draw from it?

**Q5 — κ_λ-side consistency.** All three κ_λ points use one map file, so the κ_λ ratio should
be clean. But the shards are contiguous per κ_λ and seeded per shard, so each κ_λ point sees a
*different RNG stream*. We believe this is harmless (different streams, same distribution) —
please confirm there is no route by which it correlates with κ_λ.

**Q6 — Hard clips and flat extrapolation.** `bjet_escale`, `tau_escale`, `electron_sf`,
`muon_sf` are clipped to [0.5, 2.0]; every map is flat-extrapolated above 300 GeV. A clipped
bin is silently wrong there, and the extrapolated region is frozen at the edge value. Only
~2.8% of b-jets exceed 300 GeV and κ_λ discrimination sits near threshold
(m_HH ≈ 250–400 GeV, where the triangle's s-channel propagator makes the interference most
κ_λ-sensitive) rather than in the box-dominated tail — but signal and background populate the
tail differently, so this is another per-process effect. Does it matter at the required
precision?

**Q7 — Borrowed bins.** `tau_fake_response` has its top three pT bins below the 200-entry
threshold and borrows them from neighbours. The borrowing pattern is derived per process, so
signal and tt̄ borrow differently. Same question as Q1, but concentrated in the fake-τ
population — which is also our leading suspect for the one unresolved data/MC residual (a
Delphes excess of wide-angle τ pairs at ΔR_ττ ≈ 2.5–3.2).

**Q8 — Normalisation.** The tuning changes efficiencies and therefore yields. If the NSBI
normalisation takes sum-of-weights from the tuned sample itself it is self-consistent; if it
takes a separately-quoted generated-event count it is not. Related: 4 of 1244 tt̄ shards
(0.3%) were lost to a storage fault and the merged manifest records the shortfall.

## 8. Reproducing / inspecting

```bash
# derive maps for one process
pixi run -e nsbi-env-gpu python -m delphes_pipeline.tuning.derive_maps \
    --config config.v1.yml --output cards/tuning/maps_v2.json --max-events 200000

# inspect a map
python3 -c "import json; m=json.load(open('cards/tuning/maps_v2.json'));
print(list(m['maps'])); print(m['maps']['tau_response']['centers'], m['maps']['tau_response']['values'])"

# validate tuned Delphes against CMS, per kappa_lambda
pixi run -e nsbi-env-gpu python scripts/nsbi_overlay.py --config config.v1.yml \
    --ntuple /ceph/jpan/ntuples/merged --nano-dir <cms nanoaod dir> \
    --nano-select PowhegBugFix --out plots/check --max-events 200000 --diagnostics
```

Map JSON layout is `{"provenance": {...}, "maps": {name: {"x", "centers", "values",
"counts", ...quantiles}}}`.
