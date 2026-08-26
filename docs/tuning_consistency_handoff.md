# Handoff — Delphes/CMS tuning consistency is now blocking

*For the session that owns tuning. Written 2026-08-24. Everything below is measured on the
merged productions on `/ceph`; commands to reproduce are in §6.*

## 1. Why this is no longer an audit item

The working assumption has been that the Delphes samples are a usable stand-in for the CMS
NanoAOD ones, with tuning as a quality improvement. That assumption is now in doubt.

With the *same* generator configuration, the Delphes forward model appears to place the
κ_λ = 0, 1, 5 feature distributions **closer together** than CMS NanoAOD does. Since the rest
of the pipeline is essentially common, a compressed κ_λ separation makes the density-ratio
estimation strictly harder and the resulting interval correspondingly weaker — to the point
where the measurement on Delphes is effectively unconstrained. If that holds, **there is no
Delphes baseline** in the sense the pheno paper needs one, and the deliverable becomes
"tune Delphes to CMS NanoAOD closely enough that the comparison is meaningful."

This reframes the tuning work from *make the samples nicer* to *make the samples exist*.

## 2. Q5 has been answered, and not in the way the audit assumed

`tuning_for_nsbi_audit.md` Q5 reasoned: *"All three κ_λ points use one map file, so the κ_λ
ratio should be clean"*, and asked only whether per-shard RNG streams could correlate with
κ_λ. That argument is insufficient. One map file does not imply a κ_λ-independent effect,
because the maps are functions of pT and the three κ_λ points have different pT spectra.

Measured, selected yield Delphes / CMS, semi-leptonic, 200k events per point against the
`PowhegBugFix` anchor:

| | untuned | tuned |
|---|---|---|
| κ_λ = 0 | 0.322 | 1.082 |
| κ_λ = 1 | 0.323 | **1.043** |
| κ_λ = 5 | 0.328 | 1.185 |
| **spread across the basis** | **1.9 %** | **13.6 %** |

Read that carefully. The untuned card loses two thirds of CMS's acceptance, but loses it
**uniformly** — flat to 2 % across the morphing basis. The tuning corrects the normalisation
(0.32 → ~1.0, which is the right thing) and in doing so **introduces** a 14 % variation across
exactly the {0, 1, 5} basis the morphing runs on.

The pattern matches how the map was built: the residual is smallest at κ_λ = 1, where
`maps_v1.json` was anchored, and grows away from it. On Poisson errors alone κ_λ = 5
(1.185 ± 0.011) separates from κ_λ = 1 (1.043 ± 0.008) at roughly 10σ.

**Why this is worse than a uniform error.** A κ_λ-independent offset largely divides out of
the morphing — it is a normalisation you fit or profile. A κ_λ-*dependent* one is degenerate
with the parameter being measured: it tilts the basis the parameterised likelihood is built
from. On this evidence the untuned samples may, for κ_λ specifically, be **safer** than the
tuned ones. That inverts the default assumption and should be tested rather than believed.

Leading hypothesis: κ_λ = 5 has the softest m_HH spectrum, hence the softest jets, hence the
greatest sensitivity to an energy-scale correction fitted on the harder κ_λ = 1 sample. The
untuned acceptance deficit is very likely jets falling below the 20 GeV threshold.

## 3. Q1 quantified — the per-process asymmetry

τ_h candidates in the same 200k events:

| | untuned | tuned | CMS | change | tuned / CMS |
|---|---|---|---|---|---|
| signal κ_λ = 0 | 139,613 | 118,863 | 112,697 | −14.9 % | 1.055 |
| signal κ_λ = 1 | 143,498 | 121,684 | 118,619 | −15.2 % | **1.026** |
| signal κ_λ = 5 | 133,466 | 110,245 | 96,415 | −17.4 % | 1.143 |
| tt̄ (`TTto2L2Nu`) | 42,293 | 30,760 | 25,241 | **−27.3 %** | **1.219** |

After tuning, signal sits 3 % above CMS and tt̄ sits 22 % above. The learned p_S/p_B therefore
inherits a τ_h rate mismatch of ≈ 1.22/1.03 ≈ **1.19** — a detector-model difference between
numerator and denominator that does not exist in nature. That is Q1, with a number on it.

Note the κ_λ = 5 column again: the τ_h residual is also worst there (1.143 vs 1.026), so §2
and §3 are probably the same underlying effect seen through two different observables.

## 4. Supporting evidence from the selection side

From the SBI-conversion work (other session), σ(κ_λ=5)/σ(κ_λ=1) after selection:

| | ratio |
|---|---|
| Powheg, generator level | 1.91 |
| CROWN-matched selection | 1.831 |
| CMS HIG-25-008 analysis selection | 1.379 |

The analysis-level selection erodes κ_λ contrast on its own, independently of the detector
model — the harder pT thresholds preferentially remove the soft, triangle-enhanced κ_λ = 5
events. **This compounds with §2.** Any measurement of "how much κ_λ separation does Delphes
retain" must fix the selection first, or the two effects will be confused.

## 5. What to investigate, in order

**A. Confirm the κ_λ-dependent residual is real.** The §2 numbers use the *first* 200k events
of each sample, which is not a random subset. Re-run at a different `--max-events` and, if
possible, from a different offset. Cheap, and it gates everything below.

**B. Measure the κ_λ separation directly — this is the headline number.** §1's claim is
currently an inference, not a measurement. Compute a like-for-like separation between
κ_λ = 0, 1, 5 on Delphes and on NanoAOD, with the *same* features and the *same* selection:
pairwise TV distance or AUC per feature, then a trained ratio. If Delphes separation is
materially below NanoAOD's, that is the result that makes this blocking; if it is not, §1's
premise needs revisiting. Do this before spending effort on fixes.

**C. Localise the residual.** Is the 1.043/1.082/1.185 acceptance or shape? Run `cutflow.py`
per κ_λ point on tuned and untuned and find which selection step introduces the κ_λ
dependence. If it is the ≥2-jet requirement, the escale hypothesis is confirmed.

**D. Test the escale mechanism.** If jets crossing the 20 GeV threshold are the cause, the
acceptance ratio should track jet pT — check it differentially in m_HH.

**E. Only then, choose a fix.** Options, roughly in increasing cost: condition the escale map
on a variable that tracks the m_HH spectrum; derive maps per κ_λ point and verify the ratio
is preserved; or accept the untuned samples for κ_λ work on the grounds that flat-wrong beats
tilted-wrong. Each needs B as its acceptance test.

## 6. Reproducing

Signal, tuned then untuned (three κ_λ points each):

```
pixi run python scripts/nsbi_overlay.py --config config.v1.yml \
  --ntuple /ceph/jpan/ntuples/merged \
  --nano-dir /ceph/jpan/cms_nanoaod_2024_hh2b2tau --nano-select PowhegBugFix \
  --out plots/overlay_signal_tuned --max-events 200000
```

Swap `--ntuple /ceph/jpan/ntuples_untuned/merged` and the `--out` for the untuned arm.
`--nano-select PowhegBugFix` is mandatory (two productions exist at κ_λ = 1) and is also the
correct choice: it is the campaign `maps_v1.json` was derived from.

tt̄, tuned then untuned:

```
pixi run python scripts/nsbi_overlay.py --config config.ttbar.yml \
  --ntuple /ceph/jpan/ntuples/merged_ttbar_ds \
  --nano-dir /ceph/jpan/cms_nanoaod_2024_hh2b2tau \
  --nano-select TTto2L2Nu --dataset TTto2L2Nu --background ttbar \
  --out plots/overlay_ttbar_tuned --max-events 200000
```

Untuned tt̄ is at `/ceph/jpan/ntuples_untuned/merged_ttbar`. Both tt̄ directories were
re-merged with `--allow-mixed-datasets --force` to get the `dataset_id` column; the original
`merged/` directories predate per-dataset labelling and `--dataset` will refuse on them.

## 7. Tooling added for this, already on `main`

- `--background LABEL` on `nsbi_overlay.py` — overlays a background process instead of the
  κ_λ points. The overlay was κ_λ-keyed and could not look at tt̄ at all.
- `--dataset SUBSTRING` — pins the Delphes side to one dataset via the merge manifest.
  Without it the overlay compares the full tt̄ cross-section mixture against one CMS dataset
  and shows a *composition* difference as if it were a detector one.
- Figures are now labelled from the merge manifest, not from the config. Previously an
  untuned ntuple read with a tuned config produced a figure titled "(tuned)".

## 8. Two other findings from the same work

**The tt̄ map is validated and it works** (first time it has been checked against its anchor):
τ_h rate 1.68× → 1.22× of CMS, yield 0.70 → 0.89. All ten features agree in shape *both
before and after* — which is the caution, not the reassurance: the overlays are
density-normalised, so a 1.68× rate error is invisible in them.

**FastMTT returns no solution for most events.** m_ττ, m_HH, pT^HH from FastMTT are NaN on
69 % of signal and 80 % of tt̄. The hadronic decay weight is non-zero only for
x ≥ (m_vis/m_τ)², m_τ = 1.777 GeV; a Delphes τ_h *is* an AK4 jet carrying 5–15 GeV of jet
mass, so the likelihood is identically zero. The elliptical signal region maps NaN → ∞ and
rejects those events, silently acting as a FastMTT-success filter (~31 % where CMS quotes
99 %), on survivors biased in the very mass it cuts on. **The failure rate is
process-dependent (69 % vs 80 %), so it does not cancel between signal and background.**
Mitigation is one line (clamp the τ_h leg mass to a physical τ visible mass); the real fix is
the τ_h rebuild from PF constituents. See `docs/notes/production_validation.tex` §3.2.
