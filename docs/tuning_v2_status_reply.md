# Tuning session — status against the v2 public-anchor plan

*Written 2026-08-17 (week 1, day 1 of the schedule to 30 Sep). Replies to
`delphes_tuning_v2_public_anchor_plan_handoff.md` Rev 2, and reports what landed from
`delphes_tuning_v1_audit_reply_handoff.md` §B/§C. The rules note remains binding
throughout.*

---

## 1. Week-1 schedule (§G) — where each item stands

| § | item | state | evidence |
|---|---|---|---|
| G1 | Anchors file with named placeholders | **done** | `cards/tuning/anchors_v2.yml` |
| C1–C3 | §C decisions locked | **done** | recorded in the anchors file: 13.6 TeV both sides with an explicit "do not reintroduce a √s rescale"; Run-3 tagger package; ⟨μ⟩ slot |
| C4 | PU-fake residual bound computed | **not started** | the four factor slots + escalation criterion exist as `TOVERIFY`; needs public JME/TAU numbers |
| G1 | Public refs collected | **not started** | every `citation` is `TOVERIFY`. **This is the critical path — see §5** |
| A1/E-1 | Swap-map A/B on v1 | **tool built, not run** | `scripts/swap_map_ab.py` — paired arms (same events, same seed) so the delta is attributable to the maps, not RNG |
| G1 | Begin b/τ map derivation | **blocked** | on public refs |
| G1 | τ-object rebuild started | **not started; feasibility settled** | see §3.3 |

## 2. Audit-reply items — checked and landed

**Checked, with answers:**

- **Q2 aggravation-2 — it splits.** `nsbi_overlay.py` builds m_HH from the *FastMTT-scaled*
  ττ system, so the escalation trigger fires there. But the SBI training input does **not**:
  `convert_powheg_to_sbi.py` maps `m_vis → m_tautau` and builds `m_hh` from bb + **visible**
  ττ, and `scripts/delphes_to_sbi.py` matches that. **The primary κ_λ discriminant in the fit
  is not FastMTT-scaled**, so Q2 stays moderate rather than serious. The escalation condition
  was written against the overlay's definition, not the fit's.
- **C-1 — confirmed present.** `retag_btag` and `retag_tautag` draw independent Bernoullis
  with no exclusivity, and both the overlay and the SBI selection clean jets only against the
  *selected* legs — so a τ-tagged jet that isn't selected stays in the b pool carrying
  `btag=1`. `scripts/tag_exclusivity_check.py` measures the rate (not yet run on the merged
  ntuples). Joint categorical draw remains the v2 fix.
- **C-5(iv) — consistent today.** `smear_met` reads `jet_ht` from the *unretagged* events and
  `derive_maps` measures on raw Delphes jets. Same object state both sides. ⚠ This breaks the
  moment fluctuation propagation lands (Q2) and must be re-checked then.
- **Q5 — closeable.** Shards are contiguous groups of *files*, each an independent generator
  job, so they partition events and never phase space. One nuance: shards are cut on file
  *size*, which correlates weakly with event content — harmless, since stream identity never
  enters the density, but it is not a pure index partition.
- **C-3 — answered, and it is the "barrel SF" branch.** `objects._lepton_sf` applies the SF to
  **all** reco leptons at **all** η, indexed by pT alone, while `observables.lepton_efficiency`
  derives it on `barrel=1.5` prompt leptons. So |η| > 1.5 leptons receive a **barrel-derived
  correction**, across the ECAL crack (1.44–1.57) where reco efficiency genuinely differs.
  Interacts with the 20% process difference exactly as C-3 predicted. **Second, unflagged
  issue in the same function:** the weight is the product over *all* reco leptons, not the
  selected ones — wrong for a channel that selects exactly one lepton, and the docstring
  already admits it. Both fixed by extending to (pT, |η|) and restricting to selected leptons.

**Landed:**

- **Q8 σ_eff, codified.** The merged manifest now carries `sum_genweight` (signed, negatives
  included), `sum_genweight_x_lepton_sf`, `shards` and `shards_planned` — numerator and
  denominator over the *processed* shard set, so no generated count the sample doesn't
  correspond to can be quoted.
- **Ladder-weight rule.** `delphes_to_sbi.py` folds `lepton_sf` into `weights` when present.
  1.0 on the untuned arm, correct for v2; closes the case where ratio training consumes a
  weight column the histogram baseline doesn't.
- **C-5(i).** `_MASS_QUANTILES` 21 → 101 (same tail defect measured on the response grid:
  7.2% → 0.11%).
- **C-5(v), partly.** Map provenance is pinned by **content hash**, not filename; the merge
  announces when samples carry different map sets. The **card** is still not in that chain.

## 3. New findings — neither note anticipated these

### 3.1 A live v1 bug: the MET conditioning variable is PU-inclusive [P0]

Rev 2's §D MET row flags this as a v2 design requirement. **It is already wrong in v1 and is
baked into the tuned ntuples.** `observables.jet_ht` sums jets with `pt > 20 GeV` out to
`|η| ≤ 4.7` — about the most PU-exposed activity variable available, forward jets included. On
the CMS anchor that H_T is PU-inclusive; on no-PU Delphes it is not. At equal hard-scatter
activity the anchor sits higher, so a Delphes event reads a systematically **low** bin of
`met_smear` and MET is **under-smeared**.

Direction: cleaner MET → better-constrained FastMTT → sharper m_ττ → **inflated unbinned
gain**. Optimistic, consistent with the audit reply's overall verdict.

⚠ **This partially overlaps the §C4 credit.** The perfect-mitigation framing says our no-PU
samples are the anchor's pipeline with the residual set to zero. Under-smeared MET is a
*second, independent* no-PU error in the same optimistic direction, and it is **not** covered
by the PU-jet-fake residual bound. Two terms, one sign — the paper's detector-model paragraph
should say so.

### 3.2 The mitigation hinge is now enforced in code

Per the note accompanying Rev 2: the public rates we derive from must be the
post-PU-jet-ID / post-PV-association numbers, or the residual logic double-counts. Every
rate/efficiency anchor carries `mitigation_state`, and `anchors.for_derivation()` refuses
`TOVERIFY`, refuses `pre_mitigation` naming the double-count, and accepts only
`post_mitigation` or an explicit `not_applicable`. A second guard refuses any map declaring
`conditioning_must_be_pu_independent` while binned in `sum_et` / `jet_ht` / `n_vertices` /
`rho`. The derivation/validation firewall is enforced the same way, including against
re-tagging a validation target to get past it.

### 3.3 τ rebuild from constituents: feasible, with an unpriced cost

`cards/cms_card_v1.tcl` writes `EFlowTrack`, `EFlowPhoton`, `EFlowNeutralHadron` — **the
constituents are in the existing files, no re-generation needed.** Two consequences:

- The rebuild must happen **at ntuplization**, since our schema drops those branches. Already
  produced ntuples cannot be retrofitted.
- The 35-leaf whitelist is what keeps campaign transfers at a few percent of ~32 TB. Adding
  three EFlow collections raises read volume substantially, and XRootD reads whole baskets so
  we cannot fetch only near-τ constituents. **Week 3's "apply → ntuples v2-rc" is where this
  lands; it needs a one-shard timing probe before the schedule commits to it.**
- Delphes stores `Jet.Constituents` as a TRefArray that uproot will not resolve. The practical
  path is ΔR association in a narrow HPS-like signal cone — an emulation choice to state in the
  paper, not a readout.

## 4. Blocked or needing the other session

| item | why |
|---|---|
| All §D derivation | no public references collected yet (§5) |
| Tier-3 gate, equivalent-luminosity anchoring | anchor not public; `require_verified()` refuses to run the gate on placeholders |
| §C4 residual bound | needs public JME PU-jet rates + PU-jet-ID WP efficiency + fake probability |
| E-5 for tt̄ | cannot verify from here whether the Delphes tt̄ gen matches the CMS TT campaign. If it does not, `maps_ttbar_v1.json` absorbed gen differences — **and the swap-map A/B number needs that caveat attached when quoted** |
| F1 band mechanism | not built. Several variations (±1σ on b/τ/MET resolutions, the PU-fake proxy inflating j→τ and light-mistag by the C4 bound, clip-range choices) are all "perturb a named map and re-apply" and want one generic knob, not five scripts. Week-5 work, buildable ahead |
| Card in the provenance chain | C-5(v); I would fold it in the way that session specifies rather than guess |

## 5. Critical path — and it is not the Tier-3 anchor

The plan treats the unpublished Run-3 result as the main timing risk. It is not the binding
one for **this** session: §C says derivation does not block on it, and that is right — the
anchor gates the *gate*, not the maps.

**The binding constraint is that no public reference has been collected for any map.** Every
`citation` in the anchors file is `TOVERIFY`, and §D cannot start without them. Items G6 and
G7 (begin b/τ derivation, τ-object rebuild) are scheduled for week 1 and are both downstream
of a task nobody has started. With map freeze on **Sep 11**, and the τ rebuild carrying an
unpriced transfer cost that only reveals itself at week 3, the reference-collection task is
worth pulling to the front of week 1 ahead of everything else.

Concretely, the smallest set that unblocks derivation: BTV WP efficiencies/mistags, JME
JER/JES, TAU POG efficiency + fake rate + energy scale (post-mitigation — §3.2), EGM/MUO SFs,
MET performance σ vs activity at the anchor ⟨μ⟩. Trigger parameterisations can follow.

## 6. Untuned arm (§E) — effectively complete

The fallback/band-endpoint arm is produced: `signal` (3 κ_λ), `ttbar`, and DY `0J/1J/2J`
complete; `dy_low` recovering after a memory hold. Identity maps everywhere
(`maps_sha = "untuned"`), separate campaign directory, never mixed with tuned samples.
`scripts/delphes_to_sbi.py` converts to the SBI input format, reproducing the CMS converter's
definitions exactly — including the four that differ from this repo's overlay (visible
`m_tautau`, `cos_theta_star = tanh(Δη/2)`, `pt_h1`/`pt_h2` fixed by content, visible-ττ
`m_hh`) and the `mt`/`et` channel restriction.

**Outstanding on this arm:** the four DY bins each carry their own cross section and must be
weighted before combining into one `tree_dy`. Those cross sections are not in this repo.

⚠ **A process note worth carrying to the C-5 version-locking rules.** The untuned campaign took
four attempts; none of the failures was in the physics (a sentinel absolutised into a path, a
shell guard defeated by HTCondor whitespace, a warn-instead-of-refuse on a missing grid proxy,
and a byte-based shard cap that assumes constant bytes-per-event — false for low-mass DY).
Provenance hashes recorded the *intent* correctly through all of it while zero events were
produced. **Hashes pin what was intended, not what executed.** The only check that would have
caught all four is one real batch job's exit code before releasing the queue — and note that a
shard run by hand on the submit node cannot catch the proxy case, because the submit host can
read credentials the workers cannot.
