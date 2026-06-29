# Level 1 — Standard candles (note §6.2)

Two candles on the **background** samples that together exercise every Tier-1
object. `run(ctx)` opens `config.candles.{ttbar,ztautau}` and runs both
(`ttbar.py`, `ztautau.py`); shared selections live in `selections.py`.

## tt̄ dilepton — `TTto2L2Nu` (built)
- Selection: eμ opposite-sign (removes Z by construction → high tt̄ purity).
- **`eb_insitu_closure` (GATE)**: ε_b = 2N₂/(N₁+2N₂) over events with exactly two
  truth b-jets (note Eq. 2), vs the tuned input (card formula; the NanoAOD anchor
  once wired) — extracts the per-jet b-eff from the topology itself.
- `acceptance` (A×ε) and the `met_mean` / tail (real-neutrino pᵀᵐⁱˢˢ) — info.

## Z → ττ — `DYto2Tau` (scaffolded; m_Z peak pending the estimator)
- Selection: ℓτ_h / τ_hτ_h with a **b-jet veto** (decouples from b-tag tuning).
- `visible_peak` / `visible_width` — the **visible** m_ττ (sits below m_Z).
- `peak_at_mZ` — the model-independent headline check; **pending the m_ττ
  estimator** (decision D1, `extensions/mtautau.py`). Turns on once that is built.
- `tautau_over_ltau` channel ratio and the `lowmass_sideband` jet→τ_h fake
  fraction — info.

## Remaining
The estimator-based m_ττ peak/width (D1), POG/anchor yield targets (the ±10–15%
comparisons), and the trigger emulation (note §4.1) that the selections should
fold in. Severities are mostly `info` until those targets are wired; the tt̄ ε_b
closure is the one `GATE`.
