# Level 1 — Standard candles (STUB)

Two candles that together exercise every Tier-1 object (note §6.2). This is the
first level at which the simulation can genuinely fail. **Blocked on background
sample production** (note Table 3).

## Z → ττ — the τ + pᵀᵐⁱˢˢ-estimator chain
- Selection: same object defs + trigger emulation as the analysis (ℓτ_h, τ_hτ_h);
  two opposite-sign τ; **b-jet veto** (decouples from b-tag tuning).
- Generator requirement: τ spin correlations correct (`TauDecays:externalMode`).
- Checks: estimator peak at m_Z (±2–3 GeV, model-independent); peak width vs
  anchor (10–15%); yield + channel ratios (±10%); low-mass sideband (fake rate).

## tt̄ dilepton — b-tagging, leptons, real pᵀᵐⁱˢˢ
- Selection: eμ channel, one and two b-tags (eμ removes Z by construction).
- Checks: absolute yield (±10–15%, σ known to ~3%); pᵀᵐⁱˢˢ tail shape; the
  in-situ tag-counting closure ε_b = 2N₂/(N₁+2N₂) vs the tuned input.

## Implementation contract
Expose `run(ctx) -> list[CheckResult]` here, reading the candle samples from
config (`input.candles.{ztautau,ttbar}`). Reuse `core.plotting` overlays and the
`Severity` model. See note §6.2 and the diagnostic map (Table 5).
