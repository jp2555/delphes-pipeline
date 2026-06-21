# Level 3 — Simplified binned-m_HH expected limit (STUB)

A deliberately simple cut-based / single-discriminant binned-`m_HH` analysis on
the Delphes samples; check the expected limit against published expected limits
(≈4–5× σ_SM per experiment, note §6.1). Not a reproduction of the experiments'
BDT/DeepTau/fake-factor machinery — out of scope by design.

Blocked on: the full background model (note Table 3) + `cabinetry`-style binned
fit. Expose `run(ctx) -> list[CheckResult]`.
