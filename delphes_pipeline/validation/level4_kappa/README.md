# Level 4 — κ_λ-critical: acceptance & response in m_HH (STUB)

The headline result depends on this level. Two deliverables (note §6.3):

1. **A×ε vs truth m_HH**, per channel, Delphes vs the private NanoAOD anchor and
   any published acceptance information — emphasis on the 250–400 GeV threshold
   region where the triangle amplitude sculpts the κ_λ acceptance.
2. **m_HH response/migration matrix** (reconstructed vs truth), whose width is
   dominated by the pᵀᵐⁱˢˢ-driven m_ττ estimator — doubles as the closure test
   for decision D1 (the τʈ mass-estimator choice, note §3.3).

A mismodelled low-m_HH turn-on biases the κ_λ interval directly even when all
integrated yields match, so this level cannot afford the sloppiness a yield-level
study can.

Blocked on: truth m_HH from `GenPart` (available now) **and** the private anchor
for comparison. The A×ε-vs-truth-m_HH turn-on (deliverable 1) is partly runnable
on signal alone — a natural first extension beyond the pilot gate. Expose
`run(ctx) -> list[CheckResult]`.
