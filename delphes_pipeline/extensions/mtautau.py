"""τ_h τ_h mass estimator (note §3.3, decision D1) — extension stub.

The τ_h energy response and the m_ττ estimator drive m_ττ and, through it, m_HH.
The reference analysis regresses the neutrino momenta with a per-era DNN that
consumes the pᵀᵐⁱˢˢ covariance matrix, which Delphes does not provide natively.
Decision **D1** (note §3.3) is which estimator to adopt:

1. **FastMTT / SVfit-class** — synthesise the pᵀᵐⁱˢˢ covariance from the tuned
   resolution parametrisation (σ_xx, σ_yy, ρ vs ΣE_T) and feed it to FastMTT;
2. **covariance-free** — a collinear approximation with a fallback, whose κ_λ
   interval cost is measured against option 1.

``estimate_mtautau`` is the pluggable hook; the visible-only proxy already exists
in ``plots.quantities.mtautau_visible``. The covariance synthesis depends on the
MET-resolution tuning (``observables.met_resolution``), so this follows the MET
tuning in the loop.
"""

from __future__ import annotations


def estimate_mtautau(events, *, method: str = "fastmtt"):
    """STUB: per-event τ_h τ_h mass estimate (decision D1).

    ``method='fastmtt'`` requires a synthesised pᵀᵐⁱˢˢ covariance;
    ``method='collinear'`` is the covariance-free fallback to benchmark against it.
    """
    raise NotImplementedError(
        "m_tautau estimator is decision D1 (note §3.3); the visible-mass proxy is "
        "plots.quantities.mtautau_visible. Implement FastMTT (with synthesised "
        "pTmiss covariance) or a covariance-free fallback here."
    )
