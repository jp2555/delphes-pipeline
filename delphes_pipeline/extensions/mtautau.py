"""di-τ mass estimator (note §3.3, decision D1): covariance-free FastMTT.

Reconstructs the di-τ invariant mass from the two visible legs + pᵀᵐⁱˢˢ by
maximising, per event, a likelihood over the two visible energy fractions
x1, x2 ∈ (0,1]:

    L(x1, x2) = L_MET(x1, x2) · L_dec(x1; leg1) · L_dec(x2; leg2)

- **collinear approximation:** each τ momentum is its visible momentum scaled by
  1/x, so the neutrino system is Σ_i p_vis,i·(1−x_i)/x_i and must match pᵀᵐⁱˢˢ;
- **L_MET** is Gaussian with a *fixed* resolution — covariance-free. Full
  FastMTT/SVfit uses the per-event pᵀᵐⁱˢˢ covariance, which needs the MET-resolution
  tuning; swapping the fixed σ for that synthesised covariance is the later upgrade.
- **L_dec** is the τ-decay phase-space density: flat for a hadronic τ above the
  kinematic threshold (m_vis/m_τ)², the lepton energy spectrum (5−9x²+4x³) for ℓ.

``m_ττ = m_vis / √(x1·x2)`` at the maximum (exact for massless visible legs). The
estimator is the hook the Z→ττ candle's "peak at m_Z" check consumes.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

M_TAU = 1.777


def _decay_weight(x, is_had, m_vis):
    """τ-decay phase-space density vs visible energy fraction ``x`` (per leg)."""
    xmin = (np.asarray(m_vis, dtype=float) / M_TAU) ** 2
    had = ((x >= xmin) & (x <= 1.0)).astype(float)               # flat above threshold
    lep = np.clip(5.0 - 9.0 * x * x + 4.0 * x ** 3, 0.0, None)   # lepton energy spectrum
    return np.where(is_had, had, lep)


def _pair_inv_mass(leg1, leg2, sel):
    e = leg1["e"][sel] + leg2["e"][sel]
    px = leg1["px"][sel] + leg2["px"][sel]
    py = leg1["py"][sel] + leg2["py"][sel]
    pz = leg1["pz"][sel] + leg2["pz"][sel]
    return np.sqrt(np.maximum(e * e - px * px - py * py - pz * pz, 0.0))


def fastmtt_mass(leg1, leg2, met_x, met_y, *, met_sigma=25.0, grid=40, chunk=4000):
    """Per-event covariance-free FastMTT di-τ mass over a (grid×grid) x scan.

    ``leg1``/``leg2`` are dicts of numpy arrays (``px,py,pz,e,mass,is_had``); ``met_x``,
    ``met_y`` the pᵀᵐⁱˢˢ components. Returns ``m_ττ`` per event (NaN where the
    likelihood is everywhere zero).
    """
    met_x = np.asarray(met_x, dtype=float)
    met_y = np.asarray(met_y, dtype=float)
    n = met_x.size
    xs = np.linspace(1.0 / grid, 1.0, grid)        # avoid x=0 (divergent neutrino)
    x1 = xs[None, :, None]                          # (1,G,1)
    x2 = xs[None, None, :]                          # (1,1,G)
    r1, r2 = (1.0 - x1) / x1, (1.0 - x2) / x2
    out = np.full(n, np.nan)
    for s in range(0, n, chunk):
        e = slice(s, min(s + chunk, n))
        col = lambda a: np.asarray(a[e], dtype=float)[:, None, None]
        nu_x = col(leg1["px"]) * r1 + col(leg2["px"]) * r2
        nu_y = col(leg1["py"]) * r1 + col(leg2["py"]) * r2
        dx, dy = col(met_x) - nu_x, col(met_y) - nu_y
        like = np.exp(-0.5 * (dx * dx + dy * dy) / (met_sigma * met_sigma))
        like = like * _decay_weight(x1, col(leg1["is_had"]).astype(bool), col(leg1["mass"]))
        like = like * _decay_weight(x2, col(leg2["is_had"]).astype(bool), col(leg2["mass"]))
        flat = like.reshape(like.shape[0], -1)
        best = np.argmax(flat, axis=1)
        i1, i2 = np.divmod(best, grid)
        m = _pair_inv_mass(leg1, leg2, e) / np.sqrt(xs[i1] * xs[i2])
        m[flat[np.arange(flat.shape[0]), best] <= 0.0] = np.nan
        out[e] = m
    return out


def _leg(cand):
    pt = ak.to_numpy(cand.pt); eta = ak.to_numpy(cand.eta)
    phi = ak.to_numpy(cand.phi); m = ak.to_numpy(cand.mass)
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return {"px": px, "py": py, "pz": pz, "e": np.sqrt(px * px + py * py + pz * pz + m * m),
            "mass": m, "is_had": ak.to_numpy(cand.is_tauh).astype(bool)}


def estimate_mtautau(ev, *, mask=None, method: str = "fastmtt", met_sigma=25.0, grid=40):
    """Per-event di-τ mass for the leading τ-candidate pair (τ_h jets + reco e/μ).

    ``mask`` restricts to selected events (e.g. the b-veto). Returns the ``m_ττ``
    array over events with ≥2 candidates within the mask.
    """
    if method != "fastmtt":
        raise ValueError(f"only the covariance-free FastMTT method is implemented (got {method!r})")
    from delphes_pipeline.validation.level1_candles.selections import tau_candidates

    cand = tau_candidates(ev)
    sel = ak.to_numpy(ak.num(cand) >= 2)
    if mask is not None:
        sel = sel & np.asarray(mask, dtype=bool)
    pair = cand[sel]
    pair = pair[ak.argsort(pair.pt, axis=1, ascending=False)][:, :2]
    met = ev.met[sel]
    met_x = ak.to_numpy(met.met * np.cos(met.phi))
    met_y = ak.to_numpy(met.met * np.sin(met.phi))
    return fastmtt_mass(_leg(pair[:, 0]), _leg(pair[:, 1]), met_x, met_y,
                        met_sigma=met_sigma, grid=grid)
