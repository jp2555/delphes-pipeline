"""Z→ττ candle (note §6.2): the τ+pᵀᵐⁱˢˢ estimator chain.

Runs on the ``DYto2Tau`` Delphes sample with an opposite-sign ℓτ_h / τ_hτ_h
selection and a **b-jet veto** (decouples the candle from the b-tag tuning and
suppresses tt̄). The headline GATE is the covariance-free FastMTT m_ττ estimator
(decision D1, ``extensions.mtautau``) peak sitting at m_Z — a model-independent
validation of the τ + pᵀᵐⁱˢˢ chain. The candle also reports the **visible** m_ττ
peak/width (which sits below m_Z), the channel yield ratio, and the low-mass
fake-τ_h sideband.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.matching import matched_to_any
from delphes_pipeline.core.plotting import hist_overlay
from delphes_pipeline.core.result import CheckResult, Severity, info
from delphes_pipeline.extensions.mtautau import estimate_mtautau
from . import selections

_M_Z = 91.2


def run(ctx: ValidationContext, ev) -> list[CheckResult]:
    """Run the Z→ττ candle on the candle sample ``ev``."""
    veto = selections.bjet_veto_mask(ev)
    mass, n_tauh = selections.leading_visible_pair(ev, veto)
    results: list[CheckResult] = []

    core = mass[(mass > 40) & (mass < 120)]
    peak = float(np.median(core)) if core.size else float("nan")
    width = float(np.std(core)) if core.size else float("nan")
    plot = None
    if mass.size:
        plot = ctx.rel(hist_overlay(
            [("visible di-τ", mass, None)], bins=np.linspace(0, 200, 51),
            outpath=ctx.plot_path("candle_ztautau_mvis.png"), xlabel="visible m_ττ [GeV]",
            ylabel="events (norm.)", vlines=[(_M_Z, "m_Z")],
            title="Z→ττ visible m_ττ (below m_Z; neutrinos missing)",
        ))
    results.append(info("level1.ztautau.visible_peak", "level1", peak, units="GeV",
                        detail="visible m_ττ peak (sits below m_Z)", plot_path=plot))
    results.append(info("level1.ztautau.visible_width", "level1", width, units="GeV",
                        detail="visible m_ττ core width"))

    results.append(_peak_at_mz(ctx, ev))

    n_tt = int((n_tauh == 2).sum())
    n_lt = int((n_tauh == 1).sum())
    results.append(info("level1.ztautau.tautau_over_ltau", "level1",
                        float(n_tt / n_lt) if n_lt else float("nan"),
                        detail=f"τ_hτ_h / ℓτ_h channel ratio ({n_tt}/{n_lt})"))

    results.append(_low_mass_sideband(ev, mass))
    return results


def _peak_mode(x, *, lo=40.0, hi=160.0, w=5.0, halfwidth=20.0) -> float:
    """Robust peak: the local median within ``halfwidth`` of the densest histogram bin.

    The FastMTT m_ττ distribution is right-skewed (a high-mass leptonic tail), so a
    windowed *median* tracks the window edge, not the peak. Locating the mode (densest
    3-bin-smoothed bin) and taking the median of its neighbourhood is insensitive to
    both the far tail and the exact bin width — stable at ~88 GeV across seeds/bins.
    """
    x = x[(x >= lo) & (x <= hi)]
    if x.size < 20:
        return float("nan")
    edges = np.arange(lo, hi + w, w)
    h = np.convolve(np.histogram(x, bins=edges)[0], np.ones(3) / 3.0, mode="same")
    centre = 0.5 * (edges[int(np.argmax(h))] + edges[int(np.argmax(h)) + 1])
    core = x[np.abs(x - centre) <= halfwidth]
    return float(np.median(core)) if core.size else centre


def _peak_at_mz(ctx: ValidationContext, ev) -> CheckResult:
    """FastMTT m_ττ-estimator peak vs m_Z (note §6.2; estimator = D1, covariance-free).

    Reconstructs m_ττ for the leading τ-candidate pair (b-vetoed) and gates the *mode*
    of the distribution on m_Z. The visible peak sits below m_Z; the estimator, folding
    in pᵀᵐⁱˢˢ, should restore it — a model-independent validation of the chain. The
    leptonic channel carries a known residual high bias (covariance-free limitation),
    so the per-channel peaks and the high-mass tail fraction are reported alongside.
    """
    veto = selections.bjet_veto_mask(ev)
    sigma = float(ctx.tol("level1", "mtautau_met_sigma_gev", 25.0))  # fixed (covariance-free) MET σ
    tol = float(ctx.tol("level1", "mtautau_peak_tol_gev", 10.0))
    m = estimate_mtautau(ev, mask=veto, met_sigma=sigma)
    _, n_tauh = selections.leading_visible_pair(ev, veto)  # aligned with m (same selection)
    finite = np.isfinite(m)
    m, n_tauh = m[finite], n_tauh[finite]

    if m.size < 20:
        return CheckResult(
            name="level1.ztautau.peak_at_mZ", level="level1", passed=False, severity=Severity.WARN,
            detail=f"only {m.size} reconstructed di-τ pairs; cannot evaluate the m_Z peak")

    peak = _peak_mode(m)
    tail_frac = float(np.mean(m > 150.0))
    pk_tt = _peak_mode(m[n_tauh == 2]) if (n_tauh == 2).sum() >= 20 else float("nan")
    pk_lt = _peak_mode(m[n_tauh == 1]) if (n_tauh == 1).sum() >= 20 else float("nan")
    plot = ctx.rel(hist_overlay(
        [("FastMTT m_ττ", m, None)], bins=np.linspace(0, 200, 51),
        outpath=ctx.plot_path("candle_ztautau_mtautau.png"), xlabel="FastMTT m_ττ [GeV]",
        ylabel="events (norm.)", vlines=[(_M_Z, "m_Z")],
        title="Z→ττ FastMTT m_ττ (covariance-free)",
    ))
    return CheckResult(
        name="level1.ztautau.peak_at_mZ", level="level1",
        passed=bool(abs(peak - _M_Z) <= tol), severity=Severity.GATE,
        measured=peak, target=_M_Z, tolerance=tol, units="GeV",
        detail=(f"FastMTT m_ττ peak {peak:.1f} GeV vs m_Z {_M_Z:.1f} (±{tol:.0f}); "
                f"τ_hτ_h {pk_tt:.0f} / ℓτ_h {pk_lt:.0f}, tail(m>150) {tail_frac:.0%}"),
        plot_path=plot,
        extra={"peak_tautau": pk_tt, "peak_ltau": pk_lt, "tail_fraction": tail_frac, "n_pairs": int(m.size)},
    )


def _low_mass_sideband(ev, mass) -> CheckResult:
    """Sub-Z region fake content (jet→τ_h misid check)."""
    side_frac = float(np.mean(mass < 60)) if mass.size else float("nan")
    gen_taus = ev.gen[np.abs(ev.gen.pid) == 15]
    th = ev.jets[ev.jets.tautag == 1]
    n_th = int(ak.sum(ak.num(th)))
    fake = float(ak.sum(~matched_to_any(th, gen_taus, 0.4)) / n_th) if n_th else float("nan")
    return info("level1.ztautau.lowmass_sideband", "level1", side_frac,
                detail=f"sub-Z (m<60) fraction {side_frac:.2f}; jet→τ_h fake fraction {fake:.2f}",
                extra={"fake_tauh_fraction": fake})
