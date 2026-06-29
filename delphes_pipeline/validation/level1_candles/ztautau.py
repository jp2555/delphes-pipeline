"""Z→ττ candle (note §6.2): the τ+pᵀᵐⁱˢˢ estimator chain.

Runs on the ``DYto2Tau`` Delphes sample with an opposite-sign ℓτ_h / τ_hτ_h
selection and a **b-jet veto** (decouples the candle from the b-tag tuning and
suppresses tt̄). The headline check is the m_ττ estimator peak sitting at m_Z —
which needs the m_ττ estimator (decision D1, ``extensions.mtautau``). Until that
is built this candle reports the **visible** m_ττ peak/width (which sits below
m_Z), the channel yield ratio, and the low-mass fake-τ_h sideband, and marks the
m_Z check pending.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.matching import matched_to_any
from delphes_pipeline.core.plotting import hist_overlay
from delphes_pipeline.core.result import CheckResult, info
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


def _peak_at_mz(ctx: ValidationContext, ev) -> CheckResult:
    """m_ττ-estimator peak vs m_Z — pending the estimator (decision D1)."""
    try:
        from delphes_pipeline.extensions.mtautau import estimate_mtautau

        estimate_mtautau(ev, method="collinear")  # not implemented yet
    except NotImplementedError:
        return info("level1.ztautau.peak_at_mZ", "level1",
                    detail="needs the m_ττ estimator (D1, extensions.mtautau); "
                           "the model-independent 'peak at m_Z' check turns on once it is built")
    return info("level1.ztautau.peak_at_mZ", "level1", detail="estimator available")  # pragma: no cover


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
