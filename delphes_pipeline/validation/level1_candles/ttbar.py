"""tt̄ dilepton candle (note §6.2): in-situ b-tag closure + yield + pᵀᵐⁱˢˢ shape.

Runs on the ``TTto2L2Nu`` Delphes sample. The eμ requirement gives a high-purity
tt̄ sample by construction. The headline check is the in-situ tag-counting closure
ε_b = 2N₂/(N₁+2N₂), which extracts the per-jet b-tag efficiency from the tt̄
topology itself (independent of normalisation) and compares it to the tuned input
(the card formula, or the NanoAOD anchor once wired) — a GATE that exercises the
b-tag tuning the way the experiment does.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.plotting import hist_overlay
from delphes_pipeline.core.result import CheckResult, Severity, info
from . import selections


def run(ctx: ValidationContext, ev) -> list[CheckResult]:
    """Run the tt̄ dilepton candle on the candle sample ``ev``."""
    emu = selections.emu_os_mask(ev)
    n_sel = int(emu.sum())
    results: list[CheckResult] = [
        info("level1.ttbar.n_emu_os", "level1", float(n_sel), detail="eμ opposite-sign events"),
        info("level1.ttbar.acceptance", "level1", float(n_sel / ev.n) if ev.n else float("nan"),
             detail="eμ-OS acceptance × efficiency (A×ε)"),
    ]

    # --- in-situ ε_b closure (the GATE) ---
    eb, N1, N2, bb_pt = selections.eb_insitu(ev, emu)
    rel_tol = float(ctx.tol("level1", "eb_closure_rel_tol", 0.10))
    if bb_pt.size and np.isfinite(eb):
        # tuned input: the card-formula b-efficiency averaged over the truth-b pT
        input_eff = float(np.mean(ctx.references.expected("btag_eff_b", bb_pt, np.zeros_like(bb_pt))))
        rel = abs(eb / input_eff - 1.0) if input_eff else float("inf")
        passed = rel <= rel_tol
        detail = (f"in-situ ε_b = {eb:.3f} (N1={N1}, N2={N2}) vs tuned input {input_eff:.3f} "
                  f"-> {eb / input_eff - 1.0:+.1%}")
    else:
        input_eff, passed = float("nan"), False
        detail = f"too few 2-b eμ events for the closure (N1={N1}, N2={N2})"
    results.append(CheckResult(
        name="level1.ttbar.eb_insitu_closure", level="level1", passed=passed,
        severity=Severity.GATE, measured=eb, target=input_eff, tolerance=rel_tol,
        detail=detail, extra={"N1": N1, "N2": N2, "n_two_b_events": N1 + N2},
    ))

    # --- pᵀᵐⁱˢˢ shape in eμ events (real neutrinos -> tail test) ---
    met = ak.to_numpy(ak.fill_none(ev.met.met, 0.0))[emu]
    if met.size:
        plot = hist_overlay(
            [("tt̄ eμ", met, None)], bins=np.linspace(0, 300, 61),
            outpath=ctx.plot_path("candle_ttbar_met.png"), xlabel="pT_miss [GeV]",
            ylabel="events (norm.)", title="tt̄ eμ pT_miss (real neutrinos)",
        )
        results.append(info("level1.ttbar.met_mean", "level1", float(met.mean()), units="GeV",
                            detail="mean pT_miss in eμ events", plot_path=ctx.rel(plot),
                            extra={"met_tail_frac_gt100": float(np.mean(met > 100))}))
    return results
