"""Level-0 MET resolution.

There is no card formula for missing transverse energy, so this is a resolution
*measurement* with a sanity ceiling rather than a closure test. The per-event
residuals and the binned resolution vs sum E_T live in
:mod:`delphes_pipeline.core.observables` (shared with the tuning and plot lenses);
this module gates on the overall resolution and reports the scale bias.
"""

from __future__ import annotations

import numpy as np

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.plotting import resolution_plot
from delphes_pipeline.core.result import CheckResult, gate_max, info


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Measure the MET resolution vs sum E_T and gate on its overall value."""
    dx, dy, _ = obs.met_residuals(ctx.events)
    overall_res = float(np.sqrt(0.5 * (np.var(dx) + np.var(dy)))) if dx.size else float("nan")
    scale_bias = float(np.mean(np.concatenate([dx, dy]))) if dx.size else float("nan")

    bins = ctx.opt("level0", "sumet_bins", obs.DEFAULT_SUMET_BINS)
    min_count = int(ctx.opt("level0", "min_bin_count", 25))
    prof = obs.met_resolution(ctx.events, bins=bins, min_count=min_count)

    plot_rel = None
    if prof.centers.size:
        plot_abs = resolution_plot(
            prof.centers, prof.values,
            outpath=str(ctx.plot_path("met_resolution.png")),
            xlabel="sum E_T [GeV]", ylabel="MET resolution [GeV]",
        )
        plot_rel = ctx.rel(plot_abs)

    resolution_result = gate_max(
        "level0.met.resolution", "level0", overall_res,
        ctx.tol("level0", "met_resolution_max_gev", 40.0), units="GeV",
        detail="overall sqrt(0.5*(var(dx)+var(dy))) of MET - GenMET",
        plot_path=plot_rel,
        extra={"centers": prof.centers.tolist(), "resolution": prof.values.tolist()},
    )
    bias_result = info(
        "level0.met.scale_bias", "level0", scale_bias, units="GeV",
        detail="mean of (MET - GenMET) over both components",
    )
    return [resolution_result, bias_result]
