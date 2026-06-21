"""Level-0 MET resolution.

There is no card formula for missing transverse energy, so this is a resolution
*measurement* with a sanity ceiling rather than a closure test. We compare the
reconstructed MET to the generator-level MET component-wise — ``dx = met_x -
genmet_x``, ``dy = met_y - genmet_y`` — and report the per-component RMS,
sqrt(0.5 * (var(dx) + var(dy))), both overall and binned in sum E_T.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.plotting import resolution_plot
from delphes_pipeline.core.result import CheckResult, gate_max, info


def _xy(record: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian (x, y) components of a per-event ``(met, phi)`` record.

    Empty events have a ``None`` record; ``ak.fill_none`` maps them to zero so
    the returned arrays are dense and length ``ev.n``.
    """
    met = ak.to_numpy(ak.fill_none(record.met, 0.0))
    phi = ak.to_numpy(ak.fill_none(record.phi, 0.0))
    return met * np.cos(phi), met * np.sin(phi)


def _resolution(dx: np.ndarray, dy: np.ndarray) -> float:
    """Per-component resolution sqrt(0.5 * (var(dx) + var(dy))) (nan if empty)."""
    if dx.size == 0:
        return float("nan")
    return float(np.sqrt(0.5 * (np.var(dx) + np.var(dy))))


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Measure the MET resolution vs sum E_T and gate on its overall value."""
    ev = ctx.events
    mx, my = _xy(ev.met)
    gmx, gmy = _xy(ev.genmet)
    dx, dy = mx - gmx, my - gmy
    sumet = ak.to_numpy(ak.fill_none(ev.scalar_ht.ht, 0.0))

    overall_res = _resolution(dx, dy)
    scale_bias = float(np.mean(np.concatenate([dx, dy])))

    edges = ctx.opt("level0", "sumet_bins", [0, 100, 200, 300, 500, 800, 1200])
    min_count = ctx.opt("level0", "min_bin_count", 25)
    centers: list[float] = []
    per_bin_res: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sumet >= lo) & (sumet < hi)
        if int(np.count_nonzero(sel)) < min_count:
            continue
        centers.append(0.5 * (lo + hi))
        per_bin_res.append(_resolution(dx[sel], dy[sel]))

    plot_rel = None
    if centers:
        plot_abs = resolution_plot(
            centers,
            per_bin_res,
            outpath=str(ctx.plot_path("met_resolution.png")),
            xlabel="sum E_T [GeV]",
            ylabel="MET resolution [GeV]",
        )
        plot_rel = ctx.rel(plot_abs)

    resolution_result = gate_max(
        "level0.met.resolution",
        "level0",
        overall_res,
        ctx.tol("level0", "met_resolution_max_gev", 40.0),
        units="GeV",
        detail="overall sqrt(0.5*(var(dx)+var(dy))) of MET - GenMET",
        plot_path=plot_rel,
        extra={"centers": centers, "resolution": per_bin_res},
    )
    bias_result = info(
        "level0.met.scale_bias",
        "level0",
        scale_bias,
        units="GeV",
        detail="mean of (MET - GenMET) over both components",
    )
    return [resolution_result, bias_result]
