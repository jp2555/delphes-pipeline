"""Level 4 — κ_λ-critical m_HH faithfulness (note §6.3).

κ_λ is measured from the m_HH lineshape near the 250–400 GeV threshold (the box–triangle
interference). This rung confirms Delphes preserves that observable well enough that the
κ_λ sensitivity carries over: it reconstructs **gen m_HH** (2 leading gen b-quarks + 2 gen
τ's) and **reco m_HH** (the two highest-b-tag jets + the FastMTT di-τ system), then reports
the A×ε turn-on, the m_HH response + resolution, and the gen→reco migration. The GATE is on
the threshold window: the m_HH **scale** unbiased and the **resolution** tight enough to keep
the κ_λ structure (which varies on a ~tens-of-GeV scale) from washing out.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult, Severity, info
from delphes_pipeline.extensions.mtautau import ditau_system

_MHH_BINS = [250, 300, 350, 400, 500, 700, 1000]
_THRESHOLD = (250.0, 400.0)   # the κ_λ-critical window
_HIGGS_PID = 25


def _coll_p4_sum(coll) -> tuple:
    """Sum 4-momenta of a (leading-2) jagged collection per event -> (px,py,pz,e) numpy."""
    px = coll.pt * np.cos(coll.phi)
    py = coll.pt * np.sin(coll.phi)
    pz = coll.pt * np.sinh(coll.eta)
    e = np.sqrt(px * px + py * py + pz * pz + coll.mass * coll.mass)
    return (ak.to_numpy(ak.sum(px, axis=1)), ak.to_numpy(ak.sum(py, axis=1)),
            ak.to_numpy(ak.sum(pz, axis=1)), ak.to_numpy(ak.sum(e, axis=1)))


def _mass(px, py, pz, e):
    return np.sqrt(np.maximum(e * e - px * px - py * py - pz * pz, 0.0))


def _gen_mhh(ev) -> np.ndarray:
    """gen m_HH = invariant mass of the two gen Higgs (|pid|==25); NaN if <2 are present.

    Selects the two leading-pT Higgs after preferring hard/intermediate copies (status>=20)
    — the b-quarks and τ's proliferate into many shower/status copies in the full Pythia
    record, so the Higgs are the robust gen handle (same heuristic as plots.quantities.gen_mhh).
    """
    gen = ev.gen
    h = gen[np.abs(gen.pid) == _HIGGS_PID]
    if "status" in h.fields:
        hard = h[h.status >= 20]
        h = ak.where(ak.num(hard) >= 2, hard, h)
    h2 = h[ak.argsort(h.pt, axis=1, ascending=False, stable=True)][:, :2]
    px, py, pz, e = _coll_p4_sum(h2)
    m = _mass(px, py, pz, e)
    m[ak.to_numpy(ak.num(h2) < 2)] = np.nan
    return m


def _reco_mhh(ev) -> np.ndarray:
    """reco m_HH = mass(two highest-b-tag jets + FastMTT di-τ); NaN where the reco is incomplete."""
    jets = ev.jets
    pt_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False, stable=True)]
    bb = pt_sorted[ak.argsort(pt_sorted.btag, axis=1, ascending=False, stable=True)][:, :2]
    n_jet = ak.to_numpy(ak.num(bb))
    bx, by, bz, be = _coll_p4_sum(bb)

    ditau, sel = ditau_system(ev)
    n = ev.n
    dpx, dpy, dpz, de = (np.full(n, np.nan) for _ in range(4))
    dpx[sel], dpy[sel], dpz[sel], de[sel] = ditau["px"], ditau["py"], ditau["pz"], ditau["e"]

    m = _mass(bx + dpx, by + dpy, bz + dpz, be + de)
    ok = (n_jet >= 2) & np.isfinite(dpx)
    m[~ok] = np.nan
    return m


def run(ctx: ValidationContext) -> list[CheckResult]:
    """κ_λ-critical m_HH faithfulness on the signal sample ``ctx.events``."""
    ev = ctx.events
    gen = _gen_mhh(ev)
    reco = _reco_mhh(ev)
    have_gen = np.isfinite(gen)
    if have_gen.sum() < 50:
        return [info("level4.mhh.no_gen", "level4",
                     detail="no gen HH (need gen b-quarks + τ's); is this the signal sample?")]

    bins = np.asarray(ctx.opt("level4", "mhh_bins", _MHH_BINS), dtype=float)
    lo, hi = _THRESHOLD
    rel_tol = float(ctx.tol("level4", "mhh_scale_tol", 0.10))
    res_max = float(ctx.tol("level4", "mhh_resolution_max", 0.20))

    results: list[CheckResult] = []

    # A×ε turn-on: fraction of gen events reconstructed, per gen-m_HH bin
    passed = have_gen & np.isfinite(reco)
    centers, aeps = [], []
    for klo, khi in zip(bins[:-1], bins[1:]):
        ingen = have_gen & (gen >= klo) & (gen < khi)
        if ingen.sum():
            centers.append(0.5 * (klo + khi))
            aeps.append(float(passed[ingen].sum()) / float(ingen.sum()))
    results.append(info("level4.mhh.acceptance_efficiency", "level4",
                        float(np.mean(aeps)) if aeps else float("nan"),
                        detail=f"A×ε vs gen m_HH {[round(a, 2) for a in aeps]} over {[int(c) for c in centers]}"))

    # scale + resolution in the κ_λ-critical threshold window
    win = passed & (gen >= lo) & (gen < hi)
    if win.sum() < 30:
        results.append(info("level4.mhh.threshold_stats", "level4", float(win.sum()),
                            detail=f"only {int(win.sum())} reconstructed events in [{lo:.0f},{hi:.0f}] GeV"))
        return results
    r = reco[win] / gen[win]
    scale = float(np.median(r))
    resolution = float(0.5 * (np.percentile(r, 84) - np.percentile(r, 16)))  # robust σ of reco/gen

    results.append(CheckResult(
        name="level4.mhh.scale", level="level4",
        passed=bool(abs(scale - 1.0) <= rel_tol), severity=Severity.GATE,
        measured=scale, target=1.0, tolerance=rel_tol,
        detail=f"m_HH scale median(reco/gen) {scale:.3f} in [{lo:.0f},{hi:.0f}] GeV (|·−1|≤{rel_tol:.0%})"))
    results.append(CheckResult(
        name="level4.mhh.resolution", level="level4",
        passed=bool(resolution <= res_max), severity=Severity.GATE,
        measured=resolution, target=res_max, tolerance=res_max, units="σ(reco/gen)",
        detail=(f"m_HH resolution {resolution:.1%} in [{lo:.0f},{hi:.0f}] GeV (≤{res_max:.0%}); "
                f"keeps the κ_λ lineshape from washing out")))

    # migration: gen→reco bin diagonal fraction (the lineshape-preservation summary)
    gi = np.digitize(gen[passed], bins)
    ri = np.digitize(reco[passed], bins)
    diag = float(np.mean(gi == ri)) if gi.size else float("nan")
    results.append(info("level4.mhh.migration_diagonal", "level4", diag,
                        detail=f"fraction of reconstructed events in the correct gen-m_HH bin: {diag:.2f}"))
    return results
