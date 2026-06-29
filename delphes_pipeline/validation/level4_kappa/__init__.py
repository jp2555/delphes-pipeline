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

from delphes_pipeline.core import observables as obs
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
    """gen m_HH = mass of the H decay products: the two b-quarks + two τ's whose **mother is a
    Higgs** (via the ``m1`` links). The full Pythia record has many shower/status copies of each
    b/τ *and* of each Higgs — but a copy's mother is the previous b/τ, not the Higgs, so the
    mother cut isolates exactly the four hard decay products (robust where a leading-pT or even a
    2-Higgs selection picks duplicate copies of one particle). NaN if the four aren't found.
    """
    gen = ev.gen
    from_higgs = np.abs(obs.mother_pid(gen)) == _HIGGS_PID
    b = gen[(np.abs(gen.pid) == 5) & from_higgs]
    t = gen[(np.abs(gen.pid) == 15) & from_higgs]
    b2 = b[ak.argsort(b.pt, axis=1, ascending=False, stable=True)][:, :2]
    t2 = t[ak.argsort(t.pt, axis=1, ascending=False, stable=True)][:, :2]
    bx, by, bz, be = _coll_p4_sum(b2)
    tx, ty, tz, te = _coll_p4_sum(t2)
    m = _mass(bx + tx, by + ty, bz + tz, be + te)
    m[ak.to_numpy((ak.num(b2) < 2) | (ak.num(t2) < 2))] = np.nan
    return m


def _reco_mhh(ev):
    """reco (m_HH, m_bb, m_ditau): two highest-b-tag NON-τ jets + the FastMTT di-τ.

    The bb pair is drawn from ``tautag==0`` jets so a τ_h jet can't be double-counted into
    both the bb and the di-τ system. Returns NaN where the reco is incomplete; m_bb / m_ditau
    are the per-Higgs masses so the Delphes (bb) and FastMTT (di-τ) contributions to the m_HH
    resolution can be separated (the di-τ carries the estimator floor, not a Delphes effect).
    """
    jets = ev.jets
    bjets = jets[jets.tautag == 0]                                   # exclude τ_h jets from the bb pair
    pt_sorted = bjets[ak.argsort(bjets.pt, axis=1, ascending=False, stable=True)]
    bb = pt_sorted[ak.argsort(pt_sorted.btag, axis=1, ascending=False, stable=True)][:, :2]
    n_jet = ak.to_numpy(ak.num(bb))
    bx, by, bz, be = _coll_p4_sum(bb)

    ditau, sel = ditau_system(ev)
    n = ev.n
    dpx, dpy, dpz, de = (np.full(n, np.nan) for _ in range(4))
    dpx[sel], dpy[sel], dpz[sel], de[sel] = ditau["px"], ditau["py"], ditau["pz"], ditau["e"]

    ok = (n_jet >= 2) & np.isfinite(dpx)
    m_hh = _mass(bx + dpx, by + dpy, bz + dpz, be + de)
    m_bb = _mass(bx, by, bz, be)
    m_ditau = _mass(dpx, dpy, dpz, de)
    for a in (m_hh, m_bb, m_ditau):
        a[~ok] = np.nan
    return m_hh, m_bb, m_ditau


def run(ctx: ValidationContext) -> list[CheckResult]:
    """κ_λ-critical m_HH faithfulness on the signal sample ``ctx.events``."""
    ev = ctx.events
    gen = _gen_mhh(ev)
    reco, m_bb, m_ditau = _reco_mhh(ev)
    have_gen = np.isfinite(gen)
    if have_gen.sum() < 50:
        return [info("level4.mhh.no_gen", "level4",
                     detail="no gen HH (need gen b-quarks + τ's); is this the signal sample?")]

    bins = np.asarray(ctx.opt("level4", "mhh_bins", _MHH_BINS), dtype=float)
    lo, hi = _THRESHOLD
    rel_tol = float(ctx.tol("level4", "mhh_scale_tol", 0.10))
    # ~25-30% is the *inherent* bb̄ττ m_HH resolution (ττ neutrinos + no b-energy regression,
    # per the reference analysis); the ceiling is a not-grossly-washed-out sanity, while the
    # actual "matches CMS" check is the Delphes-vs-NanoAOD overlay (scripts/mhh_overlay.py).
    res_max = float(ctx.tol("level4", "mhh_resolution_max", 0.30))

    results: list[CheckResult] = []
    gm = gen[have_gen]
    results.append(info("level4.mhh.gen_check", "level4", float(np.median(gm)),
                        detail=(f"gen m_HH median {np.median(gm):.0f} GeV, IQR "
                                f"[{np.percentile(gm, 25):.0f},{np.percentile(gm, 75):.0f}] — should span the "
                                f"spectrum; a flat ~250 GeV would signal a duplicate-Higgs gen selection")))

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

    # separate the Delphes (bb) and FastMTT (di-τ) contributions so the b-jet part isn't
    # masked in the combined m_HH (the di-τ carries the estimator floor, not a Delphes effect)
    def _peak_width(x):
        x = x[np.isfinite(x)]
        return (float(np.median(x)), float(0.5 * (np.percentile(x, 84) - np.percentile(x, 16)))) if x.size >= 30 else (float("nan"), float("nan"))
    bb_m, bb_w = _peak_width(m_bb[win])
    dt_m, dt_w = _peak_width(m_ditau[win])
    results.append(info("level4.mhh.components", "level4", bb_w / bb_m if bb_m else float("nan"),
                        detail=(f"m_bb {bb_m:.0f}±{bb_w:.0f} (Delphes b-jet); "
                                f"m_ττ {dt_m:.0f}±{dt_w:.0f} GeV (FastMTT floor) — the di-τ dominates "
                                f"the m_HH resolution, so read it next to the b-jet part")))

    # migration: gen→reco bin diagonal fraction (the lineshape-preservation summary)
    gi = np.digitize(gen[passed], bins)
    ri = np.digitize(reco[passed], bins)
    diag = float(np.mean(gi == ri)) if gi.size else float("nan")
    results.append(info("level4.mhh.migration_diagonal", "level4", diag,
                        detail=f"fraction of reconstructed events in the correct gen-m_HH bin: {diag:.2f}"))
    return results
