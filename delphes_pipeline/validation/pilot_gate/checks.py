"""Pilot Gate — six fast card-header sanity checks.

This is the cheapest rung of the validation ladder: it reads only event-level
scalars and a couple of leading-object quantities to confirm the Delphes file is
sane enough to bother with the full Level-0 closure suite. A failure here means
the production run is malformed (flavour association did not run, the sample is
mostly negative-weight, no gen taus, the m_bb peak is smeared, MET is mis-scaled,
or the per-event size blew past the storage budget). All checks are GATE
severity; the gate exits non-zero if any fails.

The Pilot Gate uses no card formulas — it is a structural smoke test, not a
closure measurement.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult, Severity, gate_max, gate_min, info


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Run the six Pilot-Gate sanity checks against ``ctx.events``."""
    ev = ctx.events
    results: list[CheckResult] = [
        info("pilot_gate.n_events", "pilot_gate", float(ev.n), detail="events read"),
        _neg_weight_fraction(ctx, ev),
        _jet_flavor_filled(ev),
        _gen_taus_present(ctx, ev),
        _mbb_width(ctx, ev),
        _met_resolution(ctx, ev),
        _kb_per_event(ctx, ev),
    ]
    return results


def _neg_weight_fraction(ctx: ValidationContext, ev) -> CheckResult:
    """Fraction of events with a negative generator weight (Powheg sanity)."""
    frac = float(np.mean(ev.weights < 0)) if ev.n else 0.0
    return gate_max(
        "pilot_gate.neg_weight_fraction",
        "pilot_gate",
        frac,
        ctx.tol("pilot_gate", "neg_weight_fraction_max", 0.30),
        detail="fraction of events with Event.Weight < 0",
    )


def _jet_flavor_filled(ev) -> CheckResult:
    """At least one jet carries Flavor != 0 (flavour association ran)."""
    if "flavor" not in ev.jets.fields:  # Jet.Flavor leaf absent -> association did not run
        n_nonzero = 0
    else:
        n_nonzero = int(ak.sum(ak.flatten(ev.jets.flavor) != 0))
    return gate_min(
        "pilot_gate.jet_flavor_filled",
        "pilot_gate",
        float(n_nonzero),
        1,
        detail="require some jet with Flavor != 0",
    )


def _gen_taus_present(ctx: ValidationContext, ev) -> CheckResult:
    """Gen record contains tau leptons (|PID| == 15)."""
    n = int(ak.sum(np.abs(ak.flatten(ev.gen.pid)) == 15))
    return gate_min(
        "pilot_gate.gen_taus_present",
        "pilot_gate",
        float(n),
        ctx.tol("pilot_gate", "min_gen_taus", 1),
        detail="require >= 1 gen tau (|PID| == 15)",
    )


def _mbb_width(ctx: ValidationContext, ev) -> CheckResult:
    """Sanity of the di-jet Higgs peak: core resolution AND core fraction.

    The two leading jets are ranked per event by (BTag desc, then pT desc — when
    no jet is b-tagged this falls back to the two leading-pT jets); the pair
    invariant mass is formed from the (pt, eta, phi, mass) 4-vectors. We check the
    standard deviation in the Higgs core window 100 < m < 150 GeV (a full-range
    std is dominated by combinatoric tails) AND the fraction of pairs landing in
    that core — a window-bounded std saturates, so a grossly smeared/shifted
    b-response is caught by the collapse of the core fraction rather than the std.
    """
    jets = ev.jets
    sel = jets[ak.num(jets) >= 2]
    pt_sorted = sel[ak.argsort(sel.pt, axis=1, ascending=False, stable=True)]
    lead = pt_sorted[ak.argsort(pt_sorted.btag, axis=1, ascending=False, stable=True)][:, :2]

    mbb = _pair_mass(lead)
    n_pairs = int(mbb.size)
    core = mbb[(mbb > 100.0) & (mbb < 150.0)]
    n_core = int(core.size)
    width = float(np.std(core)) if n_core else float("nan")
    median = float(np.median(core)) if n_core else float("nan")
    core_frac = float(n_core / n_pairs) if n_pairs else 0.0

    max_width = float(ctx.tol("pilot_gate", "mbb_width_max_gev", 30.0))
    min_frac = float(ctx.tol("pilot_gate", "mbb_core_fraction_min", 0.30))
    passed = bool(np.isfinite(width) and width <= max_width and core_frac >= min_frac)
    return CheckResult(
        name="pilot_gate.mbb_width",
        level="pilot_gate",
        passed=passed,
        severity=Severity.GATE,
        measured=width,
        target=max_width,
        units="GeV",
        detail=(
            f"m_bb core(100-150 GeV): std {width:.1f} <= {max_width} GeV "
            f"and core fraction {core_frac:.2f} >= {min_frac}"
        ),
        extra={"mbb_median": median, "n_core": n_core, "core_fraction": core_frac},
    )


def _pair_mass(pair) -> ak.Array:
    """Invariant mass of the (exactly two) jets per event in ``pair``.

    ``pair`` is a jagged array of length-2 jet lists with fields pt, eta, phi,
    mass; returns a 1-D array of pair masses (one per event).
    """
    pt = pair.pt
    eta = pair.eta
    phi = pair.phi
    mass = pair.mass
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    ex = ak.sum(e, axis=1)
    sx = ak.sum(px, axis=1)
    sy = ak.sum(py, axis=1)
    sz = ak.sum(pz, axis=1)
    m2 = ex * ex - (sx * sx + sy * sy + sz * sz)
    return np.sqrt(np.maximum(ak.to_numpy(m2), 0.0))


def _met_resolution(ctx: ValidationContext, ev) -> CheckResult:
    """Per-axis RMS-about-zero of (reco MET - gen MET), averaged over x and y.

    RMS about zero (not variance about the mean) so the gate also catches a MET
    *scale/offset* error, which a mean-subtracting variance would hide. The net
    component offset is reported separately in ``extra['mean_offset_gev']``.
    """
    if not (ev.has_branch("MissingET.MET") and ev.has_branch("GenMissingET.MET")) or not ev.n:
        return gate_max(
            "pilot_gate.met_resolution", "pilot_gate", float("nan"),
            ctx.tol("pilot_gate", "met_resolution_max_gev", 40.0), units="GeV",
            detail="MissingET/GenMissingET branch absent or empty file",
        )
    met = ak.to_numpy(ak.fill_none(ev.met.met, 0.0))
    met_phi = ak.to_numpy(ak.fill_none(ev.met.phi, 0.0))
    genmet = ak.to_numpy(ak.fill_none(ev.genmet.met, 0.0))
    genmet_phi = ak.to_numpy(ak.fill_none(ev.genmet.phi, 0.0))
    dx = met * np.cos(met_phi) - genmet * np.cos(genmet_phi)
    dy = met * np.sin(met_phi) - genmet * np.sin(genmet_phi)
    res = float(np.sqrt(0.5 * (np.mean(dx**2) + np.mean(dy**2))))
    mean_offset = float(np.mean(np.concatenate([dx, dy])))
    return gate_max(
        "pilot_gate.met_resolution",
        "pilot_gate",
        res,
        ctx.tol("pilot_gate", "met_resolution_max_gev", 40.0),
        units="GeV",
        detail="per-axis RMS-about-zero of (reco MET - gen MET)",
        extra={"mean_offset_gev": mean_offset},
    )


def _kb_per_event(ctx: ValidationContext, ev) -> CheckResult:
    """On-disk size per event in kB (storage projection ceiling)."""
    kb = ev.bytes_per_event / 1024.0
    return gate_max(
        "pilot_gate.kb_per_event",
        "pilot_gate",
        kb,
        ctx.tol("pilot_gate", "kb_per_event_max", 150.0),
        units="kB",
        detail="on-disk file size per event",
    )
