"""Validation lens: closure-test a measured ``Profile`` against the card formula.

``closure_from_profile`` is the single place the pass/fail rule lives. Given a
:class:`~delphes_pipeline.core.observables.Profile` (measured per-bin rate +
error + count), it evaluates the card-formula closure target, applies the
statistics-aware criterion, overlays the plot, and returns a ``CheckResult``.
``efficiency_closure`` is the thin (pt_values, passed) entry the Level-0 leaves
use; it bins via ``observables.binned_efficiency`` and defers to
``closure_from_profile`` so binning is never duplicated.

Closure rule (statistics-aware): a populated bin (``count >= min_bin_count``)
counts as *failing* only if the measured rate misses the card target by more
than the relative tolerance **and** by more than ``nsigma`` of the binomial
error evaluated under the *expected* rate (so a bin with zero counts still has a
finite band — the measured Wald error would collapse to zero at rate 0 or 1).
The check passes iff at most ``max_failing_bin_fraction`` of tested bins fail.
This keeps the 5% systematic floor (``closure_rel_tol``) while not flagging bins
statistically consistent with the card — the regime where a low-rate quantity
(light mistag ~1%) is Poisson-dominated.

Per-quantity severity from ``tolerances.level0.closure_severity`` lets the
pre-v0 card flag known stock-Delphes cascade effects (lepton tracking ×
isolation, tau_mistag per-parton vs per-jet) as WARN without blocking
production; the closure infrastructure still measures and surfaces them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .context import ValidationContext
from .observables import DEFAULT_PT_BINS, Profile, binned_efficiency
from .plotting import efficiency_overlay
from .result import CheckResult, Severity


def efficiency_closure(
    ctx: ValidationContext,
    *,
    name: str,
    quantity: str,
    pt_values,
    passed,
    level: str = "level0",
    xlabel: str = "pT [GeV]",
    ylabel: str | None = None,
    plot_name: str | None = None,
    severity: Severity | None = None,
) -> CheckResult:
    """Bin ``pt_values``, measure the rate of ``passed``, and closure-test it."""
    bins = ctx.opt(level, "pt_bins", DEFAULT_PT_BINS)
    prof = binned_efficiency(pt_values, passed, bins, quantity=quantity, x="pt")
    prof.xlabel, prof.ylabel = xlabel, ylabel or quantity
    return closure_from_profile(ctx, prof, name=name, level=level,
                                plot_name=plot_name, severity=severity)


def closure_from_profile(
    ctx: ValidationContext,
    profile: Profile,
    *,
    name: str,
    level: str = "level0",
    plot_name: str | None = None,
    severity: Severity | None = None,
) -> CheckResult:
    """Closure-test a measured ``Profile`` against the card-formula target."""
    quantity = profile.quantity
    min_count = int(ctx.opt(level, "min_bin_count", 25))
    rel_tol = float(ctx.tol(level, "closure_rel_tol", 0.05))
    max_fail_frac = float(ctx.tol(level, "max_failing_bin_fraction", 0.20))
    nsigma = float(ctx.tol(level, "closure_nsigma", 2.0))
    if severity is None:
        sev_map = ctx.tol(level, "closure_severity", {}) or {}
        severity = Severity(sev_map.get(quantity, "gate"))

    centers = np.asarray(profile.centers, dtype=float)
    measured = np.asarray(profile.values, dtype=float)
    measured_err = np.asarray(profile.errors, dtype=float)
    counts = np.asarray(profile.counts, dtype=int)

    if centers.size:
        expected = np.asarray(ctx.references.expected(quantity, centers, np.zeros_like(centers)), dtype=float)
        null_err = np.sqrt(np.clip(expected * (1.0 - expected), 0.0, None) / np.maximum(counts, 1))
        abs_dev = np.abs(measured - expected)
        consistent = (abs_dev <= rel_tol * np.abs(expected)) | (abs_dev <= nsigma * null_err)
        tested = counts >= min_count
        failing = tested & ~consistent
        n_tested = int(tested.sum())
        n_failing = int(failing.sum())
    else:
        expected = np.asarray([], dtype=float)
        n_tested = n_failing = 0

    passed_check = bool(n_tested > 0 and (n_failing / n_tested) <= max_fail_frac)
    fail_frac = (n_failing / n_tested) if n_tested else 1.0

    plot_rel = None
    if centers.size:
        outpath = ctx.plot_path(plot_name or f"{quantity}.png")
        ref = ctx.references.digitized(quantity)
        ref_kw = ({} if ref is None
                  else {"ref_centers": ref.centers, "ref_values": ref.values, "ref_errors": ref.errors})
        efficiency_overlay(
            centers, measured, measured_err, expected,
            outpath=str(outpath), xlabel=profile.xlabel or "pT [GeV]",
            ylabel=profile.ylabel or quantity, **ref_kw,
        )
        plot_rel = ctx.rel(Path(outpath))

    return CheckResult(
        name=name,
        level=level,
        passed=passed_check,
        severity=severity,
        measured=float(fail_frac),
        target=float(max_fail_frac),
        tolerance=float(rel_tol),
        units="failing-bin fraction",
        detail=(
            f"{quantity}: {n_failing}/{n_tested} tested pT bins miss card by "
            f">{rel_tol:.0%} rel and >{nsigma:g}σ; require failing fraction <= {max_fail_frac:.0%}"
        ),
        plot_path=plot_rel,
        extra={
            "quantity": quantity,
            "x": profile.x,
            "centers": centers.tolist(),
            "measured": measured.tolist(),
            "measured_err": measured_err.tolist(),
            "expected": expected.tolist(),
            "counts": counts.tolist(),
            "n_tested_bins": n_tested,
            "n_failing_bins": n_failing,
        },
    )
