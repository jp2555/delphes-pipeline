"""Shared per-bin efficiency closure for the Level-0 leaves.

``efficiency_closure`` is the single place the closure pass/fail rule lives, so
b-tag, tau and lepton checks behave identically. Given flat ``pT`` values and a
boolean ``passed`` (numerator) mask over the same selection (denominator), it
bins in pT, measures the per-bin rate, evaluates the card-formula closure target,
overlays the plot, and returns a standardised GATE ``CheckResult``.

Closure rule (statistics-aware): a populated bin (``count >= min_bin_count``)
counts as *failing* only if the measured rate misses the card target by more
than the relative tolerance **and** by more than ``nsigma`` of the binomial
error evaluated under the *expected* rate (so a bin with zero counts still has a
finite band — the measured Wald error would collapse to zero at rate 0 or 1).
The check passes iff at most ``max_failing_bin_fraction`` of tested bins fail.
This keeps the 5% systematic floor (note: ``closure_rel_tol``) while not
flagging bins that are statistically consistent with the card — the regime where
a low-rate quantity (light mistag ~1%) is Poisson-dominated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .context import ValidationContext
from .plotting import efficiency_overlay
from .result import CheckResult, Severity

_DEFAULT_PT_BINS = [20, 30, 40, 50, 70, 100, 150, 200, 300]


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
    pt_bins = np.asarray(ctx.opt(level, "pt_bins", _DEFAULT_PT_BINS), dtype=float)
    min_count = int(ctx.opt(level, "min_bin_count", 25))
    rel_tol = float(ctx.tol(level, "closure_rel_tol", 0.05))
    max_fail_frac = float(ctx.tol(level, "max_failing_bin_fraction", 0.20))
    nsigma = float(ctx.tol(level, "closure_nsigma", 2.0))
    # Per-quantity severity from config; default GATE. Lets the pre-v0 card flag
    # known stock-Delphes cascade effects (e.g. lepton tracking × isolation,
    # tau_mistag per-parton vs per-jet) as WARN without blocking production —
    # the closure infrastructure still measures and surfaces them in the report.
    if severity is None:
        sev_map = ctx.tol(level, "closure_severity", {}) or {}
        severity = Severity(sev_map.get(quantity, "gate"))

    pt_values = np.asarray(pt_values, dtype=float)
    passed = np.asarray(passed, dtype=bool)

    centers, measured, counts = [], [], []
    for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
        in_bin = (pt_values >= lo) & (pt_values < hi)
        n = int(in_bin.sum())
        if n == 0:  # empty bin -> skip, never emit nan
            continue
        centers.append(0.5 * (lo + hi))
        measured.append(float(passed[in_bin].sum()) / n)
        counts.append(n)

    centers = np.asarray(centers, dtype=float)
    measured = np.asarray(measured, dtype=float)
    counts = np.asarray(counts, dtype=int)

    if centers.size:
        expected = np.asarray(ctx.references.expected(quantity, centers, np.zeros_like(centers)), dtype=float)
        measured_err = np.sqrt(np.clip(measured * (1.0 - measured), 0.0, None) / np.maximum(counts, 1))
        null_err = np.sqrt(np.clip(expected * (1.0 - expected), 0.0, None) / np.maximum(counts, 1))
        abs_dev = np.abs(measured - expected)
        consistent = (abs_dev <= rel_tol * np.abs(expected)) | (abs_dev <= nsigma * null_err)
        tested = counts >= min_count
        failing = tested & ~consistent
        n_tested = int(tested.sum())
        n_failing = int(failing.sum())
    else:
        expected = np.asarray([], dtype=float)
        measured_err = np.asarray([], dtype=float)
        n_tested = n_failing = 0

    # no tested bin -> closure cannot be confirmed -> conservative GATE failure
    passed_check = bool(n_tested > 0 and (n_failing / n_tested) <= max_fail_frac)
    fail_frac = (n_failing / n_tested) if n_tested else 1.0

    plot_rel = None
    if centers.size:
        outpath = ctx.plot_path(plot_name or f"{quantity}.png")
        ref = ctx.references.digitized(quantity)
        ref_kw = (
            {}
            if ref is None
            else {"ref_centers": ref.centers, "ref_values": ref.values, "ref_errors": ref.errors}
        )
        efficiency_overlay(
            centers, measured, measured_err, expected,
            outpath=str(outpath), xlabel=xlabel, ylabel=ylabel or quantity, **ref_kw,
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
            "x": "pt",
            "centers": centers.tolist(),
            "measured": measured.tolist(),
            "measured_err": measured_err.tolist(),
            "expected": expected.tolist(),
            "counts": counts.tolist(),
            "n_tested_bins": n_tested,
            "n_failing_bins": n_failing,
        },
    )
