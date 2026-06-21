"""Structured result of a single validation check.

Every check, in every level, returns one or more ``CheckResult``. The reporter
(:mod:`delphes_pipeline.core.report`) aggregates them into ``report.json`` and
``summary.md`` and decides the gate exit code: the process exits non-zero iff a
``GATE``-severity check failed.

Leaf modules should build results with the helper constructors at the bottom
(``gate_max``, ``gate_min``, ``gate_within``, ``info``, ``warn``) rather than
filling the dataclass by hand, so pass/fail semantics stay consistent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import math


class Severity(str, Enum):
    """How a failing check affects the gate."""

    GATE = "gate"  # failure blocks full production (non-zero exit)
    WARN = "warn"  # advisory; recorded but does not fail the gate
    INFO = "info"  # informational only; ``passed`` is always True


@dataclass
class CheckResult:
    """The outcome of one validation check.

    Attributes
    ----------
    name : str
        Unique dotted name, e.g. ``"pilot_gate.neg_weight_fraction"``.
    level : str
        Level identifier, e.g. ``"pilot_gate"`` or ``"level0"``.
    passed : bool
        Whether the check met its criterion.
    severity : Severity
        Gate/warn/info; see :class:`Severity`.
    measured, target, tolerance : float | None
        The measured value, its target/threshold, and the tolerance used. Their
        precise meaning depends on the constructor (a max bound, a relative
        window, ...); ``detail`` spells it out for humans.
    units : str
        Units of ``measured`` (for the summary table).
    detail : str
        One-line human explanation.
    plot_path : str | None
        Path (relative to the output dir) of an associated plot, if any.
    extra : dict
        Any structured payload a check wants preserved in ``report.json``
        (e.g. per-bin arrays).
    """

    name: str
    level: str
    passed: bool
    severity: Severity = Severity.GATE
    measured: Optional[float] = None
    target: Optional[float] = None
    tolerance: Optional[float] = None
    units: str = ""
    detail: str = ""
    plot_path: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

    @property
    def is_gate_failure(self) -> bool:
        return self.severity is Severity.GATE and not self.passed


# --------------------------------------------------------------------------- #
# Constructors. Keep the pass/fail logic here so it is identical everywhere.
# --------------------------------------------------------------------------- #


def _finite(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def gate_max(
    name: str,
    level: str,
    measured: Optional[float],
    maximum: float,
    *,
    units: str = "",
    detail: str = "",
    severity: Severity = Severity.GATE,
    require_finite: bool = True,
    plot_path: Optional[str] = None,
    extra: Optional[dict] = None,
) -> CheckResult:
    """Pass iff ``measured <= maximum`` (and finite, if ``require_finite``)."""
    ok = _finite(measured) if require_finite else (measured is not None)
    passed = bool(ok and measured <= maximum)
    return CheckResult(
        name=name,
        level=level,
        passed=passed,
        severity=severity,
        measured=measured,
        target=maximum,
        tolerance=None,
        units=units,
        detail=detail or f"require measured <= {maximum}",
        plot_path=plot_path,
        extra=extra or {},
    )


def gate_min(
    name: str,
    level: str,
    measured: Optional[float],
    minimum: float,
    *,
    units: str = "",
    detail: str = "",
    severity: Severity = Severity.GATE,
    require_finite: bool = True,
    plot_path: Optional[str] = None,
    extra: Optional[dict] = None,
) -> CheckResult:
    """Pass iff ``measured >= minimum`` (and finite, if ``require_finite``)."""
    ok = _finite(measured) if require_finite else (measured is not None)
    passed = bool(ok and measured >= minimum)
    return CheckResult(
        name=name,
        level=level,
        passed=passed,
        severity=severity,
        measured=measured,
        target=minimum,
        tolerance=None,
        units=units,
        detail=detail or f"require measured >= {minimum}",
        plot_path=plot_path,
        extra=extra or {},
    )


def gate_within(
    name: str,
    level: str,
    measured: Optional[float],
    target: float,
    rel_tol: float,
    *,
    units: str = "",
    detail: str = "",
    severity: Severity = Severity.GATE,
    plot_path: Optional[str] = None,
    extra: Optional[dict] = None,
) -> CheckResult:
    """Pass iff ``|measured/target - 1| <= rel_tol`` (relative window)."""
    passed = bool(
        _finite(measured)
        and target != 0
        and abs(measured / target - 1.0) <= rel_tol
    )
    return CheckResult(
        name=name,
        level=level,
        passed=passed,
        severity=severity,
        measured=measured,
        target=target,
        tolerance=rel_tol,
        units=units,
        detail=detail or f"require |measured/target - 1| <= {rel_tol}",
        plot_path=plot_path,
        extra=extra or {},
    )


def info(
    name: str,
    level: str,
    measured: Optional[float] = None,
    *,
    units: str = "",
    detail: str = "",
    plot_path: Optional[str] = None,
    extra: Optional[dict] = None,
) -> CheckResult:
    """An informational result that never fails the gate."""
    return CheckResult(
        name=name,
        level=level,
        passed=True,
        severity=Severity.INFO,
        measured=measured,
        units=units,
        detail=detail,
        plot_path=plot_path,
        extra=extra or {},
    )


def warn(
    name: str,
    level: str,
    passed: bool,
    *,
    measured: Optional[float] = None,
    target: Optional[float] = None,
    tolerance: Optional[float] = None,
    units: str = "",
    detail: str = "",
    plot_path: Optional[str] = None,
    extra: Optional[dict] = None,
) -> CheckResult:
    """An advisory result: recorded, surfaced, but does not fail the gate."""
    return CheckResult(
        name=name,
        level=level,
        passed=bool(passed),
        severity=Severity.WARN,
        measured=measured,
        target=target,
        tolerance=tolerance,
        units=units,
        detail=detail,
        plot_path=plot_path,
        extra=extra or {},
    )
