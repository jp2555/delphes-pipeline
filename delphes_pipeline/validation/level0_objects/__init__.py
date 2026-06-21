"""Level 0 — object response.

Aggregates the four object-response measurements (b-tag, tau, MET, leptons) into
one ``run(ctx) -> list[CheckResult]``. Each leaf module exposes its own
``run(ctx) -> list[CheckResult]``; a leaf that raises is recorded as a GATE
failure rather than crashing the whole gate.
"""

from __future__ import annotations

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult, Severity


def run(ctx: ValidationContext) -> list[CheckResult]:
    # imported lazily so each leaf can be developed/tested independently
    from . import btag, leptons, met, tau

    leaves = (("btag", btag), ("tau", tau), ("met", met), ("leptons", leptons))
    results: list[CheckResult] = []
    for name, mod in leaves:
        try:
            results.extend(mod.run(ctx))
        except Exception as exc:  # a crashing measurement is itself a failure
            results.append(
                CheckResult(
                    name=f"level0.{name}.error",
                    level="level0",
                    passed=False,
                    severity=Severity.GATE,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
    return results
