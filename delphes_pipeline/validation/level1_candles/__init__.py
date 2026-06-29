"""Level 1 — standard candles: tt̄ dilepton + Z→ττ (note §6.2).

The candles run on the **background** samples (not the signal), so the paths come
from ``config.candles.{ttbar,ztautau}``. Each candle that has a configured,
readable sample is run; a missing path yields an informational note rather than a
failure. The tt̄ ε_b in-situ closure is the one GATE check; the rest are
informational until POG/anchor targets (or the m_ττ estimator) are wired.
"""

from __future__ import annotations

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.core.result import CheckResult, Severity, info
from . import ttbar, ztautau

_CANDLES = (("ttbar", ttbar), ("ztautau", ztautau))


def run(ctx: ValidationContext) -> list[CheckResult]:
    candles = ctx.config.get("candles", {})
    treename = ctx.config.get("input", {}).get("treename", "Delphes")
    results: list[CheckResult] = []
    for name, mod in _CANDLES:
        path = candles.get(name)
        if not path:
            results.append(info(f"level1.{name}.not_configured", "level1",
                                detail=f"set candles.{name} to the Delphes sample to enable this candle"))
            continue
        try:
            ev = DelphesEvents(path, treename=treename, entry_stop=candles.get("max_events"))
        except Exception as exc:
            results.append(CheckResult(name=f"level1.{name}.error", level="level1", passed=False,
                                       severity=Severity.WARN, detail=f"could not open {name} sample: {exc}"))
            continue
        results.extend(mod.run(ctx, ev))
    return results
