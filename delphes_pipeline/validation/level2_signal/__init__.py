"""Level 2 — per-channel signal acceptance x efficiency. STUB.

Compares A x eps for SM HH -> bb tautau against published Run-2 bb tautau
yields (note §6.1). See README.md.
"""

from __future__ import annotations

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult, info


def run(ctx: ValidationContext) -> list[CheckResult]:
    return [info("level2.not_implemented", "level2",
                 detail="Level 2 signal A x eps is a stub; see README.md")]
