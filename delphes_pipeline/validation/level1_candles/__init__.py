"""Level 1 — standard candles (Z->tautau, ttbar dilepton). STUB.

Needs the background samples of note Table 3, which are not produced yet.
See README.md for the planned checks.
"""

from __future__ import annotations

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult, info


def run(ctx: ValidationContext) -> list[CheckResult]:
    return [info("level1.not_implemented", "level1",
                 detail="Level 1 candles are a stub (need Z->tautau / ttbar samples); see README.md")]
