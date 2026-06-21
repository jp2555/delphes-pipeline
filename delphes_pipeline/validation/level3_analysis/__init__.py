"""Level 3 — simplified binned-m_HH expected limit. STUB.

A deliberately simple cut-based / single-discriminant binned analysis whose
expected limit is checked against published expected limits (note §6.1).
See README.md.
"""

from __future__ import annotations

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult, info


def run(ctx: ValidationContext) -> list[CheckResult]:
    return [info("level3.not_implemented", "level3",
                 detail="Level 3 binned-limit analysis is a stub; see README.md")]
