"""Level 4 — kappa_lambda-critical acceptance & response in m_HH. STUB.

A x eps(m_HH^truth) turn-on and the m_HH response/migration matrix, emphasis on
the 250-400 GeV threshold region (note §6.3). The kappa_lambda-critical check.
See README.md.
"""

from __future__ import annotations

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult, info


def run(ctx: ValidationContext) -> list[CheckResult]:
    return [info("level4.not_implemented", "level4",
                 detail="Level 4 m_HH response / kappa_lambda check is a stub; see README.md")]
