"""Level-0 lepton reconstruction efficiency (closure against the card).

Prompt barrel e/μ reconstruction efficiency vs pT, matching gen leptons to reco
with a unique nearest-neighbour assignment. The prompt-mother selection
(mother ``|PID| in {15, 23, 24}``) is essential: Delphes reconstructs prompt
leptons (from τ/Z/W) and does not recover non-prompt heavy-flavour leptons, which
would otherwise bias the efficiency low. The measurement lives in
:func:`delphes_pipeline.core.observables.lepton_efficiency` (shared with the
tuning and plot lenses); this module is the validation lens.
"""

from __future__ import annotations

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.closure import closure_from_profile
from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Measure barrel e/mu reconstruction efficiency vs pT and closure-test it."""
    bins = ctx.opt("level0", "pt_bins", obs.DEFAULT_PT_BINS)
    return [
        closure_from_profile(
            ctx, obs.lepton_efficiency(ctx.events, quantity, bins=bins),
            name=f"level0.leptons.{quantity}", plot_name=f"leptons_{quantity}.png",
        )
        for quantity in ("electron_eff", "muon_eff")
    ]
