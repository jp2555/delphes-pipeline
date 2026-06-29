"""Level-0 b-tagging closure: efficiency vs jet pT, by parton flavour.

Measures the b-tag efficiency for true b jets (``Jet.Flavor == 5``), c jets
(``== 4``) and the light-jet mistag rate (``== 0``) vs jet pT (|eta| <= 2.5) and
closure-tests each against the card's transcribed ``BTagging EfficiencyFormula``.
The measurement itself lives in :func:`delphes_pipeline.core.observables.btag_efficiency`
(shared with the tuning and plot lenses); this module is the validation lens.
"""

from __future__ import annotations

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.closure import closure_from_profile
from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Run the three b-tagging closure checks; return one CheckResult each."""
    bins = ctx.opt("level0", "pt_bins", obs.DEFAULT_PT_BINS)
    return [
        closure_from_profile(
            ctx, obs.btag_efficiency(ctx.events, quantity, bins=bins),
            name=f"level0.btag.{quantity}",
        )
        for quantity in obs.BTAG_FLAVORS
    ]
