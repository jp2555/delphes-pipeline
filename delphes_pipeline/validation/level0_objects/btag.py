"""Level-0 b-tagging closure: efficiency vs jet pT, by parton flavour.

Measures the b-tag efficiency for true b jets (``Jet.Flavor == 5``), c jets
(``== 4``) and the light-jet mistag rate (``== 0``) as a function of jet pT,
restricted to ``|eta| <= 2.5`` (the JetFlavorAssociation ``PartonEtaMax``), and
closure-tests each against the card's transcribed ``BTagging EfficiencyFormula``
(the "do the modules behave as configured?" check). Each of the three quantities
yields one GATE ``CheckResult`` plus an overlay PNG.

In Delphes the ``BTag`` and ``TauTag`` bits are computed by independent modules
on the same jet collection, so a tau-tagged jet is still a flavour jet with a
real ``BTag`` draw and is *not* excluded from the b-tag flavour populations.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.closure import efficiency_closure
from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult

_ETA_MAX = 2.5

# measured quantity -> the Jet.Flavor value selecting its jets
_FLAVORS = {"btag_eff_b": 5, "btag_eff_c": 4, "btag_mistag_light": 0}


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Run the three b-tagging closure checks; return one CheckResult each."""
    jets = ctx.events.jets
    pt = ak.to_numpy(ak.flatten(jets.pt))
    eta = ak.to_numpy(ak.flatten(jets.eta))
    flavor = ak.to_numpy(ak.flatten(jets.flavor))
    tagged = ak.to_numpy(ak.flatten(jets.btag)) == 1

    in_acc = np.abs(eta) <= _ETA_MAX
    pt, flavor, tagged = pt[in_acc], flavor[in_acc], tagged[in_acc]

    return [
        efficiency_closure(
            ctx,
            name=f"level0.btag.{quantity}",
            quantity=quantity,
            pt_values=pt[flavor == flav],
            passed=tagged[flavor == flav],
            xlabel="jet pT [GeV]",
            ylabel=quantity,
        )
        for quantity, flav in _FLAVORS.items()
    ]
