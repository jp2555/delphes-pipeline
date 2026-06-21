"""Level-0 hadronic-tau response: τ_h efficiency and jet→τ_h mistag.

Two closure measurements vs jet pT, overlaid against the card's transcribed
``TauTagging`` formulas:

- ``tau_eff``   — among reco jets in tau acceptance that ARE matched (ΔR<0.4) to
  a gen tau (``|PID|==15``), the fraction carrying ``Jet.TauTag==1``. Closure
  target ``tau_eff`` (=0.6).
- ``tau_mistag``— among reco jets NOT matched to any gen tau, the fraction with
  ``Jet.TauTag==1``. Closure target ``tau_mistag`` (=0.01).

Both measurements are *reco-jet based*: the denominator is reconstructed jets,
not gen taus. This is real-data safe — a leptonically-decaying tau produces no
reco jet near the gen tau, so it is naturally excluded from the efficiency
denominator (which targets the hadronic τ_h that Delphes' TauTagging actually
tags), and a gen record with several status copies of one tau does not inflate
the count (a jet is near the tau or it is not). The acceptance matches the card:
``|eta| <= TauEtaMax = 2.5``.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.closure import efficiency_closure
from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.matching import matched_to_any, nearest_target_field
from delphes_pipeline.core.result import CheckResult

_GEN_TAU_PID = 15
_DR_MATCH = 0.4
_TAU_ETA_MAX = 2.5
_TAU_PT_MIN = 20.0


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Measure τ_h efficiency and jet→τ_h mistag and closure-test both."""
    jets = ctx.events.jets
    gen = ctx.events.gen
    gen_taus = gen[np.abs(gen.pid) == _GEN_TAU_PID]

    acc = jets[(np.abs(jets.eta) <= _TAU_ETA_MAX) & (jets.pt > _TAU_PT_MIN)]

    # tau_eff: each acceptance gen tau is measured on its UNIQUE nearest jet, so a
    # jet that happens to sit near the tau does not dilute the rate. Denominator =
    # gen taus with a matched jet; numerator = that jet is TauTag==1; bin by gen τ pT.
    taus_acc = gen_taus[(np.abs(gen_taus.eta) <= _TAU_ETA_MAX) & (gen_taus.pt > _TAU_PT_MIN)]
    matched, jet_tautag = nearest_target_field(taus_acc, acc, _DR_MATCH, "tautag")
    tau_pt = ak.to_numpy(ak.flatten(taus_acc.pt))

    # tau_mistag: TauTag rate among jets NOT near any gen tau (clean fake candidates).
    near_tau = matched_to_any(acc, gen_taus, _DR_MATCH)
    fake_jets = acc[~near_tau]

    return [
        efficiency_closure(
            ctx,
            name="level0.tau.tau_eff",
            quantity="tau_eff",
            pt_values=tau_pt[matched],
            passed=jet_tautag[matched] == 1,
            xlabel="tau pT [GeV]",
            ylabel="tau_eff",
        ),
        efficiency_closure(
            ctx,
            name="level0.tau.tau_mistag",
            quantity="tau_mistag",
            pt_values=ak.to_numpy(ak.flatten(fake_jets.pt)),
            passed=ak.to_numpy(ak.flatten(fake_jets.tautag)) == 1,
            xlabel="jet pT [GeV]",
            ylabel="tau_mistag",
        ),
    ]
