"""Level-0 lepton reconstruction efficiency (closure against the card).

Measures electron and muon reconstruction efficiency vs pT in the barrel
(``|eta| <= 1.5``), where the card plateau is flat at 0.95, by matching gen
leptons to reco leptons within ``ΔR < 0.2`` with a unique nearest-neighbour
assignment (each reco lepton claimed once, so the efficiency is not biased high
by double counting):

- ``electron_eff``: prompt gen electrons (``abs(pid)==11``, ``status==1``,
  mother ``|PID| in {15, 23, 24}``) -> ``ev.electrons``.
- ``muon_eff``:     prompt gen muons     (``abs(pid)==13``, ``status==1``,
  mother ``|PID| in {15, 23, 24}``) -> ``ev.muons``.

The prompt selection on the mother is essential: Delphes' ``ElectronEfficiency``
/ ``MuonEfficiency`` modules reconstruct *prompt* leptons (from τ/Z/W) and do
not recover non-prompt leptons from heavy-flavor (B/D) decay. Without this
filter, B-meson leptons enter the denominator but cannot enter the numerator,
biasing the measured efficiency low and producing a spurious turn-on shape
that has nothing to do with the card's flat plateau.

Only the plateau is closure-tested: gen leptons are selected above the first pT
bin edge (``pt_bins[0]``), so the card turn-on region below it is out of scope
here (it needs finer threshold-region binning and is left to a later refinement).
Each quantity yields one GATE ``CheckResult`` plus an overlay PNG.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from delphes_pipeline.core.closure import efficiency_closure
from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.matching import unique_match
from delphes_pipeline.core.result import CheckResult

_BARREL_ETA = 1.5
_DR_MATCH = 0.2
_PROMPT_MOTHER_PIDS = (15, 23, 24)  # tau, Z, W -- the standard prompt-lepton sources


def _mother_pid(gen: ak.Array) -> ak.Array:
    """Per-entry mother PID via ``gen.m1`` lookup, 0 where the mother index is
    out of range (Pythia stage-0 entries with ``M1 == -1``, or guarded malformed
    indices). Returns a jagged array aligned with ``gen``."""
    n = ak.num(gen)
    m1 = gen.m1
    valid = (m1 >= 0) & (m1 < n)
    safe = ak.where(valid, m1, 0)
    mother = gen.pid[safe]
    return ak.where(valid, mother, 0)


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Measure barrel e/mu reconstruction efficiency vs pT and closure-test it."""
    pt_min = float(np.asarray(ctx.opt("level0", "pt_bins", [20, 30, 40, 50, 70, 100, 150, 200, 300]), dtype=float)[0])
    return [
        _efficiency(ctx, "electron_eff", ctx.events.gen, ctx.events.electrons, pid=11, pt_min=pt_min),
        _efficiency(ctx, "muon_eff", ctx.events.gen, ctx.events.muons, pid=13, pt_min=pt_min),
    ]


def _efficiency(
    ctx: ValidationContext,
    quantity: str,
    gen: ak.Array,
    reco: ak.Array,
    *,
    pid: int,
    pt_min: float,
) -> CheckResult:
    """Select barrel prompt gen leptons, uniquely match to reco, and closure-test vs pT."""
    mother_abs = np.abs(_mother_pid(gen))
    prompt = (mother_abs == _PROMPT_MOTHER_PIDS[0])
    for src in _PROMPT_MOTHER_PIDS[1:]:
        prompt = prompt | (mother_abs == src)
    sel = (
        (np.abs(gen.pid) == pid)
        & (gen.status == 1)
        & (np.abs(gen.eta) <= _BARREL_ETA)
        & (gen.pt > pt_min)
        & prompt
    )
    g = gen[sel]
    matched = unique_match(g, reco, _DR_MATCH)
    return efficiency_closure(
        ctx,
        name=f"level0.leptons.{quantity}",
        quantity=quantity,
        pt_values=ak.to_numpy(ak.flatten(g.pt)),
        passed=matched,
        xlabel="lepton pT [GeV]",
        ylabel=quantity,
        plot_name=f"leptons_{quantity}.png",
    )
