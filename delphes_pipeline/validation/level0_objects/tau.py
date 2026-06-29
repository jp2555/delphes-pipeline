"""Level-0 hadronic-tau response: τ_h efficiency and jet→τ_h mistag.

Two reco-jet-based closure measurements vs jet pT, overlaid against the card's
transcribed ``TauTagging`` formulas:

- ``tau_eff``   — TauTag rate of the unique nearest jet to each acceptance gen τ.
- ``tau_mistag``— TauTag rate among jets not near any gen τ.

The measurements live in :mod:`delphes_pipeline.core.observables` (shared with
the tuning and plot lenses); this module is the validation lens. Per-quantity
severity (e.g. ``tau_mistag`` → warn for the pre-v0 card, where the stock
per-parton EfficiencyFormula under-counts the per-jet TauTag rate in busy bb̄ττ
topologies) comes from ``tolerances.level0.closure_severity`` via
``closure_from_profile``.
"""

from __future__ import annotations

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.closure import closure_from_profile
from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.result import CheckResult


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Measure τ_h efficiency and jet→τ_h mistag and closure-test both."""
    bins = ctx.opt("level0", "pt_bins", obs.DEFAULT_PT_BINS)
    return [
        closure_from_profile(ctx, obs.tau_efficiency(ctx.events, bins=bins), name="level0.tau.tau_eff"),
        closure_from_profile(ctx, obs.tau_mistag(ctx.events, bins=bins), name="level0.tau.tau_mistag"),
    ]
