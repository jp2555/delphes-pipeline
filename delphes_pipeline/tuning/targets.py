"""Tuning targets and the observable registry.

Loads the diagnostic map (``cards/tuning/diagnostic_map.json``) and scalar
targets (``cards/tuning/targets.json``), and registers each tuning observable
with the shared extractor that measures it. A target is one of:

- ``curve``  — a digitised POG/anchor curve dropped into
  ``validation/references/data/<observable>.json`` (via the core ``ReferenceStore``);
- ``unity``  — the response should equal 1.0 (energy-scale corrections);
- ``peak``   — a mass-peak position (``targets.json`` ``mbb_peak_gev``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from delphes_pipeline.core import observables as obs

_CARDS_TUNING = Path(__file__).resolve().parents[2] / "cards" / "tuning"

# observable -> extractor(events, bins) -> Profile (the same selections the gate uses)
PROFILE_OBSERVABLES = {
    "btag_eff_b": lambda ev, bins: obs.btag_efficiency(ev, "btag_eff_b", bins=bins),
    "btag_eff_c": lambda ev, bins: obs.btag_efficiency(ev, "btag_eff_c", bins=bins),
    "btag_mistag_light": lambda ev, bins: obs.btag_efficiency(ev, "btag_mistag_light", bins=bins),
    "tau_eff": lambda ev, bins: obs.tau_efficiency(ev, bins=bins),
    "tau_mistag": lambda ev, bins: obs.tau_mistag(ev, bins=bins),
    "electron_eff": lambda ev, bins: obs.lepton_efficiency(ev, "electron_eff", bins=bins),
    "muon_eff": lambda ev, bins: obs.lepton_efficiency(ev, "muon_eff", bins=bins),
    "tau_energy_response": lambda ev, bins: obs.tau_energy_response(ev, bins=bins),
    "bjet_energy_response": lambda ev, bins: obs.bjet_energy_response(ev, bins=bins),
    "met_resolution": lambda ev, bins: obs.met_resolution(ev),
}


@lru_cache(maxsize=1)
def diagnostic_map() -> dict:
    with open(_CARDS_TUNING / "diagnostic_map.json") as fh:
        return {k: v for k, v in json.load(fh).items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def scalar_targets() -> dict:
    with open(_CARDS_TUNING / "targets.json") as fh:
        return json.load(fh)


def tuning_observables() -> list[str]:
    """Observable names in diagnostic-map order (profile observables + mbb_peak)."""
    return list(diagnostic_map().keys())
