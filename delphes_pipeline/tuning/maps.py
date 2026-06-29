"""Downstream tuning maps: derive from the anchor, apply by re-tagging (note §3, D2-A).

The Delphes card's tagger bits are *ignored*; the tuned b-tag decision is re-applied
**downstream** from ``Jet.Flavor`` + the efficiency map measured on the NanoAOD
anchor — no card edit, no Delphes re-production. This is the automated tuning:

- ``derive_maps`` runs the anchor and serialises the per-flavour efficiency curves;
- ``retag_btag`` applies them stochastically: ``BTag = Bernoulli(ε_map(flavour, pT))``,
  keyed off the truth ``Jet.Flavor`` (the note's "working-point-level tagging").

After re-tagging, the Delphes b-tag matches the NanoAOD by construction; re-running
the validation/tuning lenses on the re-tagged jets confirms it (residual → 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import numpy as np

from delphes_pipeline.core import observables as obs

# flavour -> the map quantity that gives its tag probability
_FLAVOUR_QUANTITY = {5: "btag_eff_b", 4: "btag_eff_c"}  # everything else -> light
BTAG_MAP_QUANTITIES = ("btag_eff_b", "btag_eff_c", "btag_mistag_light")


def derive_maps(config: dict, *, bins=None, max_events=None) -> dict:
    """Measure the per-flavour efficiency curves on the NanoAOD anchor."""
    from .anchor import anchor_profiles

    profiles = anchor_profiles(config, bins=bins or obs.DEFAULT_PT_BINS, max_events=max_events)
    if not profiles:
        raise ValueError("anchor must be enabled (anchor.enabled: true) to derive tuning maps")
    return {
        q: {"x": p.x, "centers": np.asarray(p.centers).tolist(),
            "values": np.asarray(p.values).tolist(), "counts": np.asarray(p.counts).tolist()}
        for q, p in profiles.items()
    }


def save_maps(maps: dict, path, provenance: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"provenance": provenance, "maps": maps}, fh, indent=2, default=str)


class TuningMaps:
    """A loaded set of pT-binned efficiency curves with interpolation."""

    def __init__(self, maps: dict):
        self.maps = maps

    @classmethod
    def load(cls, path) -> "TuningMaps":
        with open(path) as fh:
            return cls(json.load(fh)["maps"])

    def efficiency(self, quantity: str, pt) -> np.ndarray:
        m = self.maps[quantity]
        c = np.asarray(m["centers"], dtype=float)
        v = np.asarray(m["values"], dtype=float)
        pt = np.asarray(pt, dtype=float)
        if c.size == 0:
            return np.zeros_like(pt)
        return np.interp(pt, c, v, left=v[0], right=v[-1])  # flat extrapolation


def retag_btag(events, maps: TuningMaps, rng: np.random.Generator) -> ak.Array:
    """Stochastic b-tag from ``Jet.Flavor`` + the map: BTag = Bernoulli(ε(flavour, pT)).

    Returns a jagged int array aligned with ``events.jets`` (replaces ``Jet.BTag``).
    """
    jets = events.jets
    counts = ak.num(jets)
    pt = ak.to_numpy(ak.flatten(jets.pt))
    flavour = ak.to_numpy(ak.flatten(jets.flavor))
    eff = np.empty(pt.shape, dtype=float)
    is_b = flavour == 5
    is_c = flavour == 4
    is_l = ~(is_b | is_c)
    eff[is_b] = maps.efficiency("btag_eff_b", pt[is_b])
    eff[is_c] = maps.efficiency("btag_eff_c", pt[is_c])
    eff[is_l] = maps.efficiency("btag_mistag_light", pt[is_l])
    tag = (rng.random(pt.shape) < eff).astype(np.int32)
    return ak.unflatten(tag, counts)
