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
from delphes_pipeline.core.matching import matched_to_any

# flavour -> the map quantity that gives its tag probability
_FLAVOUR_QUANTITY = {5: "btag_eff_b", 4: "btag_eff_c"}  # everything else -> light
BTAG_MAP_QUANTITIES = ("btag_eff_b", "btag_eff_c", "btag_mistag_light")
TAU_MAP_QUANTITIES = ("tau_eff", "tau_mistag")
ESCALE_MAP_QUANTITIES = ("bjet_escale", "tau_escale")   # jet-pT energy-scale corrections
LEPTON_SF_QUANTITIES = ("electron_sf", "muon_sf")       # efficiency scale factors (weights)
_GEN_TAU_PID = 15


def _serialise(p) -> dict:
    return {"x": p.x, "centers": np.asarray(p.centers).tolist(),
            "values": np.asarray(p.values).tolist(), "counts": np.asarray(p.counts).tolist()}


def _invert_to_unity(p) -> dict:
    """Energy-scale correction = 1/response (so the corrected reco/gen response -> 1)."""
    v = np.asarray(p.values, dtype=float)
    corr = np.clip(np.where(v > 0, 1.0 / v, 1.0), 0.5, 2.0)
    return {"x": p.x, "centers": np.asarray(p.centers).tolist(),
            "values": corr.tolist(), "counts": np.asarray(p.counts).tolist()}


def _scale_factor(anchor_p, delphes_p) -> dict:
    """Lepton efficiency scale factor = anchor_eff / delphes_eff (on the Delphes grid)."""
    dc = np.asarray(delphes_p.centers, dtype=float)
    dv = np.asarray(delphes_p.values, dtype=float)
    av = np.interp(dc, np.asarray(anchor_p.centers, dtype=float), np.asarray(anchor_p.values, dtype=float))
    sf = np.clip(np.where(dv > 0, av / dv, 1.0), 0.5, 2.0)
    return {"x": "pt", "centers": dc.tolist(), "values": sf.tolist(),
            "counts": np.asarray(delphes_p.counts).tolist()}


def derive_maps(config: dict, *, bins=None, max_events=None) -> dict:
    """Derive all tuning-v0 corrections.

    Efficiency maps (b-tag, τ_h, lepton eff) come from the NanoAOD anchor; the
    **energy-scale** corrections (1/response toward unity) and the **lepton scale
    factors** (anchor_eff/delphes_eff) need the *Delphes* response, measured from
    ``input.delphes_root`` when present.
    """
    from .anchor import anchor_profiles

    bins = bins or obs.DEFAULT_PT_BINS
    profiles = anchor_profiles(config, bins=bins, max_events=max_events)
    if not profiles:
        raise ValueError("anchor must be enabled (anchor.enabled: true) to derive tuning maps")
    maps = {q: _serialise(p) for q, p in profiles.items()}

    delphes_root = config.get("input", {}).get("delphes_root")
    if delphes_root:
        from delphes_pipeline.core.io import DelphesEvents
        print("[maps] measuring the Delphes energy response + lepton efficiency ...", flush=True)
        ev = DelphesEvents(delphes_root, treename=config.get("input", {}).get("treename", "Delphes"),
                           entry_stop=max_events)
        maps["bjet_escale"] = _invert_to_unity(obs.bjet_energy_response(ev, bins=bins))
        maps["tau_escale"] = _invert_to_unity(obs.tau_energy_response(ev, bins=bins))
        for sf_name, eff in (("electron_sf", "electron_eff"), ("muon_sf", "muon_eff")):
            if eff in profiles:
                maps[sf_name] = _scale_factor(profiles[eff], obs.lepton_efficiency(ev, eff, bins=bins))
    return maps


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


def retag_tautag(events, maps: TuningMaps, rng: np.random.Generator) -> ak.Array:
    """Stochastic τ_h tag from the gen record + maps (note D2-A): TauTag = Bernoulli(ε),
    ε = ``tau_eff`` for jets matched (ΔR<0.4) to a gen hadronic τ, else ``tau_mistag``.

    Returns a jagged int array aligned with ``events.jets`` (replaces ``Jet.TauTag``).
    Mirrors the ``observables.tau_efficiency`` / ``tau_mistag`` genuine-vs-fake split, so
    re-measuring those observables on the re-tagged jets recovers the maps by construction.
    """
    jets = events.jets
    counts = ak.num(jets)
    gen_taus = events.gen[np.abs(events.gen.pid) == _GEN_TAU_PID]
    genuine = ak.to_numpy(ak.flatten(matched_to_any(jets, gen_taus, 0.4)))
    # ε is evaluated at the jet pT (≈ the visible-τ pT for a genuine τ-jet); a steeply
    # pT-dependent tau_eff carries a mild jet-vs-gen-τ-pT smearing in the closure.
    pt = ak.to_numpy(ak.flatten(jets.pt))
    eff = np.where(genuine, maps.efficiency("tau_eff", pt), maps.efficiency("tau_mistag", pt))
    tag = (rng.random(pt.shape) < eff).astype(np.int32)
    return ak.unflatten(tag, counts)


def escale_factor(jets: ak.Array, maps: TuningMaps) -> ak.Array:
    """Per-jet energy-scale factor: ``bjet_escale`` for b-jets (flavor==5), ``tau_escale``
    for τ-jets (tautag==1, τ taking precedence), 1 otherwise. Reads the (re-tagged) jets."""
    counts = ak.num(jets)
    pt = ak.to_numpy(ak.flatten(jets.pt))
    flavour = ak.to_numpy(ak.flatten(jets.flavor))
    tautag = ak.to_numpy(ak.flatten(jets.tautag))
    esc = np.ones(pt.shape, dtype=float)
    is_b = flavour == 5
    esc[is_b] = maps.efficiency("bjet_escale", pt[is_b])
    is_t = tautag == 1
    esc[is_t] = maps.efficiency("tau_escale", pt[is_t])
    return ak.unflatten(esc, counts)


def retag_jets(events, maps: TuningMaps, rng: np.random.Generator):
    """Apply the downstream tuning-v0 corrections to the jets.

    b-tag (flavour maps) and τ_h (tau_eff+tau_mistag) tags are re-derived; b-jet/τ-jet
    pT+mass are rescaled by the energy-scale maps. Fixed order (btag, tautag, escale) so
    the same seed yields identical output in the tuning lens and the ntuplizer. Returns
    ``(jets, fields)`` with the set of corrections actually applied.
    """
    jets = events.jets
    fields = set()
    if all(q in maps.maps for q in BTAG_MAP_QUANTITIES):
        jets = ak.with_field(jets, retag_btag(events, maps, rng), "btag")
        fields.add("btag")
    if all(q in maps.maps for q in TAU_MAP_QUANTITIES):
        jets = ak.with_field(jets, retag_tautag(events, maps, rng), "tautag")
        fields.add("tautag")
    if all(q in maps.maps for q in ESCALE_MAP_QUANTITIES):
        esc = escale_factor(jets, maps)
        jets = ak.with_field(jets, jets.pt * esc, "pt")
        jets = ak.with_field(jets, jets.mass * esc, "mass")
        fields.add("escale")
    return jets, frozenset(fields)


class RetaggedEvents:
    """An events view with the downstream tuning-v0 corrections applied to ``.jets``.

    Proxies every attribute to the wrapped events; only ``.jets`` is overridden — its
    tag bits re-derived from ``Jet.Flavor`` (b-tag) and the gen record (τ_h), and its
    b-jet/τ-jet pT+mass rescaled by the energy-scale maps. All other collections are
    unchanged, so every observable re-measures the *tuned* response through the same
    ``core.observables`` path. ``retagged_fields`` reports which corrections were applied
    ({'btag','tautag','escale'}).
    """

    def __init__(self, events, maps: TuningMaps, rng: np.random.Generator):
        self._events = events
        self._jets, self.retagged_fields = retag_jets(events, maps, rng)

    @property
    def jets(self) -> ak.Array:
        return self._jets

    def __getattr__(self, name):
        # Reject dunders and the pre-init state so copy/pickle fail with a normal
        # AttributeError instead of recursing on a missing self._events.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            events = object.__getattribute__(self, "_events")
        except AttributeError:
            raise AttributeError(name)
        return getattr(events, name)
