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
# a τ_h visible mass must stay below m_τ = 1.777 GeV or the FastMTT hadronic decay
# prior has no valid solution; cap just under it (CMS medians sit near 0.8-1.2).
_MAX_TAU_VIS_MASS = 1.70


def _serialise(p) -> dict:
    d = {"x": p.x, "centers": np.asarray(p.centers).tolist(),
         "values": np.asarray(p.values).tolist(), "counts": np.asarray(p.counts).tolist()}
    if getattr(p, "aux", None):
        d.update(p.aux)          # e.g. the τ_h mass quantiles, needed to sample the map
    return d


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
        # b-jets: corrected to their own GenJet. CMS jet energies are JEC-calibrated to
        # gen, so self-anchoring is adequate here — and m_bb agrees with CMS in the overlay.
        maps["bjet_escale"] = _invert_to_unity(obs.bjet_energy_response(ev, bins=bins))
        # τ-jets: corrected to the CMS τ energy scale, NOT to gen. A Delphes τ_h is a jet
        # (jet-level pT, UE inside R=0.4) while a CMS Tau carries the clean HPS visible-τ
        # pT; inverting to unity against a GenJet — also a jet — would preserve that gap
        # and leave Delphes τ harder than CMS, inflating m_vis and so m_ττ.
        d_tau_resp = obs.tau_energy_response(ev, bins=bins)
        if "tau_energy_response" in profiles:
            maps["tau_escale"] = _scale_factor(profiles["tau_energy_response"], d_tau_resp)
        else:
            maps["tau_escale"] = _invert_to_unity(d_tau_resp)
        # MET: the quadrature gap to the anchor resolution (pileup + detector noise the
        # no-pileup card cannot produce). Measured with the SAME estimator on both sides.
        if "met_resolution" in profiles:
            dx, dy, _ = obs.met_residuals(ev)
            d_res = float(np.sqrt(0.5 * (np.var(dx) + np.var(dy)))) if dx.size else 0.0
            a_res = float(np.asarray(profiles["met_resolution"].values)[0])
            sigma = float(np.sqrt(max(a_res ** 2 - d_res ** 2, 0.0)))
            maps["met_smear"] = {"x": "overall", "centers": [0.0], "values": [sigma],
                                 "counts": [int(dx.size)],
                                 "anchor_resolution_gev": a_res,
                                 "delphes_resolution_gev": d_res}
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

    def efficiency(self, quantity: str, pt, *, default: float = 0.0) -> np.ndarray:
        """Interpolate the curve at ``pt``. ``default`` is returned for an empty grid —
        0 for efficiencies, but 1.0 for *multiplicative* corrections (escale / SF) so a
        missing/empty map is a no-op rather than zeroing the jet pT or the event weight."""
        m = self.maps[quantity]
        c = np.asarray(m["centers"], dtype=float)
        v = np.asarray(m["values"], dtype=float)
        pt = np.asarray(pt, dtype=float)
        if c.size == 0:
            return np.full(pt.shape, default)
        return np.interp(pt, c, v, left=v[0], right=v[-1])  # flat extrapolation


def retag_btag(events, maps: TuningMaps, rng: np.random.Generator, jets=None) -> ak.Array:
    """Stochastic b-tag from ``Jet.Flavor`` + the map: BTag = Bernoulli(ε(flavour, pT)).

    ``jets`` defaults to ``events.jets`` but is taken explicitly by ``retag_jets`` so the
    efficiency is evaluated on the ENERGY-CORRECTED jets: the map is measured against
    CMS-scale pT, so it must be applied at a CMS-scale pT too.
    Returns a jagged int array aligned with ``jets`` (replaces ``Jet.BTag``).
    """
    jets = events.jets if jets is None else jets
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


def retag_tautag(events, maps: TuningMaps, rng: np.random.Generator, jets=None) -> ak.Array:
    """Stochastic τ_h tag from the gen record + maps (note D2-A): TauTag = Bernoulli(ε),
    ε = ``tau_eff`` for jets matched (ΔR<0.4) to a gen hadronic τ, else ``tau_mistag``.

    Returns a jagged int array aligned with ``jets`` (replaces ``Jet.TauTag``).
    Mirrors the ``observables.tau_efficiency`` / ``tau_mistag`` genuine-vs-fake split, so
    re-measuring those observables on the re-tagged jets recovers the maps by construction.
    ``jets`` defaults to ``events.jets``; ``retag_jets`` passes the energy-corrected jets.
    """
    jets = events.jets if jets is None else jets
    counts = ak.num(jets)
    # HADRONIC gen τ only: tau_eff is measured on the anchor's GenVisTau (hadronic by
    # construction), so handing a leptonic τ's jet that efficiency instead of the fake
    # rate fabricates τ_hτ_h events out of τ_hτ_ℓ / τ_ℓτ_ℓ ones.
    taus = obs.gen_taus(events.gen, hadronic_only=True)
    genuine = ak.to_numpy(ak.flatten(matched_to_any(jets, taus, 0.4)))
    # ε is evaluated at the jet pT (≈ the visible-τ pT for a genuine τ-jet); a steeply
    # pT-dependent tau_eff carries a mild jet-vs-gen-τ-pT smearing in the closure.
    pt = ak.to_numpy(ak.flatten(jets.pt))
    eff = np.where(genuine, maps.efficiency("tau_eff", pt), maps.efficiency("tau_mistag", pt))
    tag = (rng.random(pt.shape) < eff).astype(np.int32)
    return ak.unflatten(tag, counts)


def _escale_lookup(maps: TuningMaps, quantity: str, reco_pt: np.ndarray) -> np.ndarray:
    """The energy-scale map is binned in GenJet pT but we hold reco pT; one Newton step
    (gen_pT ≈ reco_pT · escale) re-evaluates at the estimated gen pT so a pT-sloped
    response still closes to unity (exact for a flat response)."""
    e0 = maps.efficiency(quantity, reco_pt, default=1.0)
    return maps.efficiency(quantity, reco_pt * e0, default=1.0)


def escale_factor(events, jets: ak.Array, maps: TuningMaps) -> ak.Array:
    """Per-jet energy-scale factor on the SAME populations the map is derived/validated on:
    ``tau_escale`` for jets gen-matched (ΔR<0.4) to a hadronic τ (τ precedence), else
    ``bjet_escale`` for b-jets (flavor==5), 1 otherwise. Applying by the gen-matched τ set
    (not the stochastic tautag bit) keeps derive/apply/re-validate on one population."""
    counts = ak.num(jets)
    pt = ak.to_numpy(ak.flatten(jets.pt))
    flavour = ak.to_numpy(ak.flatten(jets.flavor))
    gen_taus = events.gen[np.abs(events.gen.pid) == _GEN_TAU_PID]
    is_tau = ak.to_numpy(ak.flatten(matched_to_any(jets, gen_taus, 0.4)))
    esc = np.ones(pt.shape, dtype=float)
    is_b = flavour == 5
    esc[is_b] = _escale_lookup(maps, "bjet_escale", pt[is_b])
    esc[is_tau] = _escale_lookup(maps, "tau_escale", pt[is_tau])   # τ precedence
    return ak.unflatten(esc, counts)


def _sample_tau_mass(m: dict, pt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a τ_h visible mass per jet from the anchor's per-pT quantiles.

    Sampling — not the median — because in FastMTT the visible mass is a one-sided
    *floor* on the energy fraction, ``xmin = (m_vis/m_τ)²``, not a smearing kernel.
    Assigning one value to every τ-jet relaxes that floor for every leg whose true mass
    lies above it, opening the low-x region where ``m_ττ = m_vis/√(x₁x₂)`` diverges: the
    m_ττ width inflates and the mean is dragged up, one-sided. Drawing from the measured
    distribution reproduces the CMS spread instead, so the floor is right on average
    *and* per object.
    """
    levels = np.asarray(m["quantile_levels"], dtype=float)
    qvals = np.asarray(m["quantile_values"], dtype=float)          # (n_bins, n_levels)
    centers = np.asarray(m["centers"], dtype=float)
    idx = np.abs(pt[:, None] - centers[None, :]).argmin(axis=1)    # nearest pT bin
    # levels are a uniform grid, so the quantile inverse is a gather + linear blend
    pos = rng.random(pt.shape) * (levels.size - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, levels.size - 1)
    frac = pos - lo
    return qvals[idx, lo] * (1.0 - frac) + qvals[idx, hi] * frac


def tau_visible_mass(jets: ak.Array, maps: TuningMaps, rng: np.random.Generator) -> ak.Array:
    """Replace the τ-tagged jets' mass with a CMS-like τ_h visible mass.

    A Delphes τ_h *is* a jet, so its mass is the AK4 jet mass (multi-GeV: UE and extra
    particles inside R=0.4). A CMS τ_h carries its decay-mode visible mass, ≲ m_τ =
    1.777 GeV. The gap is not cosmetic: the FastMTT hadronic decay prior is zero unless
    ``(m_vis/m_τ)² ≤ 1``, so a τ-jet keeping its jet mass admits *no* solution and m_ττ
    comes back NaN — which silently removed ~89% of Delphes τ_hτ_h events. Non-τ jets
    keep their own mass.

    The mass is *drawn* from the anchor's per-pT quantiles (see ``_sample_tau_mass``);
    a maps file carrying only the median falls back to that value, which fixes the yield
    but distorts the m_ττ shape, so re-derive rather than rely on it.
    """
    m = maps.maps.get("tau_mass", {})
    if not np.asarray(m.get("centers", []), dtype=float).size:
        return jets.mass                      # empty map -> no-op, never NaN
    counts = ak.num(jets)
    pt = ak.to_numpy(ak.flatten(jets.pt))
    mass = ak.to_numpy(ak.flatten(jets.mass)).copy()
    is_tau = ak.to_numpy(ak.flatten(jets.tautag)) == 1
    if is_tau.any():
        if m.get("quantile_values"):
            drawn = _sample_tau_mass(m, pt[is_tau], rng)
        else:                                  # legacy median-only map
            drawn = maps.efficiency("tau_mass", pt[is_tau])
        # hard-capped below m_τ = 1.777: a visible mass at or above it leaves the FastMTT
        # hadronic prior with no valid x, which is the failure this map exists to remove.
        mass[is_tau] = np.clip(drawn, 0.0, _MAX_TAU_VIS_MASS)
    return ak.unflatten(mass, counts)


def retag_jets(events, maps: TuningMaps, rng: np.random.Generator):
    """Apply the downstream tuning-v0 corrections to the jets.

    Fixed order — **energy scale first**, then the tag draws, then the visible mass:

        escale -> btag -> tautag -> tau_mass

    Every map is measured against a CMS-scale quantity, so every map must be *applied* at
    a CMS-scale pT. Drawing the tags on the raw jets instead evaluates the efficiency at
    a pT that is ~17% too high at low pT (the underlying event inside the R=0.4 cone),
    and ``tau_eff`` rises steeply there (0.39 -> 0.52 between 25 and 35 GeV), so soft τ
    were being over-efficient. ``tau_mass`` stays last so the assigned mass is final and
    is not then rescaled. The order is fixed so the same seed yields identical output in
    the tuning lens and the ntuplizer.
    Returns ``(jets, fields)`` with the set of corrections actually applied.
    """
    jets = events.jets
    fields = set()
    if all(q in maps.maps for q in ESCALE_MAP_QUANTITIES):
        esc = escale_factor(events, jets, maps)
        jets = ak.with_field(jets, jets.pt * esc, "pt")
        jets = ak.with_field(jets, jets.mass * esc, "mass")
        fields.add("escale")
    if all(q in maps.maps for q in BTAG_MAP_QUANTITIES):
        jets = ak.with_field(jets, retag_btag(events, maps, rng, jets), "btag")
        fields.add("btag")
    if all(q in maps.maps for q in TAU_MAP_QUANTITIES):
        jets = ak.with_field(jets, retag_tautag(events, maps, rng, jets), "tautag")
        fields.add("tautag")
    if "tau_mass" in maps.maps:
        jets = ak.with_field(jets, tau_visible_mass(jets, maps, rng), "mass")
        fields.add("tau_mass")
    elif "tautag" in fields:
        print("[maps] WARNING: no 'tau_mass' map -> tau_h jets keep the AK4 jet mass "
              "(> m_tau), so FastMTT will return NaN for most pairs. Re-derive the maps.",
              flush=True)
    return jets, frozenset(fields)


def smear_met(events, maps: TuningMaps, rng: np.random.Generator):
    """Degrade the Delphes pT_miss to the CMS resolution measured on the anchor.

    The card runs WITHOUT pileup (D3 option A) and its header is explicit that
    "pileup enters through the tuning maps" — this is that map. Delphes MET is
    correspondingly unphysically clean (~16 GeV per component against ~33 on the CMS
    anchor), and MET is what FastMTT fits the τ energy fractions against, so the gap
    propagates straight into m_ττ and everything built from the di-τ system.

    Gaussian noise of the stored σ is added per component; σ is the quadrature
    difference, so the smeared resolution lands on the anchor's by construction.
    """
    vals = (maps.maps.get("met_smear") or {}).get("values") or []
    sigma = float(vals[0]) if vals else 0.0
    met = events.met
    if sigma <= 0:
        return met
    x = ak.to_numpy(ak.fill_none(met.met * np.cos(met.phi), 0.0))
    y = ak.to_numpy(ak.fill_none(met.met * np.sin(met.phi), 0.0))
    x = x + rng.normal(0.0, sigma, size=x.shape)
    y = y + rng.normal(0.0, sigma, size=y.shape)
    return ak.zip({"met": np.hypot(x, y), "eta": np.zeros_like(x), "phi": np.arctan2(y, x)})


class RetaggedEvents:
    """An events view with the downstream tuning-v0 corrections applied to ``.jets``.

    Proxies every attribute to the wrapped events; only ``.jets`` is overridden — its
    tag bits re-derived from ``Jet.Flavor`` (b-tag) and the gen record (τ_h), and its
    b-jet/τ-jet pT+mass rescaled by the energy-scale maps. All other collections are
    unchanged, so every observable re-measures the *tuned* response through the same
    ``core.observables`` path. ``.met`` is also overridden when a ``met_smear`` map is
    present (the card has no pileup; the map puts that resolution back).
    ``retagged_fields`` reports which corrections were applied
    ({'btag','tautag','escale','tau_mass','met_smear'}).
    """

    def __init__(self, events, maps: TuningMaps, rng: np.random.Generator):
        self._events = events
        # fixed order (jets, then MET) so the shared rng yields identical output in the
        # tuning lens and in the ntuplizer
        self._jets, fields = retag_jets(events, maps, rng)
        self._met = None
        if (maps.maps.get("met_smear") or {}).get("values"):
            self._met = smear_met(events, maps, rng)
            fields = fields | {"met_smear"}
        self.retagged_fields = frozenset(fields)

    @property
    def jets(self) -> ak.Array:
        return self._jets

    @property
    def met(self) -> ak.Array:
        return self._events.met if self._met is None else self._met

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
