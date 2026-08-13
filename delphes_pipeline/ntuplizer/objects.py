"""Builders from a ``DelphesEvents`` to flat schema collections.

Each ``build_*`` returns a jagged awkward record array whose field names and
dtypes match ``schema.FLAT_SCHEMA`` (NanoAOD-compatible). ``scalars`` returns the
per-event scalar fields of ``schema.SCALARS`` as a dict of numpy arrays. Fields
are cast to the schema dtype so the parquet matches the declared contract
(kinematics float32; tags/charge/pdgId/status/mother int32). The Delphes-leaf ->
flat-name derivation is the one documented in ``schema.DELPHES_SOURCE``.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..core.io import DelphesEvents
from . import schema


def _cast(arr: ak.Array, dtype) -> ak.Array:
    return ak.values_astype(arr, dtype)


def build_jets(ev: DelphesEvents) -> ak.Array:
    """``Jet`` collection: {pt,eta,phi,mass,btag,tautag,hadronFlavour}.

    When ``ev`` is a ``tuning.maps.RetaggedEvents`` view, ``ev.jets`` already carries
    the downstream-re-tagged ``btag``/``tautag``, so the ntuple inherits the tuned tags.
    """
    j = ev.jets
    s = schema.FLAT_SCHEMA["Jet"]
    return ak.zip(
        {
            "pt": _cast(j.pt, s["pt"]),
            "eta": _cast(j.eta, s["eta"]),
            "phi": _cast(j.phi, s["phi"]),
            "mass": _cast(j.mass, s["mass"]),
            "btag": _cast(j.btag, s["btag"]),
            "tautag": _cast(j.tautag, s["tautag"]),
            "hadronFlavour": _cast(j.flavor, s["hadronFlavour"]),
        }
    )


def build_taus(ev: DelphesEvents) -> ak.Array:
    """``Tau`` collection: jets with ``tautag == 1``, kinematics only."""
    j = ev.jets
    sel = j.tautag == 1
    s = schema.FLAT_SCHEMA["Tau"]
    return ak.zip(
        {
            "pt": _cast(j.pt[sel], s["pt"]),
            "eta": _cast(j.eta[sel], s["eta"]),
            "phi": _cast(j.phi[sel], s["phi"]),
            "mass": _cast(j.mass[sel], s["mass"]),
        }
    )


def build_electrons(ev: DelphesEvents) -> ak.Array:
    """``Electron`` collection: {pt,eta,phi,charge}."""
    return _lepton(ev.electrons, schema.FLAT_SCHEMA["Electron"])


def build_muons(ev: DelphesEvents) -> ak.Array:
    """``Muon`` collection: {pt,eta,phi,charge}."""
    return _lepton(ev.muons, schema.FLAT_SCHEMA["Muon"])


def _lepton(coll: ak.Array, s: dict) -> ak.Array:
    return ak.zip(
        {
            "pt": _cast(coll.pt, s["pt"]),
            "eta": _cast(coll.eta, s["eta"]),
            "phi": _cast(coll.phi, s["phi"]),
            "charge": _cast(coll.charge, s["charge"]),
        }
    )


# Keep these and their full m1 ancestry when pruning: τ (the analysis object), ν_τ
# (``gen_visible_taus`` reconstructs p_vis = p_τ − p_ν, so dropping it would make the
# visible τ unrecoverable from the ntuple), and the H/Z/W/top a τ can come from.
PRUNE_KEEP_PIDS = (15, 16, 23, 24, 25, 6)


def prune_genpart(gen: ak.Array, keep_pids=PRUNE_KEEP_PIDS, max_depth: int = 24):
    """Boolean mask over ``gen``: ``keep_pids`` plus every m1 ancestor of those.

    The full Delphes ``Particle`` record is ~99.5% of the ntuple (measured 29.9 of
    30.0 kB/event), i.e. ~9 TB over a 300M-event campaign, and almost none of it is used
    downstream. Keeping the ancestry CLOSURE rather than a flat PID filter is what makes
    ``genPartIdxMother`` still resolvable after pruning: every kept particle's mother is
    kept too, so the chain terminates at a root instead of dangling.
    """
    counts = ak.to_numpy(ak.num(gen))
    apid = np.abs(ak.to_numpy(ak.flatten(gen.pid)))
    m1 = ak.to_numpy(ak.flatten(gen.m1))
    keep = np.isin(apid, np.asarray(keep_pids))
    base = np.repeat(np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    lim = np.repeat(counts, counts)
    valid = (m1 >= 0) & (m1 < lim)
    tgt = base + np.where(valid, m1, 0)
    for _ in range(max_depth):
        add = np.zeros_like(keep)
        add[tgt[keep & valid]] = True
        new = add & ~keep
        if not new.any():
            break
        keep |= add
    return ak.unflatten(keep, counts), keep, counts, m1, valid, base


def _reindex_mothers(keep_flat, counts, m1, valid, base, kept_per_ev):
    """``genPartIdxMother`` renumbered onto the pruned array (−1 where the mother is gone).

    ``kept_per_ev`` is passed in from the jagged mask rather than recomputed with
    ``np.add.reduceat``: reduceat returns ``a[i]`` instead of 0 for an empty slice, so an
    event with no gen particles silently corrupts every offset after it.
    """
    starts = (np.concatenate([[0], np.cumsum(kept_per_ev)[:-1]]) if counts.size
              else np.zeros(0, dtype=np.int64))
    glob = np.cumsum(keep_flat) - 1
    newidx = np.where(keep_flat, glob - np.repeat(starts, counts), -1)
    mother_new = np.where(valid, newidx[base + np.where(valid, m1, 0)], -1)
    return mother_new[keep_flat]


def build_genpart(ev: DelphesEvents, *, prune: bool = False) -> ak.Array:
    """``GenPart`` collection: {pt,eta,phi,mass,pdgId,status,genPartIdxMother}.

    ``GenPart`` is the unpruned Delphes ``Particle`` collection, so
    ``genPartIdxMother`` (= ``Particle.M1``) indexes into this same array — no
    NanoAOD-style pruning is applied, so the mother indices are valid only here.
    """
    g = ev.gen
    s = schema.FLAT_SCHEMA["GenPart"]
    if not prune:
        return ak.zip({
            "pt": _cast(g.pt, s["pt"]), "eta": _cast(g.eta, s["eta"]),
            "phi": _cast(g.phi, s["phi"]), "mass": _cast(g.mass, s["mass"]),
            "pdgId": _cast(g.pid, s["pdgId"]), "status": _cast(g.status, s["status"]),
            "genPartIdxMother": _cast(g.m1, s["genPartIdxMother"]),
        })
    mask, keep_flat, counts, m1, valid, base = prune_genpart(g)
    kept_per_ev = ak.to_numpy(ak.sum(mask, axis=1)).astype(np.int64)
    mother = _reindex_mothers(keep_flat, counts, m1, valid, base, kept_per_ev)
    take = lambda f, t: _cast(ak.unflatten(ak.to_numpy(ak.flatten(f))[keep_flat],
                                           kept_per_ev), t)
    return ak.zip({
        "pt": take(g.pt, s["pt"]), "eta": take(g.eta, s["eta"]),
        "phi": take(g.phi, s["phi"]), "mass": take(g.mass, s["mass"]),
        "pdgId": take(g.pid, s["pdgId"]), "status": take(g.status, s["status"]),
        "genPartIdxMother": _cast(ak.unflatten(mother, kept_per_ev),
                                  s["genPartIdxMother"]),
    })


def _lepton_sf(ev: DelphesEvents, tuning_maps) -> np.ndarray:
    """Per-event lepton-efficiency weight: Π over reco e/μ of the tuning-v0 SF (1.0 if no maps).

    Applied as a weight (an under-reconstructed lepton can't be re-tagged kinematically),
    the analogue of CMS lepton scale factors; the analysis multiplies this into the event
    weight (it is the product over *all* reco leptons — refine to the selected leptons if
    the channel vetoes extra leptons)."""
    sf = np.ones(ev.n, dtype=np.float64)
    if tuning_maps is None:
        return sf
    for coll, q in (("electrons", "electron_sf"), ("muons", "muon_sf")):
        lep = getattr(ev, coll)
        if q not in tuning_maps.maps or not ak.fields(lep):   # no map / branch absent -> no SF
            continue
        vals = tuning_maps.efficiency(q, ak.to_numpy(ak.flatten(lep.pt)), default=1.0)
        sf = sf * ak.to_numpy(ak.prod(ak.unflatten(vals, ak.num(lep)), axis=1))
    return sf


def scalars(ev: DelphesEvents, tuning_maps=None) -> dict[str, np.ndarray]:
    """Per-event scalar fields of ``schema.SCALARS`` as numpy arrays.

    Robust to a missing scalar branch: an absent MissingET/GenMissingET/ScalarHT
    collection falls back to zeros rather than raising. ``genWeight`` is the first
    ``Event.Weight`` per event (1.0 where absent, handled by the reader). ``lepton_sf``
    is the tuning-v0 lepton-efficiency weight (1.0 unless lepton SF maps are configured).
    """
    f32 = np.float32

    def scal(branch: str, getter, default: float) -> np.ndarray:
        if ev.has_branch(branch):
            return ak.to_numpy(ak.fill_none(getter(), default)).astype(f32)
        return np.full(ev.n, default, dtype=f32)

    return {
        "MET_pt": scal("MissingET.MET", lambda: ev.met.met, 0.0),
        "MET_phi": scal("MissingET.Phi", lambda: ev.met.phi, 0.0),
        "GenMET_pt": scal("GenMissingET.MET", lambda: ev.genmet.met, 0.0),
        "GenMET_phi": scal("GenMissingET.Phi", lambda: ev.genmet.phi, 0.0),
        "HT": scal("ScalarHT.HT", lambda: ev.scalar_ht.ht, 0.0),
        "genWeight": ev.weights.astype(f32),
        "lepton_sf": _lepton_sf(ev, tuning_maps).astype(f32),
    }
