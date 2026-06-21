"""Lazy readers for Delphes ROOT output and flat ntuples.

``DelphesEvents`` wraps an uproot tree and exposes each Delphes collection as a
jagged awkward record array with lower-cased field names, so downstream checks
never touch raw branch strings. Field names follow the convention in
``DESIGN.md`` (``pt, eta, phi, mass, flavor, btag, tautag, ...``).

The branch set is exactly what the full ``TreeWriter`` of ``cms_card_v0.tcl``
writes: ``Particle`` (gen), ``Jet``, ``GenJet``, ``Electron``, ``Muon``,
``Photon``, ``FatJet``, ``MissingET``, ``GenMissingET``, ``ScalarHT``, and the
``Event`` weight.
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Optional

import awkward as ak
import numpy as np
import uproot

# flat field name -> Delphes leaf under the collection branch
_JET_FIELDS = {
    "pt": "PT",
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "flavor": "Flavor",
    "btag": "BTag",
    "tautag": "TauTag",
    "charge": "Charge",
}
_GENJET_FIELDS = {"pt": "PT", "eta": "Eta", "phi": "Phi", "mass": "Mass", "flavor": "Flavor"}
_LEP_FIELDS = {"pt": "PT", "eta": "Eta", "phi": "Phi", "charge": "Charge"}
_PHOTON_FIELDS = {"pt": "PT", "eta": "Eta", "phi": "Phi"}
_FATJET_FIELDS = {"pt": "PT", "eta": "Eta", "phi": "Phi", "mass": "Mass"}
_GEN_FIELDS = {
    "pid": "PID",
    "status": "Status",
    "pt": "PT",
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "charge": "Charge",
    "m1": "M1",
    "m2": "M2",
    "d1": "D1",
    "d2": "D2",
}
_MET_FIELDS = {"met": "MET", "eta": "Eta", "phi": "Phi"}
_SCALARHT_FIELDS = {"ht": "HT"}


def _zip_collection(tree, prefix, fields, entry_stop, available) -> ak.Array:
    """Read ``prefix.<leaf>`` branches and zip them into a jagged record array.

    Membership is tested against ``available`` (the set of ``tree.keys()``),
    which reliably contains the dotted sub-branch names ``Jet.PT`` on both real
    Delphes TTrees and the test fixture. Missing leaves are silently skipped so
    the reader is robust to card variants; a collection with no present branches
    yields a per-event empty array.
    """
    want = {flat: f"{prefix}.{leaf}" for flat, leaf in fields.items()}
    present = {flat: br for flat, br in want.items() if br in available}
    if not present:
        return ak.Array([[] for _ in range(_num_entries(tree, entry_stop))])
    # Read each sub-branch individually: tree["Jet.PT"].array() resolves on both
    # real Delphes TTrees and the RNTuple fixture, whereas a single
    # tree.arrays(["Jet.PT", ...]) does not key reliably for dotted names.
    cols = {flat: tree[br].array(entry_stop=entry_stop) for flat, br in present.items()}
    return ak.zip(cols)


def _num_entries(tree, entry_stop) -> int:
    n = tree.num_entries
    return n if entry_stop is None else min(n, entry_stop)


class DelphesEvents:
    """Lazy uproot-backed view of a Delphes ROOT file.

    Parameters
    ----------
    path : str
        Path to the Delphes ROOT file.
    treename : str
        Tree name (Delphes default is ``"Delphes"``).
    entry_stop : int | None
        If set, only the first ``entry_stop`` events are read (fast gate runs).
    """

    def __init__(self, path: str, treename: str = "Delphes", entry_stop: Optional[int] = None):
        self.path = path
        self.treename = treename
        self.entry_stop = entry_stop
        self._file = uproot.open(path)
        self._tree = self._file[treename]
        # full recursive key set, includes dotted sub-branches ("Jet.PT")
        self._keys = set(self._tree.keys())

    @property
    def n(self) -> int:
        """Number of events read (respecting ``entry_stop``)."""
        return _num_entries(self._tree, self.entry_stop)

    def has_branch(self, name: str) -> bool:
        return name in self._keys

    def array(self, branch: str) -> ak.Array:
        """Raw read of a single branch (used for ad-hoc checks)."""
        return self._tree[branch].array(entry_stop=self.entry_stop)

    # ----- collections (jagged record arrays, one list per event) --------- #
    @cached_property
    def jets(self) -> ak.Array:
        return _zip_collection(self._tree, "Jet", _JET_FIELDS, self.entry_stop, self._keys)

    @cached_property
    def genjets(self) -> ak.Array:
        return _zip_collection(self._tree, "GenJet", _GENJET_FIELDS, self.entry_stop, self._keys)

    @cached_property
    def electrons(self) -> ak.Array:
        return _zip_collection(self._tree, "Electron", _LEP_FIELDS, self.entry_stop, self._keys)

    @cached_property
    def muons(self) -> ak.Array:
        return _zip_collection(self._tree, "Muon", _LEP_FIELDS, self.entry_stop, self._keys)

    @cached_property
    def photons(self) -> ak.Array:
        return _zip_collection(self._tree, "Photon", _PHOTON_FIELDS, self.entry_stop, self._keys)

    @cached_property
    def fatjets(self) -> ak.Array:
        return _zip_collection(self._tree, "FatJet", _FATJET_FIELDS, self.entry_stop, self._keys)

    @cached_property
    def gen(self) -> ak.Array:
        """Gen ``Particle`` collection (GenParticle)."""
        return _zip_collection(self._tree, "Particle", _GEN_FIELDS, self.entry_stop, self._keys)

    # ----- scalar-per-event collections ----------------------------------- #
    @cached_property
    def met(self) -> ak.Array:
        """``MissingET`` as one record per event (fields: met, eta, phi)."""
        return ak.firsts(_zip_collection(self._tree, "MissingET", _MET_FIELDS, self.entry_stop, self._keys))

    @cached_property
    def genmet(self) -> ak.Array:
        return ak.firsts(_zip_collection(self._tree, "GenMissingET", _MET_FIELDS, self.entry_stop, self._keys))

    @cached_property
    def scalar_ht(self) -> ak.Array:
        z = _zip_collection(self._tree, "ScalarHT", _SCALARHT_FIELDS, self.entry_stop, self._keys)
        return ak.firsts(z)

    @cached_property
    def weights(self) -> np.ndarray:
        """First ``Event.Weight`` per event (1.0 where absent)."""
        if "Event.Weight" in self._keys:
            w = self._tree["Event.Weight"].array(entry_stop=self.entry_stop)
            return ak.to_numpy(ak.fill_none(ak.firsts(w), 1.0))
        return np.ones(self.n)

    @property
    def bytes_per_event(self) -> float:
        """On-disk file size per event (storage projection)."""
        try:
            return os.path.getsize(self.path) / max(self.n, 1)
        except OSError:
            return float("nan")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "DelphesEvents":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_ntuple(path: str) -> ak.Array:
    """Load a flat NanoAOD-compatible ntuple written by the ntuplizer (parquet)."""
    return ak.from_parquet(path)
