"""Lazy readers for Delphes ROOT output and flat ntuples.

``DelphesEvents`` wraps one or more uproot trees and exposes each Delphes
collection as a jagged awkward record array with lower-cased field names, so
downstream checks never touch raw branch strings. Field names follow the
convention in ``DESIGN.md`` (``pt, eta, phi, mass, flavor, btag, tautag, ...``).

The input may be a single ROOT file, a glob pattern, a directory (every ``*.root``
under it is read, recursively), or an explicit list of files — a Delphes "sample"
is a directory of ROOT files. Files are read in sorted order and concatenated;
``entry_stop``, when set, caps the *total* number of events across files.

The branch set is exactly what the full ``TreeWriter`` of ``cms_card_v0.tcl``
writes: ``Particle`` (gen), ``Jet``, ``GenJet``, ``Electron``, ``Muon``,
``Photon``, ``FatJet``, ``MissingET``, ``GenMissingET``, ``ScalarHT``, and the
``Event`` weight.
"""

from __future__ import annotations

import glob
import os
from functools import cached_property
from typing import Optional, Sequence, Union

import awkward as ak
import numpy as np
import uproot

PathLike = Union[str, Sequence[str]]

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


def resolve_paths(path: PathLike) -> list[str]:
    """Expand ``path`` (file / glob / directory / list) into a sorted file list."""
    if isinstance(path, (list, tuple)):
        files = [f for p in path for f in resolve_paths(p)]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.root"), recursive=True))
    elif any(ch in str(path) for ch in "*?["):
        files = sorted(glob.glob(str(path), recursive=True))
    else:
        files = [str(path)]
    if not files:
        raise FileNotFoundError(f"no ROOT files matched input {path!r}")
    return files


class DelphesEvents:
    """Lazy uproot-backed view of one or more Delphes ROOT files.

    Parameters
    ----------
    path : str | list[str]
        A ROOT file, glob, directory, or list of files (see module docstring).
    treename : str
        Tree name (Delphes default is ``"Delphes"``).
    entry_stop : int | None
        If set, read only the first ``entry_stop`` events in total (fast runs).
    """

    def __init__(self, path: PathLike, treename: str = "Delphes", entry_stop: Optional[int] = None):
        self.path = path
        self.treename = treename
        self.entry_stop = entry_stop
        self.paths = resolve_paths(path)
        # Open files lazily and stop once entry_stop is satisfied, so a fast run
        # over a 200-file sample does not open all 200 just to read the first few.
        self._files, self._trees, self._stops, self._used = [], [], [], []
        remaining = entry_stop
        for p in self.paths:
            if entry_stop is not None and remaining <= 0:
                break
            f = uproot.open(p)
            t = f[treename]
            count = t.num_entries
            stop = count if entry_stop is None else min(count, remaining)
            self._files.append(f)
            self._trees.append(t)
            self._stops.append(stop)
            self._used.append(p)
            if entry_stop is not None:
                remaining -= stop
        if not self._trees:
            raise ValueError(f"no readable trees for input {path!r}")
        # branches present in every opened file (dotted sub-branches like "Jet.PT")
        self._keys = set.intersection(*[set(t.keys()) for t in self._trees])
        self._n = int(sum(self._stops))

    @property
    def n(self) -> int:
        """Total number of events read across all files (respecting ``entry_stop``)."""
        return self._n

    def has_branch(self, name: str) -> bool:
        return name in self._keys

    def array(self, branch: str) -> ak.Array:
        """Read a branch across all files and concatenate (respecting per-file caps)."""
        arrs = [t[branch].array(entry_stop=s) for t, s in zip(self._trees, self._stops) if s > 0]
        if not arrs:
            return ak.Array([])
        return arrs[0] if len(arrs) == 1 else ak.concatenate(arrs)

    def _zip(self, prefix: str, fields: dict) -> ak.Array:
        """Read ``prefix.<leaf>`` branches and zip them into a jagged record array.

        Missing leaves are skipped; a collection with no present branches yields a
        per-event empty array of the right length.
        """
        present = {flat: f"{prefix}.{leaf}" for flat, leaf in fields.items()
                   if f"{prefix}.{leaf}" in self._keys}
        if not present:
            return ak.Array([[] for _ in range(self.n)])
        return ak.zip({flat: self.array(br) for flat, br in present.items()})

    # ----- collections (jagged record arrays, one list per event) --------- #
    @cached_property
    def jets(self) -> ak.Array:
        return self._zip("Jet", _JET_FIELDS)

    @cached_property
    def genjets(self) -> ak.Array:
        return self._zip("GenJet", _GENJET_FIELDS)

    @cached_property
    def electrons(self) -> ak.Array:
        return self._zip("Electron", _LEP_FIELDS)

    @cached_property
    def muons(self) -> ak.Array:
        return self._zip("Muon", _LEP_FIELDS)

    @cached_property
    def photons(self) -> ak.Array:
        return self._zip("Photon", _PHOTON_FIELDS)

    @cached_property
    def fatjets(self) -> ak.Array:
        return self._zip("FatJet", _FATJET_FIELDS)

    @cached_property
    def gen(self) -> ak.Array:
        """Gen ``Particle`` collection (GenParticle)."""
        return self._zip("Particle", _GEN_FIELDS)

    # ----- scalar-per-event collections ----------------------------------- #
    @cached_property
    def met(self) -> ak.Array:
        """``MissingET`` as one record per event (fields: met, eta, phi)."""
        return ak.firsts(self._zip("MissingET", _MET_FIELDS))

    @cached_property
    def genmet(self) -> ak.Array:
        return ak.firsts(self._zip("GenMissingET", _MET_FIELDS))

    @cached_property
    def scalar_ht(self) -> ak.Array:
        return ak.firsts(self._zip("ScalarHT", _SCALARHT_FIELDS))

    @cached_property
    def weights(self) -> np.ndarray:
        """First ``Event.Weight`` per event (1.0 where absent)."""
        if "Event.Weight" in self._keys:
            return ak.to_numpy(ak.fill_none(ak.firsts(self.array("Event.Weight")), 1.0))
        return np.ones(self.n)

    @property
    def bytes_per_event(self) -> float:
        """On-disk size per event over the files actually read (storage projection)."""
        try:
            total = sum(os.path.getsize(p) for p in self._used)
            return total / max(self.n, 1)
        except OSError:
            return float("nan")

    def close(self) -> None:
        for f in self._files:
            f.close()

    def __enter__(self) -> "DelphesEvents":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_ntuple(path: str) -> ak.Array:
    """Load a flat NanoAOD-compatible ntuple written by the ntuplizer (parquet)."""
    return ak.from_parquet(path)
