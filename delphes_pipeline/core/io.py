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
import time

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
# Particle (gen allParticles) is the heaviest branch in a Delphes file, so we
# read ONLY the leaves the pipeline consumes: pid/status (selection), pt/eta/phi/
# mass (4-vectors), m1 (mother walk). charge/m2/d1/d2 are never used downstream
# and reading them roughly triples the per-particle cost for nothing.
_GEN_FIELDS = {
    "pid": "PID",
    "status": "Status",
    "pt": "PT",
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "m1": "M1",
}
_MET_FIELDS = {"met": "MET", "eta": "Eta", "phi": "Phi"}
_SCALARHT_FIELDS = {"ht": "HT"}


# A campaign of ~1400 shards streaming from dCache sees transient XRootD failures at
# well under 1% per open -- a redirector handing out a data server whose certificate
# does not cover the address, a door briefly refusing. Without a retry each one costs a
# whole ~22 min shard, so bounded retries are far cheaper than the reruns they avoid.
OPEN_TRIES = 4
OPEN_BACKOFF_S = 5.0

# Errors worth retrying: a bad door, a TLS/redirect failure, a dropped connection. A
# genuinely absent or corrupt file raises the same OSError, so retrying costs a few
# wasted seconds in that case -- an acceptable price for not losing good shards.
_RETRYABLE = ("did not open properly", "tls", "connection", "timeout", "redirect",
              "operation expired", "socket", "no servers")


def _is_retryable(exc: BaseException) -> bool:
    m = str(exc).lower()
    return any(k in m for k in _RETRYABLE)


def open_with_retry(path, *, tries: int = OPEN_TRIES, backoff: float = OPEN_BACKOFF_S,
                    _sleep=time.sleep):
    """``uproot.open`` that survives a transient remote failure.

    Local paths fail fast on the first attempt, so a typo does not cost 15 s of
    pointless backoff; only remote (``root://``) opens are retried.
    """
    remote = isinstance(path, str) and "://" in path
    last = None
    for attempt in range(tries):
        try:
            return uproot.open(path)
        except Exception as exc:  # noqa: BLE001 - uproot wraps many transport errors
            last = exc
            if not remote or attempt == tries - 1 or not _is_retryable(exc):
                raise
            wait = backoff * (2 ** attempt)
            print(f"[io] open failed ({exc.__class__.__name__}: {exc}); "
                  f"retry {attempt + 1}/{tries - 1} in {wait:.0f}s", flush=True)
            _sleep(wait)
    raise last  # pragma: no cover - loop either returns or raises


def resolve_paths(path: PathLike) -> list[str]:
    """Expand ``path`` (file / glob / directory / list) into a sorted file list."""
    if isinstance(path, (list, tuple)):
        files = [f for p in path for f in resolve_paths(p)]
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.root"), recursive=True))
    elif any(ch in str(path) for ch in "*?["):
        # a glob may match sample directories (e.g. ".../delphes/*kl-1p00*") or
        # ROOT files directly; expand matched directories to their *.root.
        files = []
        for m in sorted(glob.glob(str(path), recursive=True)):
            if os.path.isdir(m):
                files += sorted(glob.glob(os.path.join(m, "**", "*.root"), recursive=True))
            elif m.endswith(".root"):
                files.append(m)
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
            f = open_with_retry(p)
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
        # branches present in every opened file (dotted sub-branches like "Jet.PT").
        # uproot lists them as "Jet/Jet.PT" (sub-branch path); take the trailing
        # dotted leaf so callers can query the natural name they read directly.
        self._keys = set.intersection(
            *[{k.split("/")[-1] for k in t.keys()} for t in self._trees]
        )
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
        """On-disk size per event over the files actually read (storage projection).

        NaN for remote inputs: ``os.path.getsize`` is POSIX-only, and the pilot gate
        compares this against ``kb_per_event_max`` at GATE severity — so a root:// run
        would fail on the size check rather than on anything physical.
        """
        if any(str(p).startswith("root://") for p in self._used):
            return float("nan")
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
