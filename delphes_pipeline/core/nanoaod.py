"""NanoAOD reader, duck-typed to ``DelphesEvents`` for the tuning anchor.

The note's primary tuning anchor (§3, §6.4) is the private CMS NanoAOD: tune the
Delphes object response until it matches the *same* response measured on the
NanoAOD. ``NanoAODEvents`` exposes the same collection interface as
``DelphesEvents`` (``.jets`` with a ``btag`` *bit* obtained by thresholding the
tagger discriminant at the working point, ``.electrons``, ``.muons``, ``.gen``,
``.genjets``, ``.met``, ``.genmet``, ``.scalar_ht``, ``.weights``), so the shared
``core.observables`` extractors measure the anchor target with no new code.
Branch names and working points are configurable (NanoAOD version differs); the
hadronic-τ collections (``.taus``, ``.genvistau``) are NanoAOD-specific and used
by the bespoke τ anchor in ``tuning.anchor``.

All branch names and WP thresholds live in ``BRANCHES`` / the ``wp`` config so a
different NanoAOD era is a config edit, not a code change.
"""

from __future__ import annotations

from functools import cached_property
from typing import Optional

import awkward as ak
import numpy as np

from .io import PathLike, resolve_paths
import uproot

# Default NanoAOD branch names (2024 NanoAODv14/15 conventions). Override via the
# config ``anchor.branches`` block for a different era.
BRANCHES: dict = {
    "treename": "Events",
    "jet": {"pt": "Jet_pt", "eta": "Jet_eta", "phi": "Jet_phi", "mass": "Jet_mass",
            "flavor": "Jet_hadronFlavour", "btag_disc": "Jet_btagUParTAK4B"},
    "tau": {"pt": "Tau_pt", "eta": "Tau_eta", "phi": "Tau_phi", "mass": "Tau_mass",
            "vsjet": "Tau_idDeepTau2018v2p5VSjet", "genflav": "Tau_genPartFlav"},
    "genvistau": {"pt": "GenVisTau_pt", "eta": "GenVisTau_eta", "phi": "GenVisTau_phi", "mass": "GenVisTau_mass"},
    "electron": {"pt": "Electron_pt", "eta": "Electron_eta", "phi": "Electron_phi", "charge": "Electron_charge"},
    "muon": {"pt": "Muon_pt", "eta": "Muon_eta", "phi": "Muon_phi", "charge": "Muon_charge"},
    "genpart": {"pid": "GenPart_pdgId", "status": "GenPart_status", "pt": "GenPart_pt",
                "eta": "GenPart_eta", "phi": "GenPart_phi", "mass": "GenPart_mass",
                "m1": "GenPart_genPartIdxMother"},
    "genjet": {"pt": "GenJet_pt", "eta": "GenJet_eta", "phi": "GenJet_phi", "mass": "GenJet_mass"},
    "met": {"met": "PuppiMET_pt", "phi": "PuppiMET_phi"},
    "genmet": {"met": "GenMET_pt", "phi": "GenMET_phi"},
    "weight": "genWeight",
}


class NanoAODEvents:
    """Duck-typed NanoAOD view exposing the ``DelphesEvents`` collection interface.

    Parameters
    ----------
    path : str | list[str]
        NanoAOD ROOT file, glob, directory, or list (see ``io.resolve_paths``).
    branches : dict
        Branch-name map (defaults to :data:`BRANCHES`); deep-merged with the user's.
    wp : dict
        Working points: ``btag_medium`` (discriminant threshold) and
        ``deeptau_vsjet_medium`` (the DeepTau VSjet bitmask integer for Medium).
    """

    def __init__(self, path: PathLike, *, branches: Optional[dict] = None,
                 wp: Optional[dict] = None, entry_stop: Optional[int] = None):
        self.path = path
        self.b = _deep_merge(BRANCHES, branches or {})
        self.wp = wp or {}
        self.treename = self.b["treename"]
        self.entry_stop = entry_stop
        self.paths = resolve_paths(path)
        self._files, self._trees, self._stops, self._used = [], [], [], []
        remaining = entry_stop
        for p in self.paths:
            if entry_stop is not None and remaining <= 0:
                break
            f = uproot.open(p)
            t = f[self.treename]
            count = t.num_entries
            stop = count if entry_stop is None else min(count, remaining)
            self._files.append(f); self._trees.append(t); self._stops.append(stop); self._used.append(p)
            if entry_stop is not None:
                remaining -= stop
        if not self._trees:
            raise ValueError(f"no readable trees for NanoAOD input {path!r}")
        self._keys = set.intersection(*[{k.split("/")[-1] for k in t.keys()} for t in self._trees])
        self._n = int(sum(self._stops))

    @property
    def n(self) -> int:
        return self._n

    def has_branch(self, name: str) -> bool:
        return name in self._keys

    def array(self, branch: str) -> ak.Array:
        arrs = [t[branch].array(entry_stop=s) for t, s in zip(self._trees, self._stops) if s > 0]
        if not arrs:
            return ak.Array([])
        return arrs[0] if len(arrs) == 1 else ak.concatenate(arrs)

    def _zip(self, fieldmap: dict, *, skip=()) -> ak.Array:
        present = {flat: br for flat, br in fieldmap.items() if flat not in skip and br in self._keys}
        if not present:
            return ak.Array([[] for _ in range(self.n)])
        return ak.zip({flat: self.array(br) for flat, br in present.items()})

    @cached_property
    def jets(self) -> ak.Array:
        """Jets with ``flavor`` = |hadronFlavour| and ``btag`` = (disc ≥ WP) bit."""
        jb = self.b["jet"]
        rec = {"pt": self.array(jb["pt"]), "eta": self.array(jb["eta"]),
               "phi": self.array(jb["phi"]), "mass": self.array(jb["mass"]),
               "flavor": abs(self.array(jb["flavor"]))}
        wp = self.wp.get("btag_medium")
        if wp is None:
            raise ValueError("anchor.wp.btag_medium is required (UParT/PNet Medium discriminant threshold)")
        rec["btag"] = ak.values_astype(self.array(jb["btag_disc"]) >= float(wp), np.int32)
        rec["tautag"] = ak.zeros_like(rec["pt"])  # NanoAOD taus are a separate collection
        return ak.zip(rec)

    @cached_property
    def taus(self) -> ak.Array:
        tb = self.b["tau"]
        return self._zip(tb)

    @cached_property
    def genvistau(self) -> ak.Array:
        return self._zip(self.b["genvistau"])

    @cached_property
    def electrons(self) -> ak.Array:
        return self._zip(self.b["electron"])

    @cached_property
    def muons(self) -> ak.Array:
        return self._zip(self.b["muon"])

    @cached_property
    def gen(self) -> ak.Array:
        return self._zip(self.b["genpart"])

    @cached_property
    def genjets(self) -> ak.Array:
        return self._zip(self.b["genjet"])

    @cached_property
    def met(self) -> ak.Array:
        mb = self.b["met"]
        return ak.zip({"met": self.array(mb["met"]), "phi": self.array(mb["phi"]),
                       "eta": ak.zeros_like(self.array(mb["met"]))})

    @cached_property
    def genmet(self) -> ak.Array:
        gb = self.b["genmet"]
        return ak.zip({"met": self.array(gb["met"]), "phi": self.array(gb["phi"]),
                       "eta": ak.zeros_like(self.array(gb["met"]))})

    @cached_property
    def scalar_ht(self) -> ak.Array:
        """Synthesised sum E_T = Σ jet pT (NanoAOD has no ScalarHT branch)."""
        return ak.zip({"ht": ak.sum(self.jets.pt, axis=1)})

    @cached_property
    def weights(self) -> np.ndarray:
        wname = self.b["weight"]
        if wname in self._keys:
            return ak.to_numpy(self.array(wname))
        return np.ones(self.n)

    def deeptau_medium(self):
        """The DeepTau VSjet Medium WP integer (from config)."""
        wp = self.wp.get("deeptau_vsjet_medium")
        if wp is None:
            raise ValueError("anchor.wp.deeptau_vsjet_medium is required (DeepTau VSjet Medium bitmask int)")
        return int(wp)


def _deep_merge(base: dict, override: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
