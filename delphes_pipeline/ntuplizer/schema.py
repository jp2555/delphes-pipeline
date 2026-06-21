"""NanoAOD-compatible flat schema for the Delphes ntuple (note §8.1, Table 1).

The ntuplizer mirrors a subset of NanoAODv15 branch *names* and types so the
existing NSBI analysis framework runs unchanged on either CMS NanoAOD or
Delphes output. This module is the **contract**: ``convert.py`` builds exactly
these collections/fields, and downstream (Levels 1-4) reads them by these names.

``FLAT_SCHEMA`` maps each collection to its per-object fields and numpy dtypes.
``SCALARS`` lists per-event scalar fields. ``DELPHES_SOURCE`` documents, for
each flat field, which Delphes collection/leaf it is built from (and any
derivation), so the mapping is auditable.

Tau note: Delphes has no Tau collection; hadronic-tau candidates are jets with
the ``TauTag`` bit. The ``Tau`` collection below is therefore *derived* from
``Jet`` (TauTag-passing jets) in ``convert.py``.
"""

from __future__ import annotations

import numpy as np

FLAT_SCHEMA: dict[str, dict[str, type]] = {
    "Jet": {
        "pt": np.float32,
        "eta": np.float32,
        "phi": np.float32,
        "mass": np.float32,
        "btag": np.int32,           # Delphes Jet.BTag bit
        "tautag": np.int32,         # Delphes Jet.TauTag bit
        "hadronFlavour": np.int32,  # Delphes Jet.Flavor
    },
    "Tau": {  # derived: jets passing TauTag
        "pt": np.float32,
        "eta": np.float32,
        "phi": np.float32,
        "mass": np.float32,
    },
    "Electron": {
        "pt": np.float32,
        "eta": np.float32,
        "phi": np.float32,
        "charge": np.int32,
    },
    "Muon": {
        "pt": np.float32,
        "eta": np.float32,
        "phi": np.float32,
        "charge": np.int32,
    },
    "GenPart": {
        "pt": np.float32,
        "eta": np.float32,
        "phi": np.float32,
        "mass": np.float32,
        "pdgId": np.int32,
        "status": np.int32,
        "genPartIdxMother": np.int32,  # from Delphes Particle.M1
    },
}

SCALARS: dict[str, type] = {
    "MET_pt": np.float32,
    "MET_phi": np.float32,
    "GenMET_pt": np.float32,
    "GenMET_phi": np.float32,
    "HT": np.float32,
    "genWeight": np.float32,
}

# Audit trail: flat name -> "DelphesCollection.Leaf [derivation]".
DELPHES_SOURCE: dict[str, str] = {
    "Jet": "Jet.{PT,Eta,Phi,Mass}; btag=Jet.BTag; tautag=Jet.TauTag; hadronFlavour=Jet.Flavor",
    "Tau": "Jet with Jet.TauTag==1; kinematics from Jet.{PT,Eta,Phi,Mass}",
    "Electron": "Electron.{PT,Eta,Phi,Charge}",
    "Muon": "Muon.{PT,Eta,Phi,Charge}",
    "GenPart": "Particle.{PT,Eta,Phi,Mass,PID,Status}; genPartIdxMother=Particle.M1",
    "MET_pt": "MissingET.MET",
    "MET_phi": "MissingET.Phi",
    "GenMET_pt": "GenMissingET.MET",
    "GenMET_phi": "GenMissingET.Phi",
    "HT": "ScalarHT.HT",
    "genWeight": "Event.Weight[0]",
}


def collections() -> list[str]:
    return list(FLAT_SCHEMA.keys())


def flat_field_names() -> list[str]:
    """NanoAOD-style ``Collection_field`` + scalar names (reference list).

    The ntuplizer writes *nested* parquet — top-level columns are the collection
    names (``Jet``, ``Tau``, …) plus the scalars, accessed as ``ntuple.Jet.pt``.
    This helper returns the equivalent flat NanoAOD names for documentation and
    for code that needs the per-field inventory.
    """
    names: list[str] = []
    for coll, fields in FLAT_SCHEMA.items():
        names.extend(f"{coll}_{f}" for f in fields)
    names.extend(SCALARS.keys())
    return names
