"""Delphes ROOT -> flat NanoAOD-compatible parquet ntuple (note §8.1).

``to_record`` zips the jagged object collections (Jet, Tau, Electron, Muon,
GenPart) and the per-event scalars into one record per event, keeping the
collections jagged and the scalars flat (``depth_limit=1``). ``convert`` opens
the Delphes file, builds the record, writes it to parquet, and returns the
array. The downstream reader is ``core.io.load_ntuple``.
"""

from __future__ import annotations

import sys
from typing import Optional

import awkward as ak
import numpy as np

from ..core.io import DelphesEvents
from . import objects


def to_record(ev: DelphesEvents, tuning_maps=None, seed: int = 0) -> ak.Array:
    """Combine collections and scalars into one record per event.

    When ``tuning_maps`` (a ``tuning.maps.TuningMaps``) is given, the b-tag is
    re-derived downstream from ``Jet.Flavor`` + the anchor efficiency map
    (stochastic re-tag) instead of copying the stock ``Jet.BTag``.
    """
    btag_override = None
    if tuning_maps is not None:
        from ..tuning.maps import retag_btag
        btag_override = retag_btag(ev, tuning_maps, np.random.default_rng(seed))
    fields = {
        "Jet": objects.build_jets(ev, btag_override=btag_override),
        "Tau": objects.build_taus(ev),
        "Electron": objects.build_electrons(ev),
        "Muon": objects.build_muons(ev),
        "GenPart": objects.build_genpart(ev),
    }
    fields.update(objects.scalars(ev))
    return ak.zip(fields, depth_limit=1)


def convert(
    delphes_path: str,
    out_path: str,
    treename: str = "Delphes",
    entry_stop: Optional[int] = None,
    tuning_maps=None,
    seed: int = 0,
) -> ak.Array:
    """Convert a Delphes ROOT file to a flat parquet ntuple; return the array.

    ``tuning_maps`` may be a path to a maps JSON or a ``TuningMaps``; when set,
    the b-tag is re-tagged downstream from the anchor maps.
    """
    if isinstance(tuning_maps, str):
        from ..tuning.maps import TuningMaps
        tuning_maps = TuningMaps.load(tuning_maps)
    ev = DelphesEvents(delphes_path, treename=treename, entry_stop=entry_stop)
    out = to_record(ev, tuning_maps=tuning_maps, seed=seed)
    ak.to_parquet(out, out_path)
    return out


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
