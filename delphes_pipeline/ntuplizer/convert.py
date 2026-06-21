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

from ..core.io import DelphesEvents
from . import objects


def to_record(ev: DelphesEvents) -> ak.Array:
    """Combine collections and scalars into one record per event."""
    fields = {
        "Jet": objects.build_jets(ev),
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
) -> ak.Array:
    """Convert a Delphes ROOT file to a flat parquet ntuple; return the array."""
    ev = DelphesEvents(delphes_path, treename=treename, entry_stop=entry_stop)
    out = to_record(ev)
    ak.to_parquet(out, out_path)
    return out


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
