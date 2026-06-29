"""Delphes ROOT -> flat NanoAOD-compatible parquet ntuple (note §8.1).

``to_record`` zips the jagged object collections (Jet, Tau, Electron, Muon,
GenPart) and the per-event scalars into one record per event, keeping the
collections jagged and the scalars flat (``depth_limit=1``). ``convert`` opens
the Delphes file, builds the record, writes it to parquet, and returns the
array. The downstream reader is ``core.io.load_ntuple``.
"""

from __future__ import annotations

from typing import Optional

import awkward as ak
import numpy as np

from ..core.io import DelphesEvents
from . import objects


def to_record(ev: DelphesEvents, tuning_maps=None, seed: int = 0) -> ak.Array:
    """Combine collections and scalars into one record per event.

    When ``tuning_maps`` (a ``tuning.maps.TuningMaps``) is given, the jets are wrapped
    in a ``RetaggedEvents`` view so ``Jet.BTag`` (from ``Jet.Flavor``) and ``Jet.TauTag``
    (from the gen record) are re-derived downstream from the anchor maps — and the τ_h
    collection, keyed on the re-tagged ``TauTag``, follows. Seed 0 matches the lens.
    """
    source = ev
    if tuning_maps is not None:
        from ..tuning.maps import RetaggedEvents
        source = RetaggedEvents(ev, tuning_maps, np.random.default_rng(seed))
    fields = {
        "Jet": objects.build_jets(source),
        "Tau": objects.build_taus(source),
        "Electron": objects.build_electrons(source),
        "Muon": objects.build_muons(source),
        "GenPart": objects.build_genpart(source),
    }
    fields.update(objects.scalars(source))
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


def main(argv=None) -> int:
    """CLI: convert a Delphes file, applying the downstream tuning re-tag if configured.

        python -m delphes_pipeline.ntuplizer.convert in.root out.parquet [--config config.yml]
        python -m delphes_pipeline.ntuplizer.convert in.root out.parquet --tuning-maps maps_v0.json

    With ``--config`` the maps path is read from the config's ``tuning_maps`` key, so the
    shipped ntuple carries the same tuned tags the tuning lens re-validates (seed 0 on both).
    """
    import argparse

    ap = argparse.ArgumentParser(description="Delphes ROOT -> flat parquet ntuple")
    ap.add_argument("delphes")
    ap.add_argument("out")
    ap.add_argument("--treename", default="Delphes")
    ap.add_argument("--entry-stop", type=int, default=None)
    ap.add_argument("--config", help="validation config; reads tuning_maps from it")
    ap.add_argument("--tuning-maps", help="apply this maps JSON downstream (overrides --config)")
    args = ap.parse_args(argv)

    tuning_maps = args.tuning_maps
    if tuning_maps is None and args.config:
        from ..validation.run_validation import load_config
        tuning_maps = load_config(args.config).get("tuning_maps")
    print(f"[ntuplizer] reading {args.delphes}"
          + (f" (first {args.entry_stop})" if args.entry_stop else "")
          + (f" with re-tag from {tuning_maps}" if tuning_maps else " (stock tags, no tuning_maps)")
          + " ...", flush=True)
    out = convert(args.delphes, args.out, treename=args.treename,
                  entry_stop=args.entry_stop, tuning_maps=tuning_maps)
    print(f"[ntuplizer] wrote {args.out}: {len(out)} events", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
