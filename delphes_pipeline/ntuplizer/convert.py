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


def to_record(ev: DelphesEvents, tuning_maps=None, seed: int = 0,
              prune_genpart: bool = False) -> ak.Array:
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
        "GenPart": objects.build_genpart(source, prune=prune_genpart),
    }
    fields.update(objects.scalars(source, tuning_maps))
    return ak.zip(fields, depth_limit=1)


def _no_maps_if_none(tuning_maps):
    """Treat "none" (or blank) as the UNTUNED baseline rather than a filename.

    Decided in Python, not in the submit script. The worker receives the maps path as a
    comma-separated HTCondor queue field, which arrives with its leading space intact, so
    a shell guard comparing against the literal "none" does not fire and " none" is
    resolved as a relative path -- which is how 1587 untuned jobs died in 17 seconds each
    on `FileNotFoundError: .../none`. Normalising here cannot be defeated by quoting,
    whitespace, or a stale submit file.
    """
    if isinstance(tuning_maps, str) and tuning_maps.strip().lower() in ("", "none"):
        return None
    if isinstance(tuning_maps, str):
        return tuning_maps.strip()
    return tuning_maps


def convert(
    delphes_path,
    out_path: str,
    treename: str = "Delphes",
    entry_stop: Optional[int] = None,
    tuning_maps=None,
    seed: int = 0,
    prune_genpart: bool = False,
    shard: Optional[int] = None,
) -> ak.Array:
    """Convert a Delphes ROOT file to a flat parquet ntuple; return the array.

    ``tuning_maps`` may be a path to a maps JSON or a ``TuningMaps``; when set,
    the b-tag is re-tagged downstream from the anchor maps.
    """
    tuning_maps = _no_maps_if_none(tuning_maps)
    if isinstance(tuning_maps, str):
        from ..tuning.maps import TuningMaps
        tuning_maps = TuningMaps.load(tuning_maps)
    ev = DelphesEvents(delphes_path, treename=treename, entry_stop=entry_stop)
    out = to_record(ev, tuning_maps=tuning_maps, seed=seed, prune_genpart=prune_genpart)
    if shard is not None:
        # so a duplicated or missing shard is detectable after the merge; without it a
        # 20-file campaign has no way to tell a complete set from an incomplete one
        out = ak.with_field(out, np.full(len(out), shard, dtype=np.int32), "shard")
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
    ap.add_argument("delphes", nargs="?", default=None)
    ap.add_argument("out")
    ap.add_argument("--treename", default="Delphes")
    ap.add_argument("--entry-stop", type=int, default=None)
    ap.add_argument("--config", help="validation config; reads tuning_maps from it")
    ap.add_argument("--tuning-maps", help="apply this maps JSON downstream (overrides --config)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the stochastic re-tag. Shards MUST differ: a shared "
                         "seed replays one uniform stream, which is unbiased per object but "
                         "understates the variance of aggregate yields. Leave at 0 for the "
                         "tuning-lens identity (tests/test_maps.py pins it).")
    ap.add_argument("--shard", type=int, default=None,
                    help="stamp a 'shard' column so a merged set can be audited")
    ap.add_argument("--prune-genpart", action="store_true",
                    help="write only tau/nu_tau/H/Z/W/top and their ancestry; GenPart is "
                         "99.5%% of the ntuple, ~9 TB over the campaign")
    ap.add_argument("--files-from", help="text file of input paths, one per line (a shard)")
    args = ap.parse_args(argv)

    tuning_maps = args.tuning_maps
    if tuning_maps is None and args.config:
        from ..validation.run_validation import load_config
        tuning_maps = load_config(args.config).get("tuning_maps")
    print(f"[ntuplizer] reading {args.delphes}"
          + (f" (first {args.entry_stop})" if args.entry_stop else "")
          + (f" with re-tag from {tuning_maps}" if _no_maps_if_none(tuning_maps)
             else " UNTUNED (stock Delphes tags and energies, no maps)")
          + " ...", flush=True)
    src = args.delphes
    if args.files_from:
        with open(args.files_from) as fh:
            src = [ln.strip() for ln in fh if ln.strip()]
        print(f"[ntuplizer] shard of {len(src)} files from {args.files_from}", flush=True)
    out = convert(src, args.out, treename=args.treename,
                  entry_stop=args.entry_stop, tuning_maps=tuning_maps,
                  seed=args.seed, prune_genpart=args.prune_genpart, shard=args.shard)
    print(f"[ntuplizer] wrote {args.out}: {len(out)} events", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
