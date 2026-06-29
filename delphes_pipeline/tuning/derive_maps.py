"""Entry point: derive the downstream tuning maps from the NanoAOD anchor.

    python -m delphes_pipeline.tuning.derive_maps --config <config.yml> [--output cards/tuning/maps_v0.json]

Measures the per-flavour b-tag efficiency (and τ_h efficiency) on the NanoAOD
anchor and writes them to a maps JSON with provenance. The ntuplizer applies these
maps downstream (``tuning_maps:`` in the config) so the Delphes b-tag matches the
NanoAOD — no card edit, no re-production.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from delphes_pipeline.validation.run_validation import load_config
from . import maps as M


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Derive Delphes tuning maps from the NanoAOD anchor")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="cards/tuning/maps_v0.json")
    ap.add_argument("--max-events", type=int, default=None, help="cap the anchor read")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    if not config.get("anchor", {}).get("enabled"):
        raise SystemExit("anchor.enabled must be true to derive tuning maps")

    print("[maps] measuring the NanoAOD anchor ...", flush=True)
    maps = M.derive_maps(config, max_events=args.max_events)
    ac = config["anchor"]
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "anchor_nanoaod": ac.get("nanoaod_path"),
        "anchor_wp": ac.get("wp"),
        "tuning_set": "v0",
        "apply": "stochastic re-tag from Jet.Flavor (note D2 option A)",
    }
    M.save_maps(maps, args.output, provenance)

    print(f"[maps] wrote {args.output}")
    for q in (*M.BTAG_MAP_QUANTITIES, *M.TAU_MAP_QUANTITIES,
              *M.ESCALE_MAP_QUANTITIES, *M.LEPTON_SF_QUANTITIES):
        if q in maps and maps[q]["values"]:
            v = maps[q]["values"]
            print(f"  {q:18s} {[round(x, 3) for x in v]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
