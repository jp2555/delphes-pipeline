"""Ntuplizer gate check: convert a slice and validate schema + counts.

Without this, the ntuplizer is never exercised by the gate and could be broken
while the run passes green. It converts the first ``max_events`` events to the
flat record and checks (a) every ``schema.FLAT_SCHEMA`` collection + field and
``schema.SCALARS`` scalar is present, and (b) the object counts match the raw
Delphes collections. GATE severity.
"""

from __future__ import annotations

import awkward as ak

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.core.result import CheckResult, Severity, info
from delphes_pipeline.ntuplizer import convert, schema


def run(ctx: ValidationContext) -> list[CheckResult]:
    """Convert a slice and validate the flat schema + counts."""
    n = int(ctx.opt("ntuplizer", "max_events", 2000))
    ev = DelphesEvents(ctx.events.path, treename=ctx.events.treename, entry_stop=n)
    # apply the downstream tuning re-tag if a maps file is configured, so the gate
    # validates the tuned ntuple (note D2-A); schema/counts are unchanged by it.
    maps_path = ctx.config.get("tuning_maps")
    tuning_maps = None
    if maps_path:
        from delphes_pipeline.tuning.maps import TuningMaps
        tuning_maps = TuningMaps.load(maps_path)
    rec = convert.to_record(ev, tuning_maps=tuning_maps)

    missing: list[str] = []
    for coll, fields in schema.FLAT_SCHEMA.items():
        if coll not in rec.fields:
            missing.append(coll)
            continue
        missing.extend(f"{coll}.{f}" for f in fields if f not in rec[coll].fields)
    missing.extend(s for s in schema.SCALARS if s not in rec.fields)

    coverage = CheckResult(
        name="ntuplizer.schema_coverage",
        level="ntuplizer",
        passed=not missing,
        severity=Severity.GATE,
        detail="all schema fields present" if not missing else f"missing: {missing[:8]}",
        extra={"missing": missing},
    )

    n_raw = int(ak.sum(ak.num(ev.jets)))
    n_nt = int(ak.sum(ak.num(rec.Jet))) if "Jet" in rec.fields else -1
    counts = CheckResult(
        name="ntuplizer.jet_count",
        level="ntuplizer",
        passed=(n_raw == n_nt),
        severity=Severity.GATE,
        measured=float(n_nt),
        target=float(n_raw),
        detail="ntuple jet count == raw Delphes jet count",
    )

    return [info("ntuplizer.n_events", "ntuplizer", float(len(rec)), detail="events converted"),
            coverage, counts]
