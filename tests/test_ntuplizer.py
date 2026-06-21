"""Ntuplizer round-trip: Delphes ROOT -> parquet -> ``load_ntuple``.

Asserts the flat ntuple carries the full schema (the five object collections of
``schema.FLAT_SCHEMA`` with their fields, plus the per-event scalars of
``schema.SCALARS``) and that the spec's count consistencies hold:

- one record per event (all collections and scalars share ``ev.n``),
- ``Tau`` is exactly the ``Jet.TauTag == 1`` subset of ``Jet``,
- the ``Jet`` / ``Electron`` / ``Muon`` / ``GenPart`` flat counts equal the raw
  Delphes collection counts.
"""

from __future__ import annotations

from pathlib import Path

import awkward as ak

from delphes_pipeline.core.io import DelphesEvents, load_ntuple
from delphes_pipeline.ntuplizer import schema
from delphes_pipeline.ntuplizer.convert import convert


def _flat_count(coll) -> int:
    return int(ak.sum(ak.num(coll.pt)))


def test_ntuplizer_schema_and_counts(good_fixture_path, tmp_path):
    out = str(tmp_path / "ntuple.parquet")
    convert(good_fixture_path, out)
    nt = load_ntuple(out)
    ev = DelphesEvents(good_fixture_path)

    # --- schema coverage: every collection + its fields ---------------------
    for coll, fields in schema.FLAT_SCHEMA.items():
        assert coll in nt.fields, f"missing collection {coll}"
        for f in fields:
            assert f in nt[coll].fields, f"{coll} missing field {f}"

    # --- schema coverage: per-event scalar fields ---------------------------
    for sc in schema.SCALARS:
        assert sc in nt.fields, f"missing scalar {sc}"

    # --- one record per event -----------------------------------------------
    assert len(nt) == ev.n
    for sc in schema.SCALARS:
        assert len(nt[sc]) == ev.n

    # --- Tau is the TauTag==1 subset of Jet ---------------------------------
    n_tautag = int(ak.sum(ak.flatten(ev.jets.tautag) == 1))
    assert _flat_count(nt.Tau) == n_tautag

    # --- collection counts match the raw Delphes collections ----------------
    assert _flat_count(nt.Jet) == _flat_count(ev.jets)
    assert _flat_count(nt.Electron) == _flat_count(ev.electrons)
    assert _flat_count(nt.Muon) == _flat_count(ev.muons)
    assert _flat_count(nt.GenPart) == _flat_count(ev.gen)


def test_convert_writes_parquet(good_fixture_path, tmp_path):
    out = tmp_path / "ntuple.parquet"
    arr = convert(good_fixture_path, str(out))
    assert out.exists()
    # convert returns the same array it persisted
    assert len(arr) == DelphesEvents(good_fixture_path).n
