"""Delphes sample metadata — CMS cross sections, OUR processed-event bookkeeping.

The sigma_eff rule: the denominator must correspond to the sample in hand. The ttbar
campaign lost 7 of 1244 shards to a storage fault, so quoting CMS's 470M generated
events against our processed subset would bias the yield by exactly that fraction.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_delphes_datasets as M  # noqa: E402

CMS = {
    "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X": {
        "xsec": 98.036113104, "nevents": 470123263, "generator_weight": 0.9919,
        "era": "2024", "sample_type": "ttbar", "dbs": "/TTto2L2Nu/.../NANOAODSIM"},
    "TTto4Q_TuneCP5_13p6TeV_powheg-pythia8_RunIII2024Summer24NanoAODv15-150X": {
        "xsec": 419.6938241160001, "nevents": 472535695, "generator_weight": 0.9919,
        "era": "2024", "sample_type": "ttbar", "dbs": "/TTto4Q/.../NANOAODSIM"},
    "GluGluHHto2B2Tau_Par-c2-0p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8"
    "_RunIII2024Summer24NanoAODv15-PowhegBugFix": {
        "xsec": 0.00249341001728, "nevents": 957670, "generator_weight": 0.8944,
        "era": "2024", "sample_type": "ggHH", "dbs": "/GluGluHH/.../NANOAODSIM"},
    "GluGluHHto2B2Tau_Par-c2-0p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8"
    "_RunIII2024Summer24NanoAODv15-150X-kit-private": {
        "xsec": None, "nevents": 2500000, "generator_weight": 0.8942,
        "era": "2024", "sample_type": "ggHH", "dbs": "/GluGluHH/.../NANOAODSIM"},
}


def _merged(**over):
    m = {"signal": {"events": 1_000_000, "sum_genweight": 894_400.0, "shards": 150,
                    "shards_planned": 150, "files": ["a.parquet", "b.parquet"],
                    "subtree": ["delphes-tree-61fd1c12"], "maps_sha": ["untuned"],
                    "datasets": {"0": "GluGluHHto2B2Tau_Par-c2-0p00-kl-1p00-kt-1p00"
                                      "_TuneCP5_13p6TeV_powheg-pythia8"}}}
    m.update(over)
    return m


def test_the_cross_section_comes_from_cms():
    out = M.build(_merged(), CMS)
    e = next(iter(out.values()))
    assert e["xsec"] == pytest.approx(0.00249341001728)
    assert "PowhegBugFix" in e["cms_nick"], "prefer the campaign with a real xsec"


def test_nevents_and_generator_weight_are_ours_not_cms():
    """CMS generated 957,670; we processed 1,000,000 of our own Delphes events."""
    e = next(iter(M.build(_merged(), CMS).values()))
    assert e["nevents"] == 1_000_000
    assert e["nevents"] != CMS[e["cms_nick"]]["nevents"]
    assert e["generator_weight"] == pytest.approx(0.8944)


def test_generator_weight_is_the_signed_mean_over_processed_events():
    m = _merged(signal={**_merged()["signal"], "events": 200, "sum_genweight": -50.0})
    e = next(iter(M.build(m, CMS).values()))
    assert e["generator_weight"] == pytest.approx(-0.25), "negative NLO weights kept"


def test_the_shard_shortfall_stays_visible():
    m = _merged(signal={**_merged()["signal"], "shards": 1237, "shards_planned": 1244})
    e = next(iter(M.build(m, CMS).values()))
    assert (e["shards"], e["shards_planned"]) == (1237, 1244)


def test_a_sample_spanning_two_datasets_refuses_to_invent_a_split():
    """ttbar globs three channels at 98.0 / 419.7 / 405.7 pb. Dividing the sample total
    by three would be a fabricated number, so it is left null with an instruction."""
    m = {"ttbar": {"events": 300, "sum_genweight": 297.0, "shards": 3,
                   "shards_planned": 3, "files": ["x.parquet"], "subtree": [],
                   "maps_sha": ["untuned"],
                   "datasets": {"0": "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                                "1": "TTto4Q_TuneCP5_13p6TeV_powheg-pythia8"}}}
    out = M.build(m, CMS)
    assert len(out) == 2
    xsecs = sorted(e["xsec"] for e in out.values())
    assert xsecs == pytest.approx([98.036113104, 419.6938241160001])
    for e in out.values():
        assert e["nevents"] is None and e["generator_weight"] is None
        assert "dataset_id column" in e["note"]
        assert e["dataset_id"] in (0, 1)


def test_a_sample_with_no_dataset_map_is_skipped_loudly(capsys):
    m = {"ttbar": {"events": 10, "sum_genweight": 10.0, "files": []}}
    assert M.build(m, CMS) == {}
    assert "cannot be normalised" in capsys.readouterr().out


def test_an_unknown_dataset_is_reported_rather_than_silently_dropped(capsys):
    m = _merged(signal={**_merged()["signal"], "datasets": {"0": "NotARealSample_TuneCP5"}})
    out = M.build(m, CMS)
    assert next(iter(out.values()))["xsec"] is None
    assert "no CMS entry" in capsys.readouterr().out


def test_delphes_provenance_is_carried():
    e = next(iter(M.build(_merged(), CMS).values()))
    assert e["subtree"] == ["delphes-tree-61fd1c12"]
    assert e["maps_sha"] == ["untuned"]
    assert e["delphes_sample"] == "signal"


def test_the_output_is_json_serialisable_in_the_cms_schema(tmp_path):
    out = M.build(_merged(), CMS)
    p = tmp_path / "d.json"
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    back = json.loads(p.read_text())
    e = next(iter(back.values()))
    for field in ("nick", "dbs", "era", "sample_type", "xsec", "nevents",
                  "generator_weight", "nfiles"):
        assert field in e, f"CMS schema field {field} missing"


# --------------------------------------------------------------------------- #
# Multi-dataset samples: the per-dataset counts ARE recoverable from the merged
# files' dataset_id column, and that is what makes ttbar normalisable.
# --------------------------------------------------------------------------- #
def _write_merged(tmp, rows_by_id):
    import awkward as ak
    import numpy as np
    ids, ws = [], []
    for d, n in rows_by_id.items():
        ids += [d] * n
        ws += [1.0] * (n - 1) + [-1.0]        # one negative weight per dataset
    f = tmp / "ttbar.0000.parquet"
    ak.to_parquet(ak.zip({"dataset_id": np.array(ids, dtype=np.int16),
                          "genWeight": np.array(ws, dtype=np.float32)}), str(f))
    return [str(f)]


def test_per_dataset_counts_are_measured_from_the_merged_files(tmp_path):
    files = _write_merged(tmp_path, {0: 100, 1: 50, 2: 25})
    got = M.scan_per_dataset(files)
    assert {k: v[0] for k, v in got.items()} == {0: 100, 1: 50, 2: 25}
    assert got[0][1] == pytest.approx(98.0), "signed sum keeps the negative weight"


def test_a_multi_dataset_sample_gets_real_nevents_when_scanned(tmp_path):
    files = _write_merged(tmp_path, {0: 100, 1: 50})
    m = {"ttbar": {"events": 150, "sum_genweight": 146.0, "shards": 2,
                   "shards_planned": 2, "files": files, "subtree": [],
                   "maps_sha": ["untuned"],
                   "datasets": {"0": "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                                "1": "TTto4Q_TuneCP5_13p6TeV_powheg-pythia8"}}}
    out = M.build(m, CMS)
    by_x = {e["xsec"]: e for e in out.values()}
    assert by_x[98.036113104]["nevents"] == 100
    assert by_x[419.6938241160001]["nevents"] == 50
    assert by_x[98.036113104]["generator_weight"] == pytest.approx(0.98)
    assert sum(e["nevents"] for e in out.values()) == 150, "must reconstruct the total"


def test_without_the_scan_the_counts_are_null_not_guessed(tmp_path):
    files = _write_merged(tmp_path, {0: 100, 1: 50})
    m = {"ttbar": {"events": 150, "sum_genweight": 146.0, "files": files,
                   "shards": 2, "shards_planned": 2, "subtree": [], "maps_sha": [],
                   "datasets": {"0": "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
                                "1": "TTto4Q_TuneCP5_13p6TeV_powheg-pythia8"}}}
    out = M.build(m, CMS, scan=False)
    assert all(e["nevents"] is None for e in out.values())
