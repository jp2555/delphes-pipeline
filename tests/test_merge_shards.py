"""Merging must be complete, faithful, and refuse to paper over a short campaign.

A campaign missing 3 of 1244 tt̄ shards looks exactly like a finished one on disk. The
merged sample is then quietly 0.2% short — invisible in any shape comparison, and wrong
in every normalisation. That is what the ``shard`` column and this gate exist for.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_shards  # noqa: E402
import merge_shards  # noqa: E402


def _campaign(tmp, samples=(("sig", 3), ("ttbar", 4)), rows=50, drop=(), dup=None):
    """Plan a campaign and write its shard outputs."""
    out = tmp / "out"
    plan_dir = out / "_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for name, n in samples:
        for i in range(n):
            shards.append({"sample": name, "shard": i, "seed": len(shards) + 1,
                           "maps": f"/maps/{name}.json", "subtree": "delphes-tree-1",
                           "files": [f"/in/{name}.{i}.root"],
                           "out": str(out / f"{name}.{i:04d}.parquet")})
    (plan_dir / "manifest.json").write_text(json.dumps({"shards": shards}))
    for e in shards:
        if (e["sample"], e["shard"]) in drop:
            continue
        sid = dup if dup is not None else e["shard"]
        a = ak.zip({"MET_pt": np.full(rows, 10.0 + e["shard"], dtype=np.float32),
                    "genWeight": np.ones(rows, dtype=np.float32),
                    "shard": np.full(rows, sid, dtype=np.int32)})
        ak.to_parquet(a, e["out"])
    return out


def test_merge_preserves_every_event():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td), rows=50)
        assert merge_shards.main(["--out", str(out), "--target-gb", "1e-9"]) == 0
        m = json.load(open(out / "merged" / "manifest.json"))
        assert m["sig"]["events"] == 150 and m["ttbar"]["events"] == 200


def test_merged_content_matches_the_shards():
    """Not just the count — the values must survive, in shard order."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td), samples=(("sig", 3),), rows=10)
        merge_shards.main(["--out", str(out), "--target-gb", "100"])
        merged = ak.from_parquet(str(out / "merged" / "sig.0000.parquet"))
        assert len(merged) == 30
        got = sorted(set(ak.to_numpy(merged["MET_pt"]).tolist()))
        assert got == [10.0, 11.0, 12.0]
        assert sorted(set(ak.to_numpy(merged["shard"]).tolist())) == [0, 1, 2]


def test_a_missing_shard_blocks_the_merge():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td), drop={("ttbar", 2)})
        assert merge_shards.main(["--out", str(out), "--target-gb", "1e-9"]) == 1
        assert not (out / "merged" / "ttbar.0000.parquet").exists()


def test_a_duplicated_shard_id_blocks_the_merge():
    """Two files stamped the same shard means the merge double-counts those events."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td), dup=0)
        assert merge_shards.main(["--out", str(out), "--target-gb", "1e-9"]) == 1


def test_force_overrides_but_the_manifest_records_the_shortfall():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td), samples=(("sig", 3),), drop={("sig", 1)}, rows=10)
        assert merge_shards.main(["--out", str(out), "--target-gb", "100", "--force"]) == 0
        m = json.load(open(out / "merged" / "manifest.json"))
        assert m["sig"]["shards"] == 2 and m["sig"]["events"] == 20


def test_samples_are_kept_separate():
    """Signal and ttbar carry DIFFERENT tuning maps; merging them into one file would
    lose which corrections each event received."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td))
        merge_shards.main(["--out", str(out), "--target-gb", "100"])
        m = json.load(open(out / "merged" / "manifest.json"))
        assert m["sig"]["maps"] == ["/maps/sig.json"]
        assert m["ttbar"]["maps"] == ["/maps/ttbar.json"]
        assert set(os.path.basename(f).split(".")[0] for f in m["sig"]["files"]) == {"sig"}


def test_target_size_splits_into_several_files():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td), samples=(("sig", 6),), rows=200)
        merge_shards.main(["--out", str(out), "--target-gb", "1e-9"])   # split every shard
        m = json.load(open(out / "merged" / "manifest.json"))
        assert len(m["sig"]["files"]) > 1
        assert sum(len(ak.from_parquet(f)) for f in m["sig"]["files"]) == 1200


def test_schema_mismatch_is_refused_not_coerced():
    """Shards written by different code must not be silently mixed — coercing would
    drop or reorder columns without a word."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td), samples=(("sig", 3),), rows=10)
        odd = ak.zip({"MET_pt": np.zeros(10, dtype=np.float32)})       # missing fields
        ak.to_parquet(odd, str(out / "sig.0002.parquet"))
        with pytest.raises(SystemExit):
            merge_shards.main(["--out", str(out), "--target-gb", "100"])


# --------------------------------------------------------------------------- #
# The merge is codec-bound — every shard is decompressed and recompressed with
# zstd at a few hundred MB/s on one core — so output files are written in
# parallel. That requires the group boundaries to be fixed BEFORE writing.
# --------------------------------------------------------------------------- #
def test_parallel_and_serial_merges_agree_exactly():
    """Parallelism must not change a single event, or the ntuple depends on --jobs."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        out = _campaign(tmp / "a", samples=(("sig", 6),), rows=100)
        merge_shards.main(["--out", str(out), "--target-gb", "1e-9", "--jobs", "1"])
        serial = [ak.from_parquet(f) for f in
                  json.load(open(out / "merged" / "manifest.json"))["sig"]["files"]]

        out2 = _campaign(tmp / "b", samples=(("sig", 6),), rows=100)
        merge_shards.main(["--out", str(out2), "--target-gb", "1e-9", "--jobs", "4"])
        par = [ak.from_parquet(f) for f in
               json.load(open(out2 / "merged" / "manifest.json"))["sig"]["files"]]

    assert len(serial) == len(par)
    cat = lambda xs: ak.to_numpy(ak.concatenate([x["MET_pt"] for x in xs])).tolist()
    assert cat(serial) == cat(par)


def test_grouping_is_decided_before_writing():
    """Boundaries come from the shards' own sizes; deciding them by watching the output
    grow would make the groups un-parallelisable."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td), samples=(("sig", 6),), rows=100)
        plan = json.load(open(out / "_plan" / "manifest.json"))["shards"]
        groups = merge_shards._group(plan, 1)          # 1 byte -> one group per shard
        assert len(groups) == 6 and all(len(g) == 1 for g in groups)
        big = merge_shards._group(plan, 1e12)          # huge -> a single group
        assert len(big) == 1 and len(big[0]) == 6


def test_schema_mismatch_aborts_even_in_a_worker():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = _campaign(Path(td), samples=(("sig", 4),), rows=10)
        ak.to_parquet(ak.zip({"MET_pt": np.zeros(10, dtype=np.float32)}),
                      str(out / "sig.0003.parquet"))
        with pytest.raises(SystemExit):
            merge_shards.main(["--out", str(out), "--target-gb", "100", "--jobs", "4"])


# --------------------------------------------------------------------------- #
# A gap in one sample must not hold a finished sample hostage -- but it must
# still block its own merge, or the campaign silently loses events.
# --------------------------------------------------------------------------- #
def test_a_finished_sample_merges_while_another_is_incomplete(tmp_path):
    """This is the production case: ttbar lost 4 shards to dCache, signal is done."""
    out = _campaign(tmp_path, drop={("ttbar", 2)})
    assert merge_shards.main(["--out", str(out), "--sample", "sig",
                              "--target-gb", "1e-9"]) == 0
    names = [f.name for f in (out / "merged").iterdir()]
    assert any(n.startswith("sig.") for n in names)
    assert not any(n.startswith("ttbar.") for n in names)


def test_an_incomplete_sample_still_refuses_to_merge(tmp_path, capsys):
    out = _campaign(tmp_path, drop={("ttbar", 2)})
    assert merge_shards.main(["--out", str(out), "--sample", "ttbar",
                              "--target-gb", "1e-9"]) == 1
    assert "incomplete" in capsys.readouterr().out


def test_a_duplicate_in_the_selected_sample_still_blocks(tmp_path):
    """--sample must not become a way to skip the double-counting check."""
    out = _campaign(tmp_path, dup=0)
    assert merge_shards.main(["--out", str(out), "--sample", "sig",
                              "--target-gb", "1e-9"]) == 1


def test_merging_the_second_sample_keeps_the_first_in_the_manifest(tmp_path):
    """The ttbar merge must not erase sig's provenance from manifest.json."""
    out = _campaign(tmp_path)
    merge_shards.main(["--out", str(out), "--sample", "sig", "--target-gb", "1e-9"])
    merge_shards.main(["--out", str(out), "--sample", "ttbar", "--target-gb", "1e-9"])
    man = json.load(open(out / "merged" / "manifest.json"))
    assert set(man) == {"sig", "ttbar"}
    assert man["sig"]["shards"] == 3 and man["ttbar"]["events"] == 200


def test_an_unknown_sample_name_is_rejected(tmp_path):
    out = _campaign(tmp_path)
    with pytest.raises(SystemExit, match="no such sample"):
        merge_shards.main(["--out", str(out), "--sample", "signl"])


def test_the_audit_is_scoped_to_the_selected_samples(tmp_path, capsys):
    """Auditing 1244 ttbar shards to merge 150 signal ones is minutes of wasted I/O."""
    out = _campaign(tmp_path, drop={("ttbar", 2)})
    assert merge_shards.main(["--out", str(out), "--sample", "sig",
                              "--target-gb", "1e-9"]) == 0
    said = capsys.readouterr().out
    assert "3/3 shards present" in said, "the audit must cover only sig"
    assert "ttbar" not in said.split("[merge] audit")[0]


# --------------------------------------------------------------------------- #
# kappa_lambda lives only in the input path -- the ntuple schema has no field for
# it. A merged signal sample without it is unusable for NSBI, which is entirely
# about ratios between kl hypotheses.
# --------------------------------------------------------------------------- #
def _kl_campaign(tmp, per_kl=2, rows=10):
    out = tmp / "out"
    (out / "_plan").mkdir(parents=True)
    shards, i = [], 0
    for kl in ("0p00", "1p00", "m2p50"):
        for _ in range(per_kl):
            shards.append({"sample": "signal", "shard": i, "seed": i, "maps": "/m.json",
                           "files": [f"/in/GluGluToHH_kl-{kl}_x{j}.root" for j in (0, 1)],
                           "out": str(out / f"signal.{i:04d}.parquet")})
            i += 1
    (out / "_plan" / "manifest.json").write_text(json.dumps({"shards": shards}))
    for e in shards:
        ak.to_parquet(ak.zip({"MET_pt": np.ones(rows, dtype=np.float32),
                              "shard": np.full(rows, e["shard"], dtype=np.int32)}),
                      e["out"])
    return out


def test_the_generated_kl_is_recovered_onto_every_event(tmp_path):
    out = _kl_campaign(tmp_path)
    assert merge_shards.main(["--out", str(out), "--target-gb", "1e-9"]) == 0
    t = ak.from_parquet(str(out / "merged" / "signal.*.parquet"))
    by_shard = {int(s): float(k) for s, k in zip(t.shard, t.kl)}
    assert by_shard == {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: -2.5, 5: -2.5}


def test_a_negative_kl_keeps_its_sign():
    assert merge_shards._kl_of({"files": ["/in/kl-m2p50/x.root"]}) == -2.5
    assert merge_shards._kl_of({"files": ["/in/kl-5p00/x.root"]}) == 5.0


def test_a_sample_with_no_kl_in_its_paths_gets_no_column(tmp_path):
    """ttbar has no kl; it must not acquire a meaningless one."""
    out = _campaign(tmp_path, samples=(("ttbar", 2),))
    assert merge_shards.main(["--out", str(out), "--target-gb", "1e-9"]) == 0
    t = ak.from_parquet(str(out / "merged" / "ttbar.0000.parquet"))
    assert "kl" not in ak.fields(t)


def test_a_shard_spanning_two_kl_points_aborts_rather_than_guessing():
    with pytest.raises(ValueError, match="spans multiple"):
        merge_shards._kl_of({"sample": "signal", "shard": 3,
                             "files": ["/in/kl-1p00/a.root", "/in/kl-5p00/b.root"]})


# --------------------------------------------------------------------------- #
# Provenance uniformity -- not tuning fidelity -- is what the downstream unbinned
# CI depends on. Samples corrected with different map sets are different forward
# models; a ratio trained across them inherits the map difference as S/B shape.
# That can be intended, but it must never be silent.
# --------------------------------------------------------------------------- #
def _campaign_with_maps(tmp, per_sample_maps, rows=20):
    out = tmp / "out"
    (out / "_plan").mkdir(parents=True)
    shards = []
    for i, (name, mp) in enumerate(per_sample_maps.items()):
        for k in range(2):
            shards.append({"sample": name, "shard": k, "seed": 10 * i + k,
                           "maps": f"/maps/{name}.json", "maps_sha": mp,
                           "files": [f"/in/{name}.{k}.root"],
                           "out": str(out / f"{name}.{k:04d}.parquet")})
    (out / "_plan" / "manifest.json").write_text(json.dumps({"shards": shards}))
    for e in shards:
        ak.to_parquet(ak.zip({"MET_pt": np.ones(rows, dtype=np.float32),
                              "shard": np.full(rows, e["shard"], dtype=np.int32)}),
                      e["out"])
    return out


def test_mixed_map_provenance_is_announced(tmp_path, capsys):
    out = _campaign_with_maps(tmp_path, {"signal": "aaaa1111", "ttbar": "bbbb2222"})
    assert merge_shards.main(["--out", str(out), "--target-gb", "1e-9"]) == 0
    said = capsys.readouterr().out
    assert "DIFFERENT map sets" in said
    assert "signal=aaaa1111" in said and "ttbar=bbbb2222" in said


def test_uniform_map_provenance_says_nothing(tmp_path, capsys):
    out = _campaign_with_maps(tmp_path, {"signal": "aaaa1111", "ttbar": "aaaa1111"})
    merge_shards.main(["--out", str(out), "--target-gb", "1e-9"])
    assert "DIFFERENT map sets" not in capsys.readouterr().out


def test_the_merged_manifest_records_the_map_fingerprint(tmp_path):
    out = _campaign_with_maps(tmp_path, {"signal": "aaaa1111"})
    merge_shards.main(["--out", str(out), "--target-gb", "1e-9"])
    man = json.load(open(out / "merged" / "manifest.json"))
    assert man["signal"]["maps_sha"] == ["aaaa1111"], "a path is not a version"


# --------------------------------------------------------------------------- #
# Q8, codified: sigma_eff = sigma_gen * (sum w on PROCESSED shards / sum w
# generated over the SAME shard set), negative genWeights in BOTH sums. Recording
# the sum and the shard accounting is what stops anyone quoting a generated count
# the tuned sample does not correspond to -- the 4 lost ttbar shards are that trap.
# --------------------------------------------------------------------------- #
def _weighted_campaign(tmp, n=4, rows=10, drop=()):
    out = tmp / "out"
    (out / "_plan").mkdir(parents=True)
    shards = [{"sample": "ttbar", "shard": i, "seed": i, "maps": "/m.json",
               "maps_sha": "abc", "files": [f"/in/{i}.root"],
               "out": str(out / f"ttbar.{i:04d}.parquet")} for i in range(n)]
    (out / "_plan" / "manifest.json").write_text(json.dumps({"shards": shards}))
    for e in shards:
        if e["shard"] in drop:
            continue
        w = np.where(np.arange(rows) % 5 == 0, -1.0, 1.0).astype(np.float32)
        ak.to_parquet(ak.zip({"genWeight": w,
                              "lepton_sf": np.full(rows, 1.1, dtype=np.float32),
                              "shard": np.full(rows, e["shard"], dtype=np.int32)}),
                      e["out"])
    return out


def test_the_manifest_records_signed_sum_of_weights(tmp_path):
    out = _weighted_campaign(tmp_path, n=4, rows=10)
    merge_shards.main(["--out", str(out), "--target-gb", "1e-9"])
    man = json.load(open(out / "merged" / "manifest.json"))["ttbar"]
    # 8 positive + 2 negative per shard, 4 shards
    assert man["sum_genweight"] == pytest.approx(4 * (8 - 2), rel=1e-6)
    assert man["events"] == 40


def test_lost_shards_leave_processed_and_planned_both_visible(tmp_path):
    out = _weighted_campaign(tmp_path, n=4, rows=10, drop=(2,))
    merge_shards.main(["--out", str(out), "--target-gb", "1e-9", "--force"])
    man = json.load(open(out / "merged" / "manifest.json"))["ttbar"]
    assert man["shards"] == 3 and man["shards_planned"] == 4
    assert man["sum_genweight"] == pytest.approx(3 * (8 - 2), rel=1e-6), \
        "the denominator must be the PROCESSED shard set, not the planned one"


def test_the_lepton_sf_weighted_sum_is_recorded_too(tmp_path):
    """lepton_sf is part of the event weight; the ladder needs it consistently."""
    out = _weighted_campaign(tmp_path, n=2, rows=10)
    merge_shards.main(["--out", str(out), "--target-gb", "1e-9"])
    man = json.load(open(out / "merged" / "manifest.json"))["ttbar"]
    assert man["sum_genweight_x_lepton_sf"] == pytest.approx(
        man["sum_genweight"] * 1.1, rel=1e-5)


# --------------------------------------------------------------------------- #
# One --sample can span several CMS primary datasets: ttbar globs TTto2L2Nu +
# TTto4Q + TTtoLNu2Q (98.0 / 419.7 / 405.7 pb) and DY is jet- and mass-binned.
# Without a per-event label the merged sample cannot be normalised at all.
# --------------------------------------------------------------------------- #
def _multi_dataset_campaign(tmp, rows=10):
    out = tmp / "out"
    (out / "_plan").mkdir(parents=True)
    chans = ["TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
             "TTto4Q_TuneCP5_13p6TeV_powheg-pythia8",
             "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8"]
    shards = [{"sample": "ttbar", "shard": i, "seed": i, "maps": "none",
               "maps_sha": "untuned",
               "files": [f"root://h//store/{c}_Delphes_v1/delphes-tree-2ff38f65/f.root"],
               "out": str(out / f"ttbar.{i:04d}.parquet")}
              for i, c in enumerate(chans)]
    (out / "_plan" / "manifest.json").write_text(json.dumps({"shards": shards}))
    for e in shards:
        ak.to_parquet(ak.zip({"genWeight": np.ones(rows, dtype=np.float32),
                              "shard": np.full(rows, e["shard"], dtype=np.int32)}),
                      e["out"])
    return out, chans


def test_each_ttbar_channel_gets_its_own_dataset_id(tmp_path):
    out, chans = _multi_dataset_campaign(tmp_path)
    assert merge_shards.main(["--out", str(out), "--target-gb", "1e-9"]) == 0
    t = ak.from_parquet(str(out / "merged" / "ttbar.*.parquet"))
    by_shard = {int(s): int(d) for s, d in zip(t.shard, t.dataset_id)}
    assert len(set(by_shard.values())) == 3, "three channels must be distinguishable"
    man = json.load(open(out / "merged" / "manifest.json"))["ttbar"]
    assert sorted(man["datasets"].values()) == sorted(chans)


def test_the_manifest_maps_every_id_back_to_its_primary_dataset(tmp_path):
    out, _ = _multi_dataset_campaign(tmp_path)
    merge_shards.main(["--out", str(out), "--target-gb", "1e-9"])
    man = json.load(open(out / "merged" / "manifest.json"))["ttbar"]
    t = ak.from_parquet(str(out / "merged" / "ttbar.*.parquet"))
    ids = {str(int(i)) for i in t.dataset_id}
    assert ids <= set(man["datasets"]), "every id in the data must resolve to a name"


def test_a_shard_spanning_two_datasets_aborts(tmp_path):
    """Its events could not be labelled, and the two have different cross sections."""
    with pytest.raises(ValueError, match="different cross sections"):
        merge_shards.write_group((
            "ttbar", 0,
            [{"sample": "ttbar", "shard": 0, "out": "x",
              "files": ["/s/TTto4Q_TuneCP5_x_Delphes_v1/a.root",
                        "/s/TTto2L2Nu_TuneCP5_x_Delphes_v1/b.root"]}],
            str(tmp_path), {}, False))


def test_a_boundary_straddling_shard_is_labelled_not_guessed(tmp_path):
    """Shards are cut on accumulated BYTES across the whole file list, so one straddles
    each dataset boundary. Attributing it to whichever contributed more files would be a
    fabricated cross section; it is marked -1 and counted instead."""
    out = tmp_path / "out"
    (out / "_plan").mkdir(parents=True)
    base = "root://h//store/{}_TuneCP5_13p6TeV_powheg-pythia8_Delphes_v1/t/f.root"
    shards = [
        {"sample": "ttbar", "shard": 0, "seed": 0, "maps": "none", "maps_sha": "untuned",
         "files": [base.format("TTto2L2Nu")], "out": str(out / "ttbar.0000.parquet")},
        {"sample": "ttbar", "shard": 1, "seed": 1, "maps": "none", "maps_sha": "untuned",
         "files": [base.format("TTto2L2Nu"), base.format("TTto4Q")],   # straddles
         "out": str(out / "ttbar.0001.parquet")},
    ]
    (out / "_plan" / "manifest.json").write_text(json.dumps({"shards": shards}))
    for e in shards:
        ak.to_parquet(ak.zip({"genWeight": np.ones(10, dtype=np.float32),
                              "shard": np.full(10, e["shard"], dtype=np.int32)}), e["out"])

    with pytest.raises(ValueError, match="straddles"):
        merge_shards.write_group(("ttbar", 0, [shards[1]], str(out), {}, False))

    assert merge_shards.main(["--out", str(out), "--target-gb", "1e-9",
                              "--allow-mixed-datasets"]) == 0
    t = ak.from_parquet(str(out / "merged" / "ttbar.*.parquet"))
    ids = sorted(set(int(i) for i in t.dataset_id))
    assert merge_shards.MIXED_DATASET in ids, "the straddling shard must be marked"
    man = json.load(open(out / "merged" / "manifest.json"))["ttbar"]
    assert man["mixed_dataset_events"] == 10
