"""The shard planner has to produce a plan that is COMPLETE and AUDITABLE.

Sharding is forced by memory (the reader concatenates every file it is handed and caches
the gen record, ~84 GB at 1.5M events), and it must be cut at FILE granularity: there is
no ``entry_start`` in the readers, so two jobs differing only in ``--entry-stop`` would
read overlapping heads rather than disjoint slices. These tests pin that the split covers
every input exactly once, that each shard gets its own seed, and that the verifier
actually catches a missing or duplicated shard — otherwise a silently incomplete merge
looks identical to a complete one.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_shards  # noqa: E402


def _inputs(tmp_path, n=7):
    d = tmp_path / "in"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"f{i:02d}.root").write_bytes(b"x" * 4096)
    return d


def _plan(tmp_path, n=7, shard_events=2):
    src, out = _inputs(tmp_path, n), tmp_path / "out"
    make_shards.main(["--sample", "sig", str(src / "*.root"), str(tmp_path / "m.json"),
                      "--out", str(out), "--shard-gb", "1e-9"])
    return json.load(open(out / "_plan" / "manifest.json"))["shards"], src, out


def test_every_input_file_appears_exactly_once():
    """A file dropped or duplicated is a silently wrong cross-section."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards, src, _ = _plan(Path(td), n=7)
        got = [f for s in shards for f in s["files"]]
        assert sorted(got) == sorted(str(p) for p in sorted(src.glob("*.root")))
        assert len(got) == len(set(got)), "a file was assigned to two shards"


def test_each_shard_gets_its_own_seed_and_none_is_zero():
    """A shared seed replays one uniform stream across shards, understating the variance
    of aggregate yields. Seed 0 is reserved for the tuning-lens identity."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards, _, _ = _plan(Path(td), n=7)
        seeds = [s["seed"] for s in shards]
        assert len(set(seeds)) == len(seeds) and 0 not in seeds


def test_two_samples_keep_their_own_maps():
    """Per-process maps are the whole reason config.ttbar.yml exists; mixing them silently
    applies signal-derived corrections to the background."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a, b = _inputs(tmp / "A", 3), _inputs(tmp / "B", 3)
        out = tmp / "out"
        make_shards.main(["--sample", "sig", str(a / "*.root"), "/maps/sig.json",
                          "--sample", "ttbar", str(b / "*.root"), "/maps/tt.json",
                          "--out", str(out), "--shard-gb", "1e-9"])
        sh = json.load(open(out / "_plan" / "manifest.json"))["shards"]
        for s in sh:
            assert s["maps"].endswith("sig.json" if s["sample"] == "sig" else "tt.json")


def test_plan_emits_a_runnable_submit_and_executable():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _, _, out = _plan(Path(td), n=4)
        sub = (out / "_plan" / "ntuplize.sub").read_text()
        exe = out / "_plan" / "run_shard.sh"
        assert "queue sample, shard, filelist, maps, outfile, seed from" in sub
        assert os.access(exe, os.X_OK)
        assert "--files-from" in exe.read_text() and "--seed" in exe.read_text()


def _write(path, n, shard):
    a = ak.zip({"MET_pt": np.zeros(n, dtype=np.float32),
                "shard": np.full(n, shard, dtype=np.int32)})
    ak.to_parquet(a, str(path))


def test_verify_passes_on_a_complete_campaign(capsys):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards, _, out = _plan(Path(td), n=4, shard_events=2)
        for s in shards:
            _write(s["out"], 5, s["shard"])
        assert make_shards.verify(str(out)) == 0
        assert "complete and unique" in capsys.readouterr().out


def test_verify_catches_a_missing_shard(capsys):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards, _, out = _plan(Path(td), n=4, shard_events=2)
        for s in shards[:-1]:
            _write(s["out"], 5, s["shard"])
        assert make_shards.verify(str(out)) == 1
        assert "MISSING" in capsys.readouterr().out


def test_verify_catches_a_duplicated_shard_id(capsys):
    """Two files carrying the same shard id means the merge double-counts those events."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards, _, out = _plan(Path(td), n=4, shard_events=2)
        for s in shards:
            _write(s["out"], 5, 0)          # every file stamped shard 0
        assert make_shards.verify(str(out)) == 1
        assert "DUPLICATED" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The production carries several `delphes-tree-<hash>` subtrees per dataset holding
# the SAME events (download_gen_delphes.sh SUBTREE_RE). Both resolve_paths and this
# planner expand a directory with **/*.root, so spanning two of them silently
# DOUBLE-COUNTS every event — and nothing downstream would flag it.
# --------------------------------------------------------------------------- #
def _tree(tmp, dataset, h, n=3):
    d = tmp / dataset / f"delphes-tree-{h}"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"f{i}.root").write_bytes(b"x" * 4096)
    return d


def test_spanning_two_subtrees_is_refused():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _tree(tmp, "GluGluHH_Delphes_v1", "2ff38f65")
        _tree(tmp, "GluGluHH_Delphes_v1", "9abc1234")
        files = make_shards._files(str(tmp / "*_Delphes_v1"))
        assert len(files) == 6, "the recursive expansion does pick up both"
        assert make_shards.check_subtrees("sig", files) is False


def test_a_single_subtree_passes():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _tree(tmp, "GluGluHH_Delphes_v1", "2ff38f65")
        files = make_shards._files(str(tmp / "*_Delphes_v1"))
        assert make_shards.check_subtrees("sig", files) is True


def test_subtree_filter_selects_one_of_several():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _tree(tmp, "GluGluHH_Delphes_v1", "2ff38f65")
        _tree(tmp, "GluGluHH_Delphes_v1", "9abc1234")
        files = make_shards._files(str(tmp / "*_Delphes_v1"), subtree="delphes-tree-2ff38f65")
        assert len(files) == 3 and all("2ff38f65" in f for f, _ in files)
        assert make_shards.check_subtrees("sig", files) is True


def test_planning_aborts_rather_than_emitting_a_double_counting_plan():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _tree(tmp, "GluGluHH_Delphes_v1", "2ff38f65")
        _tree(tmp, "GluGluHH_Delphes_v1", "9abc1234")
        with pytest.raises(SystemExit):
            make_shards.main(["--sample", "sig", str(tmp / "*_Delphes_v1"), "/m.json",
                              "--out", str(tmp / "out"), "--shard-gb", "1e-9"])


def test_xrootd_urls_are_not_passed_to_glob():
    """glob cannot expand a root:// URL; it would silently return nothing."""
    assert glob.glob("root://host//store/user/x/*.root") == []
    with pytest.raises(SystemExit):
        make_shards._xrd_files("not-a-url")


# --------------------------------------------------------------------------- #
# Subtrees are PER SAMPLE. The full production is delphes-tree-61fd1c12 for the
# signal but delphes-tree-2ff38f65 for ttbar/DY, so one global flag cannot express
# the campaign — and a bare hash is not a portable key, since 6d2d1cb0 is the
# signal TEST subtree and also the ttbar/DY v0 one.
# --------------------------------------------------------------------------- #
def test_each_sample_selects_its_own_subtree():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _tree(tmp, "GluGluHH_kl-1p00_Delphes_v1", "61fd1c12")     # signal full
        _tree(tmp, "GluGluHH_kl-1p00_Delphes_v1", "6d2d1cb0")     # signal TEST
        _tree(tmp, "TTto2L2Nu_Delphes_v1", "2ff38f65")            # ttbar full
        _tree(tmp, "TTto2L2Nu_Delphes_v1", "6d2d1cb0")            # ttbar v0
        out = tmp / "out"
        make_shards.main([
            "--sample", "signal", str(tmp / "*kl-*"), "/m1.json",
            "_Delphes_v1/delphes-tree-61fd1c12",
            "--sample", "ttbar", str(tmp / "*TT*"), "/m2.json",
            "_Delphes_v1/delphes-tree-2ff38f65",
            "--out", str(out), "--shard-gb", "1e-9"])
        sh = json.load(open(out / "_plan" / "manifest.json"))["shards"]
        for e in sh:
            want = "61fd1c12" if e["sample"] == "signal" else "2ff38f65"
            assert all(want in f for f in e["files"]), e


def test_a_bare_hash_shared_between_roles_is_not_enough():
    """6d2d1cb0 is the signal test AND the ttbar v0 subtree, so it must be paired with
    _Delphes_v1/ — a hash-only filter would silently keep the wrong role."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _tree(tmp, "GluGluHH_kl-1p00_Delphes_v0", "6d2d1cb0")
        _tree(tmp, "GluGluHH_kl-1p00_Delphes_v1", "6d2d1cb0")
        bare = make_shards._files(str(tmp / "*kl-*"), subtree="delphes-tree-6d2d1cb0")
        paired = make_shards._files(str(tmp / "*kl-*"),
                                    subtree="_Delphes_v1/delphes-tree-6d2d1cb0")
        assert len(bare) == 6 and len(paired) == 3
        assert all("_Delphes_v1/" in f for f, _ in paired)


def test_a_bad_sample_arity_is_rejected():
    with pytest.raises(SystemExit):
        make_shards.main(["--sample", "sig", "/g", "--out", "/tmp/x"])


def test_proxy_is_written_into_the_submit_when_given():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _tree(tmp, "GluGluHH_Delphes_v1", "61fd1c12")
        proxy = tmp / "x509up"
        proxy.write_text("")
        out = tmp / "out"
        make_shards.main(["--sample", "sig", str(tmp / "*_Delphes_v1"), "/m.json",
                          "--out", str(out), "--shard-gb", "1e-9", "--proxy", str(proxy)])
        sub = (out / "_plan" / "ntuplize.sub").read_text()
        assert "x509userproxy" in sub and "use_x509userproxy       = true" in sub


def test_remote_inputs_without_a_proxy_are_warned_about(capsys):
    """A root:// job with no proxy fails to authenticate to dCache; silence would mean
    discovering that only after the whole campaign has been submitted."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        out = tmp / "out"
        (tmp / "_plan").mkdir(parents=True, exist_ok=True)
        orig = make_shards._files
        make_shards._files = lambda p, subtree=None, cache=None, refresh=False: [
            ("root://h//store/a_Delphes_v1/delphes-tree-61fd1c12/f.root", 1 << 30)]
        try:
            make_shards.main(["--sample", "sig", "root://h//store/*", "/m.json",
                              "--out", str(out), "--shard-gb", "1e-9"])
        finally:
            make_shards._files = orig
        assert "no --proxy given" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Shards are cut on BYTES. The first version used a placeholder that divided any
# input into ~8 shards regardless of size — so a 392 GB signal sample and a 8 TB
# ttbar sample both came out at 8, which would have put ~1 TB in an 8 GB slot.
# --------------------------------------------------------------------------- #
def _sized(tmp, dataset, h, sizes):
    d = tmp / dataset / f"delphes-tree-{h}"
    d.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(sizes):
        (d / f"f{i:03d}.root").write_bytes(b"x" * n)
    return d


def _shards_for(tmp, sizes, gb):
    _sized(tmp, "S_Delphes_v1", "61fd1c12", sizes)
    out = tmp / "out"
    make_shards.main(["--sample", "s", str(tmp / "*_Delphes_v1"), "/m.json",
                      "--out", str(out), "--shard-gb", str(gb)])
    return json.load(open(out / "_plan" / "manifest.json"))["shards"]


def test_shard_count_scales_with_total_size():
    """Twice the data must give about twice the shards — the property the placeholder
    did not have."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        small = _shards_for(Path(td) / "a", [10_000] * 20, 50e-6)   # 200 kB total
        big = _shards_for(Path(td) / "b", [10_000] * 40, 50e-6)     # 400 kB total
        assert len(big) >= 1.8 * len(small), (len(small), len(big))


def test_each_shard_respects_the_byte_target():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sizes = [10_000] * 30
        shards = _shards_for(tmp, sizes, 50e-6)     # 50 kB per shard -> ~5 files each
        for s in shards[:-1]:
            assert len(s["files"]) <= 7, s["files"]


def test_file_count_ceiling_applies_when_sizes_are_zero():
    """A listing with no sizes must not put the whole sample in one job."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _sized(tmp, "S_Delphes_v1", "61fd1c12", [0] * 25)
        out = tmp / "out"
        make_shards.main(["--sample", "s", str(tmp / "*_Delphes_v1"), "/m.json",
                          "--out", str(out), "--shard-gb", "1000", "--shard-files", "10"])
        sh = json.load(open(out / "_plan" / "manifest.json"))["shards"]
        assert len(sh) == 3 and all(len(s["files"]) <= 10 for s in sh)


def test_xrdfs_listing_parses_sizes():
    """xrdfs ls -l gives '<perms> <date> <time> <size> <path>'; the size is the shard axis
    and taking it from the listing avoids opening 60k remote files for their headers."""
    import subprocess as sp
    line = ("-r-- 2026-01-01 00:00:00 1234567 "
            "/store/user/x/S_Delphes_v1/delphes-tree-61fd1c12/f.root\n")
    orig = sp.check_output
    sp.check_output = lambda *a, **k: line
    try:
        got = make_shards._xrd_files("root://h//store/user/x/*")
    finally:
        sp.check_output = orig
    assert got == [("root://h//store/user/x/S_Delphes_v1/delphes-tree-61fd1c12/f.root",
                    1234567)]


# --------------------------------------------------------------------------- #
# Re-planning must be cheap. Walking 72k files over XRootD takes minutes and the
# remote tree does not change between runs, so re-listing on every invocation
# just to re-cut the shards is wasted time.
# --------------------------------------------------------------------------- #
def test_remote_listing_is_cached_and_reused(capsys):
    import subprocess as sp
    import tempfile

    line = ("-r-- 2026-01-01 00:00:00 2200000000 "
            "/store/x/S_Delphes_v1/delphes-tree-61fd1c12/f{}.root\n")
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, "listing.tsv")
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            return "".join(line.format(i) for i in range(5))

        orig, sp.check_output = sp.check_output, fake
        try:
            first = make_shards._xrd_files("root://h//store/x/*", cache=cache)
            second = make_shards._xrd_files("root://h//store/x/*", cache=cache)
            third = make_shards._xrd_files("root://h//store/x/*", cache=cache, refresh=True)
        finally:
            sp.check_output = orig
    assert first == second == third
    assert calls["n"] == 2, "the second call must come from cache, the third must re-list"
    assert "from cache" in capsys.readouterr().out


def test_cached_listing_preserves_sizes():
    """Sizes are the shard axis; a cache that dropped them would silently fall back to
    file-count sharding and blow the memory budget."""
    import subprocess as sp
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, "l.tsv")
        orig, sp.check_output = sp.check_output, lambda *a, **k: (
            "-r-- d t 123456789 /store/x/S_Delphes_v1/delphes-tree-1/f.root\n")
        try:
            make_shards._xrd_files("root://h//store/x/*", cache=cache)
            again = make_shards._xrd_files("root://h//store/x/*", cache=cache)
        finally:
            sp.check_output = orig
    assert again[0][1] == 123456789


# --------------------------------------------------------------------------- #
# `should_transfer_files = NO` claims a shared filesystem, which makes Condor add
# TARGET.FileSystemDomain == MY.FileSystemDomain. Every machine in this pool
# advertises its own hostname as its domain, so that pinned 1394 jobs to the
# submit node — 1 matching slot out of 47.
# --------------------------------------------------------------------------- #
def _sub_text(tmp, extra=()):
    _tree(tmp, "S_Delphes_v1", "61fd1c12")
    out = tmp / "out"
    make_shards.main(["--sample", "s", str(tmp / "*_Delphes_v1"), "/m.json",
                      "--out", str(out), "--shard-gb", "1e-9", *extra])
    return (out / "_plan" / "ntuplize.sub").read_text()


def test_submit_does_not_claim_a_shared_filesystem():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sub = _sub_text(Path(td))
    assert "should_transfer_files   = NO" not in sub
    assert "should_transfer_files   = YES" in sub


def test_submit_pins_to_machines_verified_to_see_ceph():
    """A job landing on an unverified node would fail on a missing pixi env."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sub = _sub_text(Path(td))
    assert "requirements" in sub and "etp" in sub and "TARGET.Machine" in sub


def test_requirements_can_be_overridden():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sub = _sub_text(Path(td), extra=["--requirements", "TARGET.Memory > 40000"])
    assert "TARGET.Memory > 40000" in sub


def test_nothing_is_transferred_back_through_the_sandbox():
    """The job writes its parquet to /ceph by absolute path; letting Condor also ship the
    scratch dir back would duplicate ~85 GB through the schedd."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sub = _sub_text(Path(td))
    assert 'transfer_output_files   = ""' in sub
