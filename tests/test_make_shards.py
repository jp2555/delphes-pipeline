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
import subprocess
import re
import pathlib
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


# --------------------------------------------------------------------------- #
# Every job exited 127 within seconds: `pixi` is on the submit node's PATH but not
# on the workers'. The probe verified the env's interpreter is importable at its
# absolute path, so the wrapper must call that, not the launcher.
# --------------------------------------------------------------------------- #
def _exe_text(tmp):
    _tree(tmp, "S_Delphes_v1", "61fd1c12")
    out = tmp / "out"
    make_shards.main(["--sample", "s", str(tmp / "*_Delphes_v1"), "/m.json",
                      "--out", str(out), "--shard-gb", "1e-9"])
    return (out / "_plan" / "run_shard.sh").read_text()


def test_wrapper_calls_the_interpreter_not_the_pixi_launcher():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        exe = _exe_text(Path(td))
    # check what is EXECUTED, not what is mentioned — the wrapper's own comment
    # explains why pixi is avoided, so a substring test on "pixi run" self-triggers
    execs = [ln for ln in exe.splitlines() if ln.strip().startswith("exec ")]
    assert execs, exe
    assert not any("pixi" in ln for ln in execs), execs
    assert any('"$PY"' in ln for ln in execs), execs
    assert '.pixi/envs/' in exe and '/bin/python"' in exe


def test_wrapper_fails_loudly_when_the_environment_is_missing():
    """Exit 127 with no message is what made this take a batch of 20 jobs to diagnose."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        exe = _exe_text(Path(td))
    assert "no interpreter at" in exe and "exit 127" in exe


def test_verify_does_not_require_the_planning_arguments():
    """--verify audits a finished campaign; demanding --out/--sample for it made the
    documented command fail at argparse before ever reaching the check."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards, _, out = _plan(Path(td), n=4)
        for s in shards:
            _write(s["out"], 5, s["shard"])
        assert make_shards.main(["--verify", str(out)]) == 0


def test_planning_still_requires_out_and_sample():
    with pytest.raises(SystemExit):
        make_shards.main(["--sample", "s", "/g", "/m.json"])       # no --out
    with pytest.raises(SystemExit):
        make_shards.main(["--out", "/tmp/x"])                       # no --sample


def test_verify_catches_a_truncated_shard_instead_of_crashing():
    """A job killed mid-write leaves an unreadable parquet. Treating that as MISSING is
    what makes the merge gate meaningful — an exception here would just abort the audit."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards, _, out = _plan(Path(td), n=4, shard_events=2)
        for s in shards:
            _write(s["out"], 5, s["shard"])
        Path(shards[1]["out"]).write_bytes(b"not a parquet file")
        assert make_shards.verify(str(out)) == 1


def test_verify_does_not_read_the_payload():
    """It must count from the parquet footer, not by loading each file: reading the whole
    ~82 GB campaign to count it, then again to merge it, doubles the I/O for nothing."""
    src = Path(make_shards.__file__).read_text()
    body = src[src.index("def verify("):src.index("def main(")]
    assert "metadata.num_rows" in body
    assert 'columns=["shard"]' in body
    assert "from_parquet" not in body, "loading the payload defeats the point"


# --------------------------------------------------------------------------- #
# Recovering a handful of failed shards must not mean re-running the campaign,
# and must not mean hand-editing the queue file — that is how shards get skipped.
# --------------------------------------------------------------------------- #
def test_resubmit_lists_only_the_missing_shards_with_their_original_seeds():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards, _, out = _plan(Path(td), n=8, shard_events=2)
        for s in shards:
            if s["shard"] not in (2, 5):
                _write(s["out"], 5, s["shard"])
        assert make_shards.main(["--verify", str(out), "--write-missing", "--force"]) == 1
        lines = (out / "_plan" / "shards.missing.txt").read_text().strip().splitlines()
        assert len(lines) == 2
        got = {int(ln.split(",")[1]): int(ln.split(",")[-1]) for ln in lines}
        orig = {s["shard"]: s["seed"] for s in shards}
        assert set(got) == {2, 5}
        assert all(got[k] == orig[k] for k in got), "a recovered shard must reuse its seed"


def test_resubmit_is_not_written_when_the_campaign_is_complete():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards, _, out = _plan(Path(td), n=4, shard_events=2)
        for s in shards:
            _write(s["out"], 5, s["shard"])
        assert make_shards.main(["--verify", str(out), "--write-missing", "--force"]) == 0
        assert not (out / "_plan" / "shards.missing.txt").exists()


# --------------------------------------------------------------------------- #
# Resubmitting a shard whose first job is still running makes two processes write
# the same parquet path. The queue check is the only thing standing between a
# routine --verify and a corrupted output file, so it must fail closed.
# --------------------------------------------------------------------------- #
def _missing_run(tmp, present, monkeypatch, live=None, force=False, boom=None):
    shards, _, out = _plan(Path(tmp), n=8, shard_events=2)
    for s in shards:
        if s["shard"] in present:
            _write(s["out"], 5, s["shard"])

    def fake():
        if boom is not None:
            raise boom
        return live or set()

    monkeypatch.setattr(make_shards, "_condor_q_args", fake)
    argv = ["--verify", str(out), "--write-missing"] + (["--force"] if force else [])
    make_shards.main(argv)
    q = out / "_plan" / "shards.missing.txt"
    if not q.exists():
        return None
    return {int(ln.split(",")[1]) for ln in q.read_text().strip().splitlines()}


def test_running_shards_are_excluded_from_the_resubmit(tmp_path, monkeypatch):
    got = _missing_run(tmp_path, present={0, 1, 2}, monkeypatch=monkeypatch,
                       live={("sig", 5), ("sig", 6), ("sig", 7)})
    assert got == {3, 4}, "only shards nothing is working on may be resubmitted"


def test_resubmit_is_withheld_when_every_gap_is_still_running(tmp_path, monkeypatch):
    got = _missing_run(tmp_path, present={0, 1, 2, 3, 4},
                       monkeypatch=monkeypatch,
                       live={("sig", i) for i in (5, 6, 7)})
    assert got is None


def test_unreachable_schedd_withholds_the_resubmit(tmp_path, monkeypatch):
    got = _missing_run(tmp_path, present={0, 1}, monkeypatch=monkeypatch,
                       boom=OSError("condor_q: not found"))
    assert got is None, "an unreachable schedd must not be read as 'nothing running'"


def test_force_writes_the_resubmit_without_consulting_the_queue(tmp_path, monkeypatch):
    got = _missing_run(tmp_path, present={0, 1}, monkeypatch=monkeypatch,
                       boom=OSError("condor_q: not found"), force=True)
    assert got == {2, 3, 4, 5, 6, 7}


def _fake_condor(monkeypatch, by_attr):
    import subprocess as sp

    def run(cmd, **k):
        return sp.CompletedProcess(cmd, 0, stdout=by_attr.get(cmd[-1], ""), stderr="")

    monkeypatch.setattr(make_shards.subprocess, "run", run)


_ROWS = ("ttbar 974 /p/ttbar.0974.txt /m.json /o/ttbar.0974.parquet 12345\n"
         "signal 3 /p/signal.0003.txt /m.json /o/signal.0003.parquet 7\n")


def test_the_submit_template_still_puts_sample_and_shard_first(monkeypatch):
    """The parser reads _SUB's `arguments` line positionally; pin that it can."""
    assert '"$(sample) $(shard)' in make_shards._SUB


def test_new_syntax_arguments_classad_is_parsed(monkeypatch):
    _fake_condor(monkeypatch, {"Arguments": _ROWS, "Args": "undefined\nundefined\n"})
    assert make_shards._condor_q_args() == {("ttbar", 974), ("signal", 3)}


def test_old_syntax_args_classad_is_parsed(monkeypatch):
    """Double-quoted `arguments =` is new syntax, but do not bet the guard on it."""
    _fake_condor(monkeypatch, {"Arguments": "undefined\nundefined\n", "Args": _ROWS})
    assert make_shards._condor_q_args() == {("ttbar", 974), ("signal", 3)}


def test_an_empty_queue_is_not_confused_with_a_broken_parse(monkeypatch):
    _fake_condor(monkeypatch, {"Arguments": "", "Args": ""})
    assert make_shards._condor_q_args() == set()


def test_unparseable_rows_raise_instead_of_reading_as_an_empty_queue(monkeypatch):
    _fake_condor(monkeypatch, {"Arguments": "undefined\nundefined\n",
                               "Args": "undefined\nundefined\n"})
    with pytest.raises(RuntimeError, match="refusing to guess"):
        make_shards._condor_q_args()


# --------------------------------------------------------------------------- #
# DY is ten MLL-binned datasets, each with a DIFFERENT cross section. Globbing
# them into one sample loses the bin identity exactly as globbing the kl points
# lost kappa_lambda -- and there the loss was only recoverable because the plan
# still held the input paths. Listing first is how the bins get planned apart.
# --------------------------------------------------------------------------- #
def _dy_tree(tmp, bins, per=2, subtree="delphes-tree-2ff38f65"):
    for b in bins:
        d = tmp / f"DYto2Tau_Bin-MLL-{b}_TuneCP5_powheg-pythia8_Delphes_v1" / subtree
        d.mkdir(parents=True)
        for i in range(per):
            (d / f"delphes-tree_{i}.root").write_bytes(b"x" * 2048)
    return tmp


def test_list_dirs_reports_each_mll_bin_separately(tmp_path, capsys):
    _dy_tree(tmp_path, ["10to50", "50to120", "120to200"])
    assert make_shards.list_dirs(str(tmp_path / "*DYto2Tau*")) == 0
    said = capsys.readouterr().out
    assert "3 sample dir(s)" in said
    for b in ("10to50", "50to120", "120to200"):
        assert f"Bin-MLL-{b}" in said


def test_list_dirs_flags_a_directory_spanning_two_subtrees(tmp_path, capsys):
    _dy_tree(tmp_path, ["50to120"], subtree="delphes-tree-2ff38f65")
    d = tmp_path / "DYto2Tau_Bin-MLL-50to120_TuneCP5_powheg-pythia8_Delphes_v1" / "delphes-tree-50b0dcf9"
    d.mkdir(parents=True)
    (d / "x.root").write_bytes(b"x" * 10)
    make_shards.list_dirs(str(tmp_path / "*DYto2Tau*"))
    assert "SPANS SUBTREES" in capsys.readouterr().out


def test_list_dirs_says_so_when_nothing_matches(tmp_path, capsys):
    assert make_shards.list_dirs(str(tmp_path / "*nope*")) == 1
    assert "nothing matched" in capsys.readouterr().out


def test_maps_none_plans_an_untuned_campaign(tmp_path, capsys):
    """The untuned baseline is a DIFFERENT forward model, so it is spelled out."""
    src = _inputs(tmp_path, 3)
    out = tmp_path / "out"
    make_shards.main(["--sample", "sig", str(src / "*.root"), "none",
                      "--out", str(out), "--shard-gb", "1e-9"])
    assert "UNTUNED" in capsys.readouterr().out
    plan = json.load(open(out / "_plan" / "manifest.json"))["shards"]
    assert {e["maps"] for e in plan} == {"none"}
    assert {e["maps_sha"] for e in plan} == {"untuned"}
    exe = (out / "_plan" / "run_shard.sh").read_text()
    assert "--tuning-maps" in exe, "the value is passed through and normalised in Python"


# --------------------------------------------------------------------------- #
# 1587 untuned jobs died in 17 s each on FileNotFoundError: '.../none'. The submit
# script HAD a guard comparing "$4" against "none" -- but an HTCondor queue field
# from a comma-separated line arrives with its leading space, so " none" != "none",
# the guard never fired, and the value was resolved as a relative path. The decision
# therefore lives in Python, where quoting and whitespace cannot defeat it.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", ["none", " none", "NONE", "  ", ""])
def test_a_none_maps_value_means_untuned_however_it_is_spelled(raw):
    from delphes_pipeline.ntuplizer.convert import _no_maps_if_none
    assert _no_maps_if_none(raw) is None


@pytest.mark.parametrize("raw", ["/m.json", " /m.json ", "\t/m.json"])
def test_a_real_maps_path_survives_the_queue_fields_whitespace(raw):
    from delphes_pipeline.ntuplizer.convert import _no_maps_if_none
    assert _no_maps_if_none(raw) == "/m.json"


def test_convert_does_not_open_a_file_called_none(tmp_path):
    """The exact failure: 'none' resolved relative to the repo and was opened."""
    from delphes_pipeline.ntuplizer import convert as C
    assert C._no_maps_if_none(" none") is None
    assert not (pathlib.Path.cwd() / "none").exists(), \
        "nothing should ever create a file literally named 'none'"
