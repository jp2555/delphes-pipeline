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
