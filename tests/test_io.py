"""Reader handles a sample *directory* of multiple ROOT files (real layout).

A Delphes signal sample is a directory of ROOT files, so ``DelphesEvents`` must
read and concatenate them and honour a total ``entry_stop`` across files.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from make_fixture import make_fixture

from delphes_pipeline.core.io import DelphesEvents, resolve_paths


def test_reader_reads_a_directory(tmp_path):
    sample = tmp_path / "GluGluHHto2B2Tau_kl-1p00_Delphes"
    sample.mkdir()
    make_fixture(str(sample / "events_0.root"), n_events=300, seed=1)
    make_fixture(str(sample / "events_1.root"), n_events=200, seed=2)

    assert len(resolve_paths(str(sample))) == 2

    ev = DelphesEvents(str(sample))  # pass the directory, not a file
    assert ev.n == 500
    assert int(ak.sum(ak.num(ev.jets))) > 0
    assert len(ev.weights) == 500


def test_entry_stop_caps_total_across_files(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    make_fixture(str(sample / "a.root"), n_events=300, seed=1)
    make_fixture(str(sample / "b.root"), n_events=300, seed=2)

    ev = DelphesEvents(str(sample), entry_stop=400)  # spans into the second file
    assert ev.n == 400
    assert len(ev.weights) == 400


def test_reader_finds_nested_tree_subdir(tmp_path):
    # real layout: <sample>/delphes-tree-<hash>/delphes-tree_N.root
    sub = tmp_path / "kl-1p00_Delphes" / "delphes-tree-edccf8a6"
    sub.mkdir(parents=True)
    make_fixture(str(sub / "delphes-tree_0.root"), n_events=200, seed=1)
    make_fixture(str(sub / "delphes-tree_1.root"), n_events=200, seed=2)

    ev = DelphesEvents(str(tmp_path / "kl-1p00_Delphes"))  # point at the sample dir
    assert len(ev.paths) == 2
    assert ev.n == 400


def test_glob_matching_a_sample_dir_expands_to_root_files(tmp_path):
    # ".../delphes/*kl-1p00*" must match the kl=1 sample dir and read its files,
    # without matching kl=0 (the deepthought layout).
    base = tmp_path / "delphes"
    one = base / "GluGluHH_kl-1p00_Delphes" / "delphes-tree-abc"
    one.mkdir(parents=True)
    make_fixture(str(one / "t_0.root"), n_events=120, seed=1)
    make_fixture(str(one / "t_1.root"), n_events=80, seed=2)
    zero = base / "GluGluHH_kl-0p00_Delphes"
    zero.mkdir(parents=True)
    make_fixture(str(zero / "t.root"), n_events=50, seed=3)

    ev = DelphesEvents(str(base / "*kl-1p00*"))
    assert ev.n == 200  # both kl=1 files, not the kl=0 file


def test_lazy_open_stops_early(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    for i in range(3):
        make_fixture(str(sample / f"f{i}.root"), n_events=300, seed=i)

    ev = DelphesEvents(str(sample), entry_stop=100)  # satisfied by the first file
    assert ev.n == 100
    assert len(ev._used) == 1  # did not open the other two files


def _mixed_ntuple(path, per_ds=100, n_ds=3):
    """A merged ntuple holding several datasets in ONE file, as merge_shards writes it."""
    rng = np.random.default_rng(0)
    n = per_ds * n_ds
    a = ak.Array({"dataset_id": np.repeat(np.arange(n_ds), per_ds).astype("int16"),
                  "Jet": ak.unflatten(ak.Array({"pt": rng.uniform(20, 200, n * 2)}), 2),
                  "MET_pt": rng.uniform(0, 200, n)})
    ak.to_parquet(a, str(path), row_group_size=per_ds // 2)
    return n


def test_dataset_filter_selects_one_dataset(tmp_path):
    # Merged files are named per SAMPLE, so every tt decay mode shares them and only
    # dataset_id separates them. Overlaying the mixture against one CMS dataset would
    # read as a detector difference, so the filter has to actually isolate a dataset.
    from delphes_pipeline.core.io import NtupleEvents
    _mixed_ntuple(tmp_path / "ttbar.0000.parquet")

    for want in (0, 1, 2):
        ev = NtupleEvents(str(tmp_path), dataset=want)
        assert ev.n == 100
        assert set(np.unique(ak.to_numpy(ev.array["dataset_id"]))) == {want}

    assert NtupleEvents(str(tmp_path)).n == 300      # unfiltered = the whole mixture


def test_dataset_filter_refuses_an_absent_dataset(tmp_path):
    from delphes_pipeline.core.io import NtupleEvents
    _mixed_ntuple(tmp_path / "ttbar.0000.parquet")
    with pytest.raises(ValueError, match="dataset_id=99"):
        NtupleEvents(str(tmp_path), dataset=99)


def test_dataset_filter_refuses_an_unlabelled_ntuple(tmp_path):
    # An ntuple predating per-dataset labelling has no dataset_id column. Returning the
    # full mixture under a name that says one dataset is the failure mode to avoid.
    from delphes_pipeline.core.io import NtupleEvents
    ak.to_parquet(ak.Array({"MET_pt": np.arange(10.0)}),
                  str(tmp_path / "ttbar.0000.parquet"))
    with pytest.raises(ValueError, match="no dataset_id column"):
        NtupleEvents(str(tmp_path), dataset=0)
