"""Reader handles a sample *directory* of multiple ROOT files (real layout).

A Delphes signal sample is a directory of ROOT files, so ``DelphesEvents`` must
read and concatenate them and honour a total ``entry_stop`` across files.
"""

from __future__ import annotations

import awkward as ak
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


def test_lazy_open_stops_early(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    for i in range(3):
        make_fixture(str(sample / f"f{i}.root"), n_events=300, seed=i)

    ev = DelphesEvents(str(sample), entry_stop=100)  # satisfied by the first file
    assert ev.n == 100
    assert len(ev._used) == 1  # did not open the other two files
