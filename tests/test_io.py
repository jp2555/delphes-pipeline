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
