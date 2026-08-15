"""The merged ntuple must read back through the SAME names as DelphesEvents.

The overlay defines m_HH, m_ττ and the rest once, against the Delphes attribute
names. If reading a merged ntuple needed a second feature implementation, a
disagreement between a Delphes overlay and an ntuple overlay could not be
attributed to the data. These tests pin the adapter, not the physics.
"""
import awkward as ak
import numpy as np
import pytest

from delphes_pipeline.core.io import NtupleEvents


def _write(path, n, kl=None, offset=0.0):
    fields = {
        "Jet": ak.Array([[{"pt": 30.0 + offset, "eta": 0.5, "phi": 0.1, "mass": 5.0,
                           "btag": 1, "tautag": 0, "hadronFlavour": 5}]] * n),
        "Tau": ak.Array([[{"pt": 25.0, "eta": 0.2, "phi": 1.0, "mass": 1.2}]] * n),
        "Electron": ak.Array([[{"pt": 20.0, "eta": 0.1, "phi": 0.3, "charge": -1}]] * n),
        # typed-but-empty: a bare [[]] has no record type and would fail for a
        # reason the real ntuple never has
        "Muon": ak.Array([[{"pt": 9.0, "eta": 0.0, "phi": 0.0, "charge": 1}]] * n)[:, :0],
        "GenPart": ak.Array([[{"pt": 40.0, "eta": 0.4, "phi": 0.2, "mass": 1.777,
                               "pdgId": 15, "status": 2, "genPartIdxMother": 0}]] * n),
        "MET_pt": np.full(n, 60.0, dtype=np.float32),
        "MET_phi": np.full(n, 0.7, dtype=np.float32),
    }
    if kl is not None:
        fields["kl"] = np.full(n, kl, dtype=np.float32)
    ak.to_parquet(ak.zip(fields, depth_limit=1), str(path), row_group_size=10)
    return path


def test_collections_read_back_under_the_delphes_names(tmp_path):
    ev = NtupleEvents(_write(tmp_path / "a.parquet", 5))
    assert ev.n == 5
    assert ev.jets.pt[0][0] == 30.0 and ev.jets.btag[0][0] == 1
    assert ev.jets.flavor[0][0] == 5, "hadronFlavour must surface as Delphes' flavor"
    assert ev.electrons.charge[0][0] == -1
    assert ev.met.met[0] == 60.0 and ev.met.phi[0] == pytest.approx(0.7)
    assert ev.gen.pid[0][0] == 15, "pdgId must surface as Delphes' pid"
    assert ev.gen.m1[0][0] == 0


def test_jet_charge_is_supplied_so_the_shared_tau_builder_works(tmp_path):
    """The ntuple drops jet charge; tau_candidates zips it regardless."""
    from delphes_pipeline.validation.level1_candles.selections import tau_candidates
    ev = NtupleEvents(_write(tmp_path / "a.parquet", 3))
    assert ak.all(ev.jets.charge == 0)
    assert len(tau_candidates(ev)) == 3


def test_selecting_one_kl_point_returns_only_that_point(tmp_path):
    """kl points sit in contiguous blocks, so this is not a no-op filter."""
    import shutil
    d = tmp_path / "merged"
    d.mkdir()
    _write(d / "signal.0000.parquet", 40, kl=0.0)
    _write(d / "signal.0001.parquet", 40, kl=5.0)
    ev = NtupleEvents(d, kl=5.0)
    assert ev.n == 40
    assert ak.all(ev.array["kl"] == 5.0)
    assert shutil  # keep the import honest


def test_a_kl_point_is_not_starved_by_reading_the_first_events(tmp_path):
    """The bug this guards: entry_stop applied BEFORE the kl filter yields nothing.

    kl=0 fills the whole first file, so a reader that takes the first 20 events and
    then filters for kl=5 finds none.
    """
    f = tmp_path / "s.parquet"
    a0 = ak.from_parquet(str(_write(tmp_path / "_0.parquet", 100, kl=0.0)))
    a5 = ak.from_parquet(str(_write(tmp_path / "_5.parquet", 30, kl=5.0)))
    ak.to_parquet(ak.concatenate([a0, a5]), str(f), row_group_size=10)
    ev = NtupleEvents(f, kl=5.0, entry_stop=20)
    assert ev.n == 20
    assert ak.all(ev.array["kl"] == 5.0)


def test_entry_stop_spans_files_without_overrunning(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    _write(d / "a.0000.parquet", 30)
    _write(d / "a.0001.parquet", 30)
    assert NtupleEvents(d, entry_stop=45).n == 45
    assert NtupleEvents(d).n == 60


def test_an_empty_selection_is_an_error_not_a_silent_empty_overlay(tmp_path):
    with pytest.raises(ValueError, match="no events"):
        NtupleEvents(_write(tmp_path / "a.parquet", 5, kl=1.0), kl=99.0)


# --------------------------------------------------------------------------- #
# The overlay CLI: --ntuple must not silently misreport what it plotted.
# --------------------------------------------------------------------------- #
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import nsbi_overlay  # noqa: E402


def test_kl_tag_parses_to_the_number_stored_in_the_ntuple():
    assert nsbi_overlay._kl_value("0p00") == 0.0
    assert nsbi_overlay._kl_value("5p00") == 5.0
    assert nsbi_overlay._kl_value("m2p50") == -2.5


def test_one_of_delphes_dir_or_ntuple_is_required():
    with pytest.raises(SystemExit):
        nsbi_overlay.main(["--config", "x.yml", "--nano-dir", "/n"])


def test_no_tuned_is_rejected_for_a_merged_ntuple():
    """The maps are baked in; honouring --no-tuned is impossible, so say so."""
    with pytest.raises(SystemExit):
        nsbi_overlay.main(["--config", "x.yml", "--nano-dir", "/n",
                           "--ntuple", "/m", "--no-tuned"])
