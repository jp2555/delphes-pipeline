"""CMS lepton pools must mean the same thing as Delphes' efficiency-filtered ones.

NanoAOD Electron/Muon are loose enough to include jet fakes; a Delphes lepton has
already passed the card's efficiency. Compared raw, CMS gains candidates Delphes
cannot make, a fake pairs with a real tau back-to-back, and dR_tautau breaks.
"""
import awkward as ak
import numpy as np
import pytest

from delphes_pipeline.core import nanoaod


class _Fake(nanoaod.NanoAODEvents):
    def __init__(self, cols, wp=None):
        self._cols = cols
        self.b = nanoaod._deep_merge(nanoaod.BRANCHES, {
            "electron": {"id": "Electron_mvaIso_WP80"},
            "muon": {"id": "Muon_tightId", "iso": "Muon_pfRelIso04_all"}})
        self.wp = wp or {}
        self._keys = set(cols)
        self._n = len(next(iter(cols.values())))

    def array(self, br):
        return self._cols[br]

    @property
    def n(self):
        return self._n


def _cols(**over):
    base = {
        "Electron_pt": ak.Array([[30.0, 25.0]]), "Electron_eta": ak.Array([[0.1, 0.2]]),
        "Electron_phi": ak.Array([[0.0, 1.0]]), "Electron_charge": ak.Array([[1, -1]]),
        "Electron_mvaIso_WP80": ak.Array([[True, False]]),
        "Muon_pt": ak.Array([[40.0, 22.0]]), "Muon_eta": ak.Array([[0.3, 0.4]]),
        "Muon_phi": ak.Array([[2.0, 3.0]]), "Muon_charge": ak.Array([[-1, 1]]),
        "Muon_tightId": ak.Array([[True, True]]),
        "Muon_pfRelIso04_all": ak.Array([[0.05, 0.60]]),
    }
    base.update(over)
    return base


def test_a_failing_id_electron_is_dropped():
    ev = _Fake(_cols())
    assert ak.to_list(ev.electrons.pt) == [[30.0]], "the WP80-failing electron must go"


def test_a_non_isolated_muon_is_dropped_when_a_wp_is_set():
    ev = _Fake(_cols(), wp={"muon_iso_max": 0.15})
    assert ak.to_list(ev.muons.pt) == [[40.0]]


def test_isolation_needs_an_explicit_working_point():
    """Without a configured cut, do not invent one."""
    ev = _Fake(_cols())
    assert ak.to_list(ev.muons.pt) == [[40.0, 22.0]]


def test_an_unconfigured_reader_keeps_the_raw_collection():
    ev = _Fake(_cols())
    ev.b = nanoaod.BRANCHES
    assert ak.to_list(ev.electrons.pt) == [[30.0, 25.0]]
    assert np.isfinite(1.0)


def test_a_missing_id_branch_raises_instead_of_silently_disabling_the_cut():
    """_zip drops absent branches; a configured-but-unapplied cut must not pass."""
    cols = _cols()
    del cols["Electron_mvaIso_WP80"]
    with pytest.raises(KeyError, match="not in this NanoAOD"):
        _ = _Fake(cols).electrons


def test_a_missing_iso_branch_also_raises():
    cols = _cols()
    del cols["Muon_pfRelIso04_all"]
    with pytest.raises(KeyError, match="not in this NanoAOD"):
        _ = _Fake(cols, wp={"muon_iso_max": 0.15}).muons
