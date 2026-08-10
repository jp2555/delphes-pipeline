"""The NSBI overlay must apply the SAME selection to Delphes and to NanoAOD.

Delphes τ_h *are* jets (``Jet.TauTag``); CMS keeps them in a separate ``Tau``
collection while their jets stay in ``Jet``. The old overlay dropped τ-tagged jets
from the Delphes b-candidate pool but kept every jet on the NanoAOD side, so a
τ-jet could enter the CMS bb pair and never the Delphes one — a pure selection
artefact that lands on ΔR_bb.

Here one *physical* event is expressed on both sides: two b-jets (only the leading
one b-tagged) and two high-pT τ_h. Sorting by (pT, then b-tag) lets a τ-jet take the
second bb slot wherever it is not removed. Symmetric cleaning must give both sides
the same pair; the legacy path must not.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nsbi_overlay import features  # noqa: E402

_N = 200          # repeat the same event so the arrays are not degenerate
_DR = 0.4

#            pt    eta   phi   role
_B1 = (100.0, 0.0, 0.0)      # b-tagged b-jet
_B2 = (30.0, 1.2, 2.0)       # untagged b-jet -> the contested second slot
_T1 = (80.0, -1.0, -2.0)     # leading τ_h (and its jet)
_T2 = (70.0, 0.5, 1.0)       # sub-leading τ_h (and its jet)


def _jag(per_event: list) -> ak.Array:
    """A jagged (ListOffset) float array repeating ``per_event`` for every event."""
    return ak.values_astype(ak.Array([list(per_event)] * _N), np.float64)


def _col(rows, **extra):
    """A jagged per-event collection from (pt, eta, phi) rows."""
    out = {"pt": _jag([r[0] for r in rows]), "eta": _jag([r[1] for r in rows]),
           "phi": _jag([r[2] for r in rows]), "mass": _jag([0.0] * len(rows))}
    for k, v in extra.items():
        out[k] = _jag(v)
    return ak.zip(out)


def _empty(*fields):
    return ak.zip({k: _jag([]) for k in fields})


def _met():
    return ak.zip({"met": ak.Array(np.full(_N, 40.0)), "phi": ak.Array(np.full(_N, 0.5))})


def _delphes_ev():
    """Delphes view: τ_h are jets carrying TauTag=1."""
    jets = _col([_B1, _B2, _T1, _T2],
                btag=[1, 0, 0, 0], tautag=[0, 0, 1, 1], charge=[0, 0, 1, -1])
    empty = _empty("pt", "eta", "phi", "charge")
    return SimpleNamespace(jets=jets, electrons=empty, muons=empty, met=_met())


def _nano_ev():
    """CMS view: the same four jets, with the τ_h ALSO in the Tau collection."""
    jets = _col([_B1, _B2, _T1, _T2], btag=[1, 0, 0, 0])
    taus = _col([_T1, _T2], vsjet=[6, 6])
    empty = _empty("pt", "eta", "phi", "charge")
    ev = SimpleNamespace(jets=jets, taus=taus, electrons=empty, muons=empty, met=_met())
    ev.deeptau_medium = lambda: 5
    return ev


def _dR(a, b):
    dphi = abs((a[2] - b[2] + np.pi) % (2 * np.pi) - np.pi)
    return float(np.hypot(a[1] - b[1], dphi))


_KW = dict(tautau_only=True, mtautau_min=0.0, clean_dr=_DR)


def test_symmetric_cleaning_gives_both_sides_the_same_bb_pair():
    d = features(_delphes_ev(), nano=False, clean=True, **_KW)
    n = features(_nano_ev(), nano=True, clean=True, **_KW)
    assert d["dR_bb"].size and n["dR_bb"].size, "both sides must keep the event"
    # the τ-jets are removed on BOTH sides -> the pair is (B1, B2) either way
    assert d["dR_bb"][0] == pytest.approx(_dR(_B1, _B2), abs=1e-6)
    assert n["dR_bb"][0] == pytest.approx(_dR(_B1, _B2), abs=1e-6)
    assert d["dR_bb"][0] == pytest.approx(n["dR_bb"][0], abs=1e-6)


def test_legacy_selection_is_asymmetric():
    """Without the fix the two sides build the bb pair from different jet pools."""
    d = features(_delphes_ev(), nano=False, clean=False, **_KW)
    n = features(_nano_ev(), nano=True, clean=False, **_KW)
    assert d["dR_bb"][0] == pytest.approx(_dR(_B1, _B2), abs=1e-6)   # τ-jets dropped
    assert n["dR_bb"][0] == pytest.approx(_dR(_B1, _T1), abs=1e-6)   # τ-jet steals the slot
    assert d["dR_bb"][0] != pytest.approx(n["dR_bb"][0], abs=1e-3)


def test_cleaning_removes_only_jets_near_the_selected_taus():
    """A jet far from both selected τ survives; the τ-jets themselves do not."""
    d = features(_delphes_ev(), nano=False, clean=True, **_KW)
    # B2 survived (it is the second leg of the pair) -> cleaning is not over-aggressive
    assert d["mbb"][0] > 0
    assert d["dR_bb"][0] == pytest.approx(_dR(_B1, _B2), abs=1e-6)


def test_common_jet_acceptance_applies_to_both_sides():
    """Raising the jet pT floor above B2 drops the event on BOTH sides alike."""
    kw = dict(_KW, clean=True, jet_pt_min=50.0)
    d = features(_delphes_ev(), nano=False, **kw)
    n = features(_nano_ev(), nano=True, **kw)
    assert d["dR_bb"].size == 0 and n["dR_bb"].size == 0
