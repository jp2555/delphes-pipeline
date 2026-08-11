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


# --------------------------------------------------------------------------- #
# gen-matched vs fake split (diagnostic for the ΔR_ττ shoulder)
# --------------------------------------------------------------------------- #
def _jagi(vals):
    return ak.values_astype(ak.Array([list(vals)] * _N), np.int64)


def _delphes_ev_gen(tau_eta_phi):
    """Delphes view whose gen τ sit at ``tau_eta_phi`` (may or may not be on the τ-jets)."""
    ev = _delphes_ev()
    ev.gen = ak.zip({
        "pid": _jagi([15, -15]), "status": _jagi([2, 2]), "m1": _jagi([-1, -1]),
        "pt": _jag([80.0, 70.0]), "mass": _jag([1.777, 1.777]),
        "eta": _jag([tau_eta_phi[0][0], tau_eta_phi[1][0]]),
        "phi": _jag([tau_eta_phi[0][1], tau_eta_phi[1][1]]),
    })
    return ev


def _nano_ev_gen(vis_eta_phi):
    ev = _nano_ev()
    ev.genvistau = _col([(50.0, e, p) for e, p in vis_eta_phi])
    return ev


_KWM = dict(_KW, clean=True, with_match=True)


def test_split_marks_pairs_on_real_gen_taus_as_matched():
    d, dm = features(_delphes_ev_gen([(_T1[1], _T1[2]), (_T2[1], _T2[2])]), nano=False, **_KWM)
    assert dm.size and dm.all(), "τ candidates sitting on gen τ must be gen-matched"
    n, nm = features(_nano_ev_gen([(_T1[1], _T1[2]), (_T2[1], _T2[2])]), nano=True, **_KWM)
    assert nm.size and nm.all()


def test_split_marks_pairs_without_gen_taus_as_fake():
    """Gen τ parked far away -> the selected pair is a fake, on both sides."""
    far = [(4.0, 0.0), (-4.0, 3.0)]
    d, dm = features(_delphes_ev_gen(far), nano=False, **_KWM)
    assert dm.size and not dm.any()
    n, nm = features(_nano_ev_gen(far), nano=True, **_KWM)
    assert nm.size and not nm.any()


def test_split_mask_is_aligned_with_the_features():
    d, dm = features(_delphes_ev_gen([(_T1[1], _T1[2]), (_T2[1], _T2[2])]), nano=False, **_KWM)
    assert dm.shape[0] == d["mHH"].shape[0], "the match mask must align with the kept events"


# --------------------------------------------------------------------------- #
# common τ-candidate acceptance (the last unsymmetrised object)
# --------------------------------------------------------------------------- #
_SOFT = (17.0, 0.4, 0.6)      # below the 20 GeV floor
_FWD = (60.0, 2.45, 1.4)      # beyond |eta| 2.3, inside Delphes' TauEtaMax 2.5


def _pair_with(extra_tau):
    """Delphes and CMS views of the SAME event, with one extra τ candidate."""
    taus = [_T1, _T2, extra_tau]
    d = _delphes_ev()
    d.jets = _col([_B1, _B2] + taus, btag=[1, 0, 0, 0, 0],
                  tautag=[0, 0, 1, 1, 1], charge=[0, 0, 1, -1, 1])
    n = _nano_ev()
    n.jets = _col([_B1, _B2] + taus, btag=[1, 0, 0, 0, 0])
    n.taus = _col(taus, vsjet=[6, 6, 6])
    return d, n


def test_soft_tau_is_rejected_identically_on_both_sides():
    """A 17 GeV τ exists in the Delphes jet collection (JetPTMin 15) but is below the
    CMS Tau floor; the common cut must remove it on both sides, not just one."""
    d_ev, n_ev = _pair_with(_SOFT)
    d = features(d_ev, nano=False, clean=True, **_KW)
    n = features(n_ev, nano=True, clean=True, **_KW)
    # the pair is (T1, T2) either way -> identical ΔR_ττ, the soft τ never enters
    assert d["dR_tautau"][0] == pytest.approx(_dR(_T1, _T2), abs=1e-6)
    assert n["dR_tautau"][0] == pytest.approx(_dR(_T1, _T2), abs=1e-6)


def test_forward_tau_is_rejected_identically_on_both_sides():
    d_ev, n_ev = _pair_with(_FWD)
    d = features(d_ev, nano=False, clean=True, **_KW)
    n = features(n_ev, nano=True, clean=True, **_KW)
    assert d["dR_tautau"][0] == pytest.approx(_dR(_T1, _T2), abs=1e-6)
    assert n["dR_tautau"][0] == pytest.approx(_dR(_T1, _T2), abs=1e-6)


def test_acceptance_rejects_only_what_it_should():
    """The cut must not swallow good candidates: an IN-acceptance τ that outranks T2 in
    pT is still picked, so the two tests above show rejection, not blanket removal."""
    hard = (75.0, 0.4, 0.6)             # inside pT/|eta|, harder than T2 -> enters the pair
    d_ev, n_ev = _pair_with(hard)
    d = features(d_ev, nano=False, clean=True, **_KW)
    n = features(n_ev, nano=True, clean=True, **_KW)
    assert d["dR_tautau"][0] == pytest.approx(_dR(_T1, hard), abs=1e-6)
    assert n["dR_tautau"][0] == pytest.approx(_dR(_T1, hard), abs=1e-6)


def test_diagnostic_panels_expose_the_fastmtt_inputs():
    """τ pT and MET are what FastMTT consumes; if the selection matches but m_ττ does
    not, these separate a τ-energy cause from a MET cause."""
    import nsbi_overlay as ov

    d = features(_delphes_ev(), nano=False, clean=True, **_KW)
    for k in ov._DIAG:
        assert k in d and k in ov._RANGES, k
    # the leading τ candidate is T1 (80 GeV), the sub-leading T2 (70 GeV)
    assert d["tau1_pt"][0] == pytest.approx(_T1[0], abs=1e-6)
    assert d["tau2_pt"][0] == pytest.approx(_T2[0], abs=1e-6)
    assert d["met"][0] == pytest.approx(40.0, abs=1e-6)      # the fixture MET magnitude


def test_diagnostics_are_not_in_the_nsbi_feature_set():
    """The 10 NSBI features stay the deliverable; the diagnostics are opt-in."""
    import nsbi_overlay as ov

    assert len(ov._FEATURES) == 10
    assert not set(ov._DIAG) & set(ov._FEATURES)


# --------------------------------------------------------------------------- #
# CMS Run-3 DNN input set (rotated frame)
# --------------------------------------------------------------------------- #
def test_cms_dnn_features_are_all_produced():
    import nsbi_overlay as ov

    d = features(_delphes_ev(), nano=False, clean=True, cms_dnn=True, **_KW)
    missing = [k for k in ov._CMS if k not in d]
    assert not missing, missing


def test_rotation_puts_the_visible_ditau_at_phi_zero():
    """The CMS DNN frame rotates every momentum by -φ(visible di-τ). The rotated di-τ
    must therefore have py = 0 and px > 0."""
    import nsbi_overlay as ov

    d = features(_delphes_ev(), nano=False, clean=True, cms_dnn=True, **_KW)
    vis_px = d["lep1_px"][0] + d["lep2_px"][0]
    vis_py = d["lep1_py"][0] + d["lep2_py"][0]
    assert abs(vis_py) < 1e-6, vis_py
    assert vis_px > 0


def test_rotation_preserves_magnitudes():
    """A rotation about the beam changes px,py but not pT, pz or E."""
    plain = features(_delphes_ev(), nano=False, clean=True, **_KW)
    rot = features(_delphes_ev(), nano=False, clean=True, cms_dnn=True, **_KW)
    pt_rot = np.hypot(rot["lep1_px"], rot["lep1_py"])
    assert pt_rot[0] == pytest.approx(plain["tau1_pt"][0], rel=1e-6)


def test_absent_fatjet_gives_zeros_and_a_zero_flag():
    """Neither fixture has AK8 jets: the event must survive with fatjet_exist = 0
    rather than being dropped."""
    d = features(_delphes_ev(), nano=False, clean=True, cms_dnn=True, **_KW)
    assert d["fatjet_exist"][0] == 0.0
    assert d["fatjet_E"][0] == 0.0


def test_cms_dnn_works_on_the_nano_side_too():
    import nsbi_overlay as ov

    n = features(_nano_ev(), nano=True, clean=True, cms_dnn=True, **_KW)
    assert all(k in n for k in ov._CMS)
    assert n["Hbbtt_E"][0] > 0
