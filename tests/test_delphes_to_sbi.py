"""The Delphes baseline must land in the SAME format, with the SAME definitions,
as convert_powheg_to_sbi.py produces from CMS NanoAOD.

The point of the untuned baseline is comparability with the NSBI result already
obtained on CMS. Several definitions in that converter differ from this repo's
overlay, so these tests pin the CMS convention rather than ours.
"""
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import delphes_to_sbi as S  # noqa: E402

from delphes_pipeline.core.io import NtupleEvents  # noqa: E402


def _write(path, n=4, kl=None):
    """One clean mu-tau_h event per row: 1 muon, 1 tau_h jet, 2 b jets."""
    jets = [[{"pt": 90.0, "eta": 0.5, "phi": 0.0, "mass": 8.0, "btag": 1,
              "tautag": 0, "hadronFlavour": 5},
             {"pt": 70.0, "eta": -0.4, "phi": 2.8, "mass": 7.0, "btag": 1,
              "tautag": 0, "hadronFlavour": 5},
             {"pt": 45.0, "eta": 1.0, "phi": 1.4, "mass": 1.2, "btag": 0,
              "tautag": 1, "hadronFlavour": 0}]] * n
    fields = {
        "Jet": ak.Array(jets),
        "Tau": ak.Array([[{"pt": 45.0, "eta": 1.0, "phi": 1.4, "mass": 1.2}]] * n),
        "Electron": ak.Array([[{"pt": 9.0, "eta": 0.0, "phi": 0.0,
                               "charge": 1}]] * n)[:, :0],
        "Muon": ak.Array([[{"pt": 40.0, "eta": -0.2, "phi": 2.0, "charge": -1}]] * n),
        "GenPart": ak.Array([[{"pt": 1.0, "eta": 0.0, "phi": 0.0, "mass": 0.0,
                              "pdgId": 15, "status": 2, "genPartIdxMother": 0}]] * n),
        "MET_pt": np.full(n, 55.0, dtype=np.float32),
        "MET_phi": np.full(n, 1.0, dtype=np.float32),
        "genWeight": np.concatenate([np.ones(n - 1), [-1.0]]).astype(np.float32),
    }
    if kl is not None:
        fields["kl"] = np.full(n, kl, dtype=np.float32)
    ak.to_parquet(ak.zip(fields, depth_limit=1), str(path), row_group_size=4)
    return path


def _feats(tmp_path, **kw):
    return S.features(NtupleEvents(_write(tmp_path / "a.parquet", **kw)))[0]


def test_every_required_branch_is_produced(tmp_path):
    d = _feats(tmp_path)
    assert set(S.REQUIRED) <= set(d)
    assert all(v.dtype == np.float64 for v in d.values())


def test_m_tautau_is_the_VISIBLE_mass_not_fastmtt(tmp_path):
    """convert_powheg_to_sbi maps m_vis -> m_tautau; FastMTT is only optional."""
    d = _feats(tmp_path)
    lep = (40.0, -0.2, 2.0, 0.0)
    tau = (45.0, 1.0, 1.4, 1.2)
    expect = S._sum_p4(lep, tau)[3]
    assert d["m_tautau"][0] == pytest.approx(expect, rel=1e-9)
    # FastMTT would divide by sqrt(x1 x2) <= 1, so it can only be larger
    assert d["m_tautau"][0] < 200.0


def test_cos_theta_star_is_tanh_of_half_deta(tmp_path):
    """NOT the Collins-Soper polar angle nsbi_overlay.py uses."""
    d = _feats(tmp_path)
    # deta_hh is |delta eta| while cos_theta_star keeps the sign, so the invariant
    # is on the magnitude
    assert abs(d["cos_theta_star"][0]) == pytest.approx(
        np.tanh(d["deta_hh"][0] / 2.0), abs=1e-9)
    assert abs(d["cos_theta_star"][0]) <= 1.0


def test_pt_h1_is_the_bb_system_and_pt_h2_the_tautau_one(tmp_path):
    """Fixed by CONTENT, not sorted by pT the way the overlay sorts them."""
    d = _feats(tmp_path)
    bb = S._sum_p4((90.0, 0.5, 0.0, 8.0), (70.0, -0.4, 2.8, 7.0))
    tt = S._sum_p4((40.0, -0.2, 2.0, 0.0), (45.0, 1.0, 1.4, 1.2))
    assert d["pt_h1"][0] == pytest.approx(bb[0], rel=1e-9)
    assert d["pt_h2"][0] == pytest.approx(tt[0], rel=1e-9)


def test_dphi_hh_is_folded_into_zero_pi(tmp_path):
    d = _feats(tmp_path)
    assert 0.0 <= d["dphi_hh"][0] <= np.pi


def test_negative_generator_weights_survive(tmp_path):
    """NLO weights can be negative; the notebook's label-flip depends on that."""
    d = _feats(tmp_path)
    assert (d["weights"] < 0).sum() == 1


def test_a_tau_jet_cannot_also_enter_the_b_pair(tmp_path):
    """A Delphes tau_h IS an AK4 jet; without overlap removal it lands on dr_bb."""
    d = _feats(tmp_path)
    tau_b_dr = S._dR(1.0, 1.4, 0.5, 0.0)
    assert d["dr_bb"][0] == pytest.approx(S._dR(0.5, 0.0, -0.4, 2.8), rel=1e-9)
    assert d["dr_bb"][0] != pytest.approx(tau_b_dr, rel=1e-6)


def test_events_without_a_light_lepton_are_dropped(tmp_path):
    """The CMS NSBI test used mt/et only — tau_h tau_h is out of scope here."""
    p = _write(tmp_path / "b.parquet")
    a = ak.from_parquet(str(p))
    a = ak.with_field(a, a["Muon"][:, :0], "Muon")     # remove the muon
    ak.to_parquet(a, str(p))
    d, _ = S.features(NtupleEvents(p))
    assert len(d["m_hh"]) == 0


def test_signal_trees_are_named_per_kappa_lambda(tmp_path):
    for kl, want in ((0.0, "tree_sbi_lam0"), (5.0, "tree_sbi_lam5")):
        name = (f"tree_sbi_lam{int(kl)}" if kl == int(kl)
                else "tree_sbi_lam" + str(kl).replace(".", "p"))
        assert name == want
    assert S.TREES["ttbar"] == "tree_ttbar" and S.TREES["dy"] == "tree_dy"
