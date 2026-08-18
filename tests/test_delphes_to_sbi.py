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
    sel = {k: kw.pop(k) for k in ("btag_min", "lep_veto") if k in kw}
    return S.features(NtupleEvents(_write(tmp_path / "a.parquet", **kw)), **sel)[0]


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
    d, _, _ = S.features(NtupleEvents(p))
    assert len(d["m_hh"]) == 0


def test_signal_trees_are_named_per_kappa_lambda(tmp_path):
    for kl, want in ((0.0, "tree_sbi_lam0"), (5.0, "tree_sbi_lam5")):
        name = (f"tree_sbi_lam{int(kl)}" if kl == int(kl)
                else "tree_sbi_lam" + str(kl).replace(".", "p"))
        assert name == want
    assert S.TREES["ttbar"] == "tree_ttbar" and S.TREES["dy"] == "tree_dy"


def test_float32_inputs_do_not_produce_infinite_eta():
    """The ntuple is float32, where `1 - 1e-10` rounds to 1.0 and the clip stops
    guarding: arctanh(1.0) = inf, and the event is then silently dropped."""
    f32 = np.float32
    a = (f32(50.0), f32(0.0), f32(0.0), f32(5.0))
    b = (f32(50.0), f32(0.0), f32(np.pi), f32(5.0))    # exactly back-to-back -> p_z = 0
    pt, eta, phi, m = S._sum_p4(a, b)
    assert np.all(np.isfinite([pt, eta, phi, m])), (pt, eta, phi, m)


def test_a_collinear_pair_keeps_a_finite_eta():
    """Large |eta| with tiny transverse momentum is where pz/p -> 1."""
    f32 = np.float32
    leg = (f32(0.001), f32(6.0), f32(0.0), f32(0.0))
    pt, eta, phi, m = S._sum_p4(leg, leg)
    assert np.isfinite(eta) and abs(eta) < 20.0


def test_the_warning_free_path_gives_the_same_masses_as_before():
    """The float64 cast must not move any physics."""
    a = (90.0, 0.5, 0.0, 8.0)
    b = (70.0, -0.4, 2.8, 7.0)
    assert S._sum_p4(a, b)[3] == pytest.approx(
        np.sqrt(max((S._p4(*a)[3] + S._p4(*b)[3]) ** 2
                    - sum((S._p4(*a)[i] + S._p4(*b)[i]) ** 2 for i in range(3)), 0.0)),
        rel=1e-12)


# --------------------------------------------------------------------------- #
# dataset_id must travel with the events: ttbar spans three decay channels at
# 98 / 420 / 406 pb, and the -1 sentinel marks events with no defined xsec.
# --------------------------------------------------------------------------- #
def _with_ids(path, ids):
    n = len(ids)
    p = _write(path, n=n)
    a = ak.from_parquet(str(p))
    a = ak.with_field(a, np.array(ids, dtype=np.int16), "dataset_id")
    ak.to_parquet(a, str(p))
    return p


def test_dataset_id_is_carried_into_the_sbi_output(tmp_path):
    p = _with_ids(tmp_path / "a.parquet", [0, 1, 2, 0])
    d, _, _ = S.features(NtupleEvents(p))
    assert "dataset_id" in d
    assert sorted(set(d["dataset_id"].astype(int))) == [0, 1, 2]


def test_mixed_dataset_events_are_dropped_not_normalised(tmp_path):
    """-1 means the shard straddled two primary datasets: no defined cross section."""
    p = _with_ids(tmp_path / "a.parquet", [0, -1, 1, -1])
    d, _, dropped_mixed = S.features(NtupleEvents(p))
    assert dropped_mixed == 2
    assert -1 not in set(d["dataset_id"].astype(int))
    assert len(d["m_hh"]) == 2


def test_a_sample_without_dataset_id_is_unaffected(tmp_path):
    d, _, dm = S.features(NtupleEvents(_write(tmp_path / "b.parquet", n=3)))
    assert "dataset_id" not in d and dm == 0


# --------------------------------------------------------------------------- #
# The merged ttbar sample is 298M events across 145 GB. Materialising it in one
# awkward array is what made this step run for hours; GenPart alone is ~99% of
# the bytes and these features never touch it.
# --------------------------------------------------------------------------- #
def test_streaming_over_files_matches_a_single_pass(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    for i in range(3):
        _write(d / f"ttbar.{i:04d}.parquet", n=4)
    one, _, _ = S.features(NtupleEvents(d))
    many, _, _, n_read = S.features_streamed(d)
    assert n_read == 12
    assert len(many["m_hh"]) == len(one["m_hh"])
    for k in one:
        assert np.allclose(sorted(many[k]), sorted(one[k])), k


def test_genpart_is_not_among_the_columns_read():
    """It is ~99% of the ntuple and no SBI feature uses it."""
    assert "GenPart" not in S.COLUMNS
    assert {"Jet", "Electron", "Muon", "genWeight"} <= set(S.COLUMNS)


def test_the_reader_honours_a_column_subset(tmp_path):
    p = _write(tmp_path / "a.parquet", n=3)
    ev = NtupleEvents(p, columns=["Jet", "Electron", "Muon", "genWeight"])
    assert "GenPart" not in ak.fields(ev.array)
    assert ev.n == 3 and len(ev.jets) == 3


def test_a_kl_selection_still_works_with_a_column_subset(tmp_path):
    """kl must be added to the projection even when the caller forgets it."""
    p = _write(tmp_path / "a.parquet", n=6, kl=5.0)
    ev = NtupleEvents(p, kl=5.0, columns=["Jet", "Electron", "Muon", "genWeight"])
    assert ev.n == 6


def test_streaming_skips_files_holding_no_events_of_the_requested_kl(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    _write(d / "signal.0000.parquet", n=4, kl=0.0)
    _write(d / "signal.0001.parquet", n=4, kl=5.0)
    _, _, _, n_read = S.features_streamed(d, kl=5.0)
    assert n_read == 4, "the kl=0 file must contribute nothing"


# --------------------------------------------------------------------------- #
# The converter applies a PRESELECTION, not the CMS analysis selection. Making
# that explicit — and giving the knobs to tighten it — is what lets the flat
# file be studied post-hoc instead of guessed at from yields.
# --------------------------------------------------------------------------- #
def test_by_default_an_event_with_no_btag_still_passes(tmp_path):
    """This is why signal/ttbar acceptance is ~2.5 and not hundreds."""
    p = _write(tmp_path / "a.parquet", n=3)
    a = ak.from_parquet(str(p))
    j = a["Jet"]
    a = ak.with_field(a, ak.with_field(j, ak.zeros_like(j.btag), "btag"), "Jet")
    ak.to_parquet(a, str(p))
    d, _, _ = S.features(NtupleEvents(p))
    assert len(d["m_hh"]) == 3, "an untagged event passes the preselection"
    assert set(d["n_btag"]) == {0.0}


def test_btag_min_two_rejects_untagged_events(tmp_path):
    p = _write(tmp_path / "a.parquet", n=3)
    a = ak.from_parquet(str(p))
    j = a["Jet"]
    a = ak.with_field(a, ak.with_field(j, ak.zeros_like(j.btag), "btag"), "Jet")
    ak.to_parquet(a, str(p))
    d, _, _ = S.features(NtupleEvents(p), btag_min=2)
    assert len(d["m_hh"]) == 0


def test_btag_min_two_keeps_a_double_tagged_event(tmp_path):
    d = _feats(tmp_path, n=2, btag_min=2)
    assert len(d["m_hh"]) == 2 and set(d["btag_1"]) == {1.0}


def test_the_channel_label_distinguishes_mt_from_et(tmp_path):
    """Without it the flat file cannot be split into mt/et after the fact."""
    d = _feats(tmp_path, n=2)
    assert set(d["channel"]) == {float(S.CHANNEL["mt"])}, "fixture has muons"


def test_the_table_61_extras_are_present(tmp_path):
    d = _feats(tmp_path, n=2)
    for f in ("pt_l1", "pt_b1", "pt_vis", "btag_1", "btag_2", "n_jets", "n_btag",
              "channel", "mt_tot", "met"):
        assert f in d, f


def test_lepton_veto_rejects_an_event_with_two_leptons(tmp_path):
    p = _write(tmp_path / "a.parquet", n=2)
    a = ak.from_parquet(str(p))
    m = a["Muon"]
    a = ak.with_field(a, ak.concatenate([m, m], axis=1), "Muon")
    ak.to_parquet(a, str(p))
    assert len(S.features(NtupleEvents(p))[0]["m_hh"]) == 2
    assert len(S.features(NtupleEvents(p), lep_veto=True)[0]["m_hh"]) == 0


def test_btag_branches_are_bits_not_scores(tmp_path):
    """Delphes has no tagger, only a parameterised efficiency: btag is 0/1, NOT UParT."""
    d = _feats(tmp_path, n=2)
    assert set(d["btag_1"]) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# CMS HIG-25-008 resolved selection (Table 1, Eqs. 2-4). Pins the thresholds
# against the paper, and pins what Delphes CANNOT do so it stays visible.
# --------------------------------------------------------------------------- #
def test_the_thresholds_match_table_1():
    assert S.CMS_SEL["mt"] == {"lep_pt": 22.0, "lep_eta": 2.4, "tau_pt": 32.0}
    assert S.CMS_SEL["et"] == {"lep_pt": 25.0, "lep_eta": 2.5, "tau_pt": 35.0}
    assert (S.CMS_TAU_ETA, S.CMS_PAIR_DR) == (2.5, 0.5)
    assert (S.CMS_JET_PT, S.CMS_JET_ETA, S.CMS_JET_DR) == (20.0, 2.5, 0.5)


def test_the_ellipse_matches_equations_2_and_3():
    assert S.CMS_ELLIPSE["mt"] == ((116.0, 61.0), (114.0, 228.0))
    assert S.CMS_ELLIPSE["et"] == ((119.0, 57.0), (109.0, 232.0))


def _cms_event(tmp, mu_pt=40.0, tau_pt=45.0, n=4, extra_mu=False):
    jets = [[{"pt": 90.0, "eta": 0.5, "phi": 0.0, "mass": 8.0, "btag": 1,
              "tautag": 0, "hadronFlavour": 5},
             {"pt": 70.0, "eta": -0.4, "phi": 2.8, "mass": 7.0, "btag": 1,
              "tautag": 0, "hadronFlavour": 5},
             {"pt": tau_pt, "eta": 1.0, "phi": 1.4, "mass": 1.2, "btag": 0,
              "tautag": 1, "hadronFlavour": 0}]] * n
    mus = [{"pt": mu_pt, "eta": -0.2, "phi": 2.0, "charge": -1}]
    if extra_mu:
        mus = mus + [{"pt": 30.0, "eta": 0.3, "phi": 0.5, "charge": 1}]
    f = {"Jet": ak.Array(jets),
         "Tau": ak.Array([[{"pt": tau_pt, "eta": 1.0, "phi": 1.4, "mass": 1.2}]] * n),
         "Electron": ak.Array([[{"pt": 9.0, "eta": 0.0, "phi": 0.0,
                                "charge": 1}]] * n)[:, :0],
         "Muon": ak.Array([mus] * n),
         "MET_pt": np.full(n, 55.0, dtype=np.float32),
         "MET_phi": np.full(n, 1.0, dtype=np.float32),
         "genWeight": np.ones(n, dtype=np.float32)}
    p = tmp / "cms.parquet"
    ak.to_parquet(ak.zip(f, depth_limit=1), str(p))
    return p


def test_a_passing_mu_tau_event_is_kept_and_labelled_mt(tmp_path):
    p = _cms_event(tmp_path)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False)
    assert len(d["m_hh"]) == 4
    assert set(d["channel"]) == {float(S.CHANNEL["mt"])}


def test_a_soft_tau_fails_the_channel_threshold(tmp_path):
    """mu-tau_h requires tau_h pT > 32; 25 is below it."""
    p = _cms_event(tmp_path, tau_pt=25.0)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False)
    assert len(d["m_hh"]) == 0


def test_a_soft_muon_fails_the_cross_trigger_threshold(tmp_path):
    p = _cms_event(tmp_path, mu_pt=18.0)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False)
    assert len(d["m_hh"]) == 0


def test_the_additional_lepton_veto_rejects_a_second_muon(tmp_path):
    p = _cms_event(tmp_path, extra_mu=True)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False)
    assert len(d["m_hh"]) == 0


def test_the_ellipse_rejects_events_far_from_the_higgs_masses(tmp_path):
    """The visible di-tau mass here is ~60 GeV and m_bb ~200; the SR is centred at
    (116, 114) with the FastMTT mass, so this must be cut."""
    p = _cms_event(tmp_path, mu_pt=200.0, tau_pt=200.0)
    kept = len(S.features(NtupleEvents(p), cms=True, ellipse=True)[0]["m_hh"])
    loose = len(S.features(NtupleEvents(p), cms=True, ellipse=False)[0]["m_hh"])
    assert loose > 0 and kept < loose


def test_the_cms_selection_is_stricter_than_the_preselection(tmp_path):
    p = _cms_event(tmp_path, mu_pt=25.0, tau_pt=25.0)
    pre = len(S.features(NtupleEvents(p))[0]["m_hh"])
    cms = len(S.features(NtupleEvents(p), cms=True, ellipse=False)[0]["m_hh"])
    assert pre == 4 and cms == 0, "tau_h pT 25 < 32 passes preselection, fails CMS"


def test_what_delphes_cannot_apply_is_documented_not_silently_skipped():
    doc = S.cms_select.__doc__
    for missing in ("opposite charge", "isolation", "HH-BTAG", "boosted", "trigger"):
        assert missing in doc, missing


# --------------------------------------------------------------------------- #
# The CMS converter maps the BRANCH mass_tautaubb -> m_hh; ours is computed from
# the VISIBLE tautau system. Those are different observables, and a config cut of
# m_hh > 250 behaves completely differently on the two -- removing 42% of the
# kl=5 basis sample against 9% of kl=1, which distorts the morphing basis.
# --------------------------------------------------------------------------- #
def test_the_fastmtt_corrected_masses_are_carried_alongside(tmp_path):
    p = _cms_event(tmp_path)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False)
    for f in ("m_tautau_fastmtt", "m_hh_fastmtt", "pt_hh_fastmtt"):
        assert f in d, f


def test_the_corrected_masses_exceed_the_visible_ones(tmp_path):
    """FastMTT divides by x1 x2 <= 1, so it can only raise the mass."""
    p = _cms_event(tmp_path)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False)
    ok = np.isfinite(d["m_tautau_fastmtt"])
    assert ok.any()
    assert np.all(d["m_tautau_fastmtt"][ok] >= d["m_tautau"][ok] - 1e-6)
    assert np.all(d["m_hh_fastmtt"][ok] >= d["m_hh"][ok] - 1e-6)


def test_the_visible_definitions_are_unchanged(tmp_path):
    """m_hh / m_tautau must stay the CMS converter's visible ones; nothing replaced."""
    p = _cms_event(tmp_path)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False)
    lep = (40.0, -0.2, 2.0, 0.0)
    tau = (45.0, 1.0, 1.4, 1.2)
    assert d["m_tautau"][0] == pytest.approx(S._sum_p4(lep, tau)[3], rel=1e-9)


# --------------------------------------------------------------------------- #
# tau_h tau_h is CMS's most sensitive channel and a lepton-requiring converter
# drops it silently.
# --------------------------------------------------------------------------- #
def _tt_event(tmp, n=3, tau_pt=(60.0, 45.0)):
    jets = [[{"pt": 90.0, "eta": 0.5, "phi": 0.0, "mass": 8.0, "btag": 1,
              "tautag": 0, "hadronFlavour": 5},
             {"pt": 70.0, "eta": -0.4, "phi": 2.8, "mass": 7.0, "btag": 1,
              "tautag": 0, "hadronFlavour": 5},
             {"pt": tau_pt[0], "eta": 1.0, "phi": 1.4, "mass": 1.2, "btag": 0,
              "tautag": 1, "hadronFlavour": 0},
             {"pt": tau_pt[1], "eta": -1.1, "phi": 4.0, "mass": 1.1, "btag": 0,
              "tautag": 1, "hadronFlavour": 0}]] * n
    empty = ak.Array([[{"pt": 9.0, "eta": 0.0, "phi": 0.0, "charge": 1}]] * n)[:, :0]
    f = {"Jet": ak.Array(jets), "Electron": empty, "Muon": empty,
         "Tau": ak.Array([[{"pt": tau_pt[0], "eta": 1.0, "phi": 1.4, "mass": 1.2}]] * n),
         "MET_pt": np.full(n, 60.0, dtype=np.float32),
         "MET_phi": np.full(n, 1.0, dtype=np.float32),
         "genWeight": np.ones(n, dtype=np.float32)}
    p = tmp / "tt.parquet"
    ak.to_parquet(ak.zip(f, depth_limit=1), str(p))
    return p


def test_a_tau_h_tau_h_event_is_selected_and_labelled(tmp_path):
    p = _tt_event(tmp_path)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False,
                         channels=("mt", "et", "tt"))
    assert len(d["m_hh"]) == 3
    assert set(d["channel"]) == {float(S.CHANNEL["tt"])}


def test_a_soft_second_tau_fails_the_double_tau_threshold(tmp_path):
    """Table 1: the double-tau trigger requires pT > 40 on BOTH legs."""
    p = _tt_event(tmp_path, tau_pt=(60.0, 30.0))
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False,
                         channels=("mt", "et", "tt"))
    assert len(d["m_hh"]) == 0


def test_tau_h_tau_h_is_absent_from_the_loose_preselection(tmp_path):
    """The preselection requires a light lepton, so it drops the channel entirely."""
    p = _tt_event(tmp_path)
    assert len(S.features(NtupleEvents(p))[0]["m_hh"]) == 0


def test_a_leptonic_event_is_not_classified_as_tau_h_tau_h(tmp_path):
    """Sec. 5 priority: a muon makes it mt even when two tau_h are present."""
    p = _cms_event(tmp_path)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False)
    assert set(d["channel"]) == {float(S.CHANNEL["mt"])}


def test_tau_h_tau_h_is_off_by_default(tmp_path):
    """The CMS NSBI test is semi-leptonic only; including tt would make the Delphes
    sample cover a final state the comparison does not."""
    p = _tt_event(tmp_path)
    assert len(S.features(NtupleEvents(p), cms=True, ellipse=False)[0]["m_hh"]) == 0


def test_tau_h_tau_h_can_be_enabled_explicitly(tmp_path):
    p = _tt_event(tmp_path)
    d, _, _ = S.features(NtupleEvents(p), cms=True, ellipse=False,
                         channels=("mt", "et", "tt"))
    assert len(d["m_hh"]) == 3 and set(d["channel"]) == {float(S.CHANNEL["tt"])}
