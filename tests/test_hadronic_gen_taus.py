"""Only HADRONIC gen τ may be treated as genuine τ_h.

``tau_eff`` is measured on the anchor as GenVisTau → DeepTau-Medium Tau, and
``GenVisTau`` exists only for hadronic decays. Treating every gen τ as genuine hands a
leptonic τ's jet that efficiency (~0.5) instead of the jet→τ_h fake rate (~0.004) — a
~125x over-efficiency that fabricates τ_hτ_h events out of τ_hτ_ℓ and τ_ℓτ_ℓ ones.
Those fakes are not collimated, so they land as a back-to-back ΔR_ττ shoulder and drag
m_ττ up.
"""

from __future__ import annotations

from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

from delphes_pipeline.core import observables as obs
from delphes_pipeline.tuning.maps import TuningMaps, retag_tautag

_N = 400
_HAD = (0.6, 0.0)       # (eta, phi) of the hadronic τ
_LEP = (-0.9, 2.4)      # (eta, phi) of the leptonic τ and its daughter muon


def _f(v):
    return ak.values_astype(ak.Array([list(v)] * _N), np.float64)


def _i(v):
    return ak.values_astype(ak.Array([list(v)] * _N), np.int64)


def _gen():
    """Two gen τ: index 0 decays hadronically, index 1 to a muon.

    The chain is status-1 μ → μ copy → τ, as in the real record, so the fix must walk
    past the self-copy to find the τ ancestor.
    """
    #            0: had τ   1: lep τ   2: μ copy   3: status-1 μ
    return ak.zip({
        "pid":    _i([15, -15, -13, -13]),
        "status": _i([2, 2, 23, 1]),
        "m1":     _i([-1, -1, 1, 2]),
        "pt":     _f([70.0, 60.0, 40.0, 35.0]),
        "eta":    _f([_HAD[0], _LEP[0], _LEP[0], _LEP[0]]),
        "phi":    _f([_HAD[1], _LEP[1], _LEP[1], _LEP[1]]),
        "mass":   _f([1.777, 1.777, 0.105, 0.105]),
    })


def test_leptonic_tau_is_excluded_hadronic_is_kept():
    gen = _gen()
    all_taus = obs.gen_taus(gen)
    had = obs.gen_taus(gen, hadronic_only=True)
    assert ak.to_list(ak.num(all_taus))[0] == 2
    assert ak.to_list(ak.num(had))[0] == 1
    assert ak.to_numpy(had.eta)[0][0] == pytest.approx(_HAD[0])


def test_retag_gives_the_leptonic_tau_jet_the_FAKE_rate():
    """A jet on the leptonic τ must be tagged at ~tau_mistag, not ~tau_eff."""
    jets = ak.zip({
        "pt": _f([70.0, 60.0]), "eta": _f([_HAD[0], _LEP[0]]), "phi": _f([_HAD[1], _LEP[1]]),
        "mass": _f([8.0, 8.0]), "tautag": _f([0, 0]), "btag": _f([0, 0]), "flavor": _f([0, 0]),
    })
    flat = lambda v: {"x": "pt", "centers": [25.0, 175.0], "values": [v, v], "counts": [10, 10]}
    maps = TuningMaps({"tau_eff": flat(1.0), "tau_mistag": flat(0.0)})   # separable by value
    tag = ak.to_numpy(retag_tautag(SimpleNamespace(jets=jets, gen=_gen()),
                                   maps, np.random.default_rng(0)))
    assert tag[:, 0].all(), "hadronic τ jet must get tau_eff (=1 here)"
    assert not tag[:, 1].any(), "leptonic τ jet must get tau_mistag (=0 here), not tau_eff"


def test_tau_efficiency_denominator_is_optional_hadronic_only():
    """The tuning lens restricts to hadronic τ (matching the anchor's GenVisTau); the
    closure does not, because Delphes' own TauTagging has its τ→ℓνν veto commented out."""
    jets = ak.zip({
        "pt": _f([70.0, 60.0]), "eta": _f([_HAD[0], _LEP[0]]), "phi": _f([_HAD[1], _LEP[1]]),
        "mass": _f([8.0, 8.0]), "tautag": _f([1, 1]), "btag": _f([0, 0]), "flavor": _f([0, 0]),
    })
    ev = SimpleNamespace(jets=jets, gen=_gen())
    incl = obs.tau_efficiency(ev, bins=[20, 100], hadronic_only=False)
    had = obs.tau_efficiency(ev, bins=[20, 100], hadronic_only=True)
    assert incl.counts.sum() == 2 * _N        # both τ enter
    assert had.counts.sum() == _N             # only the hadronic one


def test_tau_mistag_fake_sample_matches_the_anchor_definition():
    """The anchor vetoes GenVisTau (hadronic only), so a leptonic-τ jet is a fake
    candidate on the CMS side. The tuning lens must veto the same set, or the two
    sides compare different denominators."""
    jets = ak.zip({
        "pt": _f([70.0, 60.0, 50.0]),
        "eta": _f([_HAD[0], _LEP[0], 2.0]), "phi": _f([_HAD[1], _LEP[1], -1.0]),
        "mass": _f([8.0, 8.0, 8.0]), "tautag": _f([0, 0, 0]),
        "btag": _f([0, 0, 0]), "flavor": _f([0, 0, 0]),
    })
    ev = SimpleNamespace(jets=jets, gen=_gen())
    incl = obs.tau_mistag(ev, bins=[20, 100], hadronic_only=False)   # closure: veto all τ
    had = obs.tau_mistag(ev, bins=[20, 100], hadronic_only=True)     # tuning: veto hadronic
    # closure keeps only the unrelated jet; tuning also keeps the leptonic-τ jet
    assert incl.counts.sum() == _N
    assert had.counts.sum() == 2 * _N


# --------------------------------------------------------------------------- #
# visible gen τ = τ − ν_τ (the Delphes GenVisTau analogue)
# --------------------------------------------------------------------------- #
def _gen_with_nu(tau_pt=100.0, nu_frac=0.35):
    """A hadronic τ and its ν_τ, collinear, plus an unrelated ν_τ far away."""
    return ak.zip({
        #      0: had τ        1: its ν_τ            2: an unrelated ν_τ
        "pid":    _i([15, 16, 16]),
        "status": _i([2, 1, 1]),
        "m1":     _i([-1, 0, -1]),
        "pt":     _f([tau_pt, tau_pt * nu_frac, 50.0]),
        "eta":    _f([_HAD[0], _HAD[0], -2.0]),
        "phi":    _f([_HAD[1], _HAD[1], 3.0]),
        "mass":   _f([1.777, 0.0, 0.0]),
    })


def test_visible_tau_is_the_tau_minus_its_neutrino():
    """Collinear ν_τ carrying 35% of the pT -> the visible τ keeps the other 65%."""
    vis = obs.gen_visible_taus(_gen_with_nu(100.0, 0.35))
    assert ak.to_list(ak.num(vis))[0] == 1
    assert ak.to_numpy(vis.pt)[0][0] == pytest.approx(65.0, rel=1e-6)
    assert ak.to_numpy(vis.eta)[0][0] == pytest.approx(_HAD[0], abs=1e-6)


def test_visible_tau_is_softer_than_the_full_tau():
    """The whole point: profiling against the full τ (or a GenJet) is not the same
    reference as CMS GenVisTau, and the difference is large."""
    gen = _gen_with_nu(100.0, 0.35)
    full = obs.gen_taus(gen, hadronic_only=True)
    vis = obs.gen_visible_taus(gen)
    assert ak.to_numpy(vis.pt)[0][0] < ak.to_numpy(full.pt)[0][0]


def test_unrelated_neutrino_is_not_subtracted():
    """Only a ν_τ that descends from the τ AND is collinear with it counts."""
    gen = _gen_with_nu(100.0, 0.35)
    vis = obs.gen_visible_taus(gen)
    # the far-away ν (index 2, no τ ancestor) must not have been used
    assert ak.to_numpy(vis.pt)[0][0] == pytest.approx(65.0, rel=1e-6)


def test_leptonic_taus_have_no_visible_tau_entry():
    """gen_visible_taus is hadronic-only, matching GenVisTau: the leptonic τ is excluded
    even when it has a ν_τ of its own."""
    gen = ak.zip({          # 0: had τ, 1: its ν_τ, 2: lep τ, 3: its ν_τ, 4: the μ
        "pid":    _i([15, 16, -15, -16, -13]),
        "status": _i([2, 1, 2, 1, 1]),
        "m1":     _i([-1, 0, -1, 2, 2]),
        "pt":     _f([100.0, 35.0, 90.0, 30.0, 40.0]),
        "eta":    _f([_HAD[0], _HAD[0], _LEP[0], _LEP[0], _LEP[0]]),
        "phi":    _f([_HAD[1], _HAD[1], _LEP[1], _LEP[1], _LEP[1]]),
        "mass":   _f([1.777, 0.0, 1.777, 0.0, 0.105]),
    })
    vis = obs.gen_visible_taus(gen)
    assert ak.to_list(ak.num(vis))[0] == 1          # only the hadronic one
    assert ak.to_numpy(vis.pt)[0][0] == pytest.approx(65.0, rel=1e-6)


def test_tau_without_a_found_neutrino_is_dropped_not_faked():
    """No ν_τ found -> drop the τ. Falling back to the FULL τ would silently profile the
    energy response against a too-hard reference and bias tau_escale the wrong way."""
    gen = ak.zip({                       # a hadronic τ with no ν_τ anywhere
        "pid": _i([15]), "status": _i([2]), "m1": _i([-1]),
        "pt": _f([100.0]), "eta": _f([_HAD[0]]), "phi": _f([_HAD[1]]), "mass": _f([1.777]),
    })
    assert ak.to_list(ak.num(obs.gen_taus(gen, hadronic_only=True)))[0] == 1
    assert ak.to_list(ak.num(obs.gen_visible_taus(gen)))[0] == 0


# --------------------------------------------------------------------------- #
# descent beats proximity: a wide-angle daughter must not escape the veto
# --------------------------------------------------------------------------- #
def _gen_wide_leptonic(sep=0.9):
    """A hadronic τ (+ν) and a leptonic τ whose muon is ``sep`` away in eta."""
    return ak.zip({
        #      0: had τ  1: its ν  2: lep τ  3: its ν  4: the μ, far from its parent
        "pid":    _i([15, 16, -15, -16, -13]),
        "status": _i([2, 1, 2, 1, 1]),
        "m1":     _i([-1, 0, -1, 2, 2]),
        "pt":     _f([100.0, 35.0, 90.0, 30.0, 40.0]),
        "eta":    _f([0.5, 0.5, -1.0, -1.0, -1.0 + sep]),
        "phi":    _f([0.0, 0.0, 2.0, 2.0, 2.0]),
        "mass":   _f([1.777, 0.0, 1.777, 0.0, 0.105]),
    })


def test_descent_veto_catches_a_wide_angle_leptonic_tau():
    """The decisive case: the μ is 0.9 away, so a ΔR<0.4 proximity veto misses it and
    the leptonic τ is wrongly counted as hadronic. Descent cannot miss it."""
    gen = _gen_wide_leptonic(0.9)
    geo = obs.gen_taus(gen, hadronic_only=True, veto="geometric")
    des = obs.gen_taus(gen, hadronic_only=True, veto="descent")
    assert ak.to_list(ak.num(geo))[0] == 2, "proximity veto misses it (the bug)"
    assert ak.to_list(ak.num(des))[0] == 1, "descent veto catches it (the fix)"


def test_both_vetoes_agree_when_the_daughter_is_collinear():
    gen = _gen_wide_leptonic(0.05)
    for v in ("geometric", "descent"):
        assert ak.to_list(ak.num(obs.gen_taus(gen, hadronic_only=True, veto=v)))[0] == 1


def test_tau_ancestor_index_points_at_the_right_tau():
    gen = _gen_wide_leptonic(0.9)
    anc = ak.to_numpy(obs.tau_ancestor_index(gen))[0]
    assert anc[4] == 2, "the μ descends from the τ at index 2"
    assert anc[1] == 0, "the ν_τ descends from the τ at index 0"
    assert anc[0] == -1 and anc[2] == -1, "the τ themselves descend from no τ"


def test_visible_taus_use_the_descent_veto_by_default():
    gen = _gen_wide_leptonic(0.9)
    assert ak.to_list(ak.num(obs.gen_visible_taus(gen)))[0] == 1


def test_unknown_veto_is_rejected():
    with pytest.raises(ValueError, match="descent.*geometric"):
        obs.gen_taus(_gen_wide_leptonic(), hadronic_only=True, veto="bogus")
