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
