"""The τ_hτ_h cutflow must localize a yield loss to the step that causes it.

Each test injects a loss at ONE known stage and asserts the cutflow's conditional
efficiency drops at that stage and only there — otherwise the diagnostic could point
at the wrong step and send the tuning effort somewhere useless.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tauh_cutflow import _cutflow  # noqa: E402

_N = 300
# two τ_h, two b-jets, all inside acceptance and mutually well separated
_T1, _T2 = (60.0, 0.5, 0.0), (45.0, -0.8, 2.2)
_B1, _B2 = (90.0, 1.0, -1.5), (55.0, -0.3, -2.8)


def _jag(vals):
    return ak.values_astype(ak.Array([list(vals)] * _N), np.float64)


def _jagi(vals):
    """Integer jagged array — gen pid/status/m1 are Int_t in real Delphes and NanoAOD,
    and the m1 chain walk indexes with them, so the fixture must not make them floats."""
    return ak.values_astype(ak.Array([list(vals)] * _N), np.int64)


def _col(rows, **extra):
    out = {"pt": _jag([r[0] for r in rows]), "eta": _jag([r[1] for r in rows]),
           "phi": _jag([r[2] for r in rows]), "mass": _jag([1.0] * len(rows))}
    for k, v in extra.items():
        out[k] = _jag(v)
    return ak.zip(out)


def _met():
    return ak.zip({"met": ak.Array(np.full(_N, 35.0)), "phi": ak.Array(np.full(_N, 1.0))})


_VIS_FRAC = 0.65        # visible / full gen τ pT; the rest is the ν_τ


def _delphes(*, vis_taus=(_T1, _T2), tau_jets=(_T1, _T2), tautag=(1, 1)):
    """Delphes view.

    ``vis_taus`` gives the VISIBLE gen τ kinematics; the gen record is written as the
    full τ plus its collinear ν_τ, because that is what a real record contains and what
    ``obs.gen_visible_taus`` reconstructs (τ − ν). Writing a bare τ would leave stage 1
    unable to build a visible τ at all.
    """
    jets = _col(list(tau_jets) + [_B1, _B2],
                tautag=list(tautag) + [0, 0], btag=[0] * len(tau_jets) + [1, 1],
                charge=[0] * (len(tau_jets) + 2))
    pid, status, m1, pt, eta, phi, mass = [], [], [], [], [], [], []
    for k, t in enumerate(vis_taus):
        sign = 1 if k % 2 == 0 else -1
        full = t[0] / _VIS_FRAC
        tau_at = len(pid)
        pid += [15 * sign, 16 * sign]
        status += [2, 1]
        m1 += [-1, tau_at]                       # the ν points at its own τ
        pt += [full, full - t[0]]
        eta += [t[1], t[1]]
        phi += [t[2], t[2]]
        mass += [1.777, 0.0]
    gen = ak.zip({"pid": _jagi(pid), "status": _jagi(status), "m1": _jagi(m1),
                  "pt": _jag(pt), "eta": _jag(eta), "phi": _jag(phi), "mass": _jag(mass)})
    empty = ak.zip({k: _jag([]) for k in ("pt", "eta", "phi", "charge")})
    return SimpleNamespace(jets=jets, gen=gen, genjets=_col(list(vis_taus)),
                           electrons=empty, muons=empty, met=_met())


def _nano(*, genvistau=(_T1, _T2), taus=(_T1, _T2), vsjet=(6, 6)):
    jets = _col([_T1, _T2, _B1, _B2], btag=[0, 0, 1, 1])
    ev = SimpleNamespace(jets=jets, genvistau=_col(list(genvistau)),
                         taus=_col(list(taus), vsjet=list(vsjet)),
                         electrons=ak.zip({k: _jag([]) for k in ("pt", "eta", "phi", "charge")}),
                         muons=ak.zip({k: _jag([]) for k in ("pt", "eta", "phi", "charge")}),
                         met=_met())
    ev.muons = ev.electrons
    ev.deeptau_medium = lambda: 5
    return ev


def _eff(cf):
    """Per-step conditional efficiencies."""
    return [cf[i] / cf[i - 1] if cf[i - 1] else 0.0 for i in range(1, len(cf))]


def test_full_efficiency_passes_every_step():
    for cf in (_cutflow(_delphes(), nano=False), _cutflow(_nano(), nano=True)):
        assert cf[0] == _N
        assert all(e == 1.0 for e in _eff(cf)[:4]), cf   # gen/reco/id/jets all pass


def test_loss_localises_to_the_gen_step():
    """A visible τ below acceptance -> step 1 drops, and it cascades (nothing recovers)."""
    cf = _cutflow(_delphes(vis_taus=((5.0, 0.5, 0.0), _T2)), nano=False)
    assert _eff(cf)[0] == 0.0        # step 1 kills it
    cf_n = _cutflow(_nano(genvistau=((5.0, 0.5, 0.0), _T2)), nano=True)
    assert _eff(cf_n)[0] == 0.0


def test_loss_localises_to_the_reco_step():
    """Visible τ exist but only one produces a reco object -> step 2, not step 1."""
    cf = _cutflow(_delphes(tau_jets=(_T1,), tautag=(1,)), nano=False)
    e = _eff(cf)
    assert e[0] == 1.0 and e[1] == 0.0, cf
    cf_n = _cutflow(_nano(taus=(_T1,), vsjet=(6,)), nano=True)
    e_n = _eff(cf_n)
    assert e_n[0] == 1.0 and e_n[1] == 0.0, cf_n


def test_loss_localises_to_the_id_step():
    """Objects reconstructed but failing the τ ID -> step 3, not step 2."""
    cf = _cutflow(_delphes(tautag=(1, 0)), nano=False)
    e = _eff(cf)
    assert e[0] == 1.0 and e[1] == 1.0 and e[2] == 0.0, cf
    cf_n = _cutflow(_nano(vsjet=(6, 3)), nano=True)      # 3 < Medium(5)
    e_n = _eff(cf_n)
    assert e_n[0] == 1.0 and e_n[1] == 1.0 and e_n[2] == 0.0, cf_n


def test_id_step_is_the_delphes_retag_bit():
    """The Delphes ID step reads TauTag, so a re-tag that fires less shows up there."""
    full = _cutflow(_delphes(tautag=(1, 1)), nano=False)
    half = _cutflow(_delphes(tautag=(1, 0)), nano=False)
    assert full[3] == _N and half[3] == 0


# --------------------------------------------------------------------------- #
# gen ΔR_ττ: separates a RECO difference from a SAMPLE difference
# --------------------------------------------------------------------------- #
def _dr(a, b):
    dphi = abs((a[2] - b[2] + np.pi) % (2 * np.pi) - np.pi)
    return float(np.hypot(a[1] - b[1], dphi))


def test_gen_dr_is_the_separation_of_the_two_visible_gen_taus():
    cf, d = _cutflow(_delphes(), nano=False, detail=True)
    assert cf[0] == _N
    assert np.isfinite(d["gen_dr"]).all()
    assert d["gen_dr"][0] == pytest.approx(_dr(_T1, _T2), abs=1e-6)


def test_gen_dr_is_nan_when_there_is_no_pair():
    """Fewer than two visible gen τ -> no ΔR to speak of, and it must not be faked."""
    cf, d = _cutflow(_delphes(vis_taus=(_T1,)), nano=False, detail=True)
    assert np.isnan(d["gen_dr"]).all()


def test_detail_masks_align_with_the_counts():
    cf, d = _cutflow(_delphes(), nano=False, detail=True)
    assert int(d["s1"].sum()) == cf[1]
    assert int(d["s5"].sum()) == cf[5]


def test_efficiency_vs_gen_dr_is_computable_on_both_sides():
    """The decisive panel: eff = stage5/stage1 in bins of gen ΔR, for each side."""
    _, dd = _cutflow(_delphes(), nano=False, detail=True)
    _, dn = _cutflow(_nano(), nano=True, detail=True)
    for d in (dd, dn):
        ok = np.isfinite(d["gen_dr"])
        assert ok.any()
        eff = d["s5"][ok].sum() / max(d["s1"][ok].sum(), 1)
        assert 0.0 <= eff <= 1.0


def test_default_return_is_unchanged():
    """detail=False keeps the plain list, so existing callers are untouched."""
    assert isinstance(_cutflow(_delphes(), nano=False), list)
