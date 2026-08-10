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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tauh_cutflow import _cutflow  # noqa: E402

_N = 300
# two τ_h, two b-jets, all inside acceptance and mutually well separated
_T1, _T2 = (60.0, 0.5, 0.0), (45.0, -0.8, 2.2)
_B1, _B2 = (90.0, 1.0, -1.5), (55.0, -0.3, -2.8)


def _jag(vals):
    return ak.values_astype(ak.Array([list(vals)] * _N), np.float64)


def _col(rows, **extra):
    out = {"pt": _jag([r[0] for r in rows]), "eta": _jag([r[1] for r in rows]),
           "phi": _jag([r[2] for r in rows]), "mass": _jag([1.0] * len(rows))}
    for k, v in extra.items():
        out[k] = _jag(v)
    return ak.zip(out)


def _met():
    return ak.zip({"met": ak.Array(np.full(_N, 35.0)), "phi": ak.Array(np.full(_N, 1.0))})


def _delphes(*, genjets=(_T1, _T2), tau_jets=(_T1, _T2), tautag=(1, 1)):
    """Delphes view. ``genjets`` = visible τ proxies, ``tau_jets`` = the reco jets."""
    jets = _col(list(tau_jets) + [_B1, _B2],
                tautag=list(tautag) + [0, 0], btag=[0] * len(tau_jets) + [1, 1],
                charge=[0] * (len(tau_jets) + 2))
    gen = _col([(t[0] * 1.4, t[1], t[2]) for t in (_T1, _T2)], pid=[15, -15])
    empty = ak.zip({k: _jag([]) for k in ("pt", "eta", "phi", "charge")})
    return SimpleNamespace(jets=jets, gen=gen, genjets=_col(list(genjets)),
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
    cf = _cutflow(_delphes(genjets=((5.0, 0.5, 0.0), _T2)), nano=False)
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
