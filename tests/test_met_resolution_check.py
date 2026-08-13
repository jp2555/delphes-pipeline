"""The resolution check must see anisotropy and activity-dependence where they exist.

``met_smear`` adds ONE isotropic, flat-in-activity Gaussian width. The real pT_miss
resolution is larger along the hadronic recoil and grows with event activity, and because
m_ττ = m_vis/sqrt(x1x2) is nonlinear, mis-modelled noise moves the m_ττ MEAN, not only its
width. So the estimator has to detect both properties — each is injected here and the
measurement must recover it, and must NOT report either when it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from met_resolution_check import _ht, _width, measure  # noqa: E402

_N = 3000


def _sample(sig_para, sig_perp, *, ht_slope=0.0, seed=0):
    """τ_hτ_h events whose pT_miss residual has a prescribed width along/across the
    di-τ axis, optionally growing with HT."""
    rng = np.random.default_rng(seed)
    gpt = rng.uniform(30.0, 80.0, (_N, 2))
    phi = np.stack([rng.uniform(-3.0, 3.0, _N), rng.uniform(-3.0, 3.0, _N)], axis=1)
    eta = rng.uniform(-1.2, 1.2, (_N, 2))
    rpt = gpt * 1.05
    # extra jets carry the activity
    nj_pt = rng.uniform(25.0, 260.0, (_N, 2))
    ht = rpt.sum(axis=1) + nj_pt.sum(axis=1)
    scale = 1.0 + ht_slope * (ht - ht.mean()) / 100.0

    dx = ((rpt - gpt) * np.cos(phi)).sum(axis=1)
    dy = ((rpt - gpt) * np.sin(phi)).sum(axis=1)
    d = np.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    a = rng.normal(0, 1, _N) * sig_para * scale
    b = rng.normal(0, 1, _N) * sig_perp * scale
    rx, ry = a * ux - b * uy, a * uy + b * ux       # rotate back to lab frame
    gmx, gmy = rng.normal(0, 20, _N), rng.normal(0, 20, _N)

    jag = lambda x: ak.Array(x.tolist())
    jets = ak.zip({"pt": jag(np.concatenate([rpt, nj_pt], axis=1)),
                   "eta": jag(np.concatenate([eta, rng.uniform(-3, 3, (_N, 2))], axis=1)),
                   "phi": jag(np.concatenate([phi, rng.uniform(-3, 3, (_N, 2))], axis=1)),
                   "mass": jag(np.full((_N, 4), 2.0)),
                   "tautag": ak.Array([[1, 1, 0, 0]] * _N),
                   "btag": ak.Array([[0, 0, 0, 0]] * _N)})
    full = gpt / 0.65
    gen = ak.zip({"pid": ak.Array([[15, 16, -15, -16]] * _N),
                  "status": ak.Array([[2, 1, 2, 1]] * _N),
                  "m1": ak.Array([[-1, 0, -1, 2]] * _N),
                  "pt": jag(np.stack([full[:, 0], full[:, 0] - gpt[:, 0],
                                      full[:, 1], full[:, 1] - gpt[:, 1]], axis=1)),
                  "eta": jag(np.stack([eta[:, 0], eta[:, 0], eta[:, 1], eta[:, 1]], axis=1)),
                  "phi": jag(np.stack([phi[:, 0], phi[:, 0], phi[:, 1], phi[:, 1]], axis=1)),
                  "mass": ak.Array([[1.777, 0.0, 1.777, 0.0]] * _N)})
    mk = lambda x, y: ak.zip({"met": np.hypot(x, y), "eta": np.zeros(_N),
                              "phi": np.arctan2(y, x)})
    return SimpleNamespace(jets=jets, gen=gen, n=_N,
                           met=mk(gmx + rx, gmy + ry), genmet=mk(gmx, gmy))


def _sigmas(ev):
    para, perp, ht = measure(ev, nano=False)
    assert para.size > 500
    return _width(para)[0], _width(perp)[0], ht


def test_isotropic_noise_is_reported_isotropic():
    """A round smearing — what met_smear actually applies — must read as ratio 1."""
    a, b, _ = _sigmas(_sample(20.0, 20.0))
    assert a / b == pytest.approx(1.0, abs=0.10), (a, b)


def test_anisotropic_noise_is_detected():
    """The real resolution is larger along the recoil; the estimator must see that."""
    a, b, _ = _sigmas(_sample(30.0, 15.0, seed=1))
    assert a / b > 1.6, (a, b)


def test_absolute_widths_are_recovered():
    a, b, _ = _sigmas(_sample(25.0, 12.0, seed=2))
    assert a == pytest.approx(25.0, rel=0.15) and b == pytest.approx(12.0, rel=0.20)


def test_activity_dependence_is_detected_when_present():
    ev = _sample(20.0, 20.0, ht_slope=0.35, seed=3)
    para, _, ht = measure(ev, nano=False)
    lo = _width(para[ht < np.percentile(ht, 35)])[0]
    hi = _width(para[ht > np.percentile(ht, 65)])[0]
    assert hi > lo * 1.3, (lo, hi)


def test_flat_noise_shows_no_activity_dependence():
    """The control: met_smear is flat in activity, and must be reported flat — otherwise
    the measurement would manufacture the very defect it is looking for."""
    ev = _sample(20.0, 20.0, ht_slope=0.0, seed=4)
    para, _, ht = measure(ev, nano=False)
    lo = _width(para[ht < np.percentile(ht, 35)])[0]
    hi = _width(para[ht > np.percentile(ht, 65)])[0]
    assert abs(hi / lo - 1.0) < 0.15, (lo, hi)


def test_ht_uses_jets_not_the_stored_sumet():
    """ScalarHT and NanoAOD sumEt are defined differently across the tiers (see
    anchor._nano_met_resolution), so binning by them would compare different variables."""
    ev = _sample(20.0, 20.0, seed=5)
    ht = _ht(ev.jets)
    assert ht.shape == (_N,)
    assert np.all(ht > 0) and np.median(ht) > 100.0


def test_width_is_robust_to_tails():
    """A few catastrophic events must not set the quoted resolution."""
    rng = np.random.default_rng(0)
    core = rng.normal(0, 20.0, 20000)
    spiked = np.concatenate([core, rng.normal(0, 300.0, 200)])
    assert _width(spiked)[0] == pytest.approx(_width(core)[0], rel=0.10)
