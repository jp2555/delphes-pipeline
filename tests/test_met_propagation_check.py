"""The MET-propagation measurement must distinguish two opposite pictures.

Whether pT_miss follows the τ energy redraw decides ~1/3 of the m_ττ correction, so the
test injects each picture into a synthetic sample and asserts the measurement recovers it:

  * MISSING ENERGY   — the τ excess really was lost, so pT_miss should have been larger
                       by exactly that vector. Signature: slope −1.
  * CONE DEFINITION  — the excess is UE the R=0.4 cone swept up. pT_miss already sums it
                       as visible energy, so re-labelling changes nothing. Signature: 0.

If the estimator cannot separate these it is worthless, since those are the two answers.
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
from met_propagation_check import measure  # noqa: E402

_N = 1500


def _sample(picture, seed=0, excess=0.18):
    """Two τ_h whose reco pT exceeds the gen visible pT by ``excess``; pT_miss built
    according to ``picture``."""
    rng = np.random.default_rng(seed)
    gpt = rng.uniform(25.0, 90.0, (_N, 2))
    phi = np.stack([rng.uniform(-3.0, 3.0, _N), rng.uniform(-3.0, 3.0, _N)], axis=1)
    eta = rng.uniform(-1.5, 1.5, (_N, 2))
    rpt = gpt * (1.0 + excess * rng.uniform(0.4, 1.6, (_N, 2)))

    dx = ((rpt - gpt) * np.cos(phi)).sum(axis=1)
    dy = ((rpt - gpt) * np.sin(phi)).sum(axis=1)
    gmx, gmy = rng.normal(0, 25, _N), rng.normal(0, 25, _N)     # true missing (neutrinos)
    noise_x, noise_y = rng.normal(0, 4, _N), rng.normal(0, 4, _N)
    if picture == "missing":
        mx, my = gmx - dx + noise_x, gmy - dy + noise_y   # pT_miss short by exactly D
    elif picture == "cone":
        mx, my = gmx + noise_x, gmy + noise_y             # pT_miss unaffected
    else:
        raise ValueError(picture)

    jag = lambda a: ak.Array(a.tolist())
    jets = ak.zip({"pt": jag(rpt), "eta": jag(eta), "phi": jag(phi),
                   "mass": jag(np.full((_N, 2), 2.0)),
                   "tautag": jag(np.ones((_N, 2), dtype=np.int64)),
                   "btag": jag(np.zeros((_N, 2), dtype=np.int64))})
    # gen record: each visible τ as (τ, ν) so gen_visible_taus rebuilds pT = gpt
    full = gpt / 0.65
    gen = ak.zip({
        "pid": ak.Array([[15, 16, -15, -16]] * _N),
        "status": ak.Array([[2, 1, 2, 1]] * _N),
        "m1": ak.Array([[-1, 0, -1, 2]] * _N),
        "pt": jag(np.stack([full[:, 0], full[:, 0] - gpt[:, 0],
                            full[:, 1], full[:, 1] - gpt[:, 1]], axis=1)),
        "eta": jag(np.stack([eta[:, 0], eta[:, 0], eta[:, 1], eta[:, 1]], axis=1)),
        "phi": jag(np.stack([phi[:, 0], phi[:, 0], phi[:, 1], phi[:, 1]], axis=1)),
        "mass": ak.Array([[1.777, 0.0, 1.777, 0.0]] * _N)})
    mk = lambda x, y: ak.zip({"met": np.hypot(x, y), "eta": np.zeros(_N),
                              "phi": np.arctan2(y, x)})
    return SimpleNamespace(jets=jets, gen=gen, n=_N, met=mk(mx, my), genmet=mk(gmx, gmy))


def _slope(picture, seed=0):
    d, para, _ = measure(_sample(picture, seed))
    assert d.size > 200, d.size
    return float(np.polyfit(d, para, 1)[0]), d, para


def test_missing_energy_picture_gives_slope_minus_one():
    s, _, _ = _slope("missing")
    assert s == pytest.approx(-1.0, abs=0.15), s


def test_cone_definition_picture_gives_slope_zero():
    s, _, _ = _slope("cone")
    assert abs(s) < 0.15, s


def test_the_two_pictures_are_clearly_separated():
    """They must not be within noise of each other, or the measurement decides nothing."""
    a, _, _ = _slope("missing")
    b, _, _ = _slope("cone")
    assert a < -0.6 and abs(b) < 0.2 and (b - a) > 0.6


def test_perpendicular_control_is_flat_under_both_pictures():
    """The control must not respond to either injection — a slope there would mean the
    estimator is manufacturing the signal rather than measuring it."""
    for picture in ("missing", "cone"):
        d, _, perp = measure(_sample(picture, seed=3))
        assert abs(float(np.polyfit(d, perp, 1)[0])) < 0.15, picture
        assert abs(float(np.mean(perp))) < 6.0, picture


def test_only_gen_matched_pairs_are_used():
    """An unmatched leg has no gen reference, so its 'excess' is meaningless."""
    ev = _sample("cone")
    ev.gen = ak.zip({k: ev.gen[k][:, :0] for k in ak.fields(ev.gen)})   # no gen at all
    d, para, perp = measure(ev)
    assert d.size == 0 and para.size == 0 and perp.size == 0


def _nano_sample(picture, seed=0, excess=0.18):
    """The anchor's view of the same event: τ from the Tau collection, gen from GenVisTau."""
    d = _sample(picture, seed, excess)
    n2 = len(d.jets)
    taus = ak.zip({"pt": d.jets.pt, "eta": d.jets.eta, "phi": d.jets.phi,
                   "mass": d.jets.mass,
                   "vsjet": ak.Array([[6, 6]] * n2)})
    vis = obs_gen_visible(d.gen)
    ev = SimpleNamespace(taus=taus, genvistau=vis, met=d.met, genmet=d.genmet, n=n2)
    ev.deeptau_medium = lambda: 5
    return ev


def obs_gen_visible(gen):
    from delphes_pipeline.core import observables as obs
    return obs.gen_visible_taus(gen, dr=0.4)


def test_anchor_mode_reproduces_the_delphes_answer_on_the_same_event():
    """The two tiers must be measured identically, or comparing their offsets is
    meaningless — that comparison is the whole point of the anchor mode."""
    for picture, want in (("missing", -1.0), ("cone", 0.0)):
        ev = _nano_sample(picture, seed=5)
        d, para, _ = measure(ev, nano=True)
        assert d.size > 200
        assert float(np.polyfit(d, para, 1)[0]) == pytest.approx(want, abs=0.15), picture


def test_anchor_mode_applies_the_deeptau_working_point():
    ev = _nano_sample("cone", seed=6)
    ev.taus = ak.with_field(ev.taus, ak.Array([[1, 1]] * ev.n), "vsjet")   # all fail Medium
    d, _, _ = measure(ev, nano=True)
    assert d.size == 0
