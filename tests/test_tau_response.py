"""The τ response panel must isolate SHAPE from SCALE, and expose the gate asymmetry.

``tau_escale`` is a per-pT-bin median ratio applied multiplicatively: it can align
medians and nothing else. So the study only means anything if (a) a pure scale difference
vanishes once each side is divided by its own median, and (b) a one-sided tail does NOT.
Those two are asserted directly here on constructed responses.

The third fixture pins the derivation asymmetry the script was written to expose: the
anchor gates its acceptance on the GEN visible τ while the Delphes side gates on the RECO
jet, so the two measure the response on different populations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tau_response import _response  # noqa: E402

_N = 400


def _col(pt, eta, phi, **extra):
    out = {"pt": ak.Array([[p] for p in pt]), "eta": ak.Array([[e] for e in eta]),
           "phi": ak.Array([[f] for f in phi])}
    for k, v in extra.items():
        out[k] = ak.Array([[x] for x in v])
    return ak.zip(out)


def _pair(reco_pt, gen_pt=None):
    """One gen τ and one reco object per event, collinear so they always match."""
    gen_pt = np.full(_N, 50.0) if gen_pt is None else gen_pt
    eta = np.zeros(_N)
    phi = np.linspace(-3.0, 3.0, _N)
    return _col(gen_pt, eta, phi), _col(reco_pt, eta, phi)


def test_pure_scale_difference_vanishes_under_median_normalisation():
    """Two sides differing ONLY by a scale must be identical after dividing by their own
    medians — that is exactly the part tau_escale can fix, so it must not show up."""
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.08, _N)
    g_a, r_a = _pair(50.0 * base)
    g_b, r_b = _pair(50.0 * base * 1.25)          # same shape, 25% harder
    _, resp_a = _response(g_a, r_a, gate="gen")
    _, resp_b = _response(g_b, r_b, gate="gen")
    na = resp_a / np.median(resp_a)
    nb = resp_b / np.median(resp_b)
    assert np.allclose(np.sort(na), np.sort(nb), rtol=1e-9)


def test_one_sided_tail_survives_median_normalisation():
    """A one-sided high tail at matched median is what the map CANNOT remove, so it must
    still be visible after normalisation — otherwise the panel proves nothing."""
    rng = np.random.default_rng(1)
    clean = rng.normal(1.0, 0.05, _N)
    tailed = clean.copy()
    tailed[: _N // 5] *= rng.uniform(1.5, 2.5, _N // 5)      # 20% contaminated upward
    _, resp_c = _response(*_pair(50.0 * clean), gate="gen")
    _, resp_t = _response(*_pair(50.0 * tailed), gate="gen")
    q_c = np.quantile(resp_c / np.median(resp_c), 0.95)
    q_t = np.quantile(resp_t / np.median(resp_t), 0.95)
    assert q_t > q_c * 1.2, (q_c, q_t)


def test_reco_gate_drops_low_response_taus_that_the_gen_gate_keeps():
    """The asymmetry the script exists to expose: gating on the RECO object removes gen τ
    whose reco partner fell below the pT floor, biasing the measured response upward."""
    gen_pt = np.full(_N, 22.0)                    # just above the 20 GeV floor
    reco_pt = np.where(np.arange(_N) % 2 == 0, 18.0, 26.0)   # half fluctuate below it
    g, r = _pair(reco_pt, gen_pt)
    _, resp_gen = _response(g, r, gate="gen")
    _, resp_reco = _response(g, r, gate="reco")
    assert resp_gen.size > resp_reco.size, "the reco gate must drop the low-side half"
    assert np.median(resp_reco) > np.median(resp_gen), "and so bias the median upward"


def test_unmatched_objects_are_dropped_not_counted_as_zero():
    """A gen τ with no reco partner must leave the sample, never enter it as response 0."""
    g = _col(np.full(_N, 50.0), np.zeros(_N), np.linspace(-3, 3, _N))
    r = _col(np.full(_N, 50.0), np.full(_N, 4.0), np.linspace(-3, 3, _N))   # far in eta
    gen_pt, resp = _response(g, r, gate="gen")
    assert gen_pt.size == 0 and resp.size == 0


def test_response_is_reco_over_gen_not_the_inverse():
    g, r = _pair(np.full(_N, 60.0), np.full(_N, 50.0))
    _, resp = _response(g, r, gate="gen")
    assert resp[0] == pytest.approx(60.0 / 50.0, rel=1e-9)
