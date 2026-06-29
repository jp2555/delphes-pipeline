"""Covariance-free FastMTT di-τ mass estimator (note §3.3, decision D1)."""

from __future__ import annotations

import numpy as np
from make_candle_fixture import make_dy_fixture_nu

from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.extensions.mtautau import estimate_mtautau, fastmtt_mass
from delphes_pipeline.validation.level1_candles import selections


def _vec(pt, eta, phi, m=1.777):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.array([np.sqrt(px * px + py * py + pz * pz + m * m), px, py, pz])


def _leg(t, x, mvis=1.0):
    p3 = t[1:] * x
    return {"px": np.array([p3[0]]), "py": np.array([p3[1]]), "pz": np.array([p3[2]]),
            "e": np.array([np.sqrt(sum(p3 * p3) + mvis * mvis)]), "mass": np.array([mvis]),
            "is_had": np.array([True])}


def test_fastmtt_recovers_true_mass_noiseless():
    """Two visible legs = x_i·(τ momentum) + exact pᵀᵐⁱˢˢ recover the di-τ mass."""
    t1, t2 = _vec(45, 0.3, 0.2), _vec(50, -0.5, 2.8)
    true_m = np.sqrt((t1[0] + t2[0]) ** 2 - sum((t1[1:] + t2[1:]) ** 2))
    x1, x2 = 0.7, 0.55
    nu = t1[1:] * (1 - x1) + t2[1:] * (1 - x2)
    m = fastmtt_mass(_leg(t1, x1), _leg(t2, x2), np.array([nu[0]]), np.array([nu[1]]),
                     met_sigma=5.0, grid=60)
    assert abs(m[0] - true_m) < 3.0
    # the visible mass alone is well below the true di-τ mass (neutrinos carried energy)
    vis = np.sqrt((_leg(t1, x1)["e"] + _leg(t2, x2)["e"])[0] ** 2 - sum((t1[1:] * x1 + t2[1:] * x2) ** 2))
    assert vis < true_m - 20


def test_fastmtt_zero_likelihood_returns_nan():
    """A hadronic leg with visible mass > m_τ has no valid x -> NaN, not a crash."""
    big = lambda mv: {"px": np.array([30.0]), "py": np.array([0.0]), "pz": np.array([0.0]),
                      "e": np.array([np.sqrt(900 + mv * mv)]), "mass": np.array([mv]),
                      "is_had": np.array([True])}
    m = fastmtt_mass(big(5.0), big(5.0), np.array([10.0]), np.array([0.0]), grid=20)
    assert np.isnan(m[0])


def test_fastmtt_zero_met_gives_visible_mass():
    """Negative control: with no pᵀᵐⁱˢˢ the estimator cannot add neutrinos (x→1),
    so m_ττ collapses to the visible mass — far below the true di-τ mass."""
    t1, t2 = _vec(45, 0.3, 0.2), _vec(50, -0.5, 2.8)
    x1, x2 = 0.6, 0.6
    vis = np.sqrt((_leg(t1, x1)["e"] + _leg(t2, x2)["e"])[0] ** 2 - sum((t1[1:] * x1 + t2[1:] * x2) ** 2))
    true_m = np.sqrt((t1[0] + t2[0]) ** 2 - sum((t1[1:] + t2[1:]) ** 2))
    m = fastmtt_mass(_leg(t1, x1), _leg(t2, x2), np.array([0.0]), np.array([0.0]), met_sigma=10.0, grid=60)
    assert abs(m[0] - vis) < 0.15 * vis        # collapses to the visible mass
    assert m[0] < true_m - 20                   # nowhere near the true di-τ mass


def test_estimate_mtautau_peaks_at_mz(tmp_path):
    """On Z→ττ with modelled τ decays, FastMTT restores the m_Z peak from below."""
    from delphes_pipeline.validation.level1_candles.ztautau import _peak_mode

    p = tmp_path / "dy.root"
    make_dy_fixture_nu(str(p), n_events=6000, seed=2)
    ev = DelphesEvents(str(p))
    veto = selections.bjet_veto_mask(ev)
    vis, _ = selections.leading_visible_pair(ev, veto)
    m = estimate_mtautau(ev, mask=veto)        # production default met_sigma=25
    m = m[np.isfinite(m)]
    assert np.median(vis[(vis > 40) & (vis < 120)]) < 75.0   # visible sits below m_Z
    assert abs(_peak_mode(m) - 91.2) < 8.0                   # estimator restores m_Z (robust peak)


def test_estimate_mtautau_robust_to_non_collinear(tmp_path):
    """The m_Z peak survives resolution smearing + acollinearity + extra MET noise,
    so the recovery is not an artefact of the fixture being the estimator's own model."""
    from delphes_pipeline.validation.level1_candles.ztautau import _peak_mode

    p = tmp_path / "dy.root"
    make_dy_fixture_nu(str(p), n_events=8000, seed=5, vis_smear=0.10, acoll=0.05, met_extra=15.0)
    ev = DelphesEvents(str(p))
    veto = selections.bjet_veto_mask(ev)
    m = estimate_mtautau(ev, mask=veto)
    m = m[np.isfinite(m)]
    assert abs(_peak_mode(m) - 91.2) < 12.0    # looser tol: real perturbations degrade resolution
