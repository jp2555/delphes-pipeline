"""Downstream tuning maps: derive from the anchor, apply by stochastic re-tag."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from conftest import build_ctx
from make_nano_fixture import BTAG_WP, DEEPTAU_MEDIUM, make_nano_fixture

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.ntuplizer import convert as convert_mod
from delphes_pipeline.ntuplizer.convert import convert
from delphes_pipeline.tuning import report as treport
from delphes_pipeline.tuning.maps import (
    BTAG_MAP_QUANTITIES,
    RetaggedEvents,
    TuningMaps,
    derive_maps,
    retag_btag,
    save_maps,
)


def _wp():
    return {"btag_medium": BTAG_WP, "deeptau_vsjet_medium": DEEPTAU_MEDIUM}


def _const_maps(b: float, c: float, light: float) -> TuningMaps:
    """A flat (pT-independent) efficiency map per flavour."""
    pts = [20.0, 50.0, 100.0, 200.0, 400.0]
    mk = lambda v: {"x": "pt", "centers": pts, "values": [v] * len(pts), "counts": [100] * len(pts)}
    return TuningMaps({"btag_eff_b": mk(b), "btag_eff_c": mk(c), "btag_mistag_light": mk(light)})


def test_retag_recovers_injected_efficiency(good_fixture_path):
    """BTag = Bernoulli(ε_map(flavour, pT)) -> measured rate matches the map."""
    ev = DelphesEvents(good_fixture_path)
    tag = retag_btag(ev, _const_maps(0.8, 0.3, 0.05), np.random.default_rng(0))
    fl = ak.to_numpy(ak.flatten(ev.jets.flavor))
    tg = ak.to_numpy(ak.flatten(tag))
    assert tg.shape == fl.shape
    assert set(np.unique(tg)) <= {0, 1}
    assert abs(tg[fl == 5].mean() - 0.8) < 0.05
    assert abs(tg[fl == 4].mean() - 0.3) < 0.05
    light = (fl != 5) & (fl != 4)
    assert abs(tg[light].mean() - 0.05) < 0.03


def test_retag_is_reproducible(good_fixture_path):
    ev = DelphesEvents(good_fixture_path)
    m = _const_maps(0.7, 0.2, 0.05)
    a = retag_btag(ev, m, np.random.default_rng(42))
    b = retag_btag(ev, m, np.random.default_rng(42))
    assert ak.all(a == b)


def test_derive_maps_from_anchor(tmp_path):
    p = tmp_path / "nano.root"
    make_nano_fixture(str(p), n_events=8000, seed=2)
    cfg = {"anchor": {"enabled": True, "nanoaod_path": str(p), "wp": _wp()}}
    maps = derive_maps(cfg, bins=obs.DEFAULT_PT_BINS)

    assert set(maps) >= set(BTAG_MAP_QUANTITIES)
    b = maps["btag_eff_b"]
    assert abs(np.average(b["values"], weights=b["counts"]) - 0.70) < 0.05

    out = tmp_path / "maps_v0.json"
    save_maps(maps, out, {"tuning_set": "v0"})
    loaded = TuningMaps.load(out)
    assert abs(float(loaded.efficiency("btag_eff_b", [100.0])[0]) - 0.70) < 0.05


def test_convert_applies_maps(good_fixture_path, tmp_path):
    """The ntuplizer re-tags downstream when a maps object is configured."""
    out = tmp_path / "tuned.parquet"
    arr = convert(good_fixture_path, str(out), tuning_maps=_const_maps(1.0, 1.0, 0.0))
    jets = arr["Jet"]
    fl = ak.to_numpy(ak.flatten(jets.hadronFlavour))
    bt = ak.to_numpy(ak.flatten(jets.btag))
    heavy = (fl == 5) | (fl == 4)
    assert bt[heavy].min() == 1            # b/c always tagged
    assert bt[~heavy].max() == 0           # light never tagged


def test_retag_closes_the_tuning_loop(good_fixture_path, tmp_path):
    """The tuning lens re-validates the re-tagged b-tag against the anchor (loop closed).

    The good fixture injects the *card* b-tag formula (~0.63), the anchor injects
    0.70, so the stock b-tag needs tuning. After re-tagging toward the anchor maps
    the measured b-tag matches the anchor -> on_target.
    """
    nano = tmp_path / "nano.root"
    make_nano_fixture(str(nano), n_events=8000, seed=3)
    anchor_cfg = {"enabled": True, "nanoaod_path": str(nano), "wp": _wp()}

    maps = derive_maps({"anchor": anchor_cfg}, bins=obs.DEFAULT_PT_BINS)
    mpath = tmp_path / "maps_v0.json"
    save_maps(maps, mpath, {"tuning_set": "v0"})

    ctx = build_ctx(good_fixture_path)
    ctx.config["anchor"] = anchor_cfg

    base = {r.observable: r for r in treport.run_tuning(ctx)}
    assert base["btag_eff_b"].status == "needs_tuning"
    assert not base["btag_eff_b"].extra.get("retagged")

    ctx.config["tuning_maps"] = str(mpath)
    tuned = {r.observable: r for r in treport.run_tuning(ctx)}
    assert tuned["btag_eff_b"].status == "on_target"
    assert tuned["btag_eff_b"].extra.get("retagged") is True
    assert tuned["btag_eff_b"].residual < base["btag_eff_b"].residual
    # only the b-tag observables are routed through the re-tagged view
    assert tuned["electron_eff"].residual == base["electron_eff"].residual
    assert not tuned["electron_eff"].extra.get("retagged")
    # the m_bb b-pair selection reads btag, but it stays on stock tags -> energy-scale
    # diagnostic is unconfounded by the re-tag (residual identical to the stock run)
    assert tuned["mbb_peak"].residual == base["mbb_peak"].residual
    assert not tuned["mbb_peak"].extra.get("retagged")


def test_tau_retag_closes_the_loop(good_fixture_path, tmp_path):
    """τ_h re-tag: tau_eff (genuine) and tau_mistag (fake) re-validate against the anchor.

    Stock card τ response (eff 0.60, mistag ~0.01) differs from the injected anchor
    (eff 0.72, mistag 0.06), so both need tuning; after the gen-record-keyed re-tag they
    land on target.
    """
    nano = tmp_path / "nano.root"
    make_nano_fixture(str(nano), n_events=8000, seed=4,
                      tau_eff=lambda pt, eta: 0.72, tau_mistag=lambda pt, eta: 0.06)
    anchor_cfg = {"enabled": True, "nanoaod_path": str(nano), "wp": _wp()}
    maps = derive_maps({"anchor": anchor_cfg}, bins=obs.DEFAULT_PT_BINS)
    assert {"tau_eff", "tau_mistag"} <= set(maps)
    mpath = tmp_path / "maps_v0.json"
    save_maps(maps, mpath, {"tuning_set": "v0"})

    ctx = build_ctx(good_fixture_path)
    ctx.config["anchor"] = anchor_cfg
    base = {r.observable: r for r in treport.run_tuning(ctx)}

    ctx.config["tuning_maps"] = str(mpath)
    tuned = {r.observable: r for r in treport.run_tuning(ctx)}

    for q in ("tau_eff", "tau_mistag"):
        assert base[q].status == "needs_tuning"
        assert tuned[q].status == "on_target"
        assert tuned[q].extra.get("retagged") is True
        assert tuned[q].residual < base[q].residual
    # b-tag is co-tagged in the same run; the τ-energy-scale diagnostic stays on stock tags
    assert tuned["btag_eff_b"].extra.get("retagged") is True
    assert not tuned["tau_energy_response"].extra.get("retagged")
    assert tuned["mbb_peak"].residual == base["mbb_peak"].residual


def test_convert_cli_applies_maps_from_flag(good_fixture_path, tmp_path):
    """The production CLI re-tags when --tuning-maps is passed (seed 0, matching the lens)."""
    mpath = tmp_path / "maps.json"
    save_maps(_const_maps(1.0, 1.0, 0.0).maps, mpath, {"tuning_set": "v0"})
    out = tmp_path / "tuned.parquet"
    convert_mod.main([good_fixture_path, str(out), "--tuning-maps", str(mpath)])

    arr = ak.from_parquet(str(out))
    fl = ak.to_numpy(ak.flatten(arr["Jet"].hadronFlavour))
    bt = ak.to_numpy(ak.flatten(arr["Jet"].btag))
    heavy = (fl == 5) | (fl == 4)
    assert bt[heavy].min() == 1 and bt[~heavy].max() == 0


def test_retagged_view_forwards_and_guards(good_fixture_path):
    """Proxy forwards non-jet attributes; the __getattr__ guard prevents pickle recursion."""
    ev = DelphesEvents(good_fixture_path)
    view = RetaggedEvents(ev, _const_maps(0.7, 0.2, 0.05), np.random.default_rng(0))
    assert view.n == ev.n                      # forwarded to the wrapped events
    assert view.jets is not None
    # a bare (pre-__init__) instance must raise AttributeError, not recurse forever
    bare = RetaggedEvents.__new__(RetaggedEvents)
    with pytest.raises(AttributeError):
        bare.anything
