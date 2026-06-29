"""Downstream tuning maps: derive from the anchor, apply by stochastic re-tag."""

from __future__ import annotations

import awkward as ak
import numpy as np
from make_nano_fixture import BTAG_WP, DEEPTAU_MEDIUM, make_nano_fixture

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.ntuplizer.convert import convert
from delphes_pipeline.tuning.maps import (
    BTAG_MAP_QUANTITIES,
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
