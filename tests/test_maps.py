"""Downstream tuning maps: derive from the anchor, apply by stochastic re-tag."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from conftest import build_ctx
from make_fixture import make_fixture
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
    retag_jets,
    save_maps,
)


def _wp():
    return {"btag_medium": BTAG_WP, "deeptau_vsjet_medium": DEEPTAU_MEDIUM}


def _mk(v):
    pts = [20.0, 50.0, 100.0, 200.0, 400.0]
    return {"x": "pt", "centers": pts, "values": [v] * len(pts), "counts": [100] * len(pts)}


def _const_maps(b: float, c: float, light: float) -> TuningMaps:
    """A flat (pT-independent) b-tag efficiency map per flavour."""
    return TuningMaps({"btag_eff_b": _mk(b), "btag_eff_c": _mk(c), "btag_mistag_light": _mk(light)})


def _const_maps_full(b, c, light, tau_eff, tau_mistag) -> TuningMaps:
    """A flat b-tag + τ_h map (both tag fields covered)."""
    return TuningMaps({"btag_eff_b": _mk(b), "btag_eff_c": _mk(c), "btag_mistag_light": _mk(light),
                       "tau_eff": _mk(tau_eff), "tau_mistag": _mk(tau_mistag)})


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
    # leptons are not routed through the re-tagged view
    assert tuned["electron_eff"].residual == base["electron_eff"].residual
    assert not tuned["electron_eff"].extra.get("retagged")


def test_convert_retags_tau_collection(good_fixture_path, tmp_path):
    """The ntuple Tau collection (build_taus selects tautag==1) follows the τ_h re-tag."""
    stock = convert(good_fixture_path, str(tmp_path / "stock.parquet"))
    # tau_eff = tau_mistag = 1 -> every jet is τ-tagged -> Tau collection == all jets
    maps = _const_maps_full(0.7, 0.2, 0.05, tau_eff=1.0, tau_mistag=1.0)
    tuned = convert(good_fixture_path, str(tmp_path / "tuned.parquet"), tuning_maps=maps)
    n_jets = int(ak.sum(ak.num(tuned["Jet"])))
    n_tau = int(ak.sum(ak.num(tuned["Tau"])))
    assert n_tau == n_jets                                    # every jet became a Tau
    assert n_tau != int(ak.sum(ak.num(stock["Tau"])))        # changed from stock
    assert int(ak.sum(tuned["Jet"].tautag)) == n_tau         # Tau count == #(tautag==1)


def test_lens_and_ntuplizer_tags_agree(good_fixture_path, tmp_path):
    """Invariant: the lens re-tag view and the ntuplizer emit identical tags at seed 0."""
    maps = _const_maps_full(0.7, 0.3, 0.05, tau_eff=0.6, tau_mistag=0.1)
    view = RetaggedEvents(DelphesEvents(good_fixture_path), maps, np.random.default_rng(0))
    arr = convert(good_fixture_path, str(tmp_path / "t.parquet"), tuning_maps=maps, seed=0)
    assert view.retagged_fields == frozenset({"btag", "tautag"})
    assert ak.all(view.jets.btag == arr["Jet"].btag)
    assert ak.all(view.jets.tautag == arr["Jet"].tautag)


def test_retag_jets_field_selection(good_fixture_path):
    """retag_jets re-tags a field only when ALL its maps are present (backward-compat)."""
    ev = DelphesEvents(good_fixture_path)
    fields = lambda m: retag_jets(ev, m, np.random.default_rng(0))[1]
    btag_only = _const_maps(0.7, 0.2, 0.05)
    assert fields(btag_only) == frozenset({"btag"})
    # b-tag + tau_eff but NO tau_mistag -> tautag NOT re-tagged (needs both τ maps)
    assert fields(TuningMaps({**btag_only.maps, "tau_eff": _mk(0.6)})) == frozenset({"btag"})
    # τ maps only -> tautag only
    assert fields(TuningMaps({"tau_eff": _mk(0.6), "tau_mistag": _mk(0.1)})) == frozenset({"tautag"})


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


def test_escale_rescales_bjet_and_tau_pt(good_fixture_path):
    """Energy-scale maps rescale gen-matched τ-jets (precedence) and b-jets; light untouched.

    τ-jets are selected by gen-matching — the SAME population the map is derived/validated
    on — not the stochastic tautag bit.
    """
    from delphes_pipeline.core.matching import matched_to_any

    ev = DelphesEvents(good_fixture_path)
    maps = TuningMaps({"bjet_escale": _mk(1.3), "tau_escale": _mk(0.9)})
    view = RetaggedEvents(ev, maps, np.random.default_rng(0))
    assert view.retagged_fields == frozenset({"escale"})
    fl = ak.to_numpy(ak.flatten(ev.jets.flavor))
    gen_taus = ev.gen[np.abs(ev.gen.pid) == 15]
    is_tau = ak.to_numpy(ak.flatten(matched_to_any(ev.jets, gen_taus, 0.4)))
    ratio = ak.to_numpy(ak.flatten(view.jets.pt)) / ak.to_numpy(ak.flatten(ev.jets.pt))
    assert np.allclose(ratio[is_tau], 0.9)                    # gen-matched τ-jets (precedence)
    assert np.allclose(ratio[(fl == 5) & ~is_tau], 1.3)       # b-jets (not τ)
    assert np.allclose(ratio[(fl != 5) & ~is_tau], 1.0)       # light jets unchanged


def test_escale_derive_and_apply_full_loop(tmp_path):
    """Non-circular: derive escale (1/response) from a biased Delphes sample, apply, close to unity."""
    sig = tmp_path / "sig.root"
    make_fixture(str(sig), n_events=6000, seed=3, bjet_response=0.85)
    nano = tmp_path / "nano.root"
    make_nano_fixture(str(nano), n_events=4000, seed=2)
    cfg = {"anchor": {"enabled": True, "nanoaod_path": str(nano), "wp": _wp()},
           "input": {"delphes_root": str(sig), "treename": "Delphes"}}
    maps = derive_maps(cfg, bins=obs.DEFAULT_PT_BINS)
    esc = maps["bjet_escale"]
    assert abs(np.average(esc["values"], weights=esc["counts"]) - 1.0 / 0.85) < 0.05   # ~1.176

    view = RetaggedEvents(DelphesEvents(str(sig)), TuningMaps(maps), np.random.default_rng(0))
    resp = obs.bjet_energy_response(view, bins=obs.DEFAULT_PT_BINS)
    assert abs(np.average(resp.values, weights=resp.counts) - 1.0) < 0.03


def test_escale_handles_sloped_response(tmp_path):
    """A pT-sloped b-jet response still closes — the gen-pT map is iterated to the gen pT."""
    sloped = lambda pt: 0.75 + 0.2 * np.tanh(np.asarray(pt) / 100.0)
    sig = tmp_path / "sig.root"
    make_fixture(str(sig), n_events=9000, seed=4, bjet_response=sloped)
    nano = tmp_path / "nano.root"
    make_nano_fixture(str(nano), n_events=4000, seed=2)
    cfg = {"anchor": {"enabled": True, "nanoaod_path": str(nano), "wp": _wp()},
           "input": {"delphes_root": str(sig), "treename": "Delphes"}}
    maps = TuningMaps(derive_maps(cfg, bins=obs.DEFAULT_PT_BINS))
    view = RetaggedEvents(DelphesEvents(str(sig)), maps, np.random.default_rng(0))
    resp = obs.bjet_energy_response(view, bins=obs.DEFAULT_PT_BINS)
    # every pT bin corrected to within 4% of unity (the reco-vs-gen-pT iteration)
    assert np.all(np.abs(np.asarray(resp.values) - 1.0) < 0.04)


def test_empty_sf_map_is_a_noop(good_fixture_path, tmp_path):
    """A present-but-empty SF map must default to 1.0, not zero the event weight."""
    empty = {"x": "pt", "centers": [], "values": [], "counts": []}
    maps = TuningMaps({"electron_sf": empty, "muon_sf": empty})
    arr = convert(good_fixture_path, str(tmp_path / "sf.parquet"), tuning_maps=maps)
    assert np.allclose(ak.to_numpy(arr["lepton_sf"]), 1.0)


def test_escale_closes_the_loop(tmp_path):
    """A biased b-jet energy response (0.85) is corrected to unity by the escale map."""
    sig = tmp_path / "sig.root"
    make_fixture(str(sig), n_events=6000, seed=3, bjet_response=0.85)
    ctx = build_ctx(str(sig))
    base = {r.observable: r for r in treport.run_tuning(ctx)}
    assert base["bjet_energy_response"].status == "needs_tuning"   # reco/gen = 0.85 != 1

    mpath = tmp_path / "escale.json"
    save_maps({"bjet_escale": _mk(1.0 / 0.85), "tau_escale": _mk(1.0)}, mpath, {"tuning_set": "v0"})
    ctx.config["tuning_maps"] = str(mpath)
    tuned = {r.observable: r for r in treport.run_tuning(ctx)}
    assert tuned["bjet_energy_response"].status == "on_target"
    assert tuned["bjet_energy_response"].extra.get("retagged") is True
    # m_bb (stock-tagged) stays out of the escale routing
    assert not tuned["mbb_peak"].extra.get("retagged")


def test_derive_maps_includes_escale_and_sf(good_fixture_path, tmp_path):
    """derive_maps adds energy-scale (Delphes 1/response) + lepton SF (anchor/delphes) maps."""
    nano = tmp_path / "nano.root"
    make_nano_fixture(str(nano), n_events=6000, seed=2)
    cfg = {"anchor": {"enabled": True, "nanoaod_path": str(nano), "wp": _wp()},
           "input": {"delphes_root": good_fixture_path, "treename": "Delphes"}}
    maps = derive_maps(cfg, bins=obs.DEFAULT_PT_BINS)
    assert {"bjet_escale", "tau_escale", "electron_sf", "muon_sf"} <= set(maps)
    # the good fixture has reco==GenJet -> response 1 -> escale ~1
    esc = maps["bjet_escale"]
    assert abs(np.average(esc["values"], weights=esc["counts"]) - 1.0) < 0.1
    # lepton SF = anchor_eff / delphes_eff, clipped to a sane range
    assert all(0.5 <= v <= 2.0 for v in maps["electron_sf"]["values"])


def test_convert_applies_lepton_sf_weight(good_fixture_path, tmp_path):
    """The ntuple carries per-event lepton_sf = Π over reco e/μ of the SF (1.0 without maps)."""
    stock = convert(good_fixture_path, str(tmp_path / "stock.parquet"))
    assert np.allclose(ak.to_numpy(stock["lepton_sf"]), 1.0)

    maps = TuningMaps({"electron_sf": _mk(0.9), "muon_sf": _mk(0.9)})
    arr = convert(good_fixture_path, str(tmp_path / "sf.parquet"), tuning_maps=maps)
    ev = DelphesEvents(good_fixture_path)
    nlep = ak.to_numpy(ak.num(ev.electrons) + ak.num(ev.muons))
    assert np.allclose(ak.to_numpy(arr["lepton_sf"]), 0.9 ** nlep, atol=1e-5)


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
