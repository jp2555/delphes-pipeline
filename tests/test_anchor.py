"""NanoAOD-anchor tuning: reader + anchor measurement + tuning integration."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from conftest import build_ctx
from make_nano_fixture import BTAG_WP, DEEPTAU_MEDIUM, make_nano_fixture

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.nanoaod import NanoAODEvents
from delphes_pipeline.tuning import anchor as A
from delphes_pipeline.tuning import report as treport


def _wp():
    return {"btag_medium": BTAG_WP, "deeptau_vsjet_medium": DEEPTAU_MEDIUM}


def test_reader_thresholds_btag_at_wp(tmp_path):
    p = tmp_path / "nano.root"
    make_nano_fixture(str(p), n_events=2000, seed=1)
    ev = NanoAODEvents(str(p), wp=_wp())
    assert ev.n == 2000
    assert set(ev.jets.fields) >= {"pt", "eta", "flavor", "btag", "tautag"}
    btag_vals = set(np.unique(ak.to_numpy(ak.flatten(ev.jets.btag))))
    assert btag_vals <= {0, 1}  # discriminant thresholded to a bit


def test_anchor_profiles_recover_injection(tmp_path):
    p = tmp_path / "nano.root"
    truth = make_nano_fixture(str(p), n_events=8000, seed=2, met_resolution_gev=18.0)
    cfg = {"anchor": {"enabled": True, "nanoaod_path": str(p), "wp": _wp()}}
    prof = A.anchor_profiles(cfg, bins=obs.DEFAULT_PT_BINS)

    assert set(prof) >= set(A.ANCHOR_OBSERVABLES)
    # b-tag efficiency recovers the injected 0.70 (count-weighted over populated bins)
    b = prof["btag_eff_b"]
    w = np.average(b.values, weights=b.counts)
    assert abs(w - truth.btag_eff(0, 0)) < 0.05
    # tau efficiency recovers ~0.60
    t = prof["tau_eff"]
    assert abs(np.average(t.values, weights=t.counts) - truth.tau_eff(0, 0)) < 0.06
    # overall MET resolution ~ injected 18 GeV
    assert abs(prof["met_resolution"].values[0] - 18.0) < 2.5


def test_tuning_uses_the_anchor_target(good_fixture_path, tmp_path):
    nano = tmp_path / "nano.root"
    make_nano_fixture(str(nano), n_events=6000, seed=3)
    ctx = build_ctx(good_fixture_path)
    ctx.config["anchor"] = {"enabled": True, "nanoaod_path": str(nano), "wp": _wp()}

    by = {r.observable: r for r in treport.run_tuning(ctx)}
    r = by["btag_eff_b"]
    # the b-tag closure now has a target (the anchor), not 'no_target'
    assert r.status in ("on_target", "needs_tuning")
    assert r.extra.get("target") == "nanoaod_anchor"


def test_correctionlib_wp_loader_resolves_medium(tmp_path):
    pytest.importorskip("correctionlib")
    import correctionlib.schemav2 as schema

    corr = schema.Correction(
        name="UParTAK4_wp_values", version=1,
        inputs=[schema.Variable(name="working_point", type="string")],
        output=schema.Variable(name="cut", type="real"),
        data=schema.Category(
            nodetype="category", input="working_point",
            content=[schema.CategoryItem(key="L", value=0.10),
                     schema.CategoryItem(key="M", value=0.40),
                     schema.CategoryItem(key="T", value=0.80)],
        ),
    )
    cset = schema.CorrectionSet(schema_version=2, corrections=[corr])
    p = tmp_path / "btagging.json"
    dump = getattr(cset, "model_dump_json", getattr(cset, "json", None))
    p.write_text(dump())

    from delphes_pipeline.tuning import correctionlib_wp as W
    assert W.find_wp_correction(str(p), "UParT") == "UParTAK4_wp_values"
    assert abs(W.load_wp(str(p), "UParTAK4_wp_values", "M") - 0.40) < 1e-9
    assert abs(W.resolve_btag_wp({"json": str(p), "tagger": "UParT", "wp": "M"}) - 0.40) < 1e-9
