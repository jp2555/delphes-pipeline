"""The quantitative gates must actually FAIL on the failure modes they target.

A gate that can only ever pass is worthless. These tests build fixtures with a
specific defect and assert the relevant check fails:

- a MET *scale offset* must fail ``pilot_gate.met_resolution`` (the RMS-about-zero
  form is sensitive to a bias that a mean-subtracting variance would hide);
- a grossly smeared m_bb peak must fail ``pilot_gate.mbb_width`` (via the core
  fraction collapsing, since a window-bounded std saturates);
- the ntuplizer must honour the schema dtype contract.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from make_fixture import make_fixture

from conftest import build_ctx
from delphes_pipeline.ntuplizer import schema
from delphes_pipeline.ntuplizer.convert import convert
from delphes_pipeline.validation.pilot_gate import checks


def _by_name(results):
    return {r.name: r for r in results}


def test_met_scale_offset_fails_gate(tmp_path):
    path = tmp_path / "met_bias.root"
    make_fixture(str(path), n_events=3000, seed=2, met_bias_gev=60.0)
    res = _by_name(checks.run(build_ctx(str(path))))
    r = res["pilot_gate.met_resolution"]
    assert not r.passed, "a 60 GeV MET offset must fail the resolution gate"
    assert r.extra["mean_offset_gev"] > 30.0


def test_smeared_mbb_fails_gate(tmp_path):
    path = tmp_path / "wide_mbb.root"
    make_fixture(str(path), n_events=4000, seed=3, mbb_width_gev=70.0)
    res = _by_name(checks.run(build_ctx(str(path))))
    r = res["pilot_gate.mbb_width"]
    assert not r.passed, "a grossly smeared m_bb peak must fail the mbb gate"
    assert r.extra["core_fraction"] < 0.30


def test_ntuplizer_honours_schema_dtypes(good_fixture_path, tmp_path):
    rec = convert(good_fixture_path, str(tmp_path / "nt.parquet"))
    # kinematics float32, tag/flavour int32 (schema.FLAT_SCHEMA contract)
    assert ak.to_numpy(ak.flatten(rec.Jet.pt)).dtype == schema.FLAT_SCHEMA["Jet"]["pt"]
    assert ak.to_numpy(ak.flatten(rec.Jet.btag)).dtype == schema.FLAT_SCHEMA["Jet"]["btag"]
    assert ak.to_numpy(ak.flatten(rec.GenPart.pdgId)).dtype == schema.FLAT_SCHEMA["GenPart"]["pdgId"]
    # scalars float32
    assert np.asarray(rec.MET_pt).dtype == schema.SCALARS["MET_pt"]
    assert np.asarray(rec.genWeight).dtype == schema.SCALARS["genWeight"]
