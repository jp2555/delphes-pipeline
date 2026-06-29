"""Session-scoped fixtures for the Delphes card-validation test suite.

Two synthetic Delphes-like ROOT files are built once per session:

- ``good_fixture_path`` injects the **card's own transcribed formulas** as the
  per-object efficiency truth, so the Level-0 closure measurements recover the
  card and the gate passes.
- ``broken_fixture_path`` is the deliberately malformed file (``broken=True``):
  >50% negative weights, all-zero ``Jet.Flavor``, and no gen taus — it must fail
  the Pilot Gate.

``build_ctx`` mirrors :func:`delphes_pipeline.validation.run_validation.build_context`
but points at a local fixture (``max_events=None``), and ``make_config`` yaml-loads
``config.example.yml`` and overrides the input path and output dir so a full
``run(...)`` writes into the test's tmp area.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from make_fixture import make_fixture

from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.io import DelphesEvents
from delphes_pipeline.core.references import ReferenceStore
from delphes_pipeline.core import provenance as prov
from delphes_pipeline.validation.references import card_formulas

_CONFIG_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "delphes_pipeline"
    / "validation"
    / "config.example.yml"
)


def _cardfn(quantity: str):
    """A scalar ``(pt, eta) -> eff`` wrapper around the card formula.

    ``make_fixture`` calls each efficiency with python floats, so the card
    formula (which expects array-likes) is evaluated on length-1 arrays and the
    first element returned.
    """
    return lambda pt, eta: float(
        card_formulas.expected(quantity, np.atleast_1d(pt), np.atleast_1d(eta))[0]
    )


@pytest.fixture(scope="session")
def good_fixture_path(tmp_path_factory) -> str:
    """Card-injected good fixture: every Level-0 closure recovers the card."""
    path = tmp_path_factory.mktemp("good") / "signal_good.root"
    make_fixture(
        str(path),
        n_events=8000,
        seed=7,
        met_resolution_gev=20,
        mbb_width_gev=10,
        btag_eff=_cardfn("btag_eff_b"),
        ctag_eff=_cardfn("btag_eff_c"),
        ltag_mistag=_cardfn("btag_mistag_light"),
        tau_eff=_cardfn("tau_eff"),
        tau_mistag=_cardfn("tau_mistag"),
        electron_eff=_cardfn("electron_eff"),
        muon_eff=_cardfn("muon_eff"),
    )
    return str(path)


@pytest.fixture(scope="session")
def broken_fixture_path(tmp_path_factory) -> str:
    """Deliberately malformed fixture that must fail the Pilot Gate."""
    path = tmp_path_factory.mktemp("broken") / "signal_broken.root"
    make_fixture(str(path), broken=True)
    return str(path)


def build_ctx(path: str) -> ValidationContext:
    """Build a ValidationContext pointed at a local fixture.

    Mirrors ``run_validation.build_context`` (card formulas injected into the
    reference store, provenance collected) but reads all events from ``path``.
    """
    config = make_config(path)
    cfg_input = config["input"]
    events = DelphesEvents(cfg_input["delphes_root"], treename=cfg_input.get("treename", "Delphes"))

    output_dir = Path(config["output"]["dir"])
    plot_dir = output_dir / "plots"

    ref_dir = config.get("references", {}).get("dir", "delphes_pipeline/validation/references/data")
    references = ReferenceStore(ref_dir, card_formula_fn=card_formulas.expected)

    provenance = prov.collect(
        card_path=config.get("card", "cards/cms_card_v0.tcl"),
        input_path=cfg_input["delphes_root"],
        n_events=events.n,
        config=config,
    )
    return ValidationContext(
        config=config,
        events=events,
        references=references,
        output_dir=output_dir,
        plot_dir=plot_dir,
        provenance=provenance,
    )


def make_config(path: str) -> dict:
    """Load ``config.example.yml`` and point it at ``path`` with a tmp output dir."""
    with open(_CONFIG_EXAMPLE) as fh:
        config = yaml.safe_load(fh)
    config["input"]["delphes_root"] = str(path)
    config["input"]["max_events"] = None
    # the fixture suite has no external NanoAOD anchor (the example path points at
    # deepthought + CVMFS); tests that exercise the anchor set it explicitly.
    config.setdefault("anchor", {})["enabled"] = False
    # write the report next to the fixture so each test gets an isolated output
    config.setdefault("output", {})["dir"] = str(Path(path).parent / "validation_out")
    return config
