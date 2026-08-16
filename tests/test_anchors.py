"""The derivation/validation firewall must fail a build, not a review checklist."""
import pytest

from delphes_pipeline.tuning import anchors as A


@pytest.fixture
def cfg():
    return A.load("cards/tuning/anchors_v2.yml")


def test_the_shipped_anchors_file_parses(cfg):
    assert cfg["schema_version"] == 2
    assert cfg["era"]["sqrt_s_tev"] == 13.6, "no sqrt(s) bridge: anchor and gen are 13.6"


def test_a_derivation_anchor_is_returned(cfg):
    e = A.for_derivation(cfg, "tau_response")
    assert e["use"] == "derivation"
    assert "gen_decay_mode" in e["condition_on"]


def test_a_validation_target_cannot_be_derived_from(cfg):
    with pytest.raises(A.AnchorError, match="VALIDATION target"):
        A.for_derivation(cfg, "analysis_level_shapes")


def test_an_entry_mis_tagged_as_validation_is_refused():
    bad = {"maps": {"jet_response": {"use": "validation"}}}
    with pytest.raises(A.AnchorError, match="only .*derivation.* entries"):
        A.for_derivation(bad, "jet_response")


def test_unknown_anchor_names_raise(cfg):
    with pytest.raises(A.AnchorError, match="no anchor entry"):
        A.for_derivation(cfg, "not_a_map")


def test_every_shipped_number_is_still_flagged_unverified(cfg):
    """Nothing has been checked against a publication yet; the file must say so."""
    un = A.unverified(cfg)
    assert any(p.startswith("tier3_anchor") for p in un)
    assert any("citation" in p for p in un)


def test_the_gate_refuses_to_run_on_placeholders(cfg):
    with pytest.raises(A.AnchorError, match="still placeholders"):
        A.require_verified(cfg, "tier3_anchor")


def test_require_verified_passes_once_a_value_is_filled_in():
    cfg = {"tier3_anchor": {"luminosity_fb": 62.0, "citation": "CMS-PAS-XXX"}}
    A.require_verified(cfg, "tier3_anchor")


def test_the_systematics_share_is_present_because_the_paper_quotes_it(cfg):
    assert cfg["tier3_anchor"]["systematics_share"]["value"] == 0.15
    assert cfg["tier3_anchor"]["systematics_share"]["citation"] == A.TOVERIFY


def test_the_pileup_jet_fake_term_is_recorded_as_blocked(cfg):
    """No-PU samples contain no PU jets; a downstream map cannot create objects."""
    pu = cfg["maps"]["tau_fake"]["pu_jet_contribution"]
    assert pu["status"] == "BLOCKED"
