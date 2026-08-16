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


# --------------------------------------------------------------------------- #
# The C4 perfect-mitigation framing turns on ONE hinge: the public rates we derive
# from must be post-PU-jet-ID / post-PV-association. A pre-mitigation number counts
# the pileup term twice — once in the map, once in the residual bound.
# --------------------------------------------------------------------------- #
def test_a_placeholder_mitigation_state_blocks_derivation(cfg):
    with pytest.raises(A.AnchorError, match="post-PU-jet-ID"):
        A.for_derivation(cfg, "tau_fake")


def test_a_pre_mitigation_rate_is_refused():
    bad = {"maps": {"tau_fake": {"use": "derivation",
                                 "mitigation_state": "pre_mitigation"}}}
    with pytest.raises(A.AnchorError, match="twice"):
        A.for_derivation(bad, "tau_fake")


def test_a_post_mitigation_rate_is_accepted():
    ok = {"maps": {"tau_fake": {"use": "derivation",
                                "mitigation_state": "post_mitigation"}}}
    assert A.for_derivation(ok, "tau_fake")["mitigation_state"] == "post_mitigation"


def test_a_map_with_no_mitigation_field_is_unaffected(cfg):
    """jet_response is a response, not a rate on a mitigated population."""
    assert A.for_derivation(cfg, "jet_response")["use"] == "derivation"


# --------------------------------------------------------------------------- #
# A PU-inclusive conditioning variable applied to a no-PU event reads a low bin
# and under-corrects. Live in v1: met_smear is binned in jet H_T (pt>20, |eta|<4.7).
# --------------------------------------------------------------------------- #
def test_a_pu_inclusive_conditioning_variable_is_refused():
    bad = {"maps": {"met_resolution": {
        "use": "derivation", "mitigation_state": "not_applicable",
        "conditioning_must_be_pu_independent": True,
        "condition_on": ["sum_et", "recoil_axis"]}}}
    with pytest.raises(A.AnchorError, match="under-correct"):
        A.for_derivation(bad, "met_resolution")


def test_hard_scatter_conditioning_passes(cfg):
    e = A.for_derivation(cfg, "met_resolution")
    assert "hard_scatter_ht" in e["condition_on"]


def test_the_pu_fake_term_is_now_bounded_not_blocked(cfg):
    pu = cfg["maps"]["tau_fake"]["pu_jet_contribution"]
    assert pu["status"] == "omitted_bounded"
    assert pu["framing"] == "perfect-mitigation limit"
    assert pu["bound_per_leg"]["pairing_acceptance"] == 1.0
