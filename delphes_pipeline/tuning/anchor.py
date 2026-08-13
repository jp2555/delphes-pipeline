"""Measure tuning targets from the private NanoAOD anchor (note §3, §6.4).

The Delphes object response is tuned to match the *same* response measured on the
CMS NanoAOD. Because ``NanoAODEvents`` duck-types ``DelphesEvents``, the b-tag and
lepton targets reuse ``core.observables`` directly; the τ_h efficiency is bespoke
(NanoAOD ``GenVisTau`` matched to a ``Tau`` passing the DeepTau VSjet Medium WP);
MET is the overall resolution. Each returns a ``Profile`` used by the tuning
report as the target curve.
"""

from __future__ import annotations

from typing import Optional

import awkward as ak
import numpy as np

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.matching import matched_to_any, nearest_target_field, unique_match
from delphes_pipeline.core.nanoaod import NanoAODEvents
from delphes_pipeline.core.observables import Profile

# observables for which the NanoAOD anchor provides a target
_MASS_QUANTILES = 21          # uniform quantile grid stored for the τ_h visible mass
# 101, not 21: the grid's TOP segment runs from its last stored level to the sample MAX,
# and a linear draw across it inflates the extreme tail — measured 7.2% high at q99 with 21
# levels, 0.11% with 101, because q99 then lands exactly on a grid point. The response tail
# is the whole reason option B exists, so it must not be an interpolation artefact. (The
# τ_h MASS grid keeps 21: it is clipped at 1.70 GeV, which pins its endpoint, and its
# residual was measured inert.)
_RESPONSE_QUANTILES = 101
# Levels stop SHORT of 0 and 1 on purpose. Level 1.0 is the sample MAX — an order
# statistic that never converges — and a uniform draw into the top segment interpolates
# toward it, turning a handful of anchor outliers into a percent-level response tail. The
# response tail is the whole point of option B, so it must not be an artefact of its own
# grid. Truncating 0.5% at each end is the conservative direction.
_RESPONSE_LEVELS = np.linspace(0.005, 0.995, _RESPONSE_QUANTILES)

ANCHOR_OBSERVABLES = ("btag_eff_b", "btag_eff_c", "btag_mistag_light",
                      "electron_eff", "muon_eff", "tau_eff", "tau_mistag", "tau_mass",
                      "tau_energy_response", "tau_fake_response", "met_resolution")


def anchor_profiles(config: dict, *, bins, max_events: Optional[int] = None) -> dict[str, Profile]:
    """Measure the anchor target for each observable; empty if anchor disabled."""
    ac = config.get("anchor", {})
    if not ac.get("enabled"):
        return {}
    # cap by the smaller of anchor.max_events and the run-wide --max-events
    cap = ac.get("max_events")
    if max_events is not None:
        cap = max_events if cap is None else min(cap, max_events)
    print(f"[tuning] opening NanoAOD anchor (entry_stop={cap}) ...", flush=True)
    nano = NanoAODEvents(
        ac["nanoaod_path"], branches=ac.get("branches"), wp=_resolve_wp(ac.get("wp", {})),
        entry_stop=cap,
    )
    print(f"[tuning] anchor: {nano.n} events from {len(nano._used)} file(s); measuring ...", flush=True)
    out: dict[str, Profile] = {}
    print("[tuning] anchor: b-tag ...", flush=True)
    for q in ("btag_eff_b", "btag_eff_c", "btag_mistag_light"):
        out[q] = obs.btag_efficiency(nano, q, bins=bins)
    print("[tuning] anchor: leptons ...", flush=True)
    for q in ("electron_eff", "muon_eff"):
        out[q] = obs.lepton_efficiency(nano, q, bins=bins)
    print("[tuning] anchor: tau + MET ...", flush=True)
    out["tau_eff"] = _nano_tau_eff(nano, bins)
    out["tau_mistag"] = _nano_tau_mistag(nano, bins)
    out["tau_mass"] = _nano_tau_mass(nano, bins)
    out["tau_energy_response"] = _nano_tau_energy_response(nano, bins)
    out["tau_fake_response"] = _nano_tau_fake_response(nano, bins)
    out["met_resolution"] = _nano_met_resolution(nano)
    # label the source for the report/plot
    for p in out.values():
        p.ylabel = (p.ylabel or "") + " (NanoAOD anchor)"
    return out


def _resolve_wp(wp: dict) -> dict:
    """Fill ``btag_medium`` from jsonpog-integration (CVMFS) when not set explicitly."""
    wp = dict(wp)
    if wp.get("btag_medium") is None and wp.get("btag_correctionlib"):
        from . import correctionlib_wp
        wp["btag_medium"] = correctionlib_wp.resolve_btag_wp(wp["btag_correctionlib"])
        print(f"[tuning] resolved btag_medium = {wp['btag_medium']:.4f} from jsonpog-integration")
    return wp


def _nano_tau_eff(nano: NanoAODEvents, bins, *, dr=0.4, eta_max=2.5, pt_min=20.0) -> Profile:
    """τ_h efficiency on NanoAOD: GenVisTau matched to a DeepTau-Medium Tau."""
    gvt = nano.genvistau
    acc = gvt[(np.abs(gvt.eta) <= eta_max) & (gvt.pt > pt_min)]
    matched, vsjet = nearest_target_field(acc, nano.taus, dr, "vsjet")
    passed = matched & (np.nan_to_num(np.asarray(vsjet), nan=-1.0) >= nano.deeptau_medium())
    prof = obs.binned_efficiency(ak.to_numpy(ak.flatten(acc.pt)), passed, bins, quantity="tau_eff", x="pt")
    prof.xlabel, prof.ylabel = "tau pT [GeV]", "tau_eff"
    return prof


def _nano_tau_mistag(nano: NanoAODEvents, bins, *, dr=0.4, eta_max=2.5, pt_min=20.0) -> Profile:
    """jet→τ_h mistag on NanoAOD: acceptance jets *not* near a GenVisTau that match a
    DeepTau-Medium ``Tau`` (mirrors ``observables.tau_mistag``, where the Delphes TauTag
    bit is replaced by a reco τ match). The match is a *unique* nearest one (each Medium
    τ tags at most one jet) so a fake τ on one jet is not double-counted onto a collinear
    neighbour — that cross-jet leakage would bias the per-jet fake rate high."""
    jets = nano.jets
    acc = jets[(np.abs(jets.eta) <= eta_max) & (jets.pt > pt_min)]
    fake = acc[~matched_to_any(acc, nano.genvistau, dr)]
    medium = nano.taus[nano.taus.vsjet >= nano.deeptau_medium()]
    tagged = unique_match(fake, medium, dr)
    prof = obs.binned_efficiency(ak.to_numpy(ak.flatten(fake.pt)), tagged, bins, quantity="tau_mistag", x="pt")
    prof.xlabel, prof.ylabel = "jet pT [GeV]", "tau_mistag"
    return prof


def _nano_tau_mass(nano: NanoAODEvents, bins, *, eta_max=2.5, pt_min=20.0) -> Profile:
    """τ_h visible mass on NanoAOD: the Medium-DeepTau ``Tau`` mass vs τ pT.

    The CMS counterpart of ``observables.tau_visible_mass``. A reconstructed τ_h
    carries its decay-mode visible mass (≲ m_τ); the Delphes τ-jet carries the AK4 jet
    mass instead, which is multi-GeV and makes the FastMTT hadronic prior vanish.
    """
    taus = nano.taus[nano.taus.vsjet >= nano.deeptau_medium()]
    acc = taus[(np.abs(taus.eta) <= eta_max) & (taus.pt > pt_min)]
    pt = ak.to_numpy(ak.flatten(acc.pt))
    mass = ak.to_numpy(ak.flatten(acc.mass))
    prof = obs.binned_response(pt, mass, bins, quantity="tau_mass", x="pt")
    prof.xlabel, prof.ylabel = "tau_h pT [GeV]", "visible mass [GeV]"
    # The median alone is NOT enough downstream. In FastMTT the visible mass sets a
    # one-sided FLOOR on the energy fraction, xmin = (m_vis/m_τ)²; assigning one value to
    # every τ-jet relaxes that floor for every leg whose true mass is above it, opening
    # the low-x region where m_ττ = m_vis/√(x₁x₂) diverges. Collapsing the distribution
    # therefore inflates the m_ττ width and pulls the mean up — it does not merely smear.
    # Carry the per-bin quantiles so the map can be sampled from instead.
    levels = np.linspace(0.0, 1.0, _MASS_QUANTILES)
    edges = np.asarray(bins, dtype=float)
    qvals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (pt >= lo) & (pt < hi)
        if not in_bin.any():
            continue
        qvals.append(np.quantile(mass[in_bin], levels).tolist())
    prof.aux = {"quantile_levels": levels.tolist(), "quantile_values": qvals}
    return prof


def _nano_tau_energy_response(nano: NanoAODEvents, bins, *, dr=0.4, eta_max=2.5,
                              pt_min=20.0) -> Profile:
    """CMS τ_h energy response: Medium ``Tau`` pT / matched ``GenVisTau`` pT vs gen pT.

    The counterpart of ``observables.tau_energy_response``, which profiles the Delphes
    τ-jet against its (neutrino-filtered) GenJet. Both are therefore reco/gen against a
    *visible* reference, so their ratio is the Delphes→CMS energy correction.

    This is what makes ``tau_escale`` an anchor comparison instead of a self-comparison:
    a Delphes τ_h is a jet and carries jet-level pT (UE and everything else inside
    R=0.4), while a CMS ``Tau`` carries the clean HPS visible-τ pT. Correcting Delphes to
    its own GenJet — also a jet — preserves that contamination and leaves the Delphes τ
    systematically harder than CMS, inflating m_vis and hence m_ττ.
    """
    gvt = nano.genvistau
    acc = gvt[(np.abs(gvt.eta) <= eta_max) & (gvt.pt > pt_min)]
    medium = nano.taus[nano.taus.vsjet >= nano.deeptau_medium()]
    matched, tau_pt = nearest_target_field(acc, medium, dr, "pt")
    gen_pt = ak.to_numpy(ak.flatten(acc.pt))
    ok = matched & (np.nan_to_num(tau_pt, nan=0.0) > 0) & (gen_pt > 0)
    resp = tau_pt[ok] / gen_pt[ok]
    prof = obs.binned_response(gen_pt[ok], resp, bins,
                               quantity="tau_energy_response", x="pt")
    prof.xlabel, prof.ylabel = "gen visible-tau pT [GeV]", "reco/gen pT"
    # The median alone only supports a multiplicative escale, which by construction can
    # align medians and nothing else. Measured on this anchor the Delphes response is
    # one-sidedly BROADER at matched median (3x at 20-30 GeV), and a scale factor cannot
    # depopulate that tail — it survives into m_vis above the kinematic limit. Carrying
    # the per-pT quantiles lets the Delphes tau energy be *redrawn* from the CMS
    # distribution instead (maps.resample_tau_energy), which reproduces the shape too.
    levels = _RESPONSE_LEVELS
    edges = np.asarray(bins, dtype=float)
    qvals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (gen_pt[ok] >= lo) & (gen_pt[ok] < hi)
        if not in_bin.any():
            continue
        qvals.append(np.quantile(resp[in_bin], levels).tolist())
    prof.aux = {"quantile_levels": levels.tolist(), "quantile_values": qvals}
    return prof


def _nano_tau_fake_response(nano: NanoAODEvents, bins, *, dr=0.4, eta_max=2.5,
                            pt_min=20.0) -> Profile:
    """CMS FAKE τ_h energy response: Medium ``Tau`` with NO gen τ, over its matched GenJet.

    A fake τ_h is a quark or gluon jet that passed the τ ID. In CMS it still carries an HPS
    narrow-cone four-vector; in Delphes it is the whole AK4 jet, and it receives no energy
    correction at all — it matches neither branch of ``escale_factor``, so it keeps a raw
    jet pT. In the signal selection that is ~4% of pairs and harmless, but ``TTto4Q``
    selects fakes *exclusively*, so at production scale this is the energy scale of the
    dominant background. Delphes fakes have a GenJet too, so this map gives them a target.

    The reference is the GenJet, not a visible τ: a fake has no gen τ by definition.
    """
    empty = Profile("tau_fake_response", "pt", np.array([]), np.array([]),
                    np.array([]), np.array([], dtype=int))
    # an anchor skim without GenJet yields a field-less collection; no reference, no map
    if not ak.fields(nano.genjets) or not ak.fields(nano.genvistau):
        return empty
    medium = nano.taus[nano.taus.vsjet >= nano.deeptau_medium()]
    acc = medium[(np.abs(medium.eta) <= eta_max) & (medium.pt > pt_min)]
    fake = acc[~matched_to_any(acc, nano.genvistau, dr)]      # same ΔR as the real-τ match
    matched, gj_pt = nearest_target_field(fake, nano.genjets, dr, "pt")
    tau_pt = ak.to_numpy(ak.flatten(fake.pt))
    gj_pt = np.nan_to_num(gj_pt, nan=0.0)
    ok = matched & (gj_pt > 0)
    if not ok.any():        # no fake passed the ID, or none matched a GenJet
        return empty
    resp = tau_pt[ok] / gj_pt[ok]
    prof = obs.binned_response(gj_pt[ok], resp, bins, quantity="tau_fake_response", x="pt")
    prof.xlabel, prof.ylabel = "matched GenJet pT [GeV]", "fake tau reco/gen pT"
    levels = _RESPONSE_LEVELS
    edges = np.asarray(bins, dtype=float)
    qvals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (gj_pt[ok] >= lo) & (gj_pt[ok] < hi)
        if not in_bin.any():
            continue
        qvals.append(np.quantile(resp[in_bin], levels).tolist())
    prof.aux = {"quantile_levels": levels.tolist(), "quantile_values": qvals}
    return prof


def _nano_met_resolution(nano: NanoAODEvents) -> Profile:
    """CMS MET resolution vs jet-HT.

    This was a single bin because Delphes ``ScalarHT`` and NanoAOD ``sumEt`` are not the
    same variable, leaving nothing comparable to bin against. ``obs.jet_ht`` recomputes
    the activity from jets with one definition on both tiers, which removes that
    obstacle — and the dependence is large: measured, CMS rises +2.83 GeV per 100 GeV of
    HT. A flat smearing tuned to match on average therefore over-smears quiet events
    (~57% too wide below HT 180) and under-smears busy ones, and because
    m_ττ = m_vis/√(x₁x₂) is nonlinear in the fitted fractions, that excess noise drags the
    m_ττ MEAN up as well as its width.
    """
    return obs.met_resolution_vs_ht(nano)
