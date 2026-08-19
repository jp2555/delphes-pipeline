"""Merged Delphes ntuples -> the SBI input format the NSBI notebook reads.

    pixi run python scripts/delphes_to_sbi.py \
        --ntuple /ceph/jpan/ntuples_untuned/merged \
        --sample signal --out saved_datasets/dihiggs_delphes_data.root

Target format: one ROOT file per process, one TTree each, flat float64 branches --
identical to what ``convert_powheg_to_sbi.py`` produces from CMS NanoAOD:

    signal  dihiggs_delphes_data.root : tree_sbi_lam0, tree_sbi_lam1, tree_sbi_lam5
    ttbar   ttbar_delphes_data.root   : tree_ttbar
    DY      dy_delphes_data.root      : tree_dy

Required branches:
    m_hh, m_bb, m_tautau, pt_hh, deta_hh, dr_bb, dr_tautau,
    cos_theta_star, dphi_hh, pt_h1, pt_h2, weights

**The definitions here are the CMS side's, not this repo's overlay's, and several
differ.** The point of the exercise is a Delphes baseline comparable with the NSBI
result already obtained on CMS NanoAOD, so a mismatched definition would silently
compare two different measurements:

  * ``m_tautau`` is the VISIBLE di-tau mass (``m_vis``). It is NOT FastMTT -- the CMS
    converter maps ``m_vis -> m_tautau`` and carries FastMTT only as an optional extra.
  * ``cos_theta_star`` is ``tanh(deta_hh / 2)``, not the Collins-Soper polar angle in
    the HH rest frame that ``scripts/nsbi_overlay.py`` computes.
  * ``pt_h1`` is the bb system and ``pt_h2`` the tautau system -- fixed by content, NOT
    sorted by pT (the overlay uses max/min).
  * ``m_hh`` / ``pt_hh`` are built from bb + VISIBLE tautau, so they are not the
    FastMTT-corrected mHH either.
  * ``weights`` is genWeight (x puweight on CMS; Delphes has no pileup, so 1).

Channel: the CMS NSBI test used ``mt`` and ``et`` only -- one light lepton plus one
tau_h. Leg 1 is the lepton, leg 2 the tau_h. This applies the same requirement, so the
tau_h-tau_h and fully-leptonic channels are excluded here as they were there.
"""

from __future__ import annotations

import argparse
import os

import awkward as ak
import numpy as np

from delphes_pipeline.core.io import (NtupleEvents, available_kl,
                                      resolve_ntuple_paths)

#: the only columns these features touch. GenPart is ~99% of the ntuple and is never
#: read: no FastMTT here, so not even MET is needed -- the CMS converter's m_tautau is
#: the VISIBLE di-tau mass.
COLUMNS = ["Jet", "Electron", "Muon", "genWeight", "lepton_sf", "dataset_id", "kl",
           "MET_pt", "MET_phi"]

#: channel code on leg 1, so mt/et can be separated after the fact
CHANNEL = {"mt": 0, "et": 1, "tt": 2}

# --------------------------------------------------------------------------- #
# CMS HIG-25-008 (Run-3 bbtautau, 172/fb) selection, Table 1 and Sections 5-6.
# Cross-trigger thresholds are used (the lower, primary ones).
# --------------------------------------------------------------------------- #
CMS_SEL = {
    "mt": {"lep_pt": 22.0, "lep_eta": 2.4, "tau_pt": 32.0},   # cross mu-tau trigger
    "et": {"lep_pt": 25.0, "lep_eta": 2.5, "tau_pt": 35.0},   # cross e-tau trigger
    # tau_h tau_h: double-tau trigger, pT > 40(35) on BOTH legs (Table 1). CMS's most
    # sensitive channel, and the one a lepton-requiring converter silently drops.
    "tt": {"lep_pt": None, "lep_eta": None, "tau_pt": 40.0},
}
#: The CROWN ntuple baseline (KIT-CMS/BBTauTauAnalysis-CROWN, nmssm_config.py):
#: tight_{muon,electron,tau}_min_pt = 20, |eta| 2.4/2.5/2.5. NOT the paper's
#: trigger thresholds -- those, the elliptical SR and the categorisation are applied
#: later in the analysis. This is the stage the NSBI test's ntuples correspond to.
CROWN_SEL = {
    "mt": {"lep_pt": 20.0, "lep_eta": 2.4, "tau_pt": 20.0},
    "et": {"lep_pt": 20.0, "lep_eta": 2.5, "tau_pt": 20.0},
    "tt": {"lep_pt": None, "lep_eta": None, "tau_pt": 20.0},
}

CMS_TAU_ETA = 2.5
CMS_JET_PT, CMS_JET_ETA, CMS_JET_DR = 20.0, 2.5, 0.5     # H->bb resolved, Sec. 5
CMS_PAIR_DR = 0.5                                        # resolved tau-tau, Table 1
#: elliptical SR in (m_tautau, m_bb): (centre, width) per channel, Eqs. 2-4.
#: ~99% signal efficiency; background efficiency 62/60/54%.
CMS_ELLIPSE = {
    "mt": ((116.0, 61.0), (114.0, 228.0)),
    "et": ((119.0, 57.0), (109.0, 232.0)),
    "tt": ((117.0, 54.0), (134.0, 169.0)),
}

#: kept in step with merge_shards.MIXED_DATASET
MIXED_DATASET = -1

REQUIRED = ["m_hh", "m_bb", "m_tautau", "pt_hh", "deta_hh", "dr_bb", "dr_tautau",
            "cos_theta_star", "dphi_hh", "pt_h1", "pt_h2", "weights"]

# tree name per process, matching config_powheg.yml's Tree field
TREES = {"ttbar": "tree_ttbar", "dy": "tree_dy",
         "singletop": "tree_singletop", "wjets": "tree_wjets"}


def _p4(pt, eta, phi, mass):
    # float64 FIRST. The ntuple stores kinematics as float32, in which the guard
    # `1 - 1e-10` rounds to exactly 1.0, so the clip in _sum_p4 stops guarding and
    # arctanh(1.0) returns inf. The affected events are then silently dropped by the
    # finiteness filter rather than reconstructed.
    pt, eta, phi, mass = (np.asarray(a, dtype=np.float64) for a in (pt, eta, phi, mass))
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return px, py, pz, np.sqrt(px * px + py * py + pz * pz + mass * mass)


def _sum_p4(a, b):
    """(pt, eta, phi, mass) of the sum of two (pt, eta, phi, mass) legs."""
    p1, p2 = _p4(*a), _p4(*b)
    px, py, pz, e = (p1[i] + p2[i] for i in range(4))
    pt = np.hypot(px, py)
    p = np.sqrt(px * px + py * py + pz * pz)
    eta = np.arctanh(np.clip(pz / np.maximum(p, 1e-10), -1 + 1e-10, 1 - 1e-10))
    return pt, eta, np.arctan2(py, px), np.sqrt(np.maximum(e * e - p * p, 0.0))


def _dphi(a, b):
    d = np.abs(a - b)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def _dR(eta1, phi1, eta2, phi2):
    return np.hypot(eta1 - eta2, _dphi(phi1, phi2))


def cms_select(ev, *, btag_min=1, ellipse=True, channels=("mt", "et"),
               cutflow=None, thresholds=None):
    """The HIG-25-008 resolved selection, as far as Delphes can support it.

    APPLIED (Table 1 / Sec. 5-6):
      * channel priority: a muon makes it tau_mu-tau_h, else an electron makes it
        tau_e-tau_h (the tau_h tau_h channel is out of scope for this converter)
      * per-channel pT thresholds at the CROSS-trigger values: mu > 22 with tau_h > 32,
        e > 25 with tau_h > 35; |eta| < 2.4 (mu) / 2.5 (e, tau_h)
      * additional-lepton veto
      * dR(lepton, tau_h) > 0.5
      * H->bb: AK4 jets pT > 20, |eta| < 2.5, dR > 0.5 from BOTH tau candidates
      * b-tag requirement (the card's bit is UParT-AK4 Medium, PATCH-6)
      * elliptical (m_tautau, m_bb) signal region, Eqs. 2-4 -- evaluated on the
        FastMTT mass, since CMS's ellipse is defined on their regressed tau-tau mass,
        NOT on the visible mass the SBI feature carries

    NOT APPLIED, and why:
      * opposite charge -- the ntuple's Jet has no charge field, so the tau_h leg has no
        charge to compare. Needs `charge` added to schema.FLAT_SCHEMA['Jet'] and a
        re-ntuplisation; until then this selection is charge-blind and keeps same-sign
        pairs CMS would reject
      * lepton ID / isolation / impact parameters (d_xy, d_z) -- Delphes leptons carry
        only the card's efficiency parameterisation; there is no ID or IP to cut on
      * HH-BTAG DNN jet assignment -- approximated by the two highest-btag jets
      * DeepTau vs-e / vs-mu working points -- Delphes has one tau bit, not three
        discriminants
      * boosted (AK8) and VBF categories, and the trigger itself
    """
    TH = thresholds or CMS_SEL
    e, m = ev.electrons, ev.muons
    j = ev.jets
    tau_all = j[(j.tautag == 1) & (np.abs(j.eta) <= CMS_TAU_ETA)]
    tau_all = tau_all[ak.argsort(tau_all.pt, axis=1, ascending=False, stable=True)]

    def _lep(coll, ch):
        c = TH[ch]
        sel = coll[(coll.pt > c["lep_pt"]) & (np.abs(coll.eta) <= c["lep_eta"])]
        # zip in the massless four-vector field the builder expects; the raw ntuple
        # lepton carries only pt/eta/phi/charge
        sel = ak.zip({"pt": sel.pt, "eta": sel.eta, "phi": sel.phi,
                      "mass": ak.zeros_like(sel.pt)})
        return sel[ak.argsort(sel.pt, axis=1, ascending=False, stable=True)]

    mu, el = _lep(m, "mt"), _lep(e, "et")
    n_mu, n_el = ak.num(mu), ak.num(el)
    # channel priority, Sec. 5: a muon makes it mt, else an electron makes it et
    is_mt = n_mu >= 1
    is_et = (~is_mt) & (n_el >= 1)
    # additional-lepton veto (Table 1): no second light lepton of either flavour
    veto = ((n_mu + n_el) == 1)

    # Sec. 5: no muon and no electron, but two tau_h -> tau_h tau_h
    tau_tt = tau_all[tau_all.pt > TH["tt"]["tau_pt"]]
    no_lep = (n_mu == 0) & (n_el == 0)
    is_tt = no_lep & (ak.num(tau_tt) >= 2)

    # The cutflow is read off the SAME masks the selection uses -- never recomputed --
    # so it cannot drift away from what the converter actually does.
    def _cf(ch, label, mask):
        if cutflow is not None:
            cutflow.append((ch, label, int(ak.sum(mask))))

    out = []
    for ch, lepcoll, chan_mask in (("mt", mu, is_mt), ("et", el, is_et),
                                   ("tt", None, is_tt)):
        if ch not in channels:
            continue
        if ch == "tt":
            _cf(ch, "no light lepton", no_lep)
            _cf(ch, f">=2 tau_h pT>{TH['tt']['tau_pt']:.0f}", is_tt)
            keep = ak.to_numpy(is_tt)
            if not keep.any():
                continue
            # leg 1 is the harder tau_h (CMS sorts by isolation; Delphes has none)
            pair = tau_tt[keep][:, :2]
            L, T, J = pair[:, 0:1], pair[:, 1:2], j[keep]
        else:
            tau = tau_all[tau_all.pt > TH[ch]["tau_pt"]]
            c = TH[ch]
            lname = "muon" if ch == "mt" else "electron"
            m_chan = chan_mask
            m_veto = m_chan & veto
            m_tau = m_veto & (ak.num(tau) >= 1)
            _cf(ch, f"{lname} pT>{c['lep_pt']:.0f}, |eta|<{c['lep_eta']}", m_chan)
            _cf(ch, "additional-lepton veto", m_veto)
            _cf(ch, f">=1 tau_h pT>{c['tau_pt']:.0f}", m_tau)
            keep = ak.to_numpy(m_tau)
            if not keep.any():
                continue
            L, T, J = lepcoll[keep][:, :1], tau[keep][:, :1], j[keep]
        far_pair = _dR(L[:, 0].eta, L[:, 0].phi, T[:, 0].eta, T[:, 0].phi) > CMS_PAIR_DR
        ok = ak.to_numpy(far_pair)
        if cutflow is not None:
            cutflow.append((ch, f"dR(leg1,leg2)>{CMS_PAIR_DR}", int(ok.sum())))
        L, T, J = L[ok], T[ok], J[ok]
        idx = np.flatnonzero(keep)[ok]

        J = J[(J.pt > CMS_JET_PT) & (np.abs(J.eta) <= CMS_JET_ETA)]
        far = (_dR(J.eta, J.phi, L[:, 0].eta, L[:, 0].phi) > CMS_JET_DR) & \
              (_dR(J.eta, J.phi, T[:, 0].eta, T[:, 0].phi) > CMS_JET_DR)
        J = J[far]
        J = J[ak.argsort(J.pt, axis=1, ascending=False, stable=True)]
        bb = J[ak.argsort(J.btag, axis=1, ascending=False, stable=True)][:, :2]
        ok2 = ak.to_numpy(ak.num(bb) >= 2)
        if cutflow is not None:
            cutflow.append((ch, f">=2 jets pT>{CMS_JET_PT:.0f} "
                                f"|eta|<{CMS_JET_ETA}, dR>{CMS_JET_DR}", int(ok2.sum())))
        if btag_min:
            ok2 &= ak.to_numpy(ak.sum(bb.btag == 1, axis=1) >= btag_min)
            if cutflow is not None:
                cutflow.append((ch, f">={btag_min} b-tag", int(ok2.sum())))
        L, T, bb, J, idx = L[ok2][:, 0], T[ok2][:, 0], bb[ok2], J[ok2], idx[ok2]
        if len(idx) == 0:
            continue

        extra = {"n_jets": ak.to_numpy(ak.num(J)),
                 "n_btag": ak.to_numpy(ak.sum(J.btag == 1, axis=1)),
                 "channel": np.full(len(idx), float(CHANNEL[ch]))}
        extra["fastmtt"] = _fastmtt(ev, L, T, idx, ch)
        if ellipse:
            keep_e = _in_ellipse(ev, L, T, bb, idx, ch)
            if cutflow is not None:
                cutflow.append((ch, "elliptical SR (Eqs. 2-4)", int(keep_e.sum())))
            L, T, bb, idx = L[keep_e], T[keep_e], bb[keep_e], idx[keep_e]
            extra = {k: (tuple(a[keep_e] for a in v) if k == "fastmtt" else v[keep_e])
                     for k, v in extra.items()}
        out.append((L, T, bb, idx, extra))
    return out


def _fastmtt(ev, lep, tau, idx, ch):
    """(m_tautau, x1, x2) from the covariance-free FastMTT fit."""
    from delphes_pipeline.extensions.mtautau import _leg, fastmtt_mass
    met = ak.to_numpy(ev.array["MET_pt"])[idx].astype(np.float64)
    phi = ak.to_numpy(ev.array["MET_phi"])[idx].astype(np.float64)
    had1 = 1.0 if ch == "tt" else 0.0            # leg 1 is a tau_h only in tau_h tau_h
    l1 = ak.zip({"pt": lep.pt, "eta": lep.eta, "phi": lep.phi, "mass": lep.mass,
                 "is_tauh": ak.full_like(lep.pt, had1)})
    l2 = ak.zip({"pt": tau.pt, "eta": tau.eta, "phi": tau.phi, "mass": tau.mass,
                 "is_tauh": ak.ones_like(tau.pt)})
    return fastmtt_mass(_leg(l1), _leg(l2), met * np.cos(phi), met * np.sin(phi),
                        with_x=True)


def _in_ellipse(ev, lep, tau, bb, idx, ch):
    """Eqs. 2-4, on the FastMTT mass -- CMS's ellipse is NOT on the visible mass."""
    m_tt = _fastmtt(ev, lep, tau, idx, ch)[0]
    m_bb = _sum_p4((ak.to_numpy(bb[:, 0].pt), ak.to_numpy(bb[:, 0].eta),
                    ak.to_numpy(bb[:, 0].phi), ak.to_numpy(bb[:, 0].mass)),
                   (ak.to_numpy(bb[:, 1].pt), ak.to_numpy(bb[:, 1].eta),
                    ak.to_numpy(bb[:, 1].phi), ak.to_numpy(bb[:, 1].mass)))[3]
    (c1, w1), (c2, w2) = CMS_ELLIPSE[ch]
    r = ((m_tt - c1) / w1) ** 2 + ((m_bb - c2) / w2) ** 2
    return np.nan_to_num(r, nan=np.inf) < 1.0


def select(ev, *, lep_pt_min=20.0, mu_eta_max=2.4, el_eta_max=2.5,
           tau_pt_min=20.0, tau_eta_max=2.5,
           jet_pt_min=20.0, jet_eta_max=2.5, clean_dr=0.4,
           btag_min=0, lep_veto=False, cutflow=None):
    """The mt/et preselection: one light lepton, one tau_h, two jets.

    Jets are cleaned against BOTH selected legs before the b pair is chosen. On Delphes a
    tau_h *is* an AK4 jet, so without that removal the tau would be free to enter the b
    pair -- an artefact that lands on dr_bb and m_bb.

    **This is a PRESELECTION, and it is the counterpart of the CMS NTUPLE-level
    selection, not of the analysis.** The CROWN ntuples the NSBI test reads
    (KIT-CMS/BBTauTauAnalysis-CROWN, nmssm_config.py) apply pT > 20 on muon, electron
    and tau_h, |eta| < 2.4/2.5/2.5, and -- crucially -- keep tau candidates at the
    **VVVLoose** vs-jet working point ("looser taus needed for tau misidentification
    estimate"), applying Medium only later in the analysis. The trigger thresholds, the
    elliptical signal region and the categorisation all come afterwards.

    The thresholds here match that baseline. One difference cannot be matched: a Delphes
    tau_h carries a single tag bit fixed at a Medium-equivalent working point, so there
    is no VVVLoose to select. That accounts for most of the residual efficiency gap --
    DeepTau Medium is ~70% efficient where VVVLoose is ~98%.

    By default ``btag_min=0``: the b pair is the two highest-btag jets, and an event with
    no tag at all still passes. ``btag_min`` (the card's bit is UParT-AK4 Medium, cut
    0.1272, card PATCH-6) and ``lep_veto`` tighten it. Opposite charge, mass windows,
    m_T cuts, trigger and b-tag categorisation are NOT applied -- see cms_select for
    those.
    """
    # Per-flavour |eta|, matching the CROWN ntuple baseline (BBTauTauAnalysis-CROWN
    # nmssm_config.py): tight_muon_max_abs_eta 2.4, tight_electron_max_abs_eta 2.5.
    e, m = ev.electrons, ev.muons
    e = e[np.abs(e.eta) <= el_eta_max]
    m = m[np.abs(m.eta) <= mu_eta_max]
    lep = ak.concatenate([
        ak.zip({"pt": e.pt, "eta": e.eta, "phi": e.phi,
                "mass": ak.zeros_like(e.pt),
                "channel": ak.full_like(e.pt, CHANNEL["et"])}),
        ak.zip({"pt": m.pt, "eta": m.eta, "phi": m.phi,
                "mass": ak.zeros_like(m.pt),
                "channel": ak.full_like(m.pt, CHANNEL["mt"])}),
    ], axis=1)
    lep = lep[lep.pt > lep_pt_min]
    lep = lep[ak.argsort(lep.pt, axis=1, ascending=False, stable=True)]

    j = ev.jets
    tau = j[(j.tautag == 1) & (j.pt > tau_pt_min) & (np.abs(j.eta) <= tau_eta_max)]
    tau = tau[ak.argsort(tau.pt, axis=1, ascending=False, stable=True)]

    n_lep = ak.num(lep)
    ok_lep = (n_lep == 1) if lep_veto else (n_lep >= 1)
    has_tau = ok_lep & (ak.num(tau) >= 1)
    if cutflow is not None:
        cutflow.append(("pre", f"{'==1' if lep_veto else '>=1'} lepton "
                               f"pT>{lep_pt_min:.0f}", int(ak.sum(ok_lep))))
        cutflow.append(("pre", f">=1 tau_h pT>{tau_pt_min:.0f}, "
                               f"|eta|<{tau_eta_max}", int(ak.sum(has_tau))))
    keep = ak.to_numpy(has_tau)
    lep, tau, j = lep[keep][:, :1], tau[keep][:, :1], j[keep]

    j = j[(j.pt > jet_pt_min) & (np.abs(j.eta) <= jet_eta_max)]
    far = (_dR(j.eta, j.phi, lep[:, 0].eta, lep[:, 0].phi) > clean_dr) & \
          (_dR(j.eta, j.phi, tau[:, 0].eta, tau[:, 0].phi) > clean_dr)
    j = j[far]
    # two hardest b-tagged jets, falling back to the hardest jets, as the CMS bpair does
    j = j[ak.argsort(j.pt, axis=1, ascending=False, stable=True)]
    bb = j[ak.argsort(j.btag, axis=1, ascending=False, stable=True)][:, :2]

    ok = ak.to_numpy(ak.num(bb) >= 2)
    if cutflow is not None:
        cutflow.append(("pre", f">=2 jets pT>{jet_pt_min:.0f} |eta|<{jet_eta_max}, "
                               f"dR>{clean_dr}", int(ok.sum())))
    if btag_min:
        # count over the CLEANED jets, then require it of the pair actually used
        ok &= ak.to_numpy(ak.sum(bb.btag == 1, axis=1) >= btag_min)
        if cutflow is not None:
            cutflow.append(("pre", f">={btag_min} b-tag", int(ok.sum())))
    idx = np.flatnonzero(keep)[ok]
    extra = {"n_jets": ak.to_numpy(ak.num(j[ok])),
             "n_btag": ak.to_numpy(ak.sum(j[ok].btag == 1, axis=1))}
    return lep[ok][:, 0], tau[ok][:, 0], bb[ok], idx, extra


def _mt(pt1, phi1, pt2, phi2):
    return np.sqrt(np.maximum(2 * pt1 * pt2 * (1 - np.cos(phi1 - phi2)), 0.0))


def features(ev, *, cms=False, cutflow=None, **sel_kw):
    """The 12 required branches plus the Table 6.1 extras the CMS ntuple carries."""
    if cms:
        blocks = cms_select(ev, btag_min=sel_kw.get("btag_min", 1),
                            ellipse=sel_kw.get("ellipse", True),
                            channels=sel_kw.get("channels", ("mt", "et")),
                            cutflow=cutflow,
                            thresholds=sel_kw.get("thresholds"))
        if not blocks:
            return {k: np.array([]) for k in REQUIRED}, 0, 0
        outs = [_build(ev, *b) for b in blocks]
        keys = set(outs[0][0]).intersection(*[set(o[0]) for o in outs])
        merged = {k: np.concatenate([o[0][k] for o in outs]) for k in keys}
        return merged, sum(o[1] for o in outs), sum(o[2] for o in outs)
    return _build(ev, *select(ev, cutflow=cutflow, **sel_kw))


def _build(ev, lep, tau, bb, idx, extra):
    g = lambda c, k: ak.to_numpy(c[k])                       # noqa: E731
    b1 = (g(bb[:, 0], "pt"), g(bb[:, 0], "eta"), g(bb[:, 0], "phi"), g(bb[:, 0], "mass"))
    b2 = (g(bb[:, 1], "pt"), g(bb[:, 1], "eta"), g(bb[:, 1], "phi"), g(bb[:, 1], "mass"))
    l1 = (g(lep, "pt"), g(lep, "eta"), g(lep, "phi"), g(lep, "mass"))
    l2 = (g(tau, "pt"), g(tau, "eta"), g(tau, "phi"), g(tau, "mass"))

    if "MET_pt" in ak.fields(ev.array):
        met_pt = ak.to_numpy(ev.array["MET_pt"])[idx].astype(np.float64)
        met_phi = ak.to_numpy(ev.array["MET_phi"])[idx].astype(np.float64)
    else:
        met_pt = met_phi = None

    hbb = _sum_p4(b1, b2)
    htt = _sum_p4(l1, l2)                                     # VISIBLE, no FastMTT
    hh = _sum_p4(hbb, htt)

    deta = np.abs(hbb[1] - htt[1])
    # Ladder-weight rule: lepton_sf is part of the EVENT WEIGHT, not a correction some
    # methods apply and others don't. A weight column consumed by the ratio training and
    # not by the histogram baseline silently breaks N >= D >= H by construction. It is
    # 1.0 on an untuned sample, so this is a no-op for the baseline and correct for v2.
    w = ak.to_numpy(ev.array["genWeight"])[idx].astype(np.float64)
    if "lepton_sf" in ak.fields(ev.array):
        w = w * ak.to_numpy(ev.array["lepton_sf"])[idx].astype(np.float64)
    out = {
        "m_hh": hh[3], "pt_hh": hh[0],
        "m_bb": hbb[3], "m_tautau": htt[3],
        "dr_bb": _dR(b1[1], b1[2], b2[1], b2[2]),
        "dr_tautau": _dR(l1[1], l1[2], l2[1], l2[2]),
        "deta_hh": deta,
        "dphi_hh": _dphi(hbb[2], htt[2]),
        # tanh(deta/2), NOT the boosted-frame polar angle
        "cos_theta_star": np.tanh((hbb[1] - htt[1]) / 2.0),
        "pt_h1": hbb[0],            # bb by CONTENT, not the harder Higgs
        "pt_h2": htt[0],
        "weights": w,
        # --- Table 6.1 extras: present in the CMS ntuple, absent here until now, and
        # needed for any post-hoc channel or selection study on the flat file ---
        "pt_l1": l1[0],
        "pt_b1": b1[0],
        "pt_vis": htt[0],
        "btag_1": g(bb[:, 0], "btag").astype(np.float64),
        "btag_2": g(bb[:, 1], "btag").astype(np.float64),
        "n_jets": extra["n_jets"].astype(np.float64),
        "n_btag": extra["n_btag"].astype(np.float64),
        # channel: 0 = mu-tau_h, 1 = e-tau_h. Without it the flat file cannot be split
        # into mt/et after the fact.
        "channel": (extra["channel"] if "channel" in extra
                    else g(lep, "channel")).astype(np.float64),
    }
    if met_pt is not None and extra.get("fastmtt") is not None:
        # The CMS converter maps the BRANCH mass_tautaubb -> m_hh; ours is computed from
        # the VISIBLE tautau system, so the two are almost certainly different
        # observables. Carry the FastMTT-corrected versions alongside rather than
        # replacing anything, so the two can be compared and the right one chosen.
        m_tt, x1, x2 = extra["fastmtt"]
        def _full(leg, x):
            px, py, pz, e = _p4(*leg)
            nu = np.sqrt(px * px + py * py + pz * pz) * (1 - x) / np.maximum(x, 1e-9)
            return px / x, py / x, pz / x, e + nu
        t1f, t2f = _full(l1, x1), _full(l2, x2)
        pxs = [t1f[i] + t2f[i] for i in range(4)]
        hbbp = _p4(*hbb)
        tot = [hbbp[i] + pxs[i] for i in range(4)]
        out["m_tautau_fastmtt"] = m_tt
        out["m_hh_fastmtt"] = np.sqrt(np.maximum(
            tot[3] ** 2 - tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2, 0.0))
        out["pt_hh_fastmtt"] = np.hypot(tot[0], tot[1])

    if met_pt is not None:
        # CMS mt_tot: quadrature of the three transverse masses of (l, tau, MET)
        out["mt_tot"] = np.sqrt(_mt(l1[0], l1[2], met_pt, met_phi) ** 2
                                + _mt(l2[0], l2[2], met_pt, met_phi) ** 2
                                + _mt(l1[0], l1[2], l2[0], l2[2]) ** 2)
        out["met"] = met_pt
    # dataset_id: one sample can span several CMS primary datasets with DIFFERENT cross
    # sections (ttbar globs three decay channels at 98 / 420 / 406 pb), so the label has
    # to travel with the events or the sample cannot be normalised per channel.
    dropped_mixed = 0
    if "dataset_id" in ak.fields(ev.array):
        out["dataset_id"] = ak.to_numpy(ev.array["dataset_id"])[idx].astype(np.float64)

    out = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
    # Finiteness gates ONLY the REQUIRED branches. The optional extras include the
    # FastMTT quantities, and FastMTT returns NaN wherever its likelihood is everywhere
    # zero -- gating on those threw away ~2/3 of otherwise good events, on a diagnostic
    # the selection does not even use when the ellipse is off. A failed FastMTT leaves a
    # NaN in its own column and nothing else.
    good = np.ones(len(out["m_hh"]), dtype=bool)
    for k in REQUIRED:
        if k in out:
            good &= np.isfinite(out[k])
    if "dataset_id" in out:
        # -1 marks events whose SHARD straddled two primary datasets. Their cross section
        # is genuinely ambiguous; keeping them would mean normalising with a number
        # nobody can justify, so they are dropped here rather than downstream by memory.
        mixed = out["dataset_id"] == MIXED_DATASET
        dropped_mixed = int((good & mixed).sum())
        good &= ~mixed
    return {k: v[good] for k, v in out.items()}, int((~good).sum()), dropped_mixed


def _report(name, d, dropped, dropped_mixed=0):
    n = len(d["m_hh"])
    w = d["weights"]
    neg = int((w < 0).sum())
    print(f"  {name}: {n:,} events ({dropped:,} dropped non-finite), "
          f"sum(w)={w.sum():.1f}" + (f", {neg:,} negative weights" if neg else ""))
    if dropped_mixed:
        print(f"      dropped {dropped_mixed:,} events from shards straddling two "
              f"primary datasets (no unambiguous cross section)")
    if "dataset_id" in d and len(d["dataset_id"]):
        ids, cnt = np.unique(d["dataset_id"].astype(int), return_counts=True)
        print("      per dataset_id: "
              + ", ".join(f"{i}:{c:,}" for i, c in zip(ids, cnt))
              + "  (normalise each with its own xsec from datasets_delphes.json)")
    missing = [b for b in REQUIRED if b not in d]
    if missing:
        raise SystemExit(f"[sbi] missing required branches: {missing}")


def features_streamed(path, *, kl=None, max_events=None, sel_kw=None):
    """features() over one file at a time, concatenating only the OUTPUT.

    The merged tt-bar sample is 298M events across 145 GB; materialising it in one
    awkward array is what made this step run for hours. The selected output is ~1% of
    that and a dozen flat float64 columns, so accumulating per file is bounded by the
    result rather than by the input.
    """
    parts, dropped, mixed, n_read = [], 0, 0, 0
    for f in resolve_ntuple_paths(path):
        if max_events is not None and n_read >= max_events:
            break
        try:
            ev = NtupleEvents(f, kl=kl, columns=COLUMNS,
                              entry_stop=None if max_events is None else max_events - n_read)
        except ValueError:
            continue                        # no events of this kl in this file
        n_read += ev.n
        d, nf, nm = features(ev, **(sel_kw or {}))
        dropped += nf
        mixed += nm
        if len(d["m_hh"]):
            parts.append(d)
        print(f"    {os.path.basename(f)}: {ev.n:,} read -> {len(d['m_hh']):,} kept",
              flush=True)
    if not parts:
        raise SystemExit(f"[sbi] no events selected from {path!r}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return out, dropped, mixed, n_read


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ntuple", required=True, help="merged ntuple dir or glob")
    ap.add_argument("--sample", required=True,
                    help="'signal' (one tree per kl) or a background name (ttbar, dy, ...)")
    ap.add_argument("--out", required=True, help="output ROOT file")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--btag-min", type=int, default=0,
                    help="require this many of the two selected jets to carry the "
                         "UParT-AK4 Medium tag bit (card PATCH-6, cut 0.1272). The "
                         "default 0 is a PRESELECTION: an untagged event still passes, "
                         "which is why signal/ttbar acceptance is ~2.5 and not hundreds")
    ap.add_argument("--crown", action="store_true",
                    help="the CROWN NTUPLE baseline (pT>20 everywhere, no b-tag "
                         "requirement, no elliptical SR) -- the stage the CMS ntuples "
                         "the NSBI test reads correspond to. Implies --cms-selection.")
    ap.add_argument("--cms-selection", action="store_true",
                    help="apply the HIG-25-008 resolved selection instead of the loose "
                         "preselection (see cms_select for what Delphes cannot support)")
    ap.add_argument("--channels", default="mt,et",
                    help="channels to select. DEFAULT mt,et -- the semi-leptonic scope "
                         "the CMS NSBI test uses. Adding 'tt' brings in tau_h tau_h, "
                         "CMS's most sensitive channel, but then the Delphes sample "
                         "covers a final state the CMS test does not and the two are no "
                         "longer comparable.")
    ap.add_argument("--no-ellipse", action="store_true",
                    help="with --cms-selection: skip the elliptical (m_tautau, m_bb) SR")
    ap.add_argument("--lep-veto", action="store_true",
                    help="require EXACTLY one light lepton (CMS vetoes extra leptons)")
    ap.add_argument("--tree", default=None, help="override the output tree name")
    args = ap.parse_args(argv)

    import uproot

    if args.crown:
        args.cms_selection, args.no_ellipse = True, True
    if args.cms_selection:
        chans = tuple(c.strip() for c in args.channels.split(",") if c.strip())
        sel_kw = {"cms": True, "channels": chans,
                  "btag_min": args.btag_min if args.crown else max(args.btag_min, 1),
                  "ellipse": not args.no_ellipse,
                  "thresholds": CROWN_SEL if args.crown else CMS_SEL}
        mode = "CROWN ntuple baseline" if args.crown else "CMS HIG-25-008 resolved"
        print(f"[sbi] selection: {mode}, channels={','.join(chans)}, "
              f"btag_min={sel_kw['btag_min']}, ellipse={sel_kw['ellipse']}")
        if args.crown:
            print("[sbi]   NOT applied vs CROWN: opposite charge (Jet has no charge "
                  "field in these ntuples), tau_h at VVVLoose (Delphes has one "
                  "Medium-equivalent bit), decay-mode requirement")
        print("[sbi]   NOT applied: opposite charge (Jet has no charge field), lepton "
              "ID/isolation/IP, HH-BTAG jet assignment, boosted/VBF categories, trigger")
    else:
        sel_kw = {"btag_min": args.btag_min, "lep_veto": args.lep_veto}
        print(f"[sbi] selection: PRESELECTION — >=1 lepton"
              f"{' (exactly 1)' if args.lep_veto else ''}, >=1 tau_h, 2 jets, "
              f"btag_min={args.btag_min}")
    trees = {}
    if args.sample == "signal":
        for kl in available_kl(args.ntuple):
            name = (f"tree_sbi_lam{int(kl)}" if kl == int(kl)
                    else "tree_sbi_lam" + str(kl).replace(".", "p"))
            print(f"  {name} (kl={kl:g}) ...", flush=True)
            d, dropped, dm, n_read = features_streamed(
                args.ntuple, kl=kl, max_events=args.max_events, sel_kw=sel_kw)
            _report(f"{name} (kl={kl:g}, {n_read:,} read)", d, dropped, dm)
            trees[name] = d
    else:
        name = args.tree or TREES.get(args.sample, f"tree_{args.sample}")
        print(f"  {name} ...", flush=True)
        d, dropped, dm, n_read = features_streamed(
            args.ntuple, max_events=args.max_events, sel_kw=sel_kw)
        _report(f"{name} ({n_read:,} read)", d, dropped, dm)
        trees[name] = d

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with uproot.recreate(args.out) as fh:
        for name, d in trees.items():
            fh[name] = d
    print(f"[sbi] wrote {args.out}: {', '.join(sorted(trees))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
