"""The one measurement layer: object response measured from a ``DelphesEvents``.

Every lens consumes these same measurements:

- **validation** (the gate): ``closure.closure_from_profile`` compares a ``Profile``
  to the card-formula target and returns a pass/fail ``CheckResult``;
- **tuning** (``delphes_pipeline.tuning``): compares the same ``Profile`` to a
  digitised POG/anchor target and returns a residual + which card knob to turn;
- **plots** (``delphes_pipeline.plots``): renders the ``Profile``s and raw spectra.

A ``Profile`` is a quantity binned in one variable: per-bin value, error, and
count. ``kind`` distinguishes an efficiency/rate, a resolution, and an
energy-response so consumers format axes and errors correctly. The selection
logic for each quantity lives here once (the Level-0 leaves are thin wrappers),
so retuning a selection happens in a single place.
"""

from __future__ import annotations

from typing import Optional

from dataclasses import asdict, dataclass

import awkward as ak
import numpy as np

from .io import DelphesEvents
from .matching import matched_to_any, nearest_target_field, nearest_target_fields, unique_match

DEFAULT_PT_BINS = [20, 30, 40, 50, 70, 100, 150, 200, 300]
DEFAULT_SUMET_BINS = [0, 100, 200, 300, 500, 800, 1200]

# quantity -> Jet.Flavor selecting its jets (b-tag closure populations)
BTAG_FLAVORS = {"btag_eff_b": 5, "btag_eff_c": 4, "btag_mistag_light": 0}

_GEN_TAU_PID = 15
_PROMPT_MOTHER_PIDS = (15, 23, 24)  # tau, Z, W -- standard prompt-lepton sources


@dataclass
class Profile:
    """A measured quantity binned in one variable ``x``."""

    quantity: str
    x: str
    centers: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    counts: np.ndarray
    kind: str = "efficiency"  # efficiency | resolution | response
    xlabel: str = ""
    ylabel: str = ""
    # optional per-bin distribution shape (plain JSON-able lists), for quantities where
    # the bin summary alone is not enough to reproduce the observable downstream — e.g.
    # the τ_h visible mass, which enters FastMTT as a per-object kinematic bound rather
    # than an average, so its spread has to survive into the tuning map.
    aux: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("centers", "values", "errors", "counts"):
            d[k] = np.asarray(d[k]).tolist()
        return d


@dataclass
class PeakMetrics:
    """Scalar peak descriptors of a mass distribution (e.g. m_bb)."""

    quantity: str
    peak: float          # core-window median
    width: float         # core-window std
    core_fraction: float # fraction of pairs inside the core window
    n_core: int
    window: tuple
    n_pairs: int

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Binning primitives
# --------------------------------------------------------------------------- #
def binned_efficiency(x_values, passed, bins, *, quantity="", x="pt") -> Profile:
    """Per-bin pass rate of ``passed`` over ``x_values`` (binomial error)."""
    x_values = np.asarray(x_values, dtype=float)
    passed = np.asarray(passed, dtype=bool)
    bins = np.asarray(bins, dtype=float)
    centers, values, errors, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (x_values >= lo) & (x_values < hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        p = float(passed[in_bin].sum()) / n
        centers.append(float(x_values[in_bin].mean()))  # bin mean, not midpoint
        values.append(p)
        errors.append(float(np.sqrt(max(p * (1.0 - p), 0.0) / n)))
        counts.append(n)
    return Profile(quantity, x, np.asarray(centers), np.asarray(values),
                   np.asarray(errors), np.asarray(counts, dtype=int), kind="efficiency")


def binned_response(x_values, ratio, bins, *, quantity="", x="pt") -> Profile:
    """Per-bin median of ``ratio`` (reco/gen response) with error on the median."""
    x_values = np.asarray(x_values, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    bins = np.asarray(bins, dtype=float)
    centers, values, errors, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (x_values >= lo) & (x_values < hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        r = ratio[in_bin]
        med = float(np.median(r))
        # robust spread / sqrt(n) as the error on the median
        sigma = 1.4826 * float(np.median(np.abs(r - med)))
        centers.append(float(x_values[in_bin].mean()))  # bin mean, not midpoint
        values.append(med)
        errors.append(sigma / np.sqrt(n) if n else float("nan"))
        counts.append(n)
    return Profile(quantity, x, np.asarray(centers), np.asarray(values),
                   np.asarray(errors), np.asarray(counts, dtype=int), kind="response")


def binned_resolution(x_values, dx, dy, bins, *, min_count=25, quantity="met_resolution", x="sumet") -> Profile:
    """Per-bin per-component resolution sqrt(0.5*(var(dx)+var(dy)))."""
    x_values = np.asarray(x_values, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    bins = np.asarray(bins, dtype=float)
    centers, values, errors, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (x_values >= lo) & (x_values < hi)
        n = int(in_bin.sum())
        if n < min_count:
            continue
        res = float(np.sqrt(0.5 * (np.var(dx[in_bin]) + np.var(dy[in_bin]))))
        centers.append(float(x_values[in_bin].mean()))  # bin mean, not midpoint
        values.append(res)
        errors.append(res / np.sqrt(2.0 * n))  # ~ error on a standard deviation
        counts.append(n)
    return Profile(quantity, x, np.asarray(centers), np.asarray(values),
                   np.asarray(errors), np.asarray(counts, dtype=int), kind="resolution")


# --------------------------------------------------------------------------- #
# Gen helpers
# --------------------------------------------------------------------------- #
def mother_pid(gen: ak.Array) -> ak.Array:
    """Per-entry mother PID via ``gen.m1`` lookup (0 where the index is invalid)."""
    n = ak.num(gen)
    m1 = gen.m1
    valid = (m1 >= 0) & (m1 < n)
    safe = ak.where(valid, m1, 0)
    return ak.where(valid, gen.pid[safe], 0)


def prompt_mother_match(gen: ak.Array, prompt_pids=_PROMPT_MOTHER_PIDS, max_depth: int = 12,
                        rows: Optional[ak.Array] = None) -> ak.Array:
    """Per-entry bool: is the first non-self-copy ancestor a prompt source?

    A single ``m1`` hop is wrong on the full Pythia / pruned-NanoAOD record: a
    status-1 lepton's direct mother is usually a same-|PID| copy of itself (the
    FSR / last-copy chain), with the real τ/Z/W several links up. We walk ``m1``
    past those self-copies, then test whether the first genuinely different
    ancestor is in ``prompt_pids`` (15=τ, 23=Z, 24=W).

    ``rows`` restricts the WALK to a boolean subset, returning the answer over
    ``gen[rows]`` instead of the whole record. Every caller immediately ANDs the result
    with a narrow selection (status-1 e/μ, or ν_τ — of order 10 rows against a 2000-entry
    Pythia record), so walking everything is pure waste: each of the ``max_depth``
    iterations does two full-record jagged gathers, and this dominates the tuning
    derivation. Ancestors are still looked up in the FULL record, so the answer is
    unchanged — only the set of starting points shrinks.
    """
    n = ak.num(gen)
    sub = gen if rows is None else gen[rows]
    self_apid = np.abs(sub.pid)
    cur = sub.m1
    for _ in range(max_depth):
        valid = (cur >= 0) & (cur < n)
        safe = ak.where(valid, cur, 0)
        apid = ak.where(valid, np.abs(gen.pid[safe]), -1)
        advance = valid & (apid == self_apid)        # same-flavour copy -> keep walking
        cur = ak.where(advance, gen.m1[safe], cur)
    valid = (cur >= 0) & (cur < n)
    safe = ak.where(valid, cur, 0)
    anc = ak.where(valid, np.abs(gen.pid[safe]), 0)
    match = anc == prompt_pids[0]
    for src in prompt_pids[1:]:
        match = match | (anc == src)
    return match


def tau_ancestor_index(gen: ak.Array, max_depth: int = 12,
                       rows: Optional[ak.Array] = None) -> ak.Array:
    """Per-entry index of the τ each particle descends from (-1 if it descends from none).

    Walks ``m1`` upward and stops at the first τ. Unlike a ΔR proximity test this is
    exact: a τ→ℓνν daughter is identified by *descent*, so a soft or wide-angle lepton
    cannot escape the classification.

    ``rows`` restricts the walk to a subset; see ``prompt_mother_match``.
    """
    n = ak.num(gen)
    sub = gen if rows is None else gen[rows]
    cur = sub.m1
    found = ak.zeros_like(sub.m1) - 1
    for _ in range(max_depth):
        valid = (cur >= 0) & (cur < n)
        safe = ak.where(valid, cur, 0)
        is_tau = valid & (np.abs(gen.pid[safe]) == _GEN_TAU_PID)
        found = ak.where((found < 0) & is_tau, cur, found)
        cur = ak.where(valid & ~is_tau, gen.m1[safe], -1)   # stop once a τ is found
    return found


def gen_taus(gen: ak.Array, *, hadronic_only: bool = False, dr: float = 0.4,
             veto: str = "geometric", last_copy: bool = True) -> ak.Array:
    """Gen τ leptons, optionally only those decaying hadronically.

    Delphes' gen record carries no decay-mode flag, so a leptonic τ is identified by its
    own daughter: a status-1 e/μ descended from a τ and lying within ``dr`` of it
    (``veto="geometric"``, the default).

``veto="descent"`` instead uses the m1 chain to find *which* τ a lepton came
    from. Once generator copies are collapsed (``last_copy``) the two are equivalent;
    scored against CMS ``GenVisTau`` on 200k anchor events (``scripts/gen_tau_check.py``):

        veto        objects   eff      purity    stage-1 ratio
        geometric   0.992x    0.9908   0.9983    0.984
        descent     0.983x    0.9818   0.9989    0.965

    Geometric is kept as the default for its slightly higher efficiency. Note the history:
    BEFORE copy collapse, descent scored purity 0.831 and inflated stage 1 by 1.46, because
    it vetoes only the single copy the lepton's chain points at while the geometric test
    removes every copy at once (they are collinear). That was a property of the copies,
    not of the method.

    This matters because ``tau_eff`` is measured on the anchor as ``GenVisTau`` →
    DeepTau-Medium ``Tau``, and ``GenVisTau`` is **hadronic-only**. Treating every gen τ
    as "genuine" hands a leptonic τ's jet the hadronic efficiency (~0.5) instead of the
    jet→τ_h fake rate (~0.004) — a ~125× over-efficiency that manufactures τ_hτ_h
    events out of τ_hτ_ℓ and τ_ℓτ_ℓ ones, whose objects are not collimated.
    """
    is_tau = np.abs(gen.pid) == _GEN_TAU_PID
    if last_copy:
        # Keep one entry per PHYSICAL τ. The Delphes `allParticles` record is the full
        # Pythia history, so each τ appears several times (23 -> 22 -> 2 ...); counting
        # those copies makes ">=2 τ" satisfiable by one τ alone. A τ is the last copy iff
        # no other τ lists it as its mother. (NanoAOD GenPart is pruned and already
        # ~1 entry per τ, so this is a no-op there.)
        idx = ak.local_index(gen)[is_tau]
        m1 = gen.m1[is_tau]
        pairs = ak.cartesian({"i": idx, "m": m1}, nested=True)
        is_parent = ak.fill_none(ak.any(pairs["i"] == pairs["m"], axis=-1), False)
        taus = gen[is_tau][~is_parent]
    else:
        is_parent = None
        taus = gen[is_tau]
    if not hadronic_only:
        return taus
    is_lep = ((np.abs(gen.pid) == 11) | (np.abs(gen.pid) == 13)) & (gen.status == 1)
    if veto == "geometric":
        # legacy: veto a τ with a τ-descended lepton within dr. Misses soft/wide-angle
        # daughters, so some leptonic τ are wrongly kept — kept only for comparison.
        from_tau = gen[is_lep][prompt_mother_match(gen, (_GEN_TAU_PID,), rows=is_lep)]
        return taus[~matched_to_any(taus, from_tau, dr)]
    if veto != "descent":
        raise ValueError(f"veto must be 'descent' or 'geometric' (got {veto!r})")
    # exact: a τ is leptonic iff some status-1 e/μ descends from *that* τ
    anc_lep = tau_ancestor_index(gen, rows=is_lep)      # walk only the status-1 e/μ
    lep_anc = anc_lep[anc_lep >= 0]
    tau_idx = ak.local_index(gen)[np.abs(gen.pid) == _GEN_TAU_PID]
    if last_copy:
        tau_idx = tau_idx[~is_parent]
    pairs = ak.cartesian({"i": tau_idx, "a": lep_anc}, nested=True)
    is_leptonic = ak.fill_none(ak.any(pairs["i"] == pairs["a"], axis=-1), False)
    return taus[~is_leptonic]


def gen_visible_taus(gen: ak.Array, *, dr: float = 0.4, veto: str = "descent") -> ak.Array:
    """Visible hadronic gen τ — the Delphes analogue of NanoAOD's ``GenVisTau``.

    A hadronic τ decay has exactly ONE neutrino, so the visible four-vector is simply
    ``τ − ν_τ``; the ν_τ is found with the same m1 ancestor walk used elsewhere (no gen
    daughter links needed). This matters because it is the only reference that makes the
    Delphes and CMS τ energy responses comparable: a Delphes ``GenJet`` is an R=0.4
    cluster and still contains the underlying event (only *pileup* is absent from the
    card), so profiling against it leaves the Delphes τ ~10% harder than a CMS ``Tau``
    profiled against ``GenVisTau`` — which inflates m_vis and hence m_ττ.
    """
    taus = gen_taus(gen, hadronic_only=True, dr=dr, veto=veto)
    is_nu = (np.abs(gen.pid) == 16) & (gen.status == 1)
    nu = gen[is_nu][prompt_mother_match(gen, (_GEN_TAU_PID,), rows=is_nu)]
    matched, v = nearest_target_fields(taus, nu, dr, ("pt", "eta", "phi"))
    tpt = ak.to_numpy(ak.flatten(taus.pt))
    teta = ak.to_numpy(ak.flatten(taus.eta))
    tphi = ak.to_numpy(ak.flatten(taus.phi))
    take = lambda k: np.where(matched, np.nan_to_num(v[k], nan=0.0), 0.0)
    npt, neta, nphi = take("pt"), take("eta"), take("phi")
    px = tpt * np.cos(tphi) - npt * np.cos(nphi)
    py = tpt * np.sin(tphi) - npt * np.sin(nphi)
    pz = tpt * np.sinh(teta) - npt * np.sinh(neta)
    pt = np.hypot(px, py)
    eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pt), where=pt > 0))
    counts = ak.num(taus)
    vis = ak.zip({"pt": ak.unflatten(pt, counts), "eta": ak.unflatten(eta, counts),
                  "phi": ak.unflatten(np.arctan2(py, px), counts),
                  "mass": ak.unflatten(np.zeros_like(pt), counts)})
    # DROP τ whose ν_τ was not found rather than falling back to the full τ: that
    # fallback would silently profile against a too-hard reference and bias the escale
    # the wrong way. Measuring fewer τ is safe; measuring the wrong reference is not.
    return vis[ak.unflatten(matched, counts)]


def _vis_pt_eta_phi_mass(coll):
    return (ak.to_numpy(ak.flatten(coll.pt)), ak.to_numpy(ak.flatten(coll.eta)),
            ak.to_numpy(ak.flatten(coll.phi)), ak.to_numpy(ak.flatten(coll.mass)))


# --------------------------------------------------------------------------- #
# Efficiency / rate extractors (selection lives here once)
# --------------------------------------------------------------------------- #
def btag_efficiency(events: DelphesEvents, quantity: str, *, bins=DEFAULT_PT_BINS, eta_max=2.5) -> Profile:
    """b-tag efficiency / mistag rate vs jet pT for the flavour of ``quantity``."""
    flavor = BTAG_FLAVORS[quantity]
    jets = events.jets
    pt = ak.to_numpy(ak.flatten(jets.pt))
    eta = ak.to_numpy(ak.flatten(jets.eta))
    flav = ak.to_numpy(ak.flatten(jets.flavor))
    tagged = ak.to_numpy(ak.flatten(jets.btag)) == 1
    sel = (np.abs(eta) <= eta_max) & (flav == flavor)
    prof = binned_efficiency(pt[sel], tagged[sel], bins, quantity=quantity, x="pt")
    prof.xlabel, prof.ylabel = "jet pT [GeV]", quantity
    return prof


def tau_efficiency(events: DelphesEvents, *, bins=DEFAULT_PT_BINS, dr=0.4, eta_max=2.5,
                   pt_min=20.0, x: str = "gen_pt", hadronic_only: bool = False) -> Profile:
    """τ_h efficiency: TauTag rate of the unique nearest jet to each acceptance gen τ.

    ``x`` selects the binning variable — the same numerator/denominator either way.
    **Both lenses want** ``"jet_pt"``; the gen axis is kept only as a diagnostic.

    - ``"jet_pt"``: bin by the **matched reco jet** pT. Correct for *both* lenses:

      * *closure* — Delphes' ``TauTagging`` evaluates its ``EfficiencyFormula`` at the
        jet kinematics, so a pT-dependent card formula only closes against itself on
        this axis. The b-tag closures are jet-pT for exactly this reason.
      * *tuning* — the NanoAOD anchor is binned in ``GenVisTau`` pT, i.e. the τ's
        **visible** decay products. Its Delphes counterpart is the τ-jet (Delphes jets
        run after ``NeutrinoFilter``), not the full gen τ. The residual jet-vs-visible
        difference is the τ energy response, measured separately as ``tau_escale``.

    - ``"gen_pt"``: bin by the **full gen τ** pT, neutrinos included. This is neither
      the axis the card is applied at nor the anchor's ``GenVisTau`` definition — a
      hadronic τ carries only ~65% of its pT visibly, so comparing this axis to the
      anchor reads a pure axis mismatch as a several-percent efficiency deficit.
      Retained for diagnostics only; do not use it against the anchor.

    Verified in the Delphes source (``modules/TauTagging.cc``, identical in 3.5.0 /
    3.5.1pre05 / master): ``pt``/``eta`` are set once per jet from ``jet->Momentum``
    and never reassigned, then ``formula->Eval(pt, eta, phi, e)`` — the tau parton
    only selects *which* formula (the ``fEfficiencyMap`` key), never its arguments.
    ``BTagging.cc`` follows the same pattern.

    ``hadronic_only`` restricts the denominator to hadronically-decaying gen τ. The
    *tuning* lens wants this (the anchor's ``GenVisTau`` is hadronic-only); the
    *closure* does not, because Delphes' own ``TauTagging`` applies the {15} formula to
    any τ-matched jet — its direct τ→ℓνν veto is commented out in the source.
    """
    if x not in ("gen_pt", "jet_pt"):
        raise ValueError(f"x must be 'gen_pt' or 'jet_pt' (got {x!r})")
    jets = events.jets
    taus = gen_taus(events.gen, hadronic_only=hadronic_only, dr=dr)
    acc = jets[(np.abs(jets.eta) <= eta_max) & (jets.pt > pt_min)]
    taus_acc = taus[(np.abs(taus.eta) <= eta_max) & (taus.pt > pt_min)]
    matched, vals = nearest_target_fields(taus_acc, acc, dr, ("tautag", "pt"))
    jet_tautag = vals["tautag"]
    axis = (ak.to_numpy(ak.flatten(taus_acc.pt)) if x == "gen_pt" else vals["pt"])
    prof = binned_efficiency(axis[matched], jet_tautag[matched] == 1, bins, quantity="tau_eff", x="pt")
    prof.xlabel = "tau pT [GeV]" if x == "gen_pt" else "matched jet pT [GeV]"
    prof.ylabel = "tau_eff"
    return prof


def tau_visible_mass(events: DelphesEvents, *, bins=DEFAULT_PT_BINS, eta_max=2.5, pt_min=20.0) -> Profile:
    """Median visible mass of the τ_h candidates vs their pT.

    In Delphes a τ_h *is* a jet, so its ``mass`` is the AK4 jet mass — clustering
    contamination (UE, extra particles in the R=0.4 cone), not a τ property. A real
    τ_h visible mass is bounded by m_τ = 1.777 GeV and set by the decay mode
    (π 0.14, ρ 0.77, a₁ 1.26). The gap breaks the di-τ mass estimator outright: the
    FastMTT hadronic decay prior is zero unless ``(m_vis/m_τ)² ≤ 1``, so a τ-jet
    carrying a multi-GeV jet mass yields no valid solution at all.

    Measured here (Delphes) and on the anchor (CMS ``Tau``) so the difference becomes
    a tuning map, like ``tau_escale``. NB the map is *sampled* per τ-jet, not set to this
    median: xmin is a one-sided floor, so a single value relaxes it for every leg whose
    true mass is higher and drags m_ττ up. This profile is the lens/reporting view.
    """
    jets = events.jets
    th = jets[(jets.tautag == 1) & (np.abs(jets.eta) <= eta_max) & (jets.pt > pt_min)]
    prof = binned_response(ak.to_numpy(ak.flatten(th.pt)), ak.to_numpy(ak.flatten(th.mass)),
                           bins, quantity="tau_mass", x="pt")
    prof.xlabel, prof.ylabel = "tau_h pT [GeV]", "visible mass [GeV]"
    return prof


def tau_mistag(events: DelphesEvents, *, bins=DEFAULT_PT_BINS, dr=0.4, eta_max=2.5,
               pt_min=20.0, hadronic_only: bool = False) -> Profile:
    """jet→τ_h mistag: TauTag rate among jets not near a gen τ.

    ``hadronic_only`` sets WHICH τ are vetoed from the fake sample, and the two lenses
    need different answers. The anchor vetoes ``GenVisTau``, which exists only for
    hadronic decays, so a jet from a leptonic τ counts as a fake candidate on the CMS
    side; the *tuning* comparison is only like-for-like if Delphes does the same
    (hadronic_only=True). The *closure* keeps vetoing every τ, because that is the
    population Delphes' own TauTagging acts on.
    """
    jets = events.jets
    taus = gen_taus(events.gen, hadronic_only=hadronic_only, dr=dr)
    acc = jets[(np.abs(jets.eta) <= eta_max) & (jets.pt > pt_min)]
    fake = acc[~matched_to_any(acc, taus, dr)]
    prof = binned_efficiency(ak.to_numpy(ak.flatten(fake.pt)),
                             ak.to_numpy(ak.flatten(fake.tautag)) == 1, bins, quantity="tau_mistag", x="pt")
    prof.xlabel, prof.ylabel = "jet pT [GeV]", "tau_mistag"
    return prof


def lepton_efficiency(events: DelphesEvents, quantity: str, *, bins=DEFAULT_PT_BINS,
                      barrel=1.5, dr=0.2, prompt_pids=_PROMPT_MOTHER_PIDS) -> Profile:
    """Prompt barrel e/μ reconstruction efficiency vs pT (unique gen→reco match)."""
    pid = 11 if quantity == "electron_eff" else 13
    reco = events.electrons if pid == 11 else events.muons
    gen = events.gen
    pt_min = float(np.asarray(bins, dtype=float)[0])
    # narrow FIRST, then walk only those rows — the walk is the dominant cost
    sel = (np.abs(gen.pid) == pid) & (gen.status == 1) & (np.abs(gen.eta) <= barrel) & (gen.pt > pt_min)
    g = gen[sel][prompt_mother_match(gen, prompt_pids, rows=sel)]
    matched = unique_match(g, reco, dr)
    prof = binned_efficiency(ak.to_numpy(ak.flatten(g.pt)), matched, bins, quantity=quantity, x="pt")
    prof.xlabel, prof.ylabel = "lepton pT [GeV]", quantity
    return prof


# --------------------------------------------------------------------------- #
# MET, energy response, m_bb (tuning observables)
# --------------------------------------------------------------------------- #
def met_residuals(events: DelphesEvents):
    """``(dx, dy, sumet)`` numpy arrays of (reco MET - gen MET) and sum E_T."""
    def xy(rec):
        met = ak.to_numpy(ak.fill_none(rec.met, 0.0))
        phi = ak.to_numpy(ak.fill_none(rec.phi, 0.0))
        return met * np.cos(phi), met * np.sin(phi)

    mx, my = xy(events.met)
    gmx, gmy = xy(events.genmet)
    sumet = ak.to_numpy(ak.fill_none(events.scalar_ht.ht, 0.0))
    return mx - gmx, my - gmy, sumet


def met_resolution(events: DelphesEvents, *, bins=DEFAULT_SUMET_BINS, min_count=25) -> Profile:
    """MET resolution vs sum E_T."""
    dx, dy, sumet = met_residuals(events)
    prof = binned_resolution(sumet, dx, dy, bins, min_count=min_count)
    prof.xlabel, prof.ylabel = "sum E_T [GeV]", "MET resolution [GeV]"
    return prof


def _response_to_genjet(probe_jets, genjets, dr, quantity, xlabel, ylabel, bins) -> Profile:
    """Response = reco-jet pT / matched nearest GenJet pT, profiled vs GenJet pT.

    GenJets are neutrino-filtered in this card, so the matched GenJet is the
    *visible* reference (the visible-τ jet for τ-jets; the b-hadron jet for b-jets).
    """
    matched, genjet_pt = nearest_target_field(probe_jets, genjets, dr, "pt")
    reco_pt = ak.to_numpy(ak.flatten(probe_jets.pt))
    ok = matched & (genjet_pt > 0)
    response = reco_pt[ok] / genjet_pt[ok]
    prof = binned_response(genjet_pt[ok], response, bins, quantity=quantity, x="pt")
    prof.xlabel, prof.ylabel = xlabel, ylabel
    return prof


def tau_energy_response(events: DelphesEvents, *, bins=DEFAULT_PT_BINS, dr_tau=0.4, dr_gen=0.4,
                        eta_max=2.5, pt_min=20.0) -> Profile:
    """τ-jet energy response: reco τ-jet pT / VISIBLE gen-τ pT vs gen pT (§3.2).

    The reference is the neutrino-subtracted gen τ (``gen_visible_taus``), NOT the
    matched GenJet: a GenJet still carries the underlying event inside R=0.4, so it is
    not the same object as the CMS ``GenVisTau`` the anchor profiles against. Using it
    would leave a ~10% Delphes-vs-CMS τ energy gap that the escale ratio cannot see.
    """
    jets = events.jets
    vis = gen_visible_taus(events.gen, dr=dr_tau)
    acc = jets[(np.abs(jets.eta) <= eta_max) & (jets.pt > pt_min)]
    matched, ref = nearest_target_fields(acc, vis, dr_tau, ("pt",))
    reco_pt = ak.to_numpy(ak.flatten(acc.pt))
    ref_pt = np.nan_to_num(ref["pt"], nan=0.0)
    ok = matched & (ref_pt > 0)
    prof = binned_response(ref_pt[ok], reco_pt[ok] / ref_pt[ok], bins,
                           quantity="tau_energy_response", x="pt")
    prof.xlabel, prof.ylabel = "gen visible-tau pT [GeV]", "reco/gen pT"
    return prof


def bjet_energy_response(events: DelphesEvents, *, bins=DEFAULT_PT_BINS, dr=0.2) -> Profile:
    """b-jet energy response: reco b-jet pT / GenJet pT vs pT (drives m_bb, §4.3)."""
    jets = events.jets
    bjets = jets[jets.flavor == 5]
    return _response_to_genjet(bjets, events.genjets, dr,
                               "bjet_energy_response", "gen-jet pT [GeV]", "reco/gen pT", bins)


def mbb_values(events: DelphesEvents) -> np.ndarray:
    """Per-event visible AK4 di-jet mass of the two highest-BTag-then-pT jets."""
    jets = events.jets
    sel = jets[ak.num(jets) >= 2]
    pt_sorted = sel[ak.argsort(sel.pt, axis=1, ascending=False, stable=True)]
    lead = pt_sorted[ak.argsort(pt_sorted.btag, axis=1, ascending=False, stable=True)][:, :2]
    return _pair_mass(lead)


def mbb_peak(events: DelphesEvents, *, window=(100.0, 150.0)) -> PeakMetrics:
    """Visible AK4 di-jet (two highest-BTag-then-pT jets) peak position & width."""
    mbb = mbb_values(events)
    n_pairs = int(mbb.size)
    lo, hi = window
    core = mbb[(mbb > lo) & (mbb < hi)]
    n_core = int(core.size)
    return PeakMetrics(
        quantity="mbb",
        peak=float(np.median(core)) if n_core else float("nan"),
        width=float(np.std(core)) if n_core else float("nan"),
        core_fraction=float(n_core / n_pairs) if n_pairs else 0.0,
        n_core=n_core,
        window=tuple(window),
        n_pairs=n_pairs,
    )


def _pair_mass(pair) -> np.ndarray:
    """Invariant mass of the two jets per event (one value per event)."""
    pt, eta, phi, mass = pair.pt, pair.eta, pair.phi, pair.mass
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    m2 = ak.sum(e, axis=1) ** 2 - (ak.sum(px, axis=1) ** 2 + ak.sum(py, axis=1) ** 2 + ak.sum(pz, axis=1) ** 2)
    return np.sqrt(np.maximum(ak.to_numpy(m2), 0.0))
