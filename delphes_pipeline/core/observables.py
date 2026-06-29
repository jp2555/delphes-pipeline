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

from dataclasses import asdict, dataclass

import awkward as ak
import numpy as np

from .io import DelphesEvents
from .matching import matched_to_any, nearest_target_field, unique_match

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
        centers.append(0.5 * (lo + hi))
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
        centers.append(0.5 * (lo + hi))
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
        centers.append(0.5 * (lo + hi))
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


def tau_efficiency(events: DelphesEvents, *, bins=DEFAULT_PT_BINS, dr=0.4, eta_max=2.5, pt_min=20.0) -> Profile:
    """τ_h efficiency: TauTag rate of the unique nearest jet to each acceptance gen τ."""
    jets = events.jets
    gen = events.gen
    gen_taus = gen[np.abs(gen.pid) == _GEN_TAU_PID]
    acc = jets[(np.abs(jets.eta) <= eta_max) & (jets.pt > pt_min)]
    taus_acc = gen_taus[(np.abs(gen_taus.eta) <= eta_max) & (gen_taus.pt > pt_min)]
    matched, jet_tautag = nearest_target_field(taus_acc, acc, dr, "tautag")
    tau_pt = ak.to_numpy(ak.flatten(taus_acc.pt))
    prof = binned_efficiency(tau_pt[matched], jet_tautag[matched] == 1, bins, quantity="tau_eff", x="pt")
    prof.xlabel, prof.ylabel = "tau pT [GeV]", "tau_eff"
    return prof


def tau_mistag(events: DelphesEvents, *, bins=DEFAULT_PT_BINS, dr=0.4, eta_max=2.5, pt_min=20.0) -> Profile:
    """jet→τ_h mistag: TauTag rate among jets not near any gen τ."""
    jets = events.jets
    gen_taus = events.gen[np.abs(events.gen.pid) == _GEN_TAU_PID]
    acc = jets[(np.abs(jets.eta) <= eta_max) & (jets.pt > pt_min)]
    fake = acc[~matched_to_any(acc, gen_taus, dr)]
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
    mom = np.abs(mother_pid(gen))
    prompt = mom == prompt_pids[0]
    for src in prompt_pids[1:]:
        prompt = prompt | (mom == src)
    pt_min = float(np.asarray(bins, dtype=float)[0])
    sel = (np.abs(gen.pid) == pid) & (gen.status == 1) & (np.abs(gen.eta) <= barrel) & (gen.pt > pt_min) & prompt
    g = gen[sel]
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
    """τ-jet energy response: reco τ-jet pT / visible-τ (GenJet) pT vs pT (§3.2)."""
    jets = events.jets
    gen_taus = events.gen[np.abs(events.gen.pid) == _GEN_TAU_PID]
    acc = jets[(np.abs(jets.eta) <= eta_max) & (jets.pt > pt_min)]
    tau_jets = acc[matched_to_any(acc, gen_taus, dr_tau)]  # reco jets that are τ_h
    return _response_to_genjet(tau_jets, events.genjets, dr_gen,
                               "tau_energy_response", "gen-jet pT [GeV]", "reco/gen pT", bins)


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
