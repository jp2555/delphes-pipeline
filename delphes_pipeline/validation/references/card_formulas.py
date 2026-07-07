"""Closure targets transcribed from the production cards.

``expected(quantity, pt, eta)`` evaluates the efficiency/response the card is
*configured* to produce, so the Level-0 measurement can be overlaid against it
(the "do the modules behave as configured?" closure check). Each formula mirrors
a specific card block and line, noted inline; if a card is retuned, update its
formulas here in lockstep.

Two cards are transcribed:

- ``expected``    — ``cards/cms_card_v0.tcl`` (stock Run-1-era taggers; the
  provenance of the already-produced /ceph samples);
- ``expected_v1`` — ``cards/cms_card_v1.tcl`` (PATCH-6/7: UParT-AK4 Medium +
  DeepTau v2p5 Medium baselines fit to the 2024 NanoAODv15 anchor).

``for_card(card_path)`` picks the right one from the config's ``card:`` path.
``pt`` is in GeV; ``pt``/``eta`` are array-likes of equal length and the return
is a numpy array of the same length, clipped to [0, 1].
"""

from __future__ import annotations

import os
import re

import numpy as np

from delphes_pipeline.core.references import QUANTITIES

# explicit card -> formula-set map (extend in lockstep when a card is added);
# both the repo file name and the production card-header name are accepted
_FORMULA_SETS = {
    "cms_card_v0.tcl": "expected",
    "delphes_card_cms_hhbbtt_v0.tcl": "expected",
    "cms_card_v1.tcl": "expected_v1",
    "delphes_card_cms_hhbbtt_v1.tcl": "expected_v1",
}


def for_card(card_path: str):
    """The closure-target function for the configured card (by file name).

    Known cards dispatch via the explicit map. An unknown card that *carries a
    version marker* is an error — its formulas have not been transcribed, and
    silently validating against another card's targets is exactly the failure
    this dispatch exists to prevent. Unversioned names (fixtures, ad-hoc test
    cards) fall back to the stock v0 baseline.
    """
    name = os.path.basename(str(card_path)).lower()
    if name in _FORMULA_SETS:
        return globals()[_FORMULA_SETS[name]]
    if re.search(r"v\d+", name):
        raise ValueError(
            f"card {name!r} has a version marker but no transcribed closure formulas; "
            "transcribe its tagger blocks and add it to card_formulas._FORMULA_SETS "
            "(kept in lockstep with the tcl)"
        )
    return expected


def expected(quantity: str, pt, eta) -> np.ndarray:
    """Card-formula closure target for ``quantity`` at points ``(pt, eta)``."""
    if quantity not in QUANTITIES:
        raise KeyError(f"unknown quantity {quantity!r}; expected one of {QUANTITIES}")
    pt = np.asarray(pt, dtype=float)
    eta = np.asarray(eta, dtype=float)
    aeta = np.abs(eta)

    if quantity == "btag_eff_b":
        # BTagging EfficiencyFormula {5}:  0.85*tanh(0.0025*pt)*(25/(1+0.063*pt))
        val = 0.85 * np.tanh(0.0025 * pt) * (25.0 / (1.0 + 0.063 * pt))
    elif quantity == "btag_eff_c":
        # BTagging EfficiencyFormula {4}:  0.25*tanh(0.018*pt)*(1/(1+0.0013*pt))
        val = 0.25 * np.tanh(0.018 * pt) * (1.0 / (1.0 + 0.0013 * pt))
    elif quantity == "btag_mistag_light":
        # BTagging EfficiencyFormula {0}:  0.01 + 0.000038*pt
        val = 0.01 + 0.000038 * pt
    elif quantity == "tau_eff":
        # TauTagging EfficiencyFormula {15}: 0.6 within TauEtaMax = 2.5
        val = np.where(aeta <= 2.5, 0.6, 0.0)
    elif quantity == "tau_mistag":
        # TauTagging EfficiencyFormula {0}: 0.01 within TauEtaMax = 2.5
        val = np.where(aeta <= 2.5, 0.01, 0.0)
    elif quantity == "electron_eff":
        # ElectronEfficiency: 0 for pt<=7; 0.95 (|eta|<=1.5), 0.85 (1.5<|eta|<=2.5), 0 else
        val = np.where(
            pt <= 7.0, 0.0,
            np.where(aeta <= 1.5, 0.95, np.where(aeta <= 2.5, 0.85, 0.0)),
        )
    elif quantity == "muon_eff":
        # MuonEfficiency: 0 for pt<=5; 0.95 (|eta|<=2.4), 0 else
        val = np.where(pt <= 5.0, 0.0, np.where(aeta <= 2.4, 0.95, 0.0))
    else:  # pragma: no cover - guarded above
        raise KeyError(quantity)

    return np.clip(np.asarray(val, dtype=float), 0.0, 1.0)


def expected_v1(quantity: str, pt, eta) -> np.ndarray:
    """Closure target for ``cards/cms_card_v1.tcl`` (PATCH-6/7 Run-3 taggers).

    The tagger formulas are smooth fits to the 2024 NanoAODv15 kl=1 anchor
    (UParT-AK4 Medium 0.1272 / DeepTau v2p5 VSjet Medium, ``derive_maps``);
    lepton blocks are unchanged from v0. Keep in lockstep with the tcl.
    """
    if quantity not in QUANTITIES:
        raise KeyError(f"unknown quantity {quantity!r}; expected one of {QUANTITIES}")
    pt = np.asarray(pt, dtype=float)
    eta = np.asarray(eta, dtype=float)
    aeta = np.abs(eta)

    if quantity == "btag_eff_b":
        # PATCH-6 EfficiencyFormula {5}: (pt > 4.0) * (0.904 - 3.53/pt)
        val = np.where(pt > 4.0, 0.904 - 3.53 / np.where(pt > 0, pt, 1.0), 0.0)
    elif quantity == "btag_eff_c":
        # PATCH-6 EfficiencyFormula {4}: 0.094 + 2.22/pt
        val = 0.094 + 2.22 / np.where(pt > 0, pt, 1.0)
    elif quantity == "btag_mistag_light":
        # PATCH-6 EfficiencyFormula {0}: 0.019 + 0.32*exp(-pt/30.0)
        val = 0.019 + 0.32 * np.exp(-pt / 30.0)
    elif quantity == "tau_eff":
        # PATCH-7 EfficiencyFormula {15}: (pt > 12.5) * (0.776 - 9.7/pt), TauEtaMax 2.5
        val = np.where((aeta <= 2.5) & (pt > 12.5), 0.776 - 9.7 / np.where(pt > 0, pt, 1.0), 0.0)
    elif quantity == "tau_mistag":
        # PATCH-7 EfficiencyFormula {0}: 0.004 within TauEtaMax = 2.5
        val = np.where(aeta <= 2.5, 0.004, 0.0)
    else:
        # lepton blocks unchanged in v1 -> defer to the v0 transcription
        return expected(quantity, pt, eta)

    return np.clip(np.asarray(val, dtype=float), 0.0, 1.0)
