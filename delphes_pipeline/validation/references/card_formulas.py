"""Closure targets transcribed from ``cards/cms_card_v0.tcl``.

``expected(quantity, pt, eta)`` evaluates the efficiency/response the card is
*configured* to produce, so the Level-0 measurement can be overlaid against it
(the "do the modules behave as configured?" closure check). Each formula mirrors
a specific card block and line, noted inline; if the card is retuned (set v0),
update these in lockstep.

These are the **stock** parametrisations (the attached card is the pre-v0
baseline). ``pt`` is in GeV; ``pt``/``eta`` are array-likes of equal length and
the return is a numpy array of the same length, clipped to [0, 1].
"""

from __future__ import annotations

import numpy as np

from delphes_pipeline.core.references import QUANTITIES


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
