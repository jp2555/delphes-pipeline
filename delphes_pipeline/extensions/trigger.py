"""Trigger emulation (note §4.1, Table 2) — 2024 menu data + apply stub.

Triggers are emulated parametrically: offline kinematic thresholds plus a plateau
efficiency, not full path emulation. The τ_h/lepton thresholds sculpt the
acceptance at low m_HH — exactly the κ_λ-sensitive threshold region — so this
layer must be in place before the Level-4 κ_λ acceptance study is meaningful.

``TRIGGER_MENU_2024`` encodes the offline-matched thresholds (the right column of
Table 2). ``apply_trigger`` is the stub to implement: an offline-threshold mask
times a per-path plateau efficiency taken from public tag-and-probe curves.
"""

from __future__ import annotations

# Offline-matched thresholds (GeV) per channel, AN-25-103 Table 2 (2024 menu).
# Online tau_h ID is ParticleNet in 2024 (DeepTau in 2022-23).
TRIGGER_MENU_2024 = {
    "mutau": {
        "single_mu_pt": 26.0,
        "cross_mu_pt": 22.0, "cross_tau_pt": 32.0,
    },
    "etau": {
        "single_e_pt": 31.0,
        "cross_e_pt": 25.0, "cross_tau_pt": 35.0,
    },
    "tautau": {
        "ditau_tau_pt": 40.0,            # two tau_h
        "ditau_jet_tau_pt": 35.0,        # two tau_h + jet
        "vbf_tau_pt": 25.0,              # VBF + ditau
        "parking_quadjet_tau_pt": 25.0,  # quad-jet parking path
    },
}

# Hadronic-tau pT threshold for the VBF trigger (used only where the diTau
# trigger is inaccessible, i.e. tau_h pT < 40 GeV); see §4.1.
VBF_TRIGGER_TAU_PT = 25.0


def apply_trigger(events, channel: str, *, plateau_efficiencies=None):
    """STUB: per-event trigger decision (offline thresholds × plateau efficiency).

    Plan: build the OR of the channel's paths from ``TRIGGER_MENU_2024`` offline
    thresholds on the selected leptons/τ_h, then weight by per-path plateau
    efficiencies from public tag-and-probe curves (with the §7 data-SF tolerance).
    The DiTau+Jet path adds ~13-17% low-m_HH signal efficiency and must be
    included or the κ_λ-sensitive region is understated.
    """
    raise NotImplementedError(
        "trigger emulation is an extension point (note §4.1, Table 2); "
        "TRIGGER_MENU_2024 holds the offline thresholds to build it on."
    )
