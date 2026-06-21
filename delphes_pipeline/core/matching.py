"""ΔR object matching shared by the Level-0 leaves.

``matched_to_any`` classifies each probe by proximity to *any* target (used for
the τ_h efficiency/mistag jet split). ``unique_match`` does a per-event greedy
nearest-neighbour assignment with each target claimed at most once (used for
gen→reco lepton efficiency, where double-counting would bias the efficiency
high).
"""

from __future__ import annotations

import awkward as ak
import numpy as np


def _dphi(p1, p2):
    return (p1 - p2 + np.pi) % (2.0 * np.pi) - np.pi


def matched_to_any(probes: ak.Array, targets: ak.Array, dr_max: float) -> ak.Array:
    """Per-event, per-probe boolean: is the probe within ``dr_max`` of any target?

    ``probes`` and ``targets`` are jagged record arrays with ``eta``/``phi``
    fields. Events with no probes or no targets yield the correct empty/all-False
    result without raising.
    """
    pairs = ak.cartesian({"p": probes, "t": targets}, nested=True)
    deta = pairs["p"].eta - pairs["t"].eta
    dphi = _dphi(pairs["p"].phi, pairs["t"].phi)
    dr = np.sqrt(deta * deta + dphi * dphi)
    return ak.fill_none(ak.any(dr < dr_max, axis=-1), False)


def nearest_target_field(
    probes: ak.Array, targets: ak.Array, dr_max: float, field: str
) -> tuple[np.ndarray, np.ndarray]:
    """For each probe, the ``field`` of its unique nearest target within ``dr_max``.

    Greedy nearest-neighbour, each target claimed once. Returns two flat numpy
    arrays over the flattened probes (event-major): a matched mask and the
    matched target's ``field`` value (``nan`` where unmatched). Used for τ_h
    efficiency, where each gen τ must be measured on exactly its own reco jet so
    that jets accidentally near the τ do not dilute the rate.
    """
    out_mask: list[np.ndarray] = []
    out_val: list[np.ndarray] = []
    for p_ev, t_ev in zip(probes, targets):
        pe = ak.to_numpy(p_ev.eta)
        pp = ak.to_numpy(p_ev.phi)
        te = ak.to_numpy(t_ev.eta)
        tp = ak.to_numpy(t_ev.phi)
        tv = ak.to_numpy(t_ev[field])
        m = np.zeros(len(pe), dtype=bool)
        v = np.full(len(pe), np.nan)
        if len(pe) and len(te):
            dphi = _dphi(pp[:, None], tp[None, :])
            dr = np.sqrt((pe[:, None] - te[None, :]) ** 2 + dphi**2)
            used = np.zeros(len(te), dtype=bool)
            for idx in np.argsort(dr, axis=None):
                i, j = divmod(int(idx), dr.shape[1])
                if dr[i, j] >= dr_max:
                    break
                if not m[i] and not used[j]:
                    m[i] = used[j] = True
                    v[i] = tv[j]
        out_mask.append(m)
        out_val.append(v)
    return (
        np.concatenate(out_mask) if out_mask else np.zeros(0, dtype=bool),
        np.concatenate(out_val) if out_val else np.zeros(0),
    )


def unique_match(probes: ak.Array, targets: ak.Array, dr_max: float) -> np.ndarray:
    """Greedy nearest-neighbour match; each target used at most once.

    Returns a flat numpy boolean array over the flattened probes (event-major,
    aligned with ``ak.flatten(probes.<field>)``): True where the probe is matched
    to a unique target within ``dr_max``. Pairs are assigned in order of
    increasing ΔR.
    """
    out: list[np.ndarray] = []
    for p_ev, t_ev in zip(probes, targets):
        pe = ak.to_numpy(p_ev.eta)
        pp = ak.to_numpy(p_ev.phi)
        te = ak.to_numpy(t_ev.eta)
        tp = ak.to_numpy(t_ev.phi)
        matched = np.zeros(len(pe), dtype=bool)
        if len(pe) and len(te):
            dphi = _dphi(pp[:, None], tp[None, :])
            dr = np.sqrt((pe[:, None] - te[None, :]) ** 2 + dphi**2)
            used = np.zeros(len(te), dtype=bool)
            flat_order = np.argsort(dr, axis=None)
            for idx in flat_order:
                i, j = divmod(int(idx), dr.shape[1])
                if dr[i, j] >= dr_max:
                    break
                if not matched[i] and not used[j]:
                    matched[i] = used[j] = True
        out.append(matched)
    return np.concatenate(out) if out else np.zeros(0, dtype=bool)
