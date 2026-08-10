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
    matched, values = nearest_target_fields(probes, targets, dr_max, (field,))
    return matched, values[field]


def nearest_target_fields(
    probes: ak.Array, targets: ak.Array, dr_max: float, fields: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``nearest_target_field`` for several target fields in ONE matching pass.

    The greedy assignment is deterministic, so calling the single-field variant
    once per field would repeat identical (expensive) work and return consistent
    values; this reads them all off the same match instead. Returns the matched
    mask and ``{field: values}``, each flat and event-major over the probes.
    """
    # Extract the fields to plain Python lists ONCE (a single vectorised call each);
    # iterating those is far cheaper than iterating the awkward arrays per event.
    pe_l, pp_l = ak.to_list(probes.eta), ak.to_list(probes.phi)
    te_l, tp_l = ak.to_list(targets.eta), ak.to_list(targets.phi)
    tv_ls = [ak.to_list(targets[f]) for f in fields]
    out_mask: list[np.ndarray] = []
    out_vals: list[list[np.ndarray]] = [[] for _ in fields]
    for ev_i, (pe_, pp_, te_, tp_) in enumerate(zip(pe_l, pp_l, te_l, tp_l)):
        pe = np.asarray(pe_, dtype=float)
        pp = np.asarray(pp_, dtype=float)
        te = np.asarray(te_, dtype=float)
        tp = np.asarray(tp_, dtype=float)
        tvs = [np.asarray(tv_l[ev_i], dtype=float) for tv_l in tv_ls]
        m = np.zeros(len(pe), dtype=bool)
        vs = [np.full(len(pe), np.nan) for _ in fields]
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
                    for v, tv in zip(vs, tvs):
                        v[i] = tv[j]
        out_mask.append(m)
        for acc, v in zip(out_vals, vs):
            acc.append(v)
    mask = np.concatenate(out_mask) if out_mask else np.zeros(0, dtype=bool)
    values = {f: (np.concatenate(acc) if acc else np.zeros(0))
              for f, acc in zip(fields, out_vals)}
    return mask, values


def unique_match(probes: ak.Array, targets: ak.Array, dr_max: float) -> np.ndarray:
    """Greedy nearest-neighbour match; each target used at most once.

    Returns a flat numpy boolean array over the flattened probes (event-major,
    aligned with ``ak.flatten(probes.<field>)``): True where the probe is matched
    to a unique target within ``dr_max``. Pairs are assigned in order of
    increasing ΔR.
    """
    pe_l, pp_l = ak.to_list(probes.eta), ak.to_list(probes.phi)
    te_l, tp_l = ak.to_list(targets.eta), ak.to_list(targets.phi)
    out: list[np.ndarray] = []
    for pe_, pp_, te_, tp_ in zip(pe_l, pp_l, te_l, tp_l):
        pe = np.asarray(pe_, dtype=float)
        pp = np.asarray(pp_, dtype=float)
        te = np.asarray(te_, dtype=float)
        tp = np.asarray(tp_, dtype=float)
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
