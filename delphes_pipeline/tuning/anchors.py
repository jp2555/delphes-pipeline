"""Public-anchor reference values, with the derivation/validation firewall enforced.

The hard rule from the consistency notes: published event-level DISTRIBUTIONS are
validation targets only, never derivation targets. Deriving a map from a kinematic
distribution launders physics into the detector model — the gen-difference-absorption
failure in public clothing. The firewall lives here rather than in a review checklist,
because a checklist does not fail a build.
"""
from __future__ import annotations

from pathlib import Path

import yaml

TOVERIFY = "TOVERIFY"


class AnchorError(RuntimeError):
    pass


def load(path="cards/tuning/anchors_v2.yml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def for_derivation(anchors: dict, name: str) -> dict:
    """The entry for ``name``, or raise if it is not a legitimate derivation input."""
    entry = (anchors.get("maps") or {}).get(name)
    if entry is None:
        val = (anchors.get("validation") or {}).get(name)
        if val is not None:
            raise AnchorError(
                f"{name!r} is a VALIDATION target and must not be derived from: "
                f"deriving maps from published event-level distributions launders "
                f"physics into the detector model")
        raise AnchorError(f"no anchor entry named {name!r}")
    if entry.get("use") != "derivation":
        raise AnchorError(
            f"anchor {name!r} is tagged use={entry.get('use')!r}; only "
            f"use='derivation' entries may be read by the deriver. Re-tagging a "
            f"validation target to get past this is the error the tag exists to stop.")
    _check_mitigation(name, entry)
    _check_pu_conditioning(name, entry)
    return entry


#: rates and efficiencies measured on a population CMS has already cleaned of PU-origin
#: objects. Anything else double-counts against the C4 residual bound.
_MITIGATION_OK = ("post_mitigation", "not_applicable")


def _check_mitigation(name, entry):
    """Refuse a rate/efficiency anchor that is not the post-mitigation number.

    The C4 argument is that our no-PU samples ARE the anchor's pipeline with the residual
    PU contamination set to zero, so the model error is the published residual and not the
    whole PU-jet population. That holds only if the maps we take from public material are
    the post-PU-jet-ID, post-primary-vertex-association numbers. A pre-mitigation POG
    fake rate would count the PU term twice -- once in the map and once in the residual
    bound -- and the whole framing collapses. It is a one-line check when filling the
    file, which is exactly the kind of check that gets skipped, so it is enforced here.
    """
    state = entry.get("mitigation_state")
    if state is None:
        return                                  # not a rate/efficiency anchor
    if state == TOVERIFY:
        raise AnchorError(
            f"anchor {name!r} has mitigation_state=TOVERIFY: confirm the public number is "
            f"post-PU-jet-ID / post-PV-association before deriving from it. A "
            f"pre-mitigation rate double-counts against the C4 residual bound.")
    if state not in _MITIGATION_OK:
        raise AnchorError(
            f"anchor {name!r} has mitigation_state={state!r}; only "
            f"{' or '.join(_MITIGATION_OK)} may be derived from. A pre-mitigation number "
            f"counts the pileup term twice (map + residual bound).")


def _check_pu_conditioning(name, entry):
    """Refuse a PU-inclusive conditioning variable on a map applied to no-PU events.

    An anchor map binned in a PU-inclusive activity variable, applied to a no-PU event,
    reads a systematically LOW bin and under-corrects. This is live in v1: met_smear is
    conditioned on jet H_T with pt > 20 GeV out to |eta| <= 4.7, which is PU-inclusive on
    the CMS side and PU-free on ours.
    """
    if not entry.get("conditioning_must_be_pu_independent"):
        return
    pu_dependent = {"sum_et", "sumet", "jet_ht", "ht", "n_jets", "n_vertices", "rho"}
    bad = [v for v in entry.get("condition_on", []) if v in pu_dependent]
    if bad:
        raise AnchorError(
            f"anchor {name!r} requires PU-independent conditioning but is binned in "
            f"{bad}: a PU-inclusive anchor variable applied to a no-PU event reads a low "
            f"bin and under-corrects. Use hard-scatter H_T or visible q_T, or "
            f"offset-correct by the mean PU contribution at <mu>.")


def unverified(anchors: dict) -> list[str]:
    """Dotted paths of every value still carrying the TOVERIFY placeholder."""
    out: list[str] = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif node == TOVERIFY:
            out.append(path)

    walk(anchors, "")
    return sorted(out)


def require_verified(anchors: dict, *paths: str) -> None:
    """Raise unless every named anchor value has been checked against a publication.

    Used to gate the things that genuinely cannot proceed on a placeholder — the
    Tier-3 comparison and anything quoting an equivalent luminosity.
    """
    missing = [p for p in unverified(anchors) if any(p.startswith(q) for q in paths)]
    if missing:
        raise AnchorError(
            "these anchor values are still placeholders and must be checked against "
            "the published source first:\n  " + "\n  ".join(missing))
