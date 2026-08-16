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
    return entry


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
