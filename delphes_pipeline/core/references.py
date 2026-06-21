"""Reference curves for the Level-0 object-response overlays.

Two kinds of reference exist for each measured quantity:

1. The **card's transcribed parametrisation** — the efficiency/response formula
   written into ``cms_card_v0.tcl``. This is the *closure* target: "do the
   Delphes modules behave as configured?". It is always available and is
   provided by :mod:`delphes_pipeline.validation.references.card_formulas`,
   injected here as ``card_formula_fn`` to keep ``core`` free of upward imports.

2. A **digitised POG curve** — public performance points [note refs 5,6,7],
   dropped into the references data dir as ``<quantity>.json`` once tuning set
   v0 lands. When present, it is returned by :meth:`ReferenceStore.digitized`
   and overlaid alongside the closure target.

Recognised quantity keys (the vocabulary leaf modules and JSONs must use):
``btag_eff_b``, ``btag_eff_c``, ``btag_mistag_light``, ``tau_eff``,
``tau_mistag``, ``electron_eff``, ``muon_eff``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

QUANTITIES = (
    "btag_eff_b",
    "btag_eff_c",
    "btag_mistag_light",
    "tau_eff",
    "tau_mistag",
    "electron_eff",
    "muon_eff",
)


@dataclass
class ReferenceCurve:
    """A digitised reference curve loaded from JSON."""

    quantity: str
    x: str  # "pt" or "eta"
    centers: np.ndarray
    values: np.ndarray
    errors: Optional[np.ndarray] = None
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class ReferenceStore:
    """Provides closure targets and (optional) digitised reference curves."""

    def __init__(
        self,
        data_dir: str | Path,
        card_formula_fn: Optional[Callable[..., np.ndarray]] = None,
    ):
        self.data_dir = Path(data_dir)
        self._card_formula_fn = card_formula_fn
        self._cache: dict[str, Optional[ReferenceCurve]] = {}

    # ----- closure target (card transcription) ---------------------------- #
    def expected(self, quantity: str, pt, eta) -> np.ndarray:
        """Card-formula closure target evaluated at ``(pt, eta)`` points.

        ``pt`` and ``eta`` are array-likes of equal length; returns an array of
        the same length with the expected efficiency/response per point.
        """
        if quantity not in QUANTITIES:
            raise KeyError(f"unknown reference quantity {quantity!r}; expected one of {QUANTITIES}")
        if self._card_formula_fn is None:
            raise RuntimeError(
                "ReferenceStore has no card_formula_fn injected; the orchestrator "
                "must pass card_formulas.expected when constructing the store."
            )
        return np.asarray(self._card_formula_fn(quantity, np.asarray(pt), np.asarray(eta)))

    # ----- digitised POG curve (optional) --------------------------------- #
    def digitized(self, quantity: str) -> Optional[ReferenceCurve]:
        """Return the digitised reference for ``quantity`` if a JSON exists."""
        if quantity in self._cache:
            return self._cache[quantity]
        path = self.data_dir / f"{quantity}.json"
        curve = _load_curve(path) if path.exists() else None
        self._cache[quantity] = curve
        return curve

    def has_digitized(self, quantity: str) -> bool:
        return self.digitized(quantity) is not None


def _load_curve(path: Path) -> ReferenceCurve:
    with open(path) as fh:
        d = json.load(fh)
    return ReferenceCurve(
        quantity=d["quantity"],
        x=d.get("x", "pt"),
        centers=np.asarray(d["centers"]),
        values=np.asarray(d["values"]),
        errors=np.asarray(d["errors"]) if d.get("errors") is not None else None,
        source=d.get("source", ""),
        meta={k: v for k, v in d.items() if k not in {"quantity", "x", "centers", "values", "errors", "source"}},
    )
