"""The single argument every validation level receives.

A ``ValidationContext`` bundles the opened Delphes file, the (lazily built) flat
ntuple, the reference store, output directories, and the run provenance, plus a
``tol()`` helper for config tolerance lookups. Levels expose
``run(ctx: ValidationContext) -> list[CheckResult]`` and pull everything they
need from this object — no globals, no hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid import cycles at runtime
    import awkward as ak

    from .io import DelphesEvents
    from .references import ReferenceStore


@dataclass
class ValidationContext:
    config: dict[str, Any]
    events: "DelphesEvents"
    references: "ReferenceStore"
    output_dir: Path
    plot_dir: Path
    provenance: dict[str, Any] = field(default_factory=dict)
    ntuple: Optional["ak.Array"] = None  # built on demand by levels that need it

    def tol(self, level: str, key: str, default: Any = None) -> Any:
        """Look up ``config['tolerances'][level][key]`` with a default."""
        return self.config.get("tolerances", {}).get(level, {}).get(key, default)

    def opt(self, level: str, key: str, default: Any = None) -> Any:
        """Look up a per-level option ``config['levels'][level][key]``."""
        return self.config.get("levels", {}).get(level, {}).get(key, default)

    def plot_path(self, name: str) -> Path:
        """Absolute path for a plot named ``name`` (e.g. ``"btag_eff.png"``)."""
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        return self.plot_dir / name

    def rel(self, path: Path) -> str:
        """Path relative to the output dir, for storing in report.json."""
        try:
            return str(Path(path).relative_to(self.output_dir))
        except ValueError:
            return str(path)
