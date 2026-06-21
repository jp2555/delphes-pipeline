"""Core contracts shared by every validation level.

Public surface (imported by leaf modules and the orchestrator):

- result:     CheckResult, Severity, gate_max, gate_min, gate_within, info, warn
- context:    ValidationContext
- io:         DelphesEvents, load_ntuple
- references: ReferenceStore, ReferenceCurve
- provenance: collect
- report:     Report
- plotting:   efficiency_overlay, resolution_plot, cms_style
"""

from .result import (
    CheckResult,
    Severity,
    gate_max,
    gate_min,
    gate_within,
    info,
    warn,
)

__all__ = [
    "CheckResult",
    "Severity",
    "gate_max",
    "gate_min",
    "gate_within",
    "info",
    "warn",
]
