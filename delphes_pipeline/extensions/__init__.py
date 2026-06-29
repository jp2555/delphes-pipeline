"""Documented extension points (note Tier-2 parametric layers, decision D1).

These are scaffolded hooks the framework is designed around but does not yet
implement, because each needs more than the 50k signal samples or is a larger
sub-project:

- :mod:`trigger`  — offline-threshold + plateau trigger emulation (note §4.1,
  Table 2; the 2024 menu is encoded as data, the apply step is a stub);
- :mod:`mtautau`  — the τ_h τ_h mass estimator (note §3.3, decision D1: FastMTT
  or a covariance-free fallback).

The Level-1 candles (Z→ττ, tt̄; note §6.2) are the other extension and live as
stubs under ``validation/level1_candles``.
"""
