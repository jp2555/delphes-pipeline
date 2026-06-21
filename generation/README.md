# generation/ (STUB)

Generator run cards, gridpack configs, and the **pinned container** for the
in-flight chain (note §5): `MG5_aMC 2.9.x`, `Powheg 2.0`, `Pythia 8.3`,
`Delphes 3.5`. Versions are pinned to match Ref. [2] so future cross-comparison
is clean.

Not needed to validate the *existing* signal sample (the subject of this repo's
gate). Populate when wiring the full production:

```
generation/
├── powheg/        # ggHH kappa_lambda = {0, 1, 5} run cards
├── madgraph/      # VBF / Z+jets / single-H run cards
├── pythia/        # shower + hadronisation config (TauDecays:externalMode=2)
└── container.def  # pinned image digest -> versions.json next to the card
```

`versions.json` (placed next to `cards/cms_card_v0.tcl`) is read by
`core.provenance` and stamped into every report.
