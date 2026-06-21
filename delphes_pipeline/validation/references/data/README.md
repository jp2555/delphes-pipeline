# Digitised reference curves (drop-in)

Drop one JSON per quantity here to overlay **public POG performance points** on
the Level-0 plots, alongside the card-formula closure target. Filenames must be
`<quantity>.json` where `<quantity>` is one of:

`btag_eff_b`, `btag_eff_c`, `btag_mistag_light`, `tau_eff`, `tau_mistag`,
`electron_eff`, `muon_eff`.

Format (see `core/references.py::ReferenceCurve`):

```json
{
  "quantity": "btag_eff_b",
  "x": "pt",
  "centers": [25, 35, 50, 75, 125, 250],
  "values":  [0.45, 0.60, 0.68, 0.72, 0.70, 0.62],
  "errors":  [0.02, 0.02, 0.01, 0.01, 0.02, 0.03],
  "source": "CMS-DP-2024-066, Fig. 7 (UParT BvAll Medium)",
  "wp": "UParT Medium", "eta_range": [0.0, 2.4]
}
```

When a JSON is present, `ReferenceStore.digitized(quantity)` returns it and the
plot shows three series: measured (Delphes), card formula (closure), and POG
(digitised). Until then, only the first two appear. See `example_btag_eff_b.json`
for a non-active template (note the `example_` prefix keeps it from loading).
