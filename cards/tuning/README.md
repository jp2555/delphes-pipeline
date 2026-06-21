# cards/tuning/ — provenance for tuning set v0 (STUB)

One JSON per tuned quantity, recording exactly which public result it encodes so
the provenance chain runs from any analysis plot back to a citable figure
(note §3, §8.3). The Delphes card transcribes these; the JSONs document the
transcription.

`_schema.json` is the shape every tuning JSON follows. The `*_v0.json` files are
**stubs** — tuning set v0 (UParT-2024 Medium b-tag maps, DeepTau v2p5 maps,
PUPPI pᵀᵐⁱˢˢ) is not applied to the card yet. Fill them when v0 lands and update
the `BTagging`/`TauTagging` blocks of `cms_card_v0.tcl` to match.
