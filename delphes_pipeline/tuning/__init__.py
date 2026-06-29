"""Tuning lens: measured object response vs POG/anchor target -> which card knob.

Consumes the shared measurement layer (:mod:`delphes_pipeline.core.observables`)
— the same measurements the validation gate uses — but compares each observable
to its *tuning target* (a digitised POG curve, a unity response, or an anchor
mass peak) rather than to the card formula, and reports the residual plus the
card knob to turn (the note Sec. 3-4 diagnostic map). Re-run after each card
edit to drive the tuning loop.
"""
