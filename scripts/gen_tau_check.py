"""Score our visible-hadronic-gen-τ construction against CMS ``GenVisTau``.

    pixi run python scripts/gen_tau_check.py --config config.v1.yml [--max-events 200000]

The cutflow's stage 1 (``>=2 visible hadronic gen τ in acceptance``) sits 17 percent higher on
Delphes than on CMS at every κ_λ, and stage 1 is a *generator-level* count of the same
physics — so the gap is most likely in how we build the object, not in the samples.

The CMS NanoAOD anchor carries BOTH ``GenPart`` and ``GenVisTau``, so our construction
can be run on the very same events and scored against CMS's own definition:

* **efficiency** — of CMS's GenVisTau, how many do we find?
* **purity**     — of the τ we build, how many are a real GenVisTau?

A purity below 1 means we admit τ that CMS does not count — the expected failure mode,
since we veto a leptonic τ by looking for a status-1 e/μ within ΔR of it, which misses
soft or wide-angle daughters, whereas GenVisTau is hadronic by construction.

Matching is a unique nearest-neighbour assignment, so multiple generator *copies* of one
τ collapse onto a single GenVisTau and cannot inflate either number.
"""

from __future__ import annotations

import argparse

import awkward as ak
import numpy as np

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.matching import unique_match
from delphes_pipeline.core.nanoaod import NanoAODEvents
from delphes_pipeline.validation.run_validation import load_config


def _acc(coll, pt_min, eta_max):
    return coll[(coll.pt > pt_min) & (np.abs(coll.eta) <= eta_max)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="visible-gen-τ construction vs CMS GenVisTau")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--pt-min", type=float, default=20.0)
    ap.add_argument("--eta-max", type=float, default=2.3)
    ap.add_argument("--dr", type=float, default=0.3, help="ΔR for the truth match")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    ac = cfg.get("anchor", {})
    from delphes_pipeline.tuning.anchor import _resolve_wp
    cap = args.max_events or ac.get("max_events")
    nano = NanoAODEvents(ac["nanoaod_path"], branches=ac.get("branches"),
                         wp=_resolve_wp(ac.get("wp", {})), entry_stop=cap)
    print(f"[gencheck] {nano.n} anchor events", flush=True)

    truth = _acc(nano.genvistau, args.pt_min, args.eta_max)      # CMS definition
    n_truth = int(ak.sum(ak.num(truth)))
    pair_truth = float(np.mean(ak.to_numpy(ak.num(truth) >= 2)))
    print(f"\n  CMS GenVisTau : {n_truth} objects, >=2 in {pair_truth:.4f} of events\n")
    print(f"  {'veto':10s} {'objects':>9s} {'ratio':>7s} {'eff':>7s} {'purity':>8s} "
          f"{'>=2 /evt':>9s} {'stage-1 x':>10s}")

    results = {}
    for veto in ("geometric", "descent"):
        ours = _acc(obs.gen_visible_taus(nano.gen, veto=veto), args.pt_min, args.eta_max)
        # unique nearest match both ways: generator COPIES collapse onto one truth object
        ours_matched = unique_match(ours, truth, args.dr)
        truth_found = unique_match(truth, ours, args.dr)
        n_ours = int(ak.sum(ak.num(ours)))
        purity = float(ours_matched.sum()) / max(n_ours, 1)
        efficiency = float(truth_found.sum()) / max(n_truth, 1)
        pair_ours = float(np.mean(ak.to_numpy(ak.num(ours) >= 2)))
        results[veto] = (purity, efficiency, pair_ours / max(pair_truth, 1e-9))
        print(f"  {veto:10s} {n_ours:9d} {n_ours / max(n_truth, 1):7.3f} "
              f"{efficiency:7.4f} {purity:8.4f} {pair_ours:9.4f} "
              f"{pair_ours / max(pair_truth, 1e-9):10.3f}")

    pg, eg, sg = results["geometric"]
    pd, ed, sd = results["descent"]
    print()
    if pd > pg + 0.005:
        print(f"  -> the geometric veto admitted τ that CMS does not count "
              f"(purity {pg:.3f} -> {pd:.3f}); the descent veto is the fix, and the "
              f"stage-1 ratio moves {sg:.3f} -> {sd:.3f}.")
    elif pd < 0.97:
        print("  -> both vetoes admit τ CMS does not count: the leptonic classification "
              "is NOT the whole story.")
    elif ed < 0.97:
        print("  -> we MISS real GenVisTau; acceptance or the ν subtraction is the suspect.")
    else:
        print("  -> the construction already agreed with CMS; the stage-1 gap is NOT "
              "definitional, and the samples differ in gen acceptance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
