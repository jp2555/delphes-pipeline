"""Compute per-observable tuning residuals and render the tuning report.

For each observable: measure it (shared ``core.observables``), compare to its
target (digitised POG curve / unity response / anchor peak), compute the residual
and a status, attach the card knob + action from the diagnostic map, and overlay
measured-vs-card-vs-target. Aggregate to ``tuning_report.json`` + ``.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from delphes_pipeline.core import observables as obs
from delphes_pipeline.core.context import ValidationContext
from delphes_pipeline.core.plotting import efficiency_overlay
from delphes_pipeline.core.references import QUANTITIES
from . import targets as T

# Only the b-tag efficiency observables are re-measured on the re-tagged view. Other
# diagnostics (energy scale, leptons, τ_h, MET) read different knobs and stay on stock
# tags; in particular the m_bb b-pair selection sorts on btag, so re-tagging it would
# confound the energy-scale check.
_RETAG_OBSERVABLES = frozenset(obs.BTAG_FLAVORS)


@dataclass
class TuningResult:
    observable: str
    kind: str
    status: str            # on_target | needs_tuning | no_target | no_data
    residual: float        # overall residual metric (relative, or GeV for peak)
    tolerance: float
    knob: str
    action: str
    note_section: str
    plot_path: Optional[str] = None
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _weighted_mean(dev, counts):
    counts = np.asarray(counts, dtype=float)
    return float(np.sum(np.abs(dev) * counts) / np.sum(counts)) if counts.sum() else float("nan")


def tune_observable(ctx: ValidationContext, observable: str, *, bins, anchor_target=None,
                    measure_events=None) -> TuningResult:
    """Measure one observable, compare to its target, return a TuningResult.

    If ``anchor_target`` (a Profile measured on the NanoAOD anchor) is given it
    takes precedence over the digitised-curve / card-formula target. ``measure_events``
    overrides where the Delphes side is measured (e.g. a downstream-re-tagged view).
    """
    events = measure_events if measure_events is not None else ctx.events
    diag = T.diagnostic_map()[observable]
    kind = diag["target_kind"]
    tol = float(T.scalar_targets()["tuning_tolerance"])
    common = dict(observable=observable, kind=kind, knob=diag["card_knob"],
                  action=diag["action"], note_section=diag["note_section"])

    if observable == "mbb_peak":
        return _tune_peak(ctx, common, tol, events)

    prof = T.PROFILE_OBSERVABLES[observable](events, bins)
    if not prof.centers.size:
        return TuningResult(status="no_data", residual=float("nan"), tolerance=tol,
                            detail="no populated bins", **common)

    centers = np.asarray(prof.centers, dtype=float)
    measured = np.asarray(prof.values, dtype=float)
    counts = np.asarray(prof.counts, dtype=float)
    card = (np.asarray(ctx.references.expected(observable, centers, np.zeros_like(centers)), dtype=float)
            if observable in QUANTITIES else None)

    if anchor_target is not None and np.asarray(anchor_target.centers).size:
        a_centers = np.asarray(anchor_target.centers, dtype=float)
        a_values = np.asarray(anchor_target.values, dtype=float)
        tgt = np.interp(centers, a_centers, a_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = np.where(tgt != 0, measured / tgt - 1.0, np.nan)
        ok = np.isfinite(dev)
        residual = _weighted_mean(dev[ok], counts[ok])
        status = "on_target" if residual <= tol else "needs_tuning"
        detail = f"Delphes deviates {residual:.1%} from the NanoAOD anchor (count-weighted)"
        plot = _overlay(ctx, observable, prof, expected=card, ref_centers=a_centers,
                        ref_values=a_values, ref_errors=anchor_target.errors, ref_label="NanoAOD anchor")
        extra = {"target": "nanoaod_anchor", "anchor": anchor_target.to_dict(), **prof.to_dict()}
        return TuningResult(status=status, residual=residual, tolerance=tol,
                            plot_path=plot, detail=detail, extra=extra, **common)

    if kind == "unity":
        target = np.ones_like(centers)
        residual = _weighted_mean(measured - 1.0, counts)
        status = "on_target" if residual <= tol else "needs_tuning"
        plot = _overlay(ctx, observable, prof, expected=target, expected_label="target (1.0)")
        detail = f"median reco/gen response deviates {residual:.1%} from 1.0 (count-weighted)"
        extra = {"target": "unity", **prof.to_dict()}
    else:  # curve (efficiencies, MET resolution)
        ref = ctx.references.digitized(observable)
        if ref is None:
            residual, status = float("nan"), "no_target"
            detail = "no digitised target; drop validation/references/data/%s.json to enable tuning" % observable
            plot = _overlay(ctx, observable, prof, expected=card)
            extra = {"target": "absent", **prof.to_dict()}
        else:
            tgt = np.interp(centers, np.asarray(ref.centers, dtype=float), np.asarray(ref.values, dtype=float))
            with np.errstate(divide="ignore", invalid="ignore"):
                dev = np.where(tgt != 0, measured / tgt - 1.0, np.nan)
            residual = _weighted_mean(dev[np.isfinite(dev)], counts[np.isfinite(dev)])
            status = "on_target" if residual <= tol else "needs_tuning"
            detail = f"measured deviates {residual:.1%} from {ref.source or 'target'} (count-weighted)"
            plot = _overlay(ctx, observable, prof, expected=card,
                            ref_centers=ref.centers, ref_values=ref.values, ref_errors=ref.errors)
            extra = {"target": ref.source, **prof.to_dict()}

    return TuningResult(status=status, residual=residual, tolerance=tol,
                        plot_path=plot, detail=detail, extra=extra, **common)


def _tune_peak(ctx: ValidationContext, common: dict, tol: float, events=None) -> TuningResult:
    st = T.scalar_targets()
    target = float(st["mbb_peak_gev"])
    peaktol = float(st["mbb_peak_tolerance_gev"])
    pk = obs.mbb_peak(events if events is not None else ctx.events)
    if pk.n_core == 0:
        return TuningResult(status="no_data", residual=float("nan"), tolerance=peaktol,
                            detail="empty m_bb core window", **common)
    residual = (pk.peak - target) / target
    status = "on_target" if abs(pk.peak - target) <= peaktol else "needs_tuning"
    detail = (f"visible m_bb peak {pk.peak:.1f} GeV (width {pk.width:.1f}) vs anchor {target:.0f} GeV "
              f"-> {residual:+.1%}")
    return TuningResult(status=status, residual=float(residual), tolerance=peaktol / target,
                        detail=detail, extra=pk.to_dict(), **common)


def _overlay(ctx, observable, prof, *, expected=None, expected_label="card formula",
             ref_centers=None, ref_values=None, ref_errors=None, ref_label="tuning target") -> str:
    """Render a measured / card / target overlay for a tuning observable."""
    outpath = ctx.plot_path(f"tuning_{observable}.png")
    if expected is None:
        expected = prof.values  # degenerate: nothing to compare; plot measured against itself faintly
        expected_label = "(no card formula)"
    efficiency_overlay(
        prof.centers, prof.values, prof.errors, expected,
        outpath=str(outpath), xlabel=prof.xlabel or "pT [GeV]", ylabel=prof.ylabel or observable,
        measured_label="Delphes (measured)", expected_label=expected_label,
        ref_centers=ref_centers, ref_values=ref_values, ref_errors=ref_errors,
        ref_label=ref_label,
    )
    return ctx.rel(Path(outpath))


def run_tuning(ctx: ValidationContext) -> list[TuningResult]:
    """Measure every tuning observable, write tuning_report.{json,md}, return results."""
    import json

    from . import anchor as anchor_mod

    bins = ctx.opt("level0", "pt_bins", obs.DEFAULT_PT_BINS)
    cap = getattr(ctx.events, "entry_stop", None)  # the --max-events cap (caps the anchor too)
    print(f"[tuning] Delphes: {ctx.provenance.get('n_events', '?')} events"
          + (f" (capped at {cap})" if cap else ""), flush=True)
    anchors = anchor_mod.anchor_profiles(ctx.config, bins=bins, max_events=cap)  # {} unless enabled
    if anchors:
        print(f"[tuning] anchor targets ready: {sorted(anchors)}", flush=True)

    # Close the loop: when tuning maps are configured, re-measure the b-tag observables on a
    # downstream-re-tagged view so their residual reflects the *tuned* tags (note D2-A). Seed 0
    # matches the ntuplizer default so the re-validated tags equal the shipped ntuple's.
    retag_view = None
    maps_path = ctx.config.get("tuning_maps")
    if maps_path:
        from .maps import RetaggedEvents, TuningMaps
        retag_view = RetaggedEvents(ctx.events, TuningMaps.load(maps_path), np.random.default_rng(0))
        print(f"[tuning] re-validating the b-tag observables with the downstream re-tag "
              f"from {maps_path}", flush=True)

    results = []
    for name in T.tuning_observables():
        print(f"[tuning]   measuring Delphes {name} ...", flush=True)
        me = retag_view if (retag_view is not None and name in _RETAG_OBSERVABLES) else None
        r = tune_observable(ctx, name, bins=bins, anchor_target=anchors.get(name), measure_events=me)
        if me is not None and isinstance(r.extra, dict):
            r.extra["retagged"] = True
        results.append(r)

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": ctx.provenance, "results": [r.to_dict() for r in results]}
    with open(ctx.output_dir / "tuning_report.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    (ctx.output_dir / "tuning_report.md").write_text(_render_md(ctx, results))

    from collections import Counter

    c = Counter(r.status for r in results)
    print(f"[tuning] read {ctx.provenance.get('n_events', '?')} events from "
          f"{ctx.provenance.get('input_path', '?')}")
    print(f"[tuning] {len(results)} observables: "
          f"{c.get('on_target', 0)} on target · {c.get('needs_tuning', 0)} need tuning · "
          f"{c.get('no_target', 0)} measured but no target curve · {c.get('no_data', 0)} no data")
    for r in results:
        bins = len(r.extra.get("counts", [])) if isinstance(r.extra, dict) else 0
        n = int(np.sum(r.extra["counts"])) if isinstance(r.extra, dict) and r.extra.get("counts") else 0
        suffix = f"  ({bins} bins, {n} entries)" if bins else ""
        print(f"           {r.observable:22s} {_STATUS_MARK.get(r.status, r.status)}{suffix}")
    if c.get("no_target", 0):
        print("[tuning] 'no target' = measured fine, but no digitised POG/anchor curve to tune toward. "
              "Drop validation/references/data/<observable>.json to enable those.")
    print(f"[tuning] -> {ctx.output_dir / 'tuning_report.md'}")
    return results


_STATUS_MARK = {"on_target": "✅ on target", "needs_tuning": "🔧 needs tuning",
                "no_target": "○ no target", "no_data": "— no data"}


def _render_md(ctx: ValidationContext, results: list[TuningResult]) -> str:
    prov = ctx.provenance
    retagged = any(isinstance(r.extra, dict) and r.extra.get("retagged") for r in results)
    lines = [
        "# Delphes object-tuning report", "",
        f"input: `{prov.get('input_path', '?')}` ({prov.get('n_events', '?')} events)  ·  "
        f"card sha256 `{str(prov.get('card_sha256'))[:12]}`  ·  tuning set {prov.get('tuning_set', '?')}",
        "",
        *(["> **Downstream re-tag applied** (`tuning_maps`): the b-tag observables below are "
           "re-measured with `Jet.btag` re-derived from `Jet.Flavor` + the anchor map (tuned tags).", ""]
          if retagged else []),
        "_Status: ✅ on target · 🔧 needs tuning · ○ **measured, but no digitised target curve "
        "to tune toward** (drop `validation/references/data/<observable>.json`) · — no data._",
        "",
        "| observable | status | residual | card knob | action |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        res = "—" if not np.isfinite(r.residual) else (f"{r.residual:+.1%}")
        lines.append(f"| `{r.observable}` | {_STATUS_MARK.get(r.status, r.status)} | {res} | "
                     f"`{r.knob}` (§{r.note_section}) | {r.action} |")
    lines += ["", "## Details", ""]
    for r in results:
        lines.append(f"- **{r.observable}** — {r.detail}" + (f"  ([plot]({r.plot_path}))" if r.plot_path else ""))
    return "\n".join(lines) + "\n"
