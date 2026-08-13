"""Pruning GenPart is ~9 TB of the 300M-event campaign, and it must stay resolvable.

The full Delphes ``Particle`` record is 99.5% of the ntuple (measured 29.9 of 30.0 kB per
event) and almost none of it is read downstream. But ``genPartIdxMother`` indexes into
that same array, so a naive PID filter leaves every surviving mother pointer aimed at the
wrong particle — silently, and in every file. These tests assert the two properties that
make pruning safe: the ancestry CLOSURE is kept, and the mother indices are renumbered
onto the pruned array so the chain still walks.
"""

from __future__ import annotations

from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest

from delphes_pipeline.core import observables as obs
from delphes_pipeline.ntuplizer import objects, schema


def _gen(rows):
    """rows: per-event list of (pid, status, m1)."""
    return ak.zip({"pid": ak.Array([[r[0] for r in ev] for ev in rows]),
                   "status": ak.Array([[r[1] for r in ev] for ev in rows]),
                   "m1": ak.Array([[r[2] for r in ev] for ev in rows]),
                   "pt": ak.Array([[10.0 + i for i, _ in enumerate(ev)] for ev in rows]),
                   "eta": ak.Array([[0.1 * i for i, _ in enumerate(ev)] for ev in rows]),
                   "phi": ak.Array([[0.2 * i for i, _ in enumerate(ev)] for ev in rows]),
                   "mass": ak.Array([[0.0 for _ in ev] for ev in rows])})


# H -> tau tau, one tau with a copy chain then its nu; plus unrelated QCD junk
_EVENT = [(25, 22, -1),      # 0 H
          (15, 2, 0),        # 1 tau  <- H
          (15, 2, 1),        # 2 tau copy
          (16, 1, 2),        # 3 nu_tau <- tau copy
          (-15, 2, 0),       # 4 tau bar <- H
          (-16, 1, 4),       # 5 nu_tau bar
          (21, 1, -1),       # 6 gluon (junk)
          (1, 1, 6),         # 7 quark  (junk)
          (211, 1, 2)]       # 8 pion from the tau (junk, but its mother IS kept)


def _ev(rows):
    return SimpleNamespace(gen=_gen(rows))


def test_pruning_keeps_the_targets_and_their_ancestry():
    mask, *_ = objects.prune_genpart(_gen([_EVENT]))
    kept = ak.to_numpy(ak.flatten(mask))
    assert kept.tolist() == [True, True, True, True, True, True, False, False, False]


def test_pruning_drops_the_bulk():
    """A realistic record is mostly junk; the whole point is not writing it."""
    rows = [_EVENT + [(211, 1, -1)] * 500]
    mask, *_ = objects.prune_genpart(_gen(rows))
    frac = float(ak.sum(mask)) / float(ak.sum(ak.num(_gen(rows))))
    assert frac < 0.02, frac


def test_mother_indices_are_renumbered_and_still_walk():
    """The property a flat PID filter destroys: every kept mother pointer must resolve to
    the SAME physical particle in the pruned array."""
    g = _gen([_EVENT])
    out = objects.build_genpart(_ev([_EVENT]), prune=True)
    pid = ak.to_numpy(ak.flatten(out.pdgId))
    mom = ak.to_numpy(ak.flatten(out.genPartIdxMother))
    assert pid.tolist() == [25, 15, 15, 16, -15, -16]
    # H has no mother; tau <- H; copy <- tau; nu <- copy; taubar <- H; nubar <- taubar
    assert mom.tolist() == [-1, 0, 1, 2, 0, 4]
    for i, m in enumerate(mom):
        if m >= 0:
            assert 0 <= m < len(pid), (i, m)


def test_visible_taus_survive_pruning():
    """gen_visible_taus rebuilds p_vis = p_tau - p_nu from the record, so if pruning broke
    the tau/nu pairing the ntuple could no longer reproduce the analysis object."""
    ev = _ev([_EVENT])
    before = obs.gen_visible_taus(ev.gen, dr=0.4)
    pruned = objects.build_genpart(ev, prune=True)
    after = obs.gen_visible_taus(
        ak.zip({"pid": pruned.pdgId, "status": pruned.status, "m1": pruned.genPartIdxMother,
                "pt": pruned.pt, "eta": pruned.eta, "phi": pruned.phi, "mass": pruned.mass}),
        dr=0.4)
    assert ak.sum(ak.num(after)) == ak.sum(ak.num(before))
    assert ak.to_numpy(ak.flatten(after.pt)).tolist() == \
        pytest.approx(ak.to_numpy(ak.flatten(before.pt)).tolist())


def test_multi_event_offsets_are_not_mixed():
    """Indices are per-event; a global renumbering would silently point across events."""
    rows = [_EVENT, [(25, 22, -1), (15, 2, 0), (16, 1, 1)], _EVENT]
    out = objects.build_genpart(_ev(rows), prune=True)
    assert ak.to_numpy(ak.num(out)).tolist() == [6, 3, 6]
    for e in range(3):
        mom = ak.to_numpy(out.genPartIdxMother[e])
        assert mom.max() < len(mom), e
        assert mom[0] == -1


def test_an_event_with_nothing_to_keep_yields_an_empty_list_not_a_crash():
    rows = [[(21, 1, -1), (1, 1, 0)], _EVENT]
    out = objects.build_genpart(_ev(rows), prune=True)
    assert ak.to_numpy(ak.num(out)).tolist() == [0, 6]


def test_prune_false_is_byte_for_byte_the_old_behaviour():
    ev = _ev([_EVENT])
    full = objects.build_genpart(ev, prune=False)
    assert ak.to_numpy(ak.flatten(full.pdgId)).tolist() == [r[0] for r in _EVENT]
    assert ak.to_numpy(ak.flatten(full.genPartIdxMother)).tolist() == [r[2] for r in _EVENT]


def test_schema_fields_are_unchanged_by_pruning():
    """Downstream readers must not have to care which mode wrote the file."""
    ev = _ev([_EVENT])
    assert set(ak.fields(objects.build_genpart(ev, prune=True))) == \
        set(ak.fields(objects.build_genpart(ev, prune=False))) == \
        set(schema.FLAT_SCHEMA["GenPart"])


def test_variable_multiplicity_events_keep_their_offsets():
    """Real records have different particle counts per event — a uniform fixture cannot
    see an offset bug, and np.add.reduceat silently mis-sums an empty slice."""
    rows = [_EVENT,
            _EVENT + [(211, 1, -1)] * 13,
            [(21, 1, -1)],                       # nothing to keep
            _EVENT[:4],
            _EVENT + [(211, 1, 0)] * 2]
    out = objects.build_genpart(_ev(rows), prune=True)
    counts = ak.to_numpy(ak.num(out)).tolist()
    assert counts == [6, 6, 0, 4, 6], counts
    for e, n in enumerate(counts):
        mom = ak.to_numpy(out.genPartIdxMother[e])
        assert all(m == -1 or 0 <= m < n for m in mom.tolist()), (e, mom)


def test_an_empty_event_does_not_corrupt_later_events():
    rows = [[], _EVENT, [], _EVENT]
    out = objects.build_genpart(_ev(rows), prune=True)
    assert ak.to_numpy(ak.num(out)).tolist() == [0, 6, 0, 6]
    for e in (1, 3):
        assert ak.to_numpy(out.genPartIdxMother[e]).tolist() == [-1, 0, 1, 2, 0, 4]
