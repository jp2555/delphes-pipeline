"""Gen tau decay mode — the conditioning variable the v2 tau maps need.

Visible mass is essentially determined by decay mode (1-prong ~ m_pi; 1-prong+pi0 ~
rho(770); 3-prong ~ a1(1260)), so drawing mass and energy response independently of it
is the worst of the destroyed-correlation defects. Delphes has no decay-mode flag and no
daughter links, so this is counted by descent through the m1 chain.
"""
import awkward as ak
import numpy as np
import pytest

from delphes_pipeline.core import observables as obs

TAU, NU_TAU, PIP, PI0, K, MU, GAMMA = 15, 16, 211, 111, 321, 13, 22


def _event(decays):
    """Build one gen event. `decays` is a list of daughter-pid lists, one per tau.

    Layout: taus first (mother -1), then each tau's daughters pointing at their tau.
    Photons from pi0 are added too, to prove they are not double-counted as prongs.
    """
    pid, status, m1 = [], [], []
    for _ in decays:
        pid.append(TAU); status.append(2); m1.append(-1)
    for t, daughters in enumerate(decays):
        for d in daughters:
            pid.append(d)
            status.append(2 if d == PI0 else 1)
            m1.append(t)
            if d == PI0:                       # pi0 -> gamma gamma
                for _ in range(2):
                    pid.append(GAMMA); status.append(1); m1.append(len(pid) - 2)
    n = len(pid)
    # ak.zip gives `1 * var * {record}` — a jagged list OF records, the layout the real
    # gen record has. `ak.Array([{...list fields...}])` builds a record OF lists instead,
    # on which ak.local_index returns a record of indices and every comparison breaks.
    return ak.zip({"pid": ak.Array([pid]), "status": ak.Array([status]),
                   "m1": ak.Array([m1]), "pt": ak.Array([[10.0] * n]),
                   "eta": ak.Array([[0.0] * n]), "phi": ak.Array([[0.0] * n]),
                   "mass": ak.Array([[0.0] * n])})


def _dm(decays):
    return ak.to_list(obs.gen_tau_decay_mode(_event(decays)))[0]


def test_one_prong_no_pi0_is_dm0():
    assert _dm([[PIP, NU_TAU]]) == [0]


def test_one_prong_one_pi0_is_dm1():
    assert _dm([[PIP, PI0, NU_TAU]]) == [1]


def test_one_prong_two_pi0_is_dm2():
    assert _dm([[PIP, PI0, PI0, NU_TAU]]) == [2]


def test_three_prong_no_pi0_is_dm10():
    assert _dm([[PIP, PIP, PIP, NU_TAU]]) == [10]


def test_three_prong_one_pi0_is_dm11():
    assert _dm([[PIP, PIP, PIP, PI0, NU_TAU]]) == [11]


def test_a_charged_kaon_counts_as_a_prong():
    assert _dm([[K, NU_TAU]]) == [0]


def test_a_leptonic_tau_is_minus_one():
    """Not a 1-prong hadronic decay — handing it the hadronic response is the ~125x error."""
    assert _dm([[MU, NU_TAU, NU_TAU]]) == [-1]


def test_pi0_photons_are_not_counted_as_prongs():
    """Counting photons instead of the pi0 would double every neutral."""
    assert _dm([[PIP, PI0, NU_TAU]]) == [1]


def test_several_taus_in_one_event_do_not_mix_their_daughters():
    assert _dm([[PIP, NU_TAU], [PIP, PIP, PIP, PI0, NU_TAU]]) == [0, 11]


def test_an_unclassifiable_multiplicity_is_minus_one():
    assert _dm([[PIP, PIP, NU_TAU]]) == [-1]      # 2 prongs: not a CMS category


def test_alignment_with_gen_taus_is_one_to_one():
    """A silent misalignment would attach every tau's response to the wrong mode."""
    ev = _event([[PIP, NU_TAU], [MU, NU_TAU, NU_TAU], [PIP, PI0, NU_TAU]])
    taus = obs.gen_taus(ev, hadronic_only=False)
    dm = obs.gen_tau_decay_mode(ev)
    assert ak.to_list(ak.num(taus)) == ak.to_list(ak.num(dm)) == [3]
    assert ak.to_list(dm)[0] == [0, -1, 1]


def test_generator_copies_are_collapsed_like_gen_taus():
    """The Pythia history repeats each tau; counting copies would triple the maps."""
    ev = ak.zip({"pid": ak.Array([[TAU, TAU, PIP, NU_TAU]]),
                 "status": ak.Array([[2, 2, 1, 1]]),
                 "m1": ak.Array([[-1, 0, 1, 1]]),      # tau[1] is a copy of tau[0]
                 "pt": ak.Array([[1.0] * 4]), "eta": ak.Array([[0.0] * 4]),
                 "phi": ak.Array([[0.0] * 4]), "mass": ak.Array([[0.0] * 4])})
    assert ak.to_list(obs.gen_tau_decay_mode(ev))[0] == [0]
    # Without collapse the ancestor walk stops at the NEAREST tau, so the earlier copy
    # owns no daughters and classifies as -1 — a spurious extra tau carrying no decay
    # mode. That is exactly the population the collapse exists to remove.
    assert ak.to_list(obs.gen_tau_decay_mode(ev, last_copy=False))[0] == [-1, 0]
