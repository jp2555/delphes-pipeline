"""A transient dCache failure must not cost a whole shard.

Four shards of the first production campaign died on
``[FATAL] TLS error: Unable to validate <ip>; hostname not in SAN extension`` --
the redirector handed the job a data server whose certificate did not cover the
address. 1105 sibling shards read the same storage fine, so the failure is a bad
door node, not a bad file: exactly what a retry is for.
"""
import pytest

from delphes_pipeline.core import io

REAL_TLS_ERROR = ("File did not open properly: [FATAL] TLS error: Unable to "
                  "validate 157.180.232.69; hostname not in SAN extension.")


class _Flaky:
    """Fails the first ``n`` opens, then succeeds."""

    def __init__(self, n, exc=None):
        self.n, self.calls, self.exc = n, 0, exc or OSError(REAL_TLS_ERROR)

    def __call__(self, path):
        self.calls += 1
        if self.calls <= self.n:
            raise self.exc
        return f"tree:{path}"


def _run(monkeypatch, flaky, path="root://x.example//a.root", **kw):
    monkeypatch.setattr(io.uproot, "open", flaky)
    slept = []
    return io.open_with_retry(path, _sleep=slept.append, **kw), slept


def test_the_production_tls_failure_is_retried_and_recovers(monkeypatch):
    flaky = _Flaky(2)
    got, slept = _run(monkeypatch, flaky)
    assert got == "tree:root://x.example//a.root"
    assert flaky.calls == 3


def test_retries_back_off_instead_of_hammering_a_sick_door(monkeypatch):
    _, slept = _run(monkeypatch, _Flaky(2), backoff=5.0)
    assert slept == [5.0, 10.0], "backoff must grow, not retry in a tight loop"


def test_a_persistently_bad_file_still_fails_and_is_bounded(monkeypatch):
    flaky = _Flaky(99)
    monkeypatch.setattr(io.uproot, "open", flaky)
    with pytest.raises(OSError, match="SAN extension"):
        io.open_with_retry("root://x//a.root", tries=4, _sleep=lambda s: None)
    assert flaky.calls == 4, "retries must be bounded, not infinite"


def test_a_local_path_fails_fast_with_no_backoff(monkeypatch):
    """A typo in a local path must not cost 15 s of pointless waiting."""
    flaky = _Flaky(99)
    monkeypatch.setattr(io.uproot, "open", flaky)
    with pytest.raises(OSError):
        io.open_with_retry("/ceph/jpan/missing.root", _sleep=lambda s: None)
    assert flaky.calls == 1


def test_a_non_transport_error_is_not_retried(monkeypatch):
    """Retrying a real logic error just delays the traceback."""
    flaky = _Flaky(99, exc=KeyError("Delphes"))
    monkeypatch.setattr(io.uproot, "open", flaky)
    with pytest.raises(KeyError):
        io.open_with_retry("root://x//a.root", _sleep=lambda s: None)
    assert flaky.calls == 1


def test_a_first_attempt_success_costs_nothing(monkeypatch):
    flaky = _Flaky(0)
    got, slept = _run(monkeypatch, flaky)
    assert (flaky.calls, slept) == (1, [])
