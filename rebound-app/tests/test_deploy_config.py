"""Pins on the gunicorn config, for a bug the rest of the suite cannot see.

Every other test here drives Flask's test client in-process, so none of them fork.
The failure this file exists for only appears under a real gunicorn: with
`preload_app = True` the movement network is built in the master and the sync worker
forks without exec, inheriting a torch thread pool whose threads did not survive. The
symptom is not an exception -- `/healthz` keeps answering 200 because it touches no
tensors, and the first `/predict` hangs until gunicorn SIGKILLs the worker at the
timeout. Measured on this machine: 25 s and a dead worker with preload on, 67 ms and
a 200 with it off.

Import-time settings, so this is a cheap module-level read rather than a live server.
"""

import importlib.util
from pathlib import Path

import pytest

CONF = Path(__file__).resolve().parent.parent / "gunicorn.conf.py"


@pytest.fixture(scope="module")
def conf():
    spec = importlib.util.spec_from_file_location("gunicorn_conf", CONF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preload_is_off_so_torch_is_loaded_after_the_fork(conf):
    """If this ever goes True again, /predict hangs instead of failing."""
    assert conf.preload_app is False


def test_the_api_binds_loopback_only(conf):
    """nginx terminates TLS and proxies to 127.0.0.1:8000; see deploy/nginx.conf."""
    assert conf.bind.startswith("127.0.0.1:")


def test_workers_are_sync(conf):
    """A prediction is CPU-bound and sub-millisecond; async workers only add memory."""
    assert conf.worker_class == "sync"


def test_torch_threads_are_pinned_in_each_worker(conf):
    """N workers each sizing a thread pool to the whole machine oversubscribes it."""
    assert hasattr(conf, "post_fork")
    import os

    assert os.environ.get("OMP_NUM_THREADS") == "1"


def test_the_systemd_unit_matches_this_config():
    """The unit shipped `app:app`, which no longer exists -- the entry point is wsgi."""
    unit = (CONF.parent.parent / "deploy" / "rebound.service").read_text()
    assert "wsgi:app" in unit
    assert "-c gunicorn.conf.py" in unit
    assert "app:app" not in unit.replace("wsgi:app", "")
