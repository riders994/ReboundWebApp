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


def test_app_logs_are_wired_into_gunicorns_stream(conf):
    """Without this, everything the app logs below WARNING is silently discarded.

    gunicorn configures `gunicorn.error` and leaves the root logger alone, and nothing
    else configures it either -- `logging.basicConfig` lives in app.py's `__main__`
    block, which gunicorn never runs. Measured before the fix: a clean boot printed
    gunicorn's own lines and the access line, and neither "loaded FinalModel.pkl" nor
    the telemetry sink line, while warnings arrived through logging's `lastResort`
    handler with no timestamp, level or logger name. Both are the lines DEPLOY.md's
    troubleshooting section sends you to `journalctl -u rebound` to read.

    Like the rest of this file, this is the failure the in-process suite cannot see:
    every other test configures logging itself or does not look.
    """
    import logging

    root = logging.getLogger()
    error_log = logging.getLogger("gunicorn.error")
    saved = (root.handlers, root.level, error_log.handlers, error_log.level)

    sentinel = logging.StreamHandler()
    error_log.handlers = [sentinel]
    error_log.setLevel(logging.INFO)
    try:
        conf.post_fork(None, None)
        assert sentinel in root.handlers, "app logs have nowhere to go"
        assert root.level == logging.INFO, "root must follow gunicorn's LOG_LEVEL"
    finally:
        root.handlers, error_log.handlers = saved[0], saved[2]
        root.setLevel(saved[1])
        error_log.setLevel(saved[3])
