"""Gunicorn settings.

The last commit on the original repo was "Need to add GUnicorn capabilities to load
balance", so this is that, finally. `app.run(host='0.0.0.0', port=80)` was the Flask
development server: single-threaded, no request limits, and explicitly not for
production.

Workers are sync and few. Each one loads its own copy of the model, and a prediction is
LightGBM over eighty rows -- microseconds of CPU, no I/O to wait on -- so the async
workers that help an I/O-bound app would only add memory here.

**`preload_app` is off, and it must stay off.** The port shipped it on, to share the
loaded model across forks via copy-on-write. That is safe for LightGBM and fatal with
torch: preloading builds the movement network in the master, and gunicorn's sync worker
forks without exec, so each child inherits a torch intra-op thread pool whose threads
did not survive the fork. `/healthz` still answers -- it touches no tensors -- but the
first `/predict` deadlocks in the child and gunicorn SIGKILLs it at the timeout, so the
demo hangs rather than erroring. Loading after the fork costs about six megabytes per
worker and a second of start-up, which is the cheap side of that trade.
"""

import logging
import multiprocessing
import os
import sys
from pathlib import Path

# `rebound-app` has a hyphen and so can never be an importable package; `wsgi`, `app`,
# `config`, `coordinates` and `movement` are flat modules in this directory. Putting it
# on the path here means gunicorn can import `wsgi:app` without systemd and gunicorn
# having to agree on a working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Loopback, not 0.0.0.0: nginx terminates TLS and proxies to 127.0.0.1:8000
# (deploy/nginx.conf). Binding all interfaces would expose the API directly.
bind = f"127.0.0.1:{os.getenv('PORT', '8000')}"
workers = int(os.getenv("WEB_CONCURRENCY", min(4, multiprocessing.cpu_count())))
worker_class = "sync"
threads = 1
timeout = 30
graceful_timeout = 30
keepalive = 5

# See the module docstring: ON deadlocks the first /predict in a forked worker.
preload_app = False
# One torch thread per worker. The model is ~800k parameters over ten tokens, so a
# request is one small forward pass and intra-op parallelism buys nothing; without
# this, N sync workers each spawn a pool sized to the whole machine and oversubscribe
# it. Set before any worker imports torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def post_fork(server, worker):
    """Pin torch to one thread, and give the app's loggers somewhere to go.

    Both have to happen here rather than at import: `preload_app` is off, so the worker
    is where torch is first loaded and where `create_app` runs. gunicorn calls this
    before `init_process` loads the WSGI app, so the handlers below are in place before
    the app logs its first line.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass  # movement model is optional; the rebounder does not need torch

    # gunicorn configures its own `gunicorn.error` logger and leaves the root logger
    # alone, and nothing else configures it either -- `logging.basicConfig` is in
    # app.py's `__main__` block, which gunicorn never executes. The result is that
    # every `LOGGER.info` in the app is dropped and every warning is emitted by
    # logging's `lastResort` handler, which has no formatter: the telemetry sink and
    # model-load lines never reach `journalctl -u rebound`, and the warnings that do
    # arrive bare, with no timestamp, level or logger name. Handing the root logger
    # gunicorn's handlers puts app logs in the same stream and format as gunicorn's
    # own, under the same LOG_LEVEL.
    error_log = logging.getLogger("gunicorn.error")
    root = logging.getLogger()
    root.handlers = error_log.handlers
    root.setLevel(error_log.level)


accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")
