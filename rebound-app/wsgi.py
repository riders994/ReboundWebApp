"""Gunicorn entry point: `gunicorn -c gunicorn.conf.py wsgi:app`.

Separate from `app.py` so that importing the module does not build an app -- the
factory reads the environment and loads a 2.5 MB model, which is not something an
import should do to a test run.

The directory is put on `sys.path` explicitly because `rebound-app` has a hyphen in
its name and so can never be an importable package; `app`, `config`, `coordinates`
and `movement` are flat modules that need this directory on the path. Doing it here
means gunicorn and systemd do not have to agree on a working directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_app  # noqa: E402

app = create_app()
