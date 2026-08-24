"""Runtime configuration, all of it from the environment.

The original hardcoded three paths, a port, and an unrestricted CORS policy. Nothing
here changes behaviour by default except CORS, which is now off unless you name the
origins that need it -- the front end is served by this same app, so same-origin is
the normal case and a blanket ``CORS(app)`` was granting access nobody needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Paths are anchored to this file, not the working directory. The 2017 app resolved
# everything from `os.getcwd()`, which is why it ran from exactly one directory;
# systemd and pytest do not share a cwd with each other or with a shell.
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    # The bundle from `python -m rebounding.cli train`, copied here by scp.
    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("REBOUND_MODEL", MODEL_DIR / "FinalModel.pkl"))
    )

    # The movement bundle from `python -m rebounding.cli train-movement`, copied here
    # by the same scp as the rebounder. Optional: without it the app still predicts
    # rebounders and simply leaves every player where he was placed. There is no
    # longer an `msd.pkl` -- the normalisation constants are buffers inside the
    # bundle, so they cannot drift away from the weights they belong to.
    movement_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("REBOUND_MOVEMENT_MODEL", MODEL_DIR / "MovementModel.pkl")
        )
    )
    movement_enabled: bool = field(default_factory=lambda: _flag("REBOUND_MOVEMENT", True))

    # How many futures to sample per request. The response carries all of them; the
    # front end decides whether to animate one or fan them out as ghosts.
    movement_scenes: int = field(default_factory=lambda: int(os.getenv("REBOUND_MOVEMENT_SCENES", "12")))

    # Sampling seed. `None` (the default) means a fresh draw per request, which is the
    # point of a model that predicts a distribution. Set it only to make a run
    # reproducible -- the regression fixture pins scene 0 of seed 0.
    movement_seed: int | None = field(
        default_factory=lambda: (
            int(os.environ["REBOUND_MOVEMENT_SEED"]) if os.getenv("REBOUND_MOVEMENT_SEED") else None
        )
    )

    # Comma-separated origins, e.g. "https://example.com". Empty means same-origin only.
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            origin.strip() for origin in os.getenv("REBOUND_CORS_ORIGINS", "").split(",") if origin.strip()
        )
    )

    # Refuse to start without a model rather than serving 503s nobody notices.
    require_model: bool = field(default_factory=lambda: _flag("REBOUND_REQUIRE_MODEL", True))

    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
