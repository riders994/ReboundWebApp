"""The movement model: what the app animates, and where its extra features come from.

This replaced the 2017 `posnn.h5` adapter that used to live here. That version existed
to reproduce an eight-column input contract exactly -- a hoop at 41.65, an `arctan2`
with its arguments swapped, box-out counts assigned to the wrong team -- because those
were the vectors the old Keras weights had been trained on and correcting them would
have fed the model inputs unlike anything it had seen. None of that survives: the
model is retrained, it eats `rebounding.data.features.SERVED_FEATURES` like everything
else in the pipeline, and there is no second definition of a court in this file.

Two behaviours are worth knowing about before wiring this into a view.

**It samples.** The model predicts a distribution over whole scenes, not a point per
player, and :meth:`Movement.scenes` returns several draws. Drawing the average of
those draws is the failure the retrain was for -- a player who might crash the glass
or might leak out is not, on average, standing between the two. Animate one draw, or
draw several as ghosts to show the spread; do not mean them together first.

**It still fails soft.** If the bundle is missing or torch is not installed the app
starts anyway, `/healthz` says which model is live, and every player stays where the
user put him. The rebounder is a separate artifact and is unaffected either way.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

N_PLAYERS = 10

# How many futures to draw per request. Cheap -- one encoder pass is shared across all
# of them -- and the front end can show as few as it likes.
DEFAULT_SCENES = 12


class NoMovement:
    """The null model: everybody stays where they were placed.

    Returned rather than raised so a missing bundle degrades the animation instead of
    taking down rebound prediction, which does not need this model at all.
    """

    name = "none"

    def __init__(self, detail: str = "disabled") -> None:
        self.detail = detail

    @property
    def artifact(self):
        return None

    def scenes(self, players, n: int = DEFAULT_SCENES, seed: int | None = None) -> np.ndarray:
        held = np.array([[float(p.x), float(p.y)] for p in players], dtype=float)
        return np.repeat(held[None], n, axis=0)

    def mean(self, players) -> np.ndarray:
        return self.scenes(players, n=1)[0]


class Movement:
    """A loaded :class:`~rebounding.models.artifact.MovementArtifact`."""

    name = "set-transformer"

    def __init__(self, artifact, path: Path) -> None:
        self._artifact = artifact
        config = artifact.state.get("config")
        head = getattr(config, "head", "?")
        parameters = sum(v.size for v in artifact.state["weights"].values())
        self.detail = f"{path.name}: {head} head, {parameters:,} parameters"

    @property
    def artifact(self):
        return self._artifact

    def scenes(self, players, n: int = DEFAULT_SCENES, seed: int | None = None) -> np.ndarray:
        """``(n, 10, 2)`` sampled rim-time positions, in the caller's player order.

        ``seed`` is normally ``None``, and deliberately so: the model samples futures,
        and a visitor who nudges one defender and re-runs should see the scene move.
        Fixing the seed would make the demo look deterministic and hide the very
        uncertainty the distribution is there to show. Tests set it so a sampled
        scene can be pinned against a fixture.
        """
        from rebounding.serve import animate

        return animate(self._artifact, players, n=n, seed=seed).scenes

    def mean(self, players) -> np.ndarray:
        """The conditional-mean scene. For the rebounder, not for the screen."""
        from rebounding.serve import animate

        return animate(self._artifact, players, n=1).mean


def load(path: Path | str, enabled: bool = True) -> Movement | NoMovement:
    """Load the movement bundle, or explain in one line why there is no animation."""
    if not enabled:
        return NoMovement("disabled by REBOUND_MOVEMENT=0")

    path = Path(path)
    if not path.exists():
        return NoMovement(f"no bundle at {path}")

    try:
        from rebounding.models.artifact import load_movement
    except ImportError as exc:
        return NoMovement(f"rebounding package not importable: {exc}")

    try:
        artifact = load_movement(path)
        # Force the network to build now rather than on the first request, so a torch
        # that cannot load the weights fails at start-up where somebody is watching.
        assert artifact.model is not None
    except Exception as exc:  # noqa: BLE001 - any failure here is the same failure
        LOGGER.warning("could not load %s: %s", path, exc)
        return NoMovement(f"{type(exc).__name__} loading {path.name}: {exc}")

    LOGGER.info("loaded movement model from %s", path)
    return Movement(artifact, path)
