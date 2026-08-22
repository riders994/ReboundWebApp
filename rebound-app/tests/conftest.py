import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

# The bundles live in rebound-app/models/ here, not at the repo root as in the 2017
# layout the port was staged against. They arrive by scp and are gitignored.
MODEL = APP_ROOT / "models" / "FinalModel.pkl"
MOVEMENT = APP_ROOT / "models" / "MovementModel.pkl"


@pytest.fixture(scope="session")
def model_path() -> Path:
    if not MODEL.exists():
        pytest.skip(f"no artifact at {MODEL}; run `python -m rebounding.cli train`")
    return MODEL


@pytest.fixture(scope="session")
def movement_path() -> Path:
    if not MOVEMENT.exists():
        pytest.skip(f"no movement bundle at {MOVEMENT}")
    return MOVEMENT


@pytest.fixture(scope="session")
def client(model_path):
    from app import create_app
    from config import Config

    application = create_app(Config(model_path=model_path, movement_enabled=False))
    application.config.update(TESTING=True)
    return application.test_client()


@pytest.fixture
def bench() -> list[dict]:
    """Ten placed players in canvas feet: five offense, five defense, one shooter."""
    spots = [(25, 41), (20, 38), (30, 37), (18, 30), (32, 25),
             (24, 39), (28, 39), (21, 33), (34, 28), (15, 22)]
    return [
        {"x": float(x), "y": float(y), "isOffense": i < 5, "isShooter": i == 4}
        for i, (x, y) in enumerate(spots)
    ]
