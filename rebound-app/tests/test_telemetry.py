"""Tests for prediction logging.

The thing under test is not really the INSERT -- it is the promise that none of this
can affect a prediction. A shared Postgres box that other projects also depend on is
exactly the kind of dependency that is down at an inconvenient moment, and the whole
design is written so that when it is, the only casualty is a row. So most of what is
pinned here is absence: no database configured, a bad one configured, a queue that has
filled up, a driver that is not installed. In every case `/predict` must still answer
200 and `/healthz` must say what is wrong.

No test here talks to Postgres. The writer thread is stopped or stubbed instead, which
keeps the suite runnable on a laptop with no database -- the same reason `conftest.py`
skips rather than fails when the model bundles are absent.
"""

import json
from datetime import datetime, timezone

import pytest
import telemetry as telemetry_module
from config import Config
from coordinates import canvas_to_model
from telemetry import DEFAULT_QUEUE_SIZE, NoTelemetry, Prediction, Telemetry
from telemetry import load as load_telemetry


def a_prediction(**overrides) -> Prediction:
    base = dict(
        served_at=datetime.now(timezone.utc),
        model_commit="abc1234",
        model_built="2026-01-01T00:00:00Z",
        model_top1=0.297,
        movement="none",
        latency_ms=1.5,
        scenes=1,
        lineup=[{"x": 1.0, "y": 2.0, "is_offense": True, "is_shooter": False, "position": "G"}],
        probabilities=[1.0],
    )
    base.update(overrides)
    return Prediction(**base)


# --------------------------------------------------------------------------- #
# Nothing configured is the default, and the default must cost nothing
# --------------------------------------------------------------------------- #


def test_no_database_url_means_no_telemetry():
    """The app has no database dependency unless somebody gives it one."""
    sink = load_telemetry("")
    assert isinstance(sink, NoTelemetry)
    assert "REBOUND_DATABASE_URL" in sink.detail


def test_config_defaults_to_off(monkeypatch):
    monkeypatch.delenv("REBOUND_DATABASE_URL", raising=False)
    assert Config().database_url == ""


def test_logging_can_be_switched_off_with_the_url_left_in_place():
    """For turning it off mid-incident without editing the connection string out."""
    sink = load_telemetry("postgresql://localhost/whatever", enabled=False)
    assert sink.name == "none"
    assert "REBOUND_TELEMETRY=0" in sink.detail


def test_a_schema_name_that_is_not_an_identifier_is_refused():
    """The schema is interpolated into DDL, so it is validated rather than escaped."""
    sink = load_telemetry("postgresql://localhost/x", schema="rebound; DROP TABLE users")
    assert isinstance(sink, NoTelemetry)
    assert "invalid schema name" in sink.detail


def test_a_missing_driver_degrades_instead_of_raising(monkeypatch):
    """psycopg is not in requirements by default; its absence must not be an error."""
    import builtins

    real_import = builtins.__import__

    def no_psycopg(name, *args, **kwargs):
        if name == "psycopg":
            raise ImportError("No module named 'psycopg'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psycopg)
    sink = load_telemetry("postgresql://localhost/x")
    assert isinstance(sink, NoTelemetry)
    assert "psycopg not installed" in sink.detail


def test_the_null_sink_accepts_rows_and_reports_none_kept():
    sink = NoTelemetry()
    sink.record(a_prediction())
    assert sink.stats == {"recorded": 0, "dropped": 0}


# --------------------------------------------------------------------------- #
# The queue, which is what keeps a slow database out of the request path
# --------------------------------------------------------------------------- #


@pytest.fixture
def stalled(monkeypatch):
    """A live Telemetry whose writer never consumes, so the queue fills predictably.

    Stubbing `_run` also means no connection is ever attempted, which is what lets
    these run without a database.
    """
    monkeypatch.setattr(Telemetry, "_run", lambda self: self._stop.wait())
    sink = Telemetry("postgresql://unused/unused", "rebound", queue_size=3)
    yield sink
    sink.close(timeout=1.0)


def test_a_full_queue_drops_rows_instead_of_blocking(stalled):
    """The property the request path depends on: `record` returns, whatever happens."""
    for _ in range(10):
        stalled.record(a_prediction())
    assert stalled.stats["dropped"] == 7
    assert stalled.stats["recorded"] == 0


def test_the_queue_is_bounded(stalled):
    """Unbounded would turn a database outage into an out-of-memory kill."""
    assert stalled._queue.maxsize == 3
    assert DEFAULT_QUEUE_SIZE > 0


def test_close_is_safe_to_call_twice(stalled):
    """atexit runs after an explicit close in some shutdown paths."""
    stalled.close(timeout=1.0)
    stalled.close(timeout=1.0)


# --------------------------------------------------------------------------- #
# The row itself
# --------------------------------------------------------------------------- #


def test_a_row_is_json_serialisable_where_it_has_to_be():
    """`lineup` and `probabilities` are cast to jsonb, so they are dumped, not adapted."""
    row = a_prediction().row()
    assert json.loads(row[-2]) == [
        {"x": 1.0, "y": 2.0, "is_offense": True, "is_shooter": False, "position": "G"}
    ]
    assert json.loads(row[-1]) == [1.0]


def test_a_null_position_survives_into_the_row():
    """How often `role` is defaulted is one of the questions the table exists for."""
    lineup = [{"x": 0.0, "y": 0.0, "is_offense": True, "is_shooter": True, "position": None}]
    row = a_prediction(lineup=lineup).row()
    assert json.loads(row[-2])[0]["position"] is None


# --------------------------------------------------------------------------- #
# Through the app
# --------------------------------------------------------------------------- #


class Capture:
    """A sink that keeps rows in memory, standing in for the writer thread."""

    name = "capture"
    detail = "in memory"

    def __init__(self):
        self.rows = []

    def record(self, prediction):
        self.rows.append(prediction)

    def close(self, timeout: float = 2.0):
        pass

    @property
    def stats(self):
        return {"recorded": len(self.rows), "dropped": 0}


@pytest.fixture
def capturing(monkeypatch, model_path):
    """An app wired to `Capture` instead of Postgres."""
    from app import create_app

    sink = Capture()
    monkeypatch.setattr(telemetry_module, "load", lambda *a, **k: sink)
    application = create_app(Config(model_path=model_path, movement_enabled=False))
    application.config.update(TESTING=True)
    return application.test_client(), sink


def test_a_served_prediction_is_recorded(capturing, bench):
    client, sink = capturing
    assert client.post("/predict", json={"bench": bench}).status_code == 200
    assert len(sink.rows) == 1

    row = sink.rows[0]
    assert len(row.lineup) == 10
    assert len(row.probabilities) == 10
    assert sum(row.probabilities) == pytest.approx(1.0)
    assert row.latency_ms > 0


def test_the_recorded_lineup_is_in_the_model_frame(capturing, bench):
    """Canvas and model axes are swapped; the row keeps what the model was given."""
    client, sink = capturing
    client.post("/predict", json={"bench": bench})

    for placed, logged in zip(bench, sink.rows[0].lineup, strict=True):
        model_x, model_y = canvas_to_model(placed["x"], placed["y"])
        assert logged["x"] == pytest.approx(float(model_x))
        assert logged["y"] == pytest.approx(float(model_y))
    # And the frames really are different, or the assertion above proves nothing.
    assert sink.rows[0].lineup[0]["x"] != pytest.approx(bench[0]["x"])


def test_the_row_carries_the_bundle_provenance(capturing, bench):
    """After a retrain, this is what ties a prediction to the build that made it."""
    client, sink = capturing
    client.post("/predict", json={"bench": bench})

    row = sink.rows[0]
    assert row.model_commit
    assert 0.0 < row.model_top1 < 1.0
    assert row.movement == "none"


def test_a_refused_placement_is_not_recorded(capturing, bench):
    """A 400 is a lineup the model never saw; logging it would bias the distribution."""
    client, sink = capturing
    assert client.post("/predict", json={"bench": bench[:9]}).status_code == 400
    assert sink.rows == []


def test_healthz_reports_the_sink(capturing):
    client, _ = capturing
    body = client.get("/healthz").get_json()
    assert body["telemetry"]["sink"] == "capture"
    assert body["telemetry"]["dropped"] == 0


def test_predictions_are_served_when_the_sink_is_broken(monkeypatch, model_path, bench):
    """The whole point: a logging failure costs a row, not a response."""
    from app import create_app

    class Broken:
        name = "broken"
        detail = "raises on every row"

        def record(self, prediction):
            raise RuntimeError("database on fire")

        @property
        def stats(self):
            return {"recorded": 0, "dropped": 0}

    monkeypatch.setattr(telemetry_module, "load", lambda *a, **k: Broken())
    application = create_app(Config(model_path=model_path, movement_enabled=False))
    application.config.update(TESTING=True)

    response = application.test_client().post("/predict", json={"bench": bench})
    assert response.status_code == 200
    assert len(response.get_json()) == 10


# --------------------------------------------------------------------------- #
# The SQL, which nothing else in this suite executes
# --------------------------------------------------------------------------- #


def test_every_column_in_the_insert_has_a_value():
    """A placeholder/value mismatch is a runtime error on a background thread.

    That is the worst place for it: the request still succeeds, so the only symptom is
    `dropped` climbing on `/healthz`. Counting them here turns it into a failed test.
    """
    from telemetry import INSERT

    assert INSERT.count("%s") == len(a_prediction().row())


def test_the_ddl_is_fully_parameterised_by_the_schema():
    """A `{schema}` left unformatted would be a syntax error against a live server."""
    from telemetry import DDL, INSERT

    for statement in (DDL, INSERT):
        rendered = statement.format(schema="rebound")
        assert "{" not in rendered and "}" not in rendered
        assert "public." not in rendered

    assert DDL.format(schema="other").count("other.") >= 3


def test_close_does_not_wait_out_a_reconnect_backoff():
    """Shutting down while the database is unreachable must not hang the worker.

    The writer sleeps between failed connections, and that sleep grows to minutes. If
    `close` only queued a sentinel, the thread would not see it until the sleep ended
    -- so a deploy during a Postgres outage would block for its whole join timeout and
    then leave the thread behind. The stop event has to interrupt the sleep itself.

    Port 59999 on loopback is refused rather than filtered, so this fails fast and does
    not depend on a network timeout. No driver is needed: an absent psycopg raises out
    of `_connect` and lands on the same backoff path a refused connection does.
    """
    import time

    sink = Telemetry("postgresql://nobody@127.0.0.1:59999/nowhere", "rebound", queue_size=8)
    try:
        sink.record(a_prediction())
        time.sleep(0.5)  # long enough to fail a loopback connect and start backing off

        started = time.perf_counter()
        sink.close(timeout=3.0)
        assert time.perf_counter() - started < 2.0
        assert not sink._thread.is_alive()
        assert sink.stats["recorded"] == 0
    finally:
        sink.close(timeout=1.0)


def test_a_dropped_row_puts_the_reason_on_healthz():
    """`dropped` says something is wrong; `detail` is what says what."""
    import time

    sink = Telemetry("postgresql://nobody@127.0.0.1:59999/nowhere", "rebound", queue_size=8)
    try:
        sink.record(a_prediction())
        deadline = time.time() + 5.0
        while time.time() < deadline and sink.stats["dropped"] == 0:
            time.sleep(0.05)
        assert sink.stats["dropped"] == 1
        assert sink.detail != "schema rebound, queue 8"
    finally:
        sink.close(timeout=1.0)
