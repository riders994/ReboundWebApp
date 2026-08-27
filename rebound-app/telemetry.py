"""Prediction logging: what the model was actually asked, and what it answered.

The README states the rebounder's 29.7% top-1 was measured on real NBA lineups --
37% guards, 11% centres -- and warns that a visitor sketching five centres is outside
that distribution, so the headline number is a ceiling rather than a promise. That
warning is currently unfalsifiable: nothing anywhere records what visitors actually
place. This module is what makes it measurable. Every served prediction becomes one
row, and the distribution of real requests can then be compared against the corpus
the score was measured on.

The second thing it buys is provenance at serving time. The bundles arrive by scp and
never appear in git, so `/healthz` is the only way to ask a *running* host what it is
serving. That answer is not retained anywhere, which means after a retrain there is no
way to say which build produced a prediction somebody is asking about. Each row
carries the bundle's own commit and build time, so that question has an answer.

Three properties matter more than the schema.

**It fails soft, like the movement model.** No `REBOUND_DATABASE_URL`, no psycopg
installed, a database that is down, a table it cannot create -- all of them degrade to
:class:`NoTelemetry`, which drops rows on the floor and says why in `/healthz`. The
Postgres box is shared with other projects and is not this app's dependency to be
taken down by. A prediction must never fail because a logging table was unreachable.

**It never runs in the request path.** Rows go onto a bounded queue and a background
thread writes them in batches. A slow or wedged database costs a dropped row and
nothing else -- no added latency, no worker blocked on a socket. The queue is bounded
rather than unbounded on purpose: under a database outage an unbounded one converts a
logging failure into an out-of-memory failure, which is a worse version of the thing
this is trying to avoid.

**It connects after the fork.** The connection is opened by the writer thread, which
starts inside :func:`load`, which `create_app` calls in the gunicorn *worker* -- see
`gunicorn.conf.py` on why `preload_app` stays off. A connection opened in the master
and inherited across a fork is shared by every worker at once, and the failure looks
like corrupted protocol state rather than an error at the point of the mistake.

Nothing here identifies a visitor. No IP, no user agent, no cookie, no session id --
only the placements, which the visitor made deliberately and which are the whole point
of keeping the row. That is a deliberate limit, not an oversight: it means the table
answers "what lineups does the model see" without becoming a visitor log that has to
be reasoned about every time somebody wants to query it.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger(__name__)

# Rows buffered before new ones are dropped. Each is a few hundred bytes, so this is
# well under a megabyte per worker; it exists to bound an outage, not to hold a backlog.
DEFAULT_QUEUE_SIZE = 512

# Rows per INSERT. A prediction is sub-millisecond and bursts arrive together, so
# batching turns a burst into one round trip.
BATCH_SIZE = 64

# How long the writer waits for a batch to fill before flushing what it has.
FLUSH_SECONDS = 2.0

# Backoff after a failed connection, so a database that is down is retried without
# spinning. Doubles up to the cap.
RETRY_SECONDS = 5.0
MAX_RETRY_SECONDS = 300.0

# The schema is named rather than `public` because the server is shared with other
# projects. Everything this app writes lives under one name that can be granted,
# dumped or dropped on its own.
DDL = """
CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.predictions (
    id            bigserial PRIMARY KEY,
    served_at     timestamptz      NOT NULL,
    model_commit  text,
    model_built   text,
    model_top1    double precision,
    movement      text             NOT NULL,
    latency_ms    double precision NOT NULL,
    scenes        integer          NOT NULL,
    lineup        jsonb            NOT NULL,
    probabilities jsonb            NOT NULL
);

-- Recent-first is how this gets read by hand; by commit is how a retrain gets
-- compared against the build before it.
CREATE INDEX IF NOT EXISTS predictions_served_at_idx
    ON {schema}.predictions (served_at DESC);
CREATE INDEX IF NOT EXISTS predictions_model_commit_idx
    ON {schema}.predictions (model_commit);
"""

INSERT = """
INSERT INTO {schema}.predictions
    (served_at, model_commit, model_built, model_top1,
     movement, latency_ms, scenes, lineup, probabilities)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
"""


@dataclass(frozen=True)
class Prediction:
    """One served prediction, in the frame the model saw it.

    ``lineup`` holds **model-frame** coordinates, not the canvas feet the browser
    posted. The two differ by the axis swap in `coordinates.py`, and the model frame
    is the one the score was measured in -- storing what the model consumed means a
    later question about the distribution does not have to trust that the conversion
    was the same on the day the row was written. The swap is exact and pinned in both
    directions by `tests/test_port.py`, so canvas coordinates remain recoverable.

    ``position`` is kept even when it is ``None``. A null there is the case the README
    calls the expensive one -- the pipeline's `role` falls back to its default and the
    same model scores 26.9% instead of 29.7% -- so how often it happens in real traffic
    is one of the things this table exists to answer.
    """

    served_at: datetime
    model_commit: str | None
    model_built: str | None
    model_top1: float | None
    movement: str
    latency_ms: float
    scenes: int
    lineup: list[dict[str, Any]]
    probabilities: list[float]

    def row(self) -> tuple:
        return (
            self.served_at,
            self.model_commit,
            self.model_built,
            self.model_top1,
            self.movement,
            self.latency_ms,
            self.scenes,
            json.dumps(self.lineup),
            json.dumps(self.probabilities),
        )


class NoTelemetry:
    """The null sink: nothing is recorded and `/healthz` says why.

    Returned rather than raised for the same reason `movement.NoMovement` is. Logging
    is an observability feature; an app that refuses to answer because it could not
    record the answer has inverted its own priorities.
    """

    name = "none"

    def __init__(self, detail: str = "disabled") -> None:
        self.detail = detail
        self.dropped = 0

    def record(self, prediction: Prediction) -> None:
        return None

    def close(self, timeout: float = 2.0) -> None:
        return None

    @property
    def stats(self) -> dict[str, Any]:
        return {"recorded": 0, "dropped": 0}


class Telemetry:
    """A background writer against a Postgres table.

    :meth:`record` is called from the request thread and does one bounded, non-blocking
    put. Everything that can be slow -- connecting, reconnecting, inserting -- happens
    on the writer thread, where its worst case is a lost row.
    """

    name = "postgres"

    def __init__(self, dsn: str, schema: str, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._dsn = dsn
        self._schema = schema
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._recorded = 0
        self._dropped = 0
        self._lock = threading.Lock()
        self.detail = f"schema {schema}, queue {queue_size}"

        self._thread = threading.Thread(
            target=self._run, name="rebound-telemetry", daemon=True
        )
        self._thread.start()
        # gunicorn's sync worker exits through sys.exit on SIGTERM, so atexit runs and
        # a deploy flushes instead of dropping whatever was still queued. Daemon=True
        # is the backstop for the ways a process leaves that do not.
        atexit.register(self.close)

    # ----------------------------------------------------------------- producing

    def record(self, prediction: Prediction) -> None:
        """Queue a row. Never blocks, never raises."""
        try:
            self._queue.put_nowait(prediction)
        except queue.Full:
            with self._lock:
                self._dropped += 1
            # One line per drop would itself become the outage, so this is deliberately
            # not logged here; the count is on /healthz.

    # ----------------------------------------------------------------- consuming

    def _connect(self):
        import psycopg

        connection = psycopg.connect(self._dsn, autocommit=True)
        with connection.cursor() as cursor:
            cursor.execute(DDL.format(schema=self._schema))
        return connection

    def _drain(self, first: Prediction) -> list[Prediction]:
        """`first` plus whatever else is already waiting, up to a batch."""
        batch = [first]
        while len(batch) < BATCH_SIZE:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._stop.set()
                break
            batch.append(item)
        return batch

    def _run(self) -> None:
        connection = None
        backoff = RETRY_SECONDS
        insert = INSERT.format(schema=self._schema)

        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=FLUSH_SECONDS)
            except queue.Empty:
                continue
            if item is None:  # the close() sentinel
                self._stop.set()
                continue

            batch = self._drain(item)

            if connection is None or connection.closed:
                try:
                    connection = self._connect()
                    backoff = RETRY_SECONDS
                    LOGGER.info("telemetry connected to %s", self._schema)
                except Exception as exc:  # noqa: BLE001 - every failure is the same one
                    connection = None
                    with self._lock:
                        self._dropped += len(batch)
                    self.detail = f"{type(exc).__name__}: {exc}"
                    LOGGER.warning("telemetry unavailable, dropping %d: %s", len(batch), exc)
                    # Sleep on the stop event so close() does not wait out the backoff.
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, MAX_RETRY_SECONDS)
                    continue

            try:
                with connection.cursor() as cursor:
                    cursor.executemany(insert, [p.row() for p in batch])
                with self._lock:
                    self._recorded += len(batch)
                self.detail = f"schema {self._schema}"
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._dropped += len(batch)
                self.detail = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("telemetry insert failed, dropping %d: %s", len(batch), exc)
                try:
                    connection.close()
                finally:
                    connection = None

        if connection is not None:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - shutting down either way
                pass

    # ------------------------------------------------------------------ lifecycle

    def close(self, timeout: float = 2.0) -> None:
        """Ask the writer to flush and stop. Safe to call twice; atexit may re-enter."""
        if self._stop.is_set() and not self._thread.is_alive():
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass  # the queue is full of rows; the event below is what stops the writer
        # Set as well as queued, because the sentinel alone only stops a writer that is
        # reading the queue. One that is sitting out a reconnect backoff is waiting on
        # this event, and a database that has been down for a while backs off further
        # than any sensible shutdown timeout -- so without this, closing during an
        # outage always waits the full `timeout` and then leaves the thread running.
        # The run loop keeps draining while the queue is non-empty, so a flush of what
        # is already queued still happens.
        self._stop.set()
        self._thread.join(timeout=timeout)

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"recorded": self._recorded, "dropped": self._dropped}

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped


def load(dsn: str, schema: str = "rebound", enabled: bool = True,
         queue_size: int = DEFAULT_QUEUE_SIZE) -> Telemetry | NoTelemetry:
    """Start the writer, or explain in one line why nothing is being recorded.

    Note what is *not* done here: no connection is opened. Reaching a shared database
    at start-up would make this app's availability depend on that server being up at
    exactly the moment systemd starts it, which is the coupling the whole module is
    written to avoid. The first row pays for the connection instead, and until then
    `/healthz` reports the writer as started rather than as verified.
    """
    if not enabled:
        return NoTelemetry("disabled by REBOUND_TELEMETRY=0")
    if not dsn:
        return NoTelemetry("no REBOUND_DATABASE_URL")

    if not schema.isidentifier():
        # The schema name is interpolated into DDL, so it is checked rather than
        # quoted-and-hoped. It comes from the environment, not from a request, but a
        # typo that becomes valid SQL is a worse failure than a refusal to log.
        # Checked before the driver, so a misconfiguration is reported as itself on a
        # host that also happens to be missing psycopg.
        return NoTelemetry(f"invalid schema name {schema!r}")

    try:
        import psycopg  # noqa: F401
    except ImportError as exc:
        return NoTelemetry(f"psycopg not installed: {exc}")

    try:
        return Telemetry(dsn, schema, queue_size=queue_size)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("could not start telemetry: %s", exc)
        return NoTelemetry(f"{type(exc).__name__} starting writer: {exc}")


def now() -> datetime:
    """Request time, not insert time -- the queue delay is not part of the answer."""
    return datetime.now(timezone.utc)


def monotonic() -> float:
    return time.perf_counter()
