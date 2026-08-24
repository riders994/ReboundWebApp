"""The rebound app, on Python 3.

A rewrite of the 2017 `webapp.py` rather than a port of it. The feature code is gone --
`rebounding.serve` computes every feature now, so the app and the model cannot disagree
about what `pre_box` means. What is left is HTTP: parse a payload, convert coordinates,
call the model, convert back.

Changes worth knowing about, beyond Python 3 and current Flask:

* **`features()` and `boxgen()` are deleted.** `rebounding.serve.predict` replaces them
  and is tested against the pipeline's own output. The eight lines they occupied here
  were the source of most of §3 of the handoff brief.
* **Probabilities are not renormalised.** The model is a grouped softmax over the ten
  players, so they already sum to 1. The old `p / p.sum()` was fixing up a per-row
  random forest that had never been told only one player rebounds.
* **No static or template serving at all.** In this repo the portfolio is a separate
  static site under `site/`, served directly by nginx; this app is a pure JSON API
  proxied at `/api/rebound/` (see `deploy/nginx.conf`). The port's `static_folder`
  wiring was dropped rather than adapted.
* **Bad placements return 400 with a reason**, instead of a stack trace or a confident
  answer computed from nine players.
* **`/healthz`** reports which model is loaded, from the bundle's own provenance. Since
  the weights are deployed by scp and never appear in git, that endpoint is the only
  way to ask a running host which model it is serving.

The wire format is backward compatible, so the existing `site/assets/js/rebound.js`
works untouched: POST to `/predict`, and get back a JSON array of
`{newx, newy, probability, scenes}` in the same order the players were sent. `scenes`
is new -- the movement model now samples several futures per request and all of them
come back, so a front end can fan them out as ghosts instead of drawing one confident
dot. `newx`/`newy` are the first of those samples rather than their average, which
matters: averaging them puts each player at the midpoint of two futures he never
takes. `rebound.js` already posts a proper `application/json` body; the 2017
form-encoded blob with a lying `Content-type` is still accepted so the legacy front end
keeps working.
"""

from __future__ import annotations

import json
import logging

import movement as movement_module
import numpy as np
from config import Config
from coordinates import canvas_to_model, model_to_canvas
from flask import Flask, jsonify, request

from rebounding.models.artifact import load as load_artifact
from rebounding.serve import PlacementError, Player, predict

LOGGER = logging.getLogger(__name__)

N_PLAYERS = 10


def _payload_from_request() -> dict:
    """Accept a proper JSON body, or the form-encoded blob the 2017 front end sends."""
    body = request.get_json(silent=True)
    if isinstance(body, dict):
        return body

    # `script.js` sends JSON.stringify(payload) with an urlencoded Content-type, so the
    # whole document arrives as a single form key. The original did `d.keys()[0]`,
    # which is also why it broke on Python 3.
    for key in request.form:
        try:
            parsed = json.loads(key)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raw = request.get_data(as_text=True).strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise PlacementError(f"body is not JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed
    raise PlacementError("expected a JSON object with a 'bench' list of ten players")


def _players_from(bench: list[dict]) -> tuple[list[Player], np.ndarray, np.ndarray]:
    """Canvas-frame bench entries to model-frame players, in the order they arrived."""
    if not isinstance(bench, list):
        raise PlacementError("'bench' must be a list")
    if len(bench) != N_PLAYERS:
        raise PlacementError(f"expected {N_PLAYERS} players, got {len(bench)}")

    try:
        canvas_x = np.array([float(entry["x"]) for entry in bench])
        canvas_y = np.array([float(entry["y"]) for entry in bench])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlacementError(f"every player needs numeric x and y: {exc}") from exc

    model_x, model_y = canvas_to_model(canvas_x, canvas_y)
    players = []
    for index, entry in enumerate(bench):
        # The front end says isOffense/isShooter; serve.Player says is_offense.
        offense = entry.get("isOffense", entry.get("is_offense"))
        shooter = entry.get("isShooter", entry.get("is_shooter", False))
        if offense is None:
            raise PlacementError(f"player {index} is missing isOffense")
        players.append(
            Player(
                x=float(model_x[index]),
                y=float(model_y[index]),
                is_offense=bool(offense),
                is_shooter=bool(shooter),
                position=entry.get("position"),
                player_id=entry.get("player_id"),
            )
        )
    return players, model_x, model_y


def create_app(config: Config | None = None) -> Flask:
    config = config or Config()
    # `static_folder=None` removes Flask's default `/static` route as well. This app
    # is JSON only; nginx serves the portfolio from `site/`, so an unused static
    # handler here is just surface area.
    app = Flask(__name__, static_folder=None)
    app.config["REBOUND"] = config

    artifact = None
    if config.model_path.exists():
        artifact = load_artifact(config.model_path)
        LOGGER.info("loaded %s", config.model_path)
    elif config.require_model:
        raise FileNotFoundError(
            f"no model at {config.model_path}. Build it in the ReboundingPrediction repo "
            "with `python -m rebounding.cli train` and scp it here, or set "
            "REBOUND_REQUIRE_MODEL=0 to start without one."
        )
    else:
        LOGGER.warning("starting without a model; /predict will return 503")

    mover = movement_module.load(config.movement_model_path, enabled=config.movement_enabled)
    app.extensions["rebound_artifact"] = artifact
    app.extensions["rebound_movement"] = mover

    if config.cors_origins:
        from flask_cors import CORS

        CORS(app, resources={r"/predict": {"origins": list(config.cors_origins)}})

    @app.get("/healthz")
    def healthz():
        payload = {
            "status": "ok" if artifact is not None else "no model",
            "movement_model": mover.name,
            "movement_detail": mover.detail,
        }
        if artifact is not None:
            meta = artifact.metadata
            payload["model"] = {
                "features": len(artifact.features),
                "regime": artifact.regime,
                "built": meta.get("created_utc"),
                "commit": (meta.get("git") or {}).get("commit"),
                "fit_on": meta.get("fit_on"),
                "test_top1": (meta.get("scores", {}).get("test") or {}).get("top1"),
            }
        return jsonify(payload), (200 if artifact is not None else 503)

    @app.post("/predict")
    def predict_route():
        if artifact is None:
            return jsonify({"error": "no model loaded"}), 503

        payload = _payload_from_request()
        players, _, _ = _players_from(payload.get("bench"))
        # An artifact fitted on the served+movement regime needs the movement model to
        # build its last ten columns, so hand it over whenever there is one.
        prediction = predict(artifact, players, movement=mover.artifact)

        scenes = mover.scenes(players, n=config.movement_scenes, seed=config.movement_seed)
        canvas_scenes = [model_to_canvas(scene[:, 0], scene[:, 1]) for scene in scenes]

        # `newx`/`newy` are **one draw**, not the average of the draws. Meaning them
        # together would put every player at the midpoint of futures he never takes,
        # which is the drift the retrained model exists to stop; see movement.py.
        first_cx, first_cy = canvas_scenes[0]

        # Same order the bench arrived in, which is what script.js indexes by.
        return jsonify(
            [
                {
                    "newx": float(first_cx[i]),
                    "newy": float(first_cy[i]),
                    "probability": float(prediction.probabilities[i]),
                    # Every sampled future for this player, for a front end that wants
                    # to draw the spread rather than one confident dot.
                    "scenes": [
                        [float(cx[i]), float(cy[i])] for cx, cy in canvas_scenes
                    ],
                }
                for i in range(N_PLAYERS)
            ]
        )

    @app.errorhandler(PlacementError)
    def placement_error(exc: PlacementError):
        return jsonify({"error": str(exc)}), 400

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    local = create_app()
    # Development only. Production runs under gunicorn; see gunicorn.conf.py.
    local.run(host="127.0.0.1", port=local.config["REBOUND"].port, debug=True)
