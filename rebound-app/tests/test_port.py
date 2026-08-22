"""Tests for the ported app.

Two things carry the risk and neither is about Flask. The **coordinate frame** decides
whether anything drawn on the court is in the right place, and it is a swap that looks
like a bug (see `coordinates.py`), so it is pinned in both directions. The **wire
format** has to keep working with a `script.js` nobody is rewriting today: an array in
bench order, with `newx`, `newy` and `probability` on every entry.
"""

import json

import numpy as np
import pytest
from coordinates import (
    CANVAS_HEIGHT_FT,
    CANVAS_WIDTH_FT,
    canvas_to_model,
    model_to_canvas,
)
from movement import DEFAULT_SCENES, NoMovement
from movement import load as load_movement

from rebounding.constants import HOOP

N_PLAYERS = 10


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #


def test_canvas_axes_map_to_the_model_axes_that_match_their_range():
    """500 px / 10 is the 50 ft width; 470 px / 10 is the 47 ft half-court length."""
    assert CANVAS_WIDTH_FT == 50
    assert CANVAS_HEIGHT_FT == 47
    model_x, model_y = canvas_to_model(np.array([0.0, 50.0]), np.array([0.0, 47.0]))
    assert model_x.max() <= CANVAS_HEIGHT_FT
    assert model_y.max() <= CANVAS_WIDTH_FT


def test_round_trip_is_the_identity():
    rng = np.random.default_rng(0)
    cx = rng.uniform(0, CANVAS_WIDTH_FT, 50)
    cy = rng.uniform(0, CANVAS_HEIGHT_FT, 50)
    back_x, back_y = model_to_canvas(*canvas_to_model(cx, cy))
    np.testing.assert_allclose(back_x, cx)
    np.testing.assert_allclose(back_y, cy)


def test_the_swap_is_a_swap_not_an_identity():
    """If these ever compare equal the conversion has quietly become a no-op."""
    model_x, model_y = canvas_to_model(10.0, 40.0)
    assert (float(model_x), float(model_y)) == (40.0, 10.0)


def test_the_rim_lands_at_the_bottom_of_the_canvas():
    """Confirmed by Rohan: the canvas is the basket half, basket at the bottom.

    SVG y grows downward, so the basket end must come out at large `cy`. If someone
    sets FLIP_LENGTH this fails, which is the point -- it is the assumption every drawn
    position depends on.
    """
    cx, cy = model_to_canvas(*HOOP)
    assert (float(cx), float(cy)) == (25.0, 41.75)
    assert cy > CANVAS_HEIGHT_FT / 2, "rim should be in the lower half of the canvas"


def test_half_court_is_the_top_of_the_canvas():
    """The other end of the same assumption: model x = 0 is half court, so cy = 0."""
    _, cy_halfcourt = model_to_canvas(0.0, 25.0)
    _, cy_baseline = model_to_canvas(47.0, 25.0)
    assert float(cy_halfcourt) == 0.0
    assert float(cy_baseline) == CANVAS_HEIGHT_FT


def test_a_player_nearer_the_bottom_is_nearer_the_rim():
    """Reads the way the user sees it: down the screen is toward the basket."""
    near_x, near_y = canvas_to_model(25.0, 40.0)
    far_x, far_y = canvas_to_model(25.0, 10.0)
    near = np.hypot(near_x - HOOP[0], near_y - HOOP[1])
    far = np.hypot(far_x - HOOP[0], far_y - HOOP[1])
    assert near < far


# --------------------------------------------------------------------------- #
# The movement model
# --------------------------------------------------------------------------- #


def test_no_movement_leaves_players_where_they_were():
    """The null model, which is what a missing bundle degrades to."""
    class Placed:
        def __init__(self, x, y):
            self.x, self.y = x, y

    players = [Placed(float(i), float(i + 10)) for i in range(N_PLAYERS)]
    scenes = NoMovement().scenes(players, n=3)
    assert scenes.shape == (3, N_PLAYERS, 2)
    for scene in scenes:
        np.testing.assert_allclose(scene[:, 0], [p.x for p in players])
        np.testing.assert_allclose(scene[:, 1], [p.y for p in players])


def test_a_missing_bundle_degrades_instead_of_raising(tmp_path):
    """The rebounder does not need this model, so its absence must not stop the app."""
    mover = load_movement(tmp_path / "nothing-here.pkl")
    assert isinstance(mover, NoMovement)
    assert "no bundle at" in mover.detail


def test_movement_can_be_switched_off_by_configuration(tmp_path):
    mover = load_movement(tmp_path / "nothing-here.pkl", enabled=False)
    assert mover.name == "none"
    assert "REBOUND_MOVEMENT=0" in mover.detail


def test_a_corrupt_bundle_degrades_with_the_reason_in_healthz(tmp_path):
    """Any failure to load is the same failure: no animation, app still up."""
    broken = tmp_path / "MovementModel.pkl"
    broken.write_bytes(b"not a pickle")
    mover = load_movement(broken)
    assert isinstance(mover, NoMovement)
    assert "MovementModel.pkl" in mover.detail


def test_default_scene_count_is_more_than_one():
    """Sampling one future and drawing it is fine; sampling one and meaning it is not."""
    assert DEFAULT_SCENES > 1


# --------------------------------------------------------------------------- #
# The endpoint
# --------------------------------------------------------------------------- #


def test_predict_returns_one_entry_per_player_in_bench_order(client, bench):
    response = client.post("/predict", json={"bench": bench})
    assert response.status_code == 200

    body = response.get_json()
    assert isinstance(body, list)
    assert len(body) == N_PLAYERS
    for entry in body:
        assert set(entry) == {"newx", "newy", "probability", "scenes"}


def test_probabilities_sum_to_one_without_renormalising(client, bench):
    body = client.post("/predict", json={"bench": bench}).get_json()
    assert sum(e["probability"] for e in body) == pytest.approx(1.0)


def test_predicted_positions_are_returned_in_canvas_feet(client, bench):
    """With movement off they are the placements, so they must come back unchanged."""
    body = client.post("/predict", json={"bench": bench}).get_json()
    for entry, placed in zip(body, bench, strict=True):
        assert entry["newx"] == pytest.approx(placed["x"])
        assert entry["newy"] == pytest.approx(placed["y"])


def test_every_sampled_scene_comes_back_to_the_front_end(client, bench):
    """The spread is the generative model's whole advantage; it has to reach the wire."""
    body = client.post("/predict", json={"bench": bench}).get_json()
    for entry in body:
        assert len(entry["scenes"]) >= 1
        assert entry["scenes"][0] == [pytest.approx(entry["newx"]), pytest.approx(entry["newy"])]


def test_reordering_the_bench_follows_the_players(client, bench):
    """The response is indexed by bench position, which is what script.js assumes."""
    straight = client.post("/predict", json={"bench": bench}).get_json()
    order = [7, 2, 9, 0, 4, 1, 8, 3, 6, 5]
    shuffled = client.post("/predict", json={"bench": [bench[i] for i in order]}).get_json()
    for new_index, old_index in enumerate(order):
        assert shuffled[new_index]["probability"] == pytest.approx(
            straight[old_index]["probability"]
        )


def test_accepts_the_legacy_form_encoded_blob(client, bench):
    """`script.js` posts JSON with an urlencoded Content-type; that still works."""
    response = client.post(
        "/predict",
        data=json.dumps({"bench": bench}),
        content_type="application/x-www-form-urlencoded",
    )
    assert response.status_code == 200
    assert len(response.get_json()) == N_PLAYERS


def test_accepts_snake_case_keys_too(client, bench):
    payload = [
        {"x": p["x"], "y": p["y"], "is_offense": p["isOffense"], "is_shooter": p["isShooter"]}
        for p in bench
    ]
    assert client.post("/predict", json={"bench": payload}).status_code == 200


def test_positions_are_passed_through_when_supplied(client, bench):
    for player, position in zip(bench, ["C", "F", "F", "G", "G"] * 2, strict=True):
        player["position"] = position
    assert client.post("/predict", json={"bench": bench}).status_code == 200


# --------------------------------------------------------------------------- #
# Failure, which used to be a stack trace or a confident wrong answer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda b: b[:9], "expected 10 players"),
        (lambda b: [{**p, "isOffense": True} for p in b], "5 offensive players"),
        (lambda b: [{**p, "isShooter": False} for p in b], "exactly 1 shooter"),
        (lambda b: [{**p, "isShooter": True} for p in b], "exactly 1 shooter"),
    ],
)
def test_bad_placements_are_400_with_a_reason(client, bench, mutate, expected):
    response = client.post("/predict", json={"bench": mutate(bench)})
    assert response.status_code == 400
    assert expected in response.get_json()["error"]


def test_missing_bench_is_400(client):
    response = client.post("/predict", json={"nope": []})
    assert response.status_code == 400


def test_garbage_body_is_400_not_500(client):
    response = client.post("/predict", data="not json at all", content_type="text/plain")
    assert response.status_code == 400


def test_non_numeric_coordinates_are_400(client, bench):
    bench[3]["x"] = "left a bit"
    response = client.post("/predict", json={"bench": bench})
    assert response.status_code == 400
    assert "numeric" in response.get_json()["error"]


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


def test_healthz_reports_which_model_is_serving(client):
    """The weights arrive by scp and never appear in git; this is how you identify one."""
    body = client.get("/healthz").get_json()
    assert body["status"] == "ok"
    assert body["model"]["features"] == 27
    assert body["model"]["commit"]
    assert 0.0 < body["model"]["test_top1"] < 1.0
    assert body["movement_model"] == "none"


def test_the_api_serves_no_static_files(client):
    """This app is JSON only; nginx serves the portfolio from `site/`.

    The port shipped a `static_folder="public"` for the 2017 single-app layout, where
    Flask served the court SVG and `script.js` itself. Here `deploy/nginx.conf` serves
    `site/` directly and proxies only `/api/rebound/` to gunicorn, so that wiring was
    removed rather than adapted. Asserting a 404 on `/public/...` would pass either
    way -- with the static folder gone there is no route to hit -- so this pins the
    absence itself: no `static` endpoint on the URL map.
    """
    endpoints = {rule.endpoint for rule in client.application.url_map.iter_rules()}
    assert "static" not in endpoints
    assert endpoints == {"healthz", "predict_route"}
