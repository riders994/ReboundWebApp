"""
Rebound predictor backend — Python 3 port of the original 2017 webapp.py.

Given 10 player positions at the moment of a shot, predict each player's post-shot
position (Keras net) and their rebound probability (random forest).

Served behind nginx in production at /api/rebound/ -> gunicorn (127.0.0.1:8000).

Model files live in ./models/:
  - posnn.h5        Keras position model
  - msd.pkl         (mean, std) normalizer
  - FinalModel.pkl  scikit-learn RandomForest (retrain & drop in)

If FinalModel.pkl (or any model) is missing, the app runs in a model-free FALLBACK
mode that returns deterministic placeholder probabilities so the demo page still works.
"""

import os
import json
import pickle
import logging

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rebound")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
HOOP = np.array([41.65, 25.0])


# --------------------------------------------------------------------------- #
# Feature engineering (ported verbatim from the original, xrange -> range)
# --------------------------------------------------------------------------- #
def features(df):
    """Prepare per-player features: distance/angle to hoop, cosine similarity to
    the shooter, and a k-means-style box-out count."""
    df.sort_values(by="Off", inplace=True)
    numcols = ["x", "y"]

    nums = df[numcols].values
    diff = nums - HOOP
    df["HDist"] = np.sqrt((diff ** 2).sum(axis=1))
    angle = np.arctan2(diff[:, 0], diff[:, 1])
    df["Angle"] = angle
    df["CosSim"] = np.cos(angle - angle[df["isShoot"] == True])
    df["Box"] = boxgen(nums)
    df.sort_index(inplace=True)
    return df


def boxgen(arr):
    """One iteration of k-means-style assignment to estimate who is boxing out whom."""
    bdist = np.sqrt(((arr - HOOP) ** 2).sum(axis=1)).reshape(10, 1)
    arr = np.concatenate((arr, bdist), axis=1)
    arr1 = arr[:5, :]
    arr2 = arr[5:, :]
    dists = np.array([np.sqrt(((arr1[:, :2] - row) ** 2).sum(axis=1)) for row in arr2[:, :2]])
    d = np.argmin(dists, axis=1)
    o = np.argmin(dists, axis=0)
    dbox = [np.sum(o == i) for i in range(5)]
    obox = [np.sum(d == i) for i in range(5)]
    return np.array(dbox + obox)


class inputDecode(object):
    def __init__(self, posModel, norm):
        self.posModel = posModel
        self.normer = norm

    def CreatePre(self, df):
        self.pre = features(df)
        self.preArr = self.pre[["x", "y"]]

    def CreatePos(self):
        vals = (self.pre.values - self.normer[0]) / self.normer[1]
        self.posArr = self.posModel.predict(vals, batch_size=40)
        self.pos = self.pre.copy()
        self.pos["x"] = self.posArr[:, 0]
        self.pos["y"] = self.posArr[:, 1]
        self.pos["newy"] = self.posArr[:, 0]
        self.pos["newx"] = self.posArr[:, 1]
        self.pos = features(self.pos)
        move = self.posArr - self.preArr
        closer = ((self.pre["HDist"] < self.pos["HDist"]).astype(int) - 0.5) * 2
        self.pos.pop("Off")
        self.pos.pop("isShoot")
        self.pos["MoveV"] = np.sqrt((move ** 2).sum(axis=1)) * closer

    def Modeling(self, fitModel):
        self.modIn = np.concatenate(
            [self.pre.values,
             self.pos[["x", "y", "HDist", "Angle", "CosSim", "Box", "MoveV"]].values],
            axis=1,
        )
        probs = fitModel.predict_proba(self.modIn)
        p = probs[:, 1]
        self.pos["probability"] = p / p.sum()
        return self.pos[["newx", "newy", "probability"]]


# --------------------------------------------------------------------------- #
# Model loading (graceful — falls back if anything is missing)
# --------------------------------------------------------------------------- #
def _load_models():
    posnn = norms = final = None
    try:
        # msd.pkl was pickled under Python 2 — latin1 lets Py3 unpickle its numpy arrays.
        with open(os.path.join(MODEL_DIR, "msd.pkl"), "rb") as f:
            norms = pickle.load(f, encoding="latin1")
    except Exception as e:
        log.warning("could not load msd.pkl: %s", e)
    try:
        from tensorflow.keras.models import load_model
        posnn = load_model(os.path.join(MODEL_DIR, "posnn.h5"), compile=False)
    except Exception as e:
        log.warning("could not load posnn.h5: %s", e)
    try:
        import joblib
        final = joblib.load(os.path.join(MODEL_DIR, "FinalModel.pkl"))
    except Exception as e:
        log.warning("could not load FinalModel.pkl: %s", e)

    ready = posnn is not None and norms is not None and final is not None
    if not ready:
        log.warning("running in FALLBACK mode (one or more models unavailable)")
    return posnn, norms, final, ready


POSNN, NORMS, FINAL, MODELS_READY = _load_models()


def _build_frame(bench):
    """Reproduce the ORIGINAL column mapping deterministically.

    The 2017 app ran on old pandas, where DataFrame(list_of_dicts) ordered columns
    ALPHABETICALLY: [isOffense, isShooter, x, y]. It then renamed positionally to
    ['Off', 'isShoot', 'y', 'x'] — i.e. isOffense->Off, isShooter->isShoot, and x/y
    swap to the model's coordinate convention. Modern pandas preserves dict order,
    so we sort the columns explicitly to keep the model's expected feature order.
    """
    df = pd.DataFrame(bench)
    df = df[sorted(df.columns)]          # -> isOffense, isShooter, x, y
    df.columns = ["Off", "isShoot", "y", "x"]
    df["Off"] = df["Off"].astype(int)
    df["isShoot"] = df["isShoot"].astype(int)
    return df


def _predict_real(bench):
    df = _build_frame(bench)
    script = inputDecode(posModel=POSNN, norm=NORMS)
    script.CreatePre(df)
    script.CreatePos()
    res = script.Modeling(fitModel=FINAL).sort_index()
    return [res.loc[i].to_dict() for i in res.index]


def _predict_fallback(bench):
    """Model-free placeholder: no movement, probability inversely proportional to
    distance from the hoop. Keeps the demo interactive without any model files."""
    out = []
    dists = []
    for p in bench:
        x, y = float(p["x"]), float(p["y"])
        dists.append(np.hypot(x - HOOP[0], y - HOOP[1]))
    inv = [1.0 / (d + 1.0) for d in dists]
    total = sum(inv) or 1.0
    for p, w in zip(bench, inv):
        x, y = float(p["x"]), float(p["y"])
        # nudge each player 15% closer to the rim so the animation still moves
        out.append({
            "newx": x + 0.15 * (HOOP[0] - x),
            "newy": y + 0.15 * (HOOP[1] - y),
            "probability": w / total,
        })
    return out


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_ready": MODELS_READY})


@app.route("/predict", methods=["POST"])
def predict():
    # Accept a clean JSON body; fall back to the legacy form-encoded shape.
    data = request.get_json(silent=True)
    if data is None:
        form = request.form.to_dict()
        if form:
            data = json.loads(next(iter(form)))
    if not data or "bench" not in data:
        return jsonify({"error": "expected JSON body with a 'bench' array"}), 400

    bench = data["bench"]
    if len(bench) != 10:
        return jsonify({"error": "expected exactly 10 players"}), 400

    try:
        result = _predict_real(bench) if MODELS_READY else _predict_fallback(bench)
    except Exception as e:
        log.exception("prediction failed")
        return jsonify({"error": str(e)}), 500
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="127.0.0.1", port=port)
