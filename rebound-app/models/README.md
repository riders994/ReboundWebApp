# Model bundles

The weights live here. They are **gitignored and never committed** — this file exists so
the directory survives a clone, because git does not track empty directories and every
real file in here is ignored. Without it, `scp`-ing two bundles into
`/opt/rebound-app/models/` on a fresh box fails: the destination is not a directory.

| file | what it does | if missing |
|---|---|---|
| `FinalModel.pkl` | rebound probabilities — LightGBM, grouped softmax over the ten players | app refuses to start |
| `MovementModel.pkl` | where players move — torch CVAE; replaces the 2017 `posnn.h5` + `msd.pkl` | fails soft: probabilities unaffected, players stay put |

Build them in the **ReboundingPrediction** repo:

```bash
python -m rebounding.cli train            # -> FinalModel.pkl
python -m rebounding.cli train-movement   # -> MovementModel.pkl
```

Both are `joblib.dump` of a dataclass bundle, not bare estimators, so **unpickling imports
the `rebounding` package** — it is a hard serving dependency, installed with the `[serve]`
extra. Each bundle carries its own provenance (build commit, corpus SHA, library versions,
test scores), which is what `/healthz` reports and what makes a copy-in workflow auditable.

`SourceOfTruth.pkl` from that repo is **not** deployable here — it needs rim-time and
velocity features that do not exist when a visitor is placing dots, and `serve.predict`
refuses it.

See `DEPLOY.md` → "The model files".
