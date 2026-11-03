"""Score a trained checkpoint on every tile file in a directory (e.g. another satellite or period).

    python scripts/evaluate.py --tiles data/tiles_goes18 --model models/satwx_ir4.pt --out models/goes18_ir4.metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from satwx import goes, labels
from satwx.model import Classifier
from train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="write metrics JSON here")
    a = ap.parse_args()

    clf = Classifier.load(a.model)
    chans = [goes.BANDS.index(c) for c in clf.channels]
    X, y, extra = [], [], []
    files = sorted(a.tiles.glob("*.npz"))
    for f in files:
        d = np.load(f)
        names = list(d["feature_names"])
        X.append(d["X"][:, chans])
        y += [labels.CLASSES.index(labels.label(dict(zip(names, row)))) for row in d["features"]]
        extra.append(d["is_extra"])
    X, y, extra = np.concatenate(X), np.array(y), np.concatenate(extra)
    p = clf.calibrated(clf.logits(X))
    metrics = dict(model=a.model.name, tiles=str(a.tiles), scenes=len(files),
                   regular=evaluate(p[~extra], y[~extra]),
                   tc_centred_tiles=evaluate(p[extra], y[extra]) if extra.any() else None)
    m = metrics["regular"]
    print(f"{len(files)} scenes, {m['n']} tiles: acc {m['accuracy']}  macro-F1 {m['macro_f1']}  ECE {m['ece']}  "
          f"majority baseline {m['majority_baseline']}")
    if metrics["tc_centred_tiles"]:
        tc = metrics["tc_centred_tiles"]["per_class"]["tropical_cyclone"]
        print(f"storm-centred tiles: TC recall {tc['recall']} precision {tc['precision']} (n={tc['support']})")
    if a.out:
        a.out.write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
