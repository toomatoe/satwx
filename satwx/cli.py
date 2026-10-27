"""classify: weather-pattern probabilities for a GOES tile.

    classify --lat 25.5 --lon -80.2 --time 2025-09-26T18:00
    classify tile.npy           # (4, 64, 64) C07/C08/C13/C15 brightness temperatures in K
    classify tile.png           # 8-bit greyscale 10.3um IR, white = cold (0 -> 330 K, 255 -> 170 K)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(os.environ.get("SATWX_MODELS", Path(__file__).resolve().parent.parent / "models"))


def parse_time(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t.astimezone(dt.timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def main(argv=None):
    ap = argparse.ArgumentParser(prog="classify", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", type=Path, help=".png (IR greyscale) or .npy tile")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--time", type=parse_time, help="UTC, ISO 8601 (e.g. 2025-09-26T18:00)")
    ap.add_argument("--satellite", choices=["goes18", "goes19"], help="default: chosen from --lon")
    ap.add_argument("--model", type=Path, help="checkpoint (default: models/satwx_ir4.pt, or satwx_ir1.pt for PNG)")
    ap.add_argument("--save-tile", type=Path, help="with --lat/--lon/--time: also write the fetched tile (.npy or .png)")
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("SATWX_CACHE", "data/cache")))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from .fetch import bt_to_png, fetch_tile, png_to_bt
    from .model import Classifier

    meta = {}
    if a.image:
        if a.image.suffix.lower() == ".npy":
            x = np.load(a.image).astype(np.float32)
        else:
            x = png_to_bt(a.image)
        meta["input"] = str(a.image)
    elif None not in (a.lat, a.lon, a.time):
        x, meta = fetch_tile(a.lat, a.lon, a.time, a.satellite, a.cache)
        meta.update(lat=a.lat, lon=a.lon, requested_time=a.time.isoformat())
        if a.save_tile:
            np.save(a.save_tile, x) if a.save_tile.suffix == ".npy" else bt_to_png(x[2], a.save_tile)
    else:
        ap.error("give an image path, or all of --lat --lon --time")

    model_path = a.model or MODEL_DIR / ("satwx_ir1.pt" if x.shape[0] == 1 else "satwx_ir4.pt")
    clf = Classifier.load(model_path)
    if x.shape[0] != len(clf.channels):
        if x.shape[0] == 4 and clf.channels == ["C13"]:
            x = x[2:3]
        else:
            ap.error(f"input has {x.shape[0]} channels, model {model_path.name} expects {clf.channels}")
    p = clf.predict_proba(x[None])[0]
    order = np.argsort(-p)
    result = dict(meta, model=model_path.name, prediction=clf.classes[order[0]],
                  probabilities={clf.classes[i]: round(float(p[i]), 4) for i in order})
    if a.json:
        json.dump(result, sys.stdout, indent=1)
        print()
    else:
        for k, v in meta.items():
            print(f"{k:>15}: {v}")
        print(f"{'model':>15}: {model_path.name}")
        for i in order:
            print(f"{clf.classes[i]:>17} {p[i]:6.3f} {'#' * int(round(p[i] * 40))}")


if __name__ == "__main__":
    main()
