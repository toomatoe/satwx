"""Download GOES-18/19 ABI IR bands from NOAA's AWS open-data bucket and save a model-ready tile.

    python scripts/fetch_goes.py --lat 25.5 --lon -80.2 --time 2025-09-26T18:00 --out tile.npy
    python scripts/fetch_goes.py --lat 25.5 --lon -80.2 --time 2025-09-26T18:00 --out tile.png   # C13 greyscale
"""
import argparse
import json
from pathlib import Path

import numpy as np

from satwx.cli import parse_time
from satwx.fetch import bt_to_png, fetch_tile

ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, required=True)
ap.add_argument("--lon", type=float, required=True)
ap.add_argument("--time", type=parse_time, required=True)
ap.add_argument("--satellite", choices=["goes18", "goes19"])
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--cache", type=Path, default=Path("data/cache"))
a = ap.parse_args()
x, meta = fetch_tile(a.lat, a.lon, a.time, a.satellite, a.cache)
if a.out.suffix == ".png":
    bt_to_png(x[2], a.out)
else:
    np.save(a.out, x)
print(json.dumps(dict(meta, out=str(a.out), shape=list(x.shape))))
