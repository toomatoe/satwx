"""Fetch a model-ready tile for (lat, lon, time) from GOES on AWS, and PNG <-> brightness-temperature helpers."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from . import goes
from .scene import MAX_ZENITH, TILE, model_input

# PNG convention (matches the usual IR enhancement: white = cold): 0 -> 330 K, 255 -> 170 K
PNG_BT_WARM, PNG_BT_COLD = 330.0, 170.0


def pick_satellite(lon: float) -> str:
    """GOES-East (GOES-19, 75.2W) or GOES-West (GOES-18, 137.2W) by longitude."""
    lon = (lon + 180) % 360 - 180
    return "goes18" if -180 <= lon < -106 or lon > 150 else "goes19"


def fetch_tile(lat: float, lon: float, t: dt.datetime, satellite: str | None = None) -> tuple[np.ndarray, dict]:
    """Tile centred on (lat, lon) from the full-disk scan nearest t. Reads only the needed byte ranges
    (~2 MB per band) instead of downloading whole files."""
    sat = satellite or pick_satellite(lon)
    ref = goes.find_key(sat, "ABI-L2-CMIPF", t, "C13")
    keys = [goes.find_key(sat, "ABI-L2-CMIPF", t, b, same_scan_as=ref) for b in goes.BANDS]
    bands = []
    for key in keys:
        with goes.open_remote(sat, key) as h:
            if not bands:
                grid = goes.grid_from_h5(h)
                r, c, vis = grid.pixel(lat, lon)
                if not vis:
                    raise ValueError(f"({lat}, {lon}) is not visible from {sat}")
                r0, c0 = int(round(float(r))) - TILE // 2, int(round(float(c))) - TILE // 2
                n = len(grid.x)
                if not (0 <= r0 <= n - TILE and 0 <= c0 <= n - TILE):
                    raise ValueError(f"({lat}, {lon}) is too close to the edge of the {sat} full disk")
            bands.append(goes.read_window(h, "CMI", r0, c0, TILE))
    bt = np.stack(bands)
    if not np.isfinite(bt).all():
        raise ValueError("tile touches space or missing pixels; choose a point further from the disk edge")
    zen = float(goes.satellite_zenith(lat, lon, grid.lon0))
    meta = dict(satellite=sat, scan_start=goes.scan_start(keys[0]).isoformat(), row=r0, col=c0,
                satellite_zenith=round(zen, 1))
    if zen > MAX_ZENITH:
        meta["warning"] = f"satellite zenith {zen:.0f} deg is outside the training range (< {MAX_ZENITH:.0f} deg)"
    return model_input(bt), meta


def bt_to_png(bt: np.ndarray, path: Path):
    from PIL import Image

    v = (PNG_BT_WARM - np.asarray(bt, float)) / (PNG_BT_WARM - PNG_BT_COLD) * 255
    Image.fromarray(np.clip(v, 0, 255).round().astype(np.uint8), mode="L").save(path)


def png_to_bt(path: Path, size: int = 64) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    v = np.asarray(img, np.float32)
    return (PNG_BT_WARM - v / 255 * (PNG_BT_WARM - PNG_BT_COLD))[None]
