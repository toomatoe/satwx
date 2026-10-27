"""Fetch a model-ready tile for (lat, lon, time) from GOES on AWS, and PNG <-> brightness-temperature helpers."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from .scene import load_scene, crop_at, model_input

# PNG convention (matches the usual IR enhancement: white = cold): 0 -> 330 K, 255 -> 170 K
PNG_BT_WARM, PNG_BT_COLD = 330.0, 170.0


def pick_satellite(lon: float) -> str:
    """GOES-East (GOES-19, 75.2W) or GOES-West (GOES-18, 137.2W) by longitude."""
    lon = (lon + 180) % 360 - 180
    return "goes18" if -180 <= lon < -106 or lon > 150 else "goes19"


def fetch_tile(lat: float, lon: float, t: dt.datetime, satellite: str | None = None,
               cache_dir: Path = Path("data/cache")) -> tuple[np.ndarray, dict]:
    sat = satellite or pick_satellite(lon)
    sc = load_scene(sat, t, cache_dir / "cli", with_labels=False)
    bt, r0, c0 = crop_at(sc, lat, lon)
    if not np.isfinite(bt).all():
        raise ValueError("tile touches space / missing pixels; choose a point further from the disk edge")
    return model_input(bt), dict(satellite=sat, scan_start=sc.time.isoformat(), row=r0, col=c0)


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
