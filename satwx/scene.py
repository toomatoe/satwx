"""Load one GOES full-disk scene (IR bands + L2 cloud products + GLM) and cut it into tiles."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import goes

TILE = 128          # tile side in 2 km pixels (~256 km at nadir)
POOL = 2            # model input is TILE/POOL = 64 px (~4 km pixels)
MAX_ZENITH = 65.0   # deg; beyond this parallax/limb effects dominate


@dataclass
class Scene:
    satellite: str
    time: dt.datetime            # scan start
    bands: np.ndarray            # (4, 5424, 5424) brightness temperature, K
    grid: goes.FixedGrid
    cloud: np.ndarray | None = None    # ACM binary cloud mask (0 clear, 1 cloudy, nan)
    phase: np.ndarray | None = None    # ACTP phase (0 clear 1 liquid 2 SC-liquid 3 mixed 4 ice 5 unknown)
    height: np.ndarray | None = None   # ACHA cloud-top height, m (coarser grid)
    flashes: np.ndarray | None = None  # (N, 2) GLM flash lat, lon in the scan window


def load_scene(satellite: str, t: dt.datetime, cache_dir: Path, with_labels: bool = True,
               keep_files: bool = False) -> Scene:
    files = []

    def get(product, band=None):
        key = goes.find_key(satellite, product, t, band)
        p = goes.download(satellite, key, cache_dir)
        files.append(p)
        return key, p

    bands, grid, start = [], None, None
    for b in goes.BANDS:
        key, p = get("ABI-L2-CMIPF", b)
        data, grid = goes.read_var(p, "CMI")
        bands.append(data)
        start = goes.scan_start(key)
    scene = Scene(satellite, start, np.stack(bands), grid)

    if with_labels:
        scene.cloud = goes.read_var(get("ABI-L2-ACMF")[1], "BCM")[0]
        scene.phase = goes.read_var(get("ABI-L2-ACTPF")[1], "Phase")[0]
        scene.height = goes.read_var(get("ABI-L2-ACHAF")[1], "HT")[0]
        lats, lons = [], []
        for key in goes.keys_between(satellite, "GLM-L2-LCFA", start, start + dt.timedelta(minutes=10)):
            p = goes.download(satellite, key, cache_dir)
            files.append(p)
            la, lo = goes.read_glm_flashes(p)
            lats.append(la)
            lons.append(lo)
        scene.flashes = np.stack([np.concatenate(lats), np.concatenate(lons)], 1) if lats else np.zeros((0, 2))

    if not keep_files:
        for p in files:
            p.unlink(missing_ok=True)
    return scene


def tile_origins(scene: Scene, extra_centers: list[tuple[float, float]] = ()) -> list[tuple[int, int]]:
    """Regular non-overlapping tile grid plus tiles centred on given (lat, lon) points."""
    n = scene.bands.shape[-1]
    origins = [(r, c) for r in range(0, n - TILE + 1, TILE) for c in range(0, n - TILE + 1, TILE)]
    for lat, lon in extra_centers:
        r, c, vis = scene.grid.pixel(lat, lon)
        if vis:
            r0, c0 = int(round(float(r))) - TILE // 2, int(round(float(c))) - TILE // 2
            if 0 <= r0 <= n - TILE and 0 <= c0 <= n - TILE:
                origins.append((r0, c0))
    return origins


def tile_center(scene: Scene, r0: int, c0: int) -> tuple[float, float]:
    xc = scene.grid.x[c0 + TILE // 2]
    yc = scene.grid.y[r0 + TILE // 2]
    lat, lon = scene.grid.latlon(np.array([xc]), np.array([yc]))
    return float(lat[0]), float(lon[0])


def model_input(bt_tile: np.ndarray) -> np.ndarray:
    """(4, TILE, TILE) K -> (4, 64, 64) float32, mean-pooled."""
    c, h, w = bt_tile.shape
    return bt_tile.reshape(c, h // POOL, POOL, w // POOL, POOL).mean(axis=(2, 4)).astype(np.float32)


def crop_at(scene: Scene, lat: float, lon: float) -> tuple[np.ndarray, int, int]:
    r, c, vis = scene.grid.pixel(lat, lon)
    if not vis:
        raise ValueError(f"({lat}, {lon}) is not visible from {scene.satellite}")
    n = scene.bands.shape[-1]
    r0 = int(np.clip(round(float(r)) - TILE // 2, 0, n - TILE))
    c0 = int(np.clip(round(float(c)) - TILE // 2, 0, n - TILE))
    return scene.bands[:, r0:r0 + TILE, c0:c0 + TILE], r0, c0
