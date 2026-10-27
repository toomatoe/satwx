"""Build a weakly labelled tile dataset from GOES-19 full-disk scenes.

One .npz per scene in --out, containing model inputs (N, 4, 64, 64) float16, labels, rule features and
tile centres. Raw NetCDF files are deleted after each scene.

    python scripts/build_dataset.py --out data/tiles --start 2025-06-01 --end 2026-05-31 --every-days 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from satwx import ancillary, goes, labels
from satwx.scene import MAX_ZENITH, TILE, load_scene, model_input, tile_center, tile_origins

TC_JITTER_PX = (-40, 0, 40)


def scene_times(start: dt.date, end: dt.date, every: int) -> list[dt.datetime]:
    out, d, i = [], start, 0
    while d <= end:
        out.append(dt.datetime(d.year, d.month, d.day, (i % 4) * 6))  # cycle 00/06/12/18 UTC
        d += dt.timedelta(days=every)
        i += 1
    return out


def visible_pixels(grid, latlon: np.ndarray):
    if len(latlon) == 0:
        return np.array([]), np.array([])
    r, c, vis = grid.pixel(latlon[:, 0], latlon[:, 1])
    vis = vis & np.isfinite(r) & np.isfinite(c)
    return r[vis], c[vis]


def count_in_tiles(rows, cols, origins):
    o = np.asarray(origins)
    if len(rows) == 0:
        return np.zeros(len(o), int)
    r0, c0 = o[:, :1], o[:, 1:]
    return ((rows >= r0) & (rows < r0 + TILE) & (cols >= c0) & (cols < c0 + TILE)).sum(1)


def tc_scene_times(tracks, start: dt.date, end: dt.date, every: int, lon0: float = -75.0) -> list[dt.datetime]:
    """Times (03/15 UTC, so never equal to a regular scene) with a tropical storm well inside the disk."""
    out, d, i = [], start, 0
    while d <= end:
        t = dt.datetime(d.year, d.month, d.day, 3 if i % 2 else 15)
        if any(goes.satellite_zenith(tc["lat"], tc["lon"], lon0) < MAX_ZENITH - 5
               for tc in ancillary.cyclones_at(tracks, t)):
            out.append(t)
            i += 1
        d += dt.timedelta(days=every)
    return out


def build_one(satellite: str, t: dt.datetime, out_dir: Path, cache: Path, tc_only: bool = False) -> str:
    out = out_dir / f"{satellite}_{t:%Y%m%dT%H%M}{'_tc' if tc_only else ''}.npz"
    if out.exists():
        return f"{out.name} exists"
    sc = load_scene(satellite, t, cache / f"{t:%Y%m%dT%H}")
    tracks = ancillary.load_ibtracs(cache)
    tcs = ancillary.cyclones_at(tracks, sc.time)
    fronts = ancillary.fronts_at(sc.time, cache) or []

    extra = []
    for tc in tcs:
        r, c, vis = sc.grid.pixel(tc["lat"], tc["lon"])
        if not vis:
            continue
        for dr in TC_JITTER_PX:
            for dc in TC_JITTER_PX:
                if dr or dc:
                    la, lo = sc.grid.latlon(np.interp(float(c) + dc, np.arange(len(sc.grid.x)), sc.grid.x),
                                            np.interp(float(r) + dr, np.arange(len(sc.grid.y)), sc.grid.y))
                    extra.append((float(la), float(lo)))
                else:
                    extra.append((tc["lat"], tc["lon"]))
    origins = tile_origins(sc, extra)
    if tc_only:  # only the storm-centred tiles; the regular grid is covered by the regular scenes
        origins = origins[len(origins) - len(extra):]
        if not origins:
            return f"{out.name}: no visible TC"

    flash_counts = count_in_tiles(*visible_pixels(sc.grid, sc.flashes), origins)
    pts = [ancillary.densify(line) for _, line in fronts]
    front_counts = count_in_tiles(*visible_pixels(sc.grid, np.concatenate(pts) if pts else np.zeros((0, 2))), origins)

    hf = sc.height.shape[0] / sc.bands.shape[-1]
    X, y, feats, centers, is_extra = [], [], [], [], []
    n_regular = len(origins) - len(extra)
    for i, (r0, c0) in enumerate(origins):
        bt = sc.bands[:, r0:r0 + TILE, c0:c0 + TILE]
        if not np.isfinite(bt).all():
            continue
        lat, lon = tile_center(sc, r0, c0)
        if not np.isfinite(lat) or goes.satellite_zenith(lat, lon, sc.grid.lon0) > MAX_ZENITH:
            continue
        hr = slice(int(r0 * hf), int(np.ceil((r0 + TILE) * hf)))
        hc = slice(int(c0 * hf), int(np.ceil((c0 + TILE) * hf)))
        tc_d = min((labels.haversine_km(lat, lon, tc["lat"], tc["lon"]) for tc in tcs), default=np.inf)
        f = labels.tile_features(bt[2], sc.cloud[r0:r0 + TILE, c0:c0 + TILE], sc.phase[r0:r0 + TILE, c0:c0 + TILE],
                                 sc.height[hr, hc], flash_counts[i], tc_d, front_counts[i])
        X.append(model_input(bt).astype(np.float16))
        y.append(labels.label(f))
        feats.append(f)
        centers.append((lat, lon))
        is_extra.append(tc_only or i >= n_regular)

    keys = list(feats[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=np.stack(X), y=np.array(y), centers=np.array(centers, np.float32), is_extra=np.array(is_extra),
        features=np.array([[f[k] for k in keys] for f in feats], np.float32), feature_names=np.array(keys),
        time=np.array(sc.time.isoformat()), n_tc=np.array(len(tcs)), has_fronts=np.array(bool(fronts)))
    counts = {c: int((np.array(y) == c).sum()) for c in labels.CLASSES}
    return f"{out.name}: {len(y)} tiles, TCs={len(tcs)}, fronts={len(fronts)} {counts}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--satellite", default="goes19")
    ap.add_argument("--out", type=Path, default=Path("data/tiles"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2025, 6, 1))
    ap.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 5, 31))
    ap.add_argument("--every-days", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tc-only", action="store_true",
                    help="only scenes with an active tropical storm, saving only storm-centred tiles")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tracks = ancillary.load_ibtracs(a.cache)  # fetch once before forking
    times = (tc_scene_times(tracks, a.start, a.end, a.every_days) if a.tc_only
             else scene_times(a.start, a.end, a.every_days))
    print(len(times), "scenes", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(build_one, a.satellite, t, a.out, a.cache, a.tc_only): t for t in times}
        for fut in as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception:
                print(f"FAILED {futs[fut]}:\n{traceback.format_exc(limit=3)}", flush=True)


if __name__ == "__main__":
    main()
