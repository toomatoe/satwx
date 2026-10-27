"""GOES-R ABI / GLM access from the NOAA open-data buckets on AWS (anonymous HTTPS, no account).

Bucket layout: s3://noaa-goes19/<product>/<YYYY>/<DOY>/<HH>/OR_<product>-M6..._s<YYYYDOYHHMMSSs>_e..._c....nc
"""
from __future__ import annotations

import datetime as dt
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
FULL_DISK_2KM = 5424  # pixels per side of the 2 km full-disk grid

# Infrared channels used as model input (all available day and night).
#   C07 3.9um shortwave IR (fog / low cloud at night via C13-C07)
#   C08 6.2um upper-level water vapour
#   C13 10.3um clean IR window (cloud-top temperature)
#   C15 12.3um dirty IR window (split-window, thin cirrus)
BANDS = ("C07", "C08", "C13", "C15")


def bucket(satellite: str) -> str:
    return f"https://noaa-{satellite}.s3.amazonaws.com"


def list_keys(satellite: str, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        url = f"{bucket(satellite)}/?list-type=2&prefix={prefix}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        keys += [c.find(S3_NS + "Key").text for c in root.iter(S3_NS + "Contents")]
        nxt = root.find(S3_NS + "NextContinuationToken")
        if nxt is None:
            return keys
        token = nxt.text


_START_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def scan_start(key: str) -> dt.datetime:
    y, doy, h, m, s = map(int, _START_RE.search(key).groups())
    return dt.datetime(y, 1, 1, h, m, s) + dt.timedelta(days=doy - 1)


def _hour_prefix(product: str, t: dt.datetime) -> str:
    return f"{product}/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"


def find_key(satellite: str, product: str, t: dt.datetime, band: str | None = None,
             max_offset_min: float = 15) -> str:
    """Key of the scan of `product` (optionally one band) whose start time is nearest `t`."""
    keys = []
    for dh in (-1, 0, 1):
        keys += list_keys(satellite, _hour_prefix(product, t + dt.timedelta(hours=dh)))
    if band:
        keys = [k for k in keys if f"M6{band}_" in k or f"M3{band}_" in k or f"M4{band}_" in k]
    if not keys:
        raise FileNotFoundError(f"no {product} {band or ''} files near {t} in {satellite}")
    best = min(keys, key=lambda k: abs((scan_start(k) - t).total_seconds()))
    off = abs((scan_start(best) - t).total_seconds()) / 60
    if off > max_offset_min:
        raise FileNotFoundError(f"nearest {product} {band or ''} scan is {off:.0f} min from {t}")
    return best


def keys_between(satellite: str, product: str, t0: dt.datetime, t1: dt.datetime) -> list[str]:
    keys, t = [], t0.replace(minute=0, second=0)
    while t <= t1:
        keys += list_keys(satellite, _hour_prefix(product, t))
        t += dt.timedelta(hours=1)
    return [k for k in keys if t0 <= scan_start(k) < t1]


def download(satellite: str, key: str, cache_dir: str | os.PathLike) -> Path:
    out = Path(cache_dir) / Path(key).name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(f"{bucket(satellite)}/{key}", tmp)
        tmp.replace(out)
    return out


# ---------------------------------------------------------------- fixed-grid geometry (GOES-R PUG vol. 4)

class FixedGrid:
    """ABI fixed-grid projection: scan angles (x, y) in radians <-> geodetic lat/lon."""

    def __init__(self, lon0: float, h_sat: float, r_eq: float, r_pol: float,
                 x: np.ndarray, y: np.ndarray):
        self.lon0, self.r_eq, self.r_pol = lon0, r_eq, r_pol
        self.H = h_sat + r_eq
        self.x, self.y = x, y  # 1-D scan-angle coordinates of the grid (x ascending, y descending)

    @classmethod
    def from_dataset(cls, ds) -> "FixedGrid":
        p = ds.variables["goes_imager_projection"]
        return cls(float(p.longitude_of_projection_origin), float(p.perspective_point_height),
                   float(p.semi_major_axis), float(p.semi_minor_axis),
                   np.asarray(ds.variables["x"][:], dtype=np.float64),
                   np.asarray(ds.variables["y"][:], dtype=np.float64))

    def latlon(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        H, re2, rp2 = self.H, self.r_eq ** 2, self.r_pol ** 2
        a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + re2 / rp2 * np.sin(y) ** 2)
        b = -2 * H * np.cos(x) * np.cos(y)
        c = H ** 2 - re2
        disc = b ** 2 - 4 * a * c
        with np.errstate(invalid="ignore"):
            rs = (-b - np.sqrt(disc)) / (2 * a)
        sx, sy, sz = rs * np.cos(x) * np.cos(y), -rs * np.sin(x), rs * np.cos(x) * np.sin(y)
        lat = np.degrees(np.arctan(re2 / rp2 * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
        lon = self.lon0 - np.degrees(np.arctan(sy / (H - sx)))
        bad = disc < 0
        return np.where(bad, np.nan, lat), np.where(bad, np.nan, lon)

    def scan_angles(self, lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (x, y, visible)."""
        lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
        H, re2, rp2 = self.H, self.r_eq ** 2, self.r_pol ** 2
        phi_c = np.arctan(rp2 / re2 * np.tan(lat))
        e2 = (re2 - rp2) / re2
        rc = self.r_pol / np.sqrt(1 - e2 * np.cos(phi_c) ** 2)
        dlon = lon - np.radians(self.lon0)
        sx = H - rc * np.cos(phi_c) * np.cos(dlon)
        sy = -rc * np.cos(phi_c) * np.sin(dlon)
        sz = rc * np.sin(phi_c)
        visible = H * (H - sx) >= sy ** 2 + re2 / rp2 * sz ** 2
        y = np.arctan(sz / sx)
        x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
        return x, y, visible

    def pixel(self, lat, lon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fractional (row, col) indices on this grid, plus visibility."""
        x, y, vis = self.scan_angles(lat, lon)
        dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        return (y - self.y[0]) / dy, (x - self.x[0]) / dx, vis


def satellite_zenith(lat, lon, lon0: float) -> np.ndarray:
    """Approximate satellite zenith angle (deg), spherical Earth."""
    lat, dlon = np.radians(lat), np.radians(np.asarray(lon) - lon0)
    cg = np.cos(lat) * np.cos(dlon)
    g = np.arccos(np.clip(cg, -1, 1))
    ratio = 6371.0 / (6371.0 + 35786.0)
    return np.degrees(np.arctan2(np.sin(g), cg - ratio))


# ---------------------------------------------------------------- readers

def read_var(path: Path, var: str, dtype=np.float32) -> tuple[np.ndarray, "FixedGrid | None"]:
    """Read one gridded variable (masked values -> NaN) and its fixed grid."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        v = ds.variables[var]
        data = np.ma.filled(v[:].astype(dtype), np.nan)
        grid = FixedGrid.from_dataset(ds) if "goes_imager_projection" in ds.variables else None
    return data, grid


def read_glm_flashes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if "flash_lat" not in ds.variables or ds.dimensions["number_of_flashes"].size == 0:
            return np.empty(0, np.float32), np.empty(0, np.float32)
        return (np.asarray(ds.variables["flash_lat"][:], np.float32),
                np.asarray(ds.variables["flash_lon"][:], np.float32))
