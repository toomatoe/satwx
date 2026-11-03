"""GOES-R ABI / GLM access from the NOAA open-data buckets on AWS (anonymous HTTPS, no account).

Bucket layout: s3://noaa-goes19/<product>/<YYYY>/<DOY>/<HH>/OR_<product>-M6..._s<YYYYDOYHHMMSSs>_e..._c....nc
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import os
import re
import ssl
import time
import urllib.error
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


def http(req: "str | urllib.request.Request", retries: int = 5) -> tuple[dict, bytes]:
    """(headers, body) of an HTTP request, retrying dropped connections and 5xx with backoff."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries - 1:
                raise
        except (urllib.error.URLError, ssl.SSLError, ConnectionError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(min(2 ** attempt, 20))


def bucket(satellite: str) -> str:
    return f"https://noaa-{satellite}.s3.amazonaws.com"


def list_keys(satellite: str, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        url = f"{bucket(satellite)}/?list-type=2&prefix={prefix}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        root = ET.fromstring(http(url)[1])
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
             max_offset_min: float = 15, same_scan_as: str | None = None) -> str:
    """Key of the scan of `product` (optionally one band) whose start time is nearest `t`.
    With `same_scan_as`, only a file from exactly that scan (same start time) is accepted."""
    keys = []
    for dh in (-1, 0, 1):
        keys += list_keys(satellite, _hour_prefix(product, t + dt.timedelta(hours=dh)))
    if band:
        keys = [k for k in keys if f"M6{band}_" in k or f"M3{band}_" in k or f"M4{band}_" in k]
    if same_scan_as:
        sid = _START_RE.search(same_scan_as)[0]
        keys = [k for k in keys if _START_RE.search(k)[0] == sid]
        if not keys:
            raise FileNotFoundError(f"{product} {band or ''} for scan {sid[2:]} is not (yet) in {satellite}")
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


def recent_keys(satellite: str, product: str, band: str, hours: int = 2) -> list[str]:
    """Keys of `product`/`band` from the last `hours` hours, newest first."""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    keys = keys_between(satellite, product, now - dt.timedelta(hours=hours), now + dt.timedelta(minutes=1))
    return sorted((k for k in keys if f"{band}_" in k), key=scan_start, reverse=True)


def download(satellite: str, key: str, cache_dir: str | os.PathLike) -> Path:
    out = Path(cache_dir) / Path(key).name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        tmp.write_bytes(http(f"{bucket(satellite)}/{key}")[1])
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


# ---------------------------------------------------------------- remote windowed reads (HTTP range requests)

class RangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests with a block cache, for h5py."""

    def __init__(self, url: str, block: int = 1 << 18):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        self.size = int(http(urllib.request.Request(url, method="HEAD"))[0]["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            lo, hi = i * self.block, min(self.size, (i + 1) * self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            self.cache[i] = http(req)[1]
        return self.cache[i]

    def readinto(self, buf) -> int:
        n = max(0, min(len(buf), self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            i, off = divmod(self.pos + len(out), self.block)
            out += self._block(i)[off:off + n - len(out)]
        buf[:n] = out
        self.pos += n
        return n


@contextlib.contextmanager
def open_remote(satellite: str, key: str):
    """h5py view of a bucket object; only the byte ranges actually read are downloaded."""
    import h5py

    f = RangeFile(f"{bucket(satellite)}/{key}")
    h = h5py.File(f, "r")
    try:
        yield h
    finally:  # close h5py before the file object, or HDF5 can crash at interpreter exit
        h.close()
        f.close()


def _scaled(var, sel=()) -> np.ndarray:
    raw = var[sel] if sel else var[()]
    out = raw.astype(np.float64)
    fill = var.attrs.get("_FillValue")
    if fill is not None:
        out[raw == np.asarray(fill).ravel()[0]] = np.nan
    out = out * var.attrs.get("scale_factor", [1.0])[0] + var.attrs.get("add_offset", [0.0])[0]
    return out


def grid_from_h5(h) -> FixedGrid:
    p = h["goes_imager_projection"].attrs
    return FixedGrid(float(p["longitude_of_projection_origin"][0]), float(p["perspective_point_height"][0]),
                     float(p["semi_major_axis"][0]), float(p["semi_minor_axis"][0]), _scaled(h["x"]), _scaled(h["y"]))


def read_window(h, var: str, r0: int, c0: int, size: int) -> np.ndarray:
    return _scaled(h[var], (slice(r0, r0 + size), slice(c0, c0 + size))).astype(np.float32)
