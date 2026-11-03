"""Independent label sources: IBTrACS best tracks and WPC coded surface fronts (via IEM AFOS archive)."""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .goes import http

IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.last3years.list.v04r01.csv")
CODSUS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=CODSUS&sdate={s}&edate={e}&limit=100&fmt=text"
FRONT_TYPES = ("COLD", "WARM", "STNRY", "OCFNT")


def load_ibtracs(cache_dir: str | Path) -> pd.DataFrame:
    path = Path(cache_dir) / "ibtracs.last3years.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(http(IBTRACS_URL)[1])
    df = pd.read_csv(path, skiprows=[1], low_memory=False,
                     usecols=["SID", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "WMO_WIND", "USA_WIND"])
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
    for c in ("LAT", "LON", "WMO_WIND", "USA_WIND"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["WIND"] = df[["WMO_WIND", "USA_WIND"]].max(axis=1)
    return df


def cyclones_at(tracks: pd.DataFrame, t: dt.datetime, min_wind_kt: float = 34) -> list[dict]:
    """Tropical storms (nature TS, >= min_wind_kt) active at time t, positions linearly interpolated."""
    out = []
    win = tracks[(tracks.ISO_TIME >= t - dt.timedelta(hours=6)) & (tracks.ISO_TIME <= t + dt.timedelta(hours=6))]
    for sid, g in win.groupby("SID"):
        g = g.sort_values("ISO_TIME")
        before, after = g[g.ISO_TIME <= t], g[g.ISO_TIME >= t]
        if before.empty or after.empty:
            continue
        a, b = before.iloc[-1], after.iloc[0]
        span = (b.ISO_TIME - a.ISO_TIME).total_seconds()
        w = 0.0 if span == 0 else (pd.Timestamp(t) - a.ISO_TIME).total_seconds() / span
        wind = np.fmax(a.WIND, b.WIND)
        dlon = (b.LON - a.LON + 180) % 360 - 180  # tracks crossing the dateline
        if a.NATURE != "TS" or not wind >= min_wind_kt:
            continue
        out.append(dict(sid=sid, name=a.NAME, lat=a.LAT + w * (b.LAT - a.LAT),
                        lon=a.LON + w * dlon, wind_kt=float(wind)))
    return out


def _coord(tok: str) -> tuple[float, float] | None:
    """Low-res CODSUS point LLooo / LLoo (deg N, deg W) or high-res LLLoooo (tenths); None if not a point."""
    if not tok.isdigit() or len(tok) not in (4, 5, 7):
        return None
    lat, lon = (int(tok[:3]) / 10, int(tok[3:]) / 10) if len(tok) == 7 else (float(tok[:2]), float(tok[2:]))
    return (lat, -lon) if lat <= 90 and lon <= 180 else None


_MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()


def parse_codsus(text: str) -> dict[dt.datetime, list[tuple[str, np.ndarray]]]:
    """Map valid time -> list of (front type, Nx2 array of lat, lon). Year taken from the bulletin header."""
    out: dict[dt.datetime, list] = {}
    for block in re.split(r"(?=VALID \d{6}Z)", text)[1:]:
        m = re.match(r"VALID (\d{2})(\d{2})(\d{2})Z", block)
        # Issue date from the header ("... MON SEP 08 2025"); a Dec 31 bulletin can be valid on Jan 1.
        hdr = re.findall(r"\b(%s) \d{1,2} (20\d{2})\b" % "|".join(_MONTHS), text[: text.find(block)])
        year = int(hdr[-1][1]) if hdr else dt.datetime.utcnow().year
        if hdr and hdr[-1][0] == "DEC" and int(m[1]) == 1:
            year += 1
        valid = dt.datetime(year, int(m[1]), int(m[2]), int(m[3]))
        fronts, cur = [], None
        for line in block.splitlines()[1:]:
            toks = line.split()
            if not toks:
                continue
            if toks[0] in FRONT_TYPES:
                cur = (toks[0], [])
                fronts.append(cur)
                toks = toks[2:] if len(toks) > 1 and toks[1].isalpha() else toks[1:]  # skip WK/MDT/STG
            pts = [_coord(t) for t in toks]
            if cur is None or None in pts:  # any other line (HIGHS, TROF, next header, $$) ends the front
                cur = None
                continue
            cur[1].extend(pts)
        out[valid] = [(k, np.array(p)) for k, p in fronts if len(p) >= 2]
    return out


def fronts_at(t: dt.datetime, cache_dir: str | Path, max_offset_h: float = 1.6) -> list[tuple[str, np.ndarray]] | None:
    """WPC analysed fronts valid nearest t (None if no analysis within max_offset_h)."""
    path = Path(cache_dir) / f"codsus_{t:%Y%m%d}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        s = (t - dt.timedelta(hours=t.hour)).strftime("%Y-%m-%dT00:00Z")
        e = (t - dt.timedelta(hours=t.hour) + dt.timedelta(hours=30)).strftime("%Y-%m-%dT06:00Z")
        path.write_bytes(http(CODSUS_URL.format(s=s, e=e))[1])
    analyses = parse_codsus(path.read_text(errors="replace"))
    if not analyses:
        return None
    valid = min(analyses, key=lambda v: abs((v - t).total_seconds()))
    if abs((valid - t).total_seconds()) > max_offset_h * 3600:
        return None
    return analyses[valid]


def densify(line: np.ndarray, step_deg: float = 0.25) -> np.ndarray:
    """Interpolate a lat/lon polyline so consecutive points are <= step_deg apart."""
    pts = [line[0]]
    for a, b in zip(line[:-1], line[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / step_deg)))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(pts)
