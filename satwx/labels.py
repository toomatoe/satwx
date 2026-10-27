"""Rule-based (weak) labels for tiles, derived from products the model never sees as input.

Inputs to the rules: ABI L2 clear-sky mask, cloud-top phase, cloud-top height, GLM lightning,
IBTrACS best-track positions and WPC analysed surface fronts. The model only sees IR radiances,
so it has to learn to recognise these regimes from imagery alone.
"""
from __future__ import annotations

import numpy as np

CLASSES = ("clear", "cumulus", "stratiform", "low_stratus_fog", "cirrus", "convective_storm",
           "tropical_cyclone", "frontal_band", "other")

TC_RADIUS_KM = 200.0
MIN_FLASHES = 10            # GLM flashes in the tile during the 10-min scan
COLD_TOP_K = 235.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def tile_features(bt13, cloud, phase, height, n_flashes, tc_dist_km, n_front_pts) -> dict:
    cloudy = cloud == 1
    valid = np.isfinite(cloud)
    cf = float(cloudy[valid].mean()) if valid.any() else np.nan
    ph = phase[cloudy & np.isfinite(phase)]
    liquid = float(np.isin(ph, (1, 2)).mean()) if ph.size else np.nan
    ice = float((ph == 4).mean()) if ph.size else np.nan
    ht = height[np.isfinite(height)]
    return dict(
        cloud_frac=cf, liquid_frac=liquid, ice_frac=ice,
        top_height_med=float(np.median(ht)) if ht.size else np.nan,
        bt13_min=float(np.nanmin(bt13)), bt13_std=float(np.nanstd(bt13)),
        cold_frac=float(np.mean(bt13 < COLD_TOP_K)),
        flashes=int(n_flashes), tc_dist_km=float(tc_dist_km), front_pts=int(n_front_pts),
    )


def label(f: dict) -> str:
    cf = f["cloud_frac"]
    if not np.isfinite(cf):
        return "other"
    if f["tc_dist_km"] <= TC_RADIUS_KM and cf > 0.5:
        return "tropical_cyclone"
    if f["flashes"] >= MIN_FLASHES and f["cold_frac"] > 0.02:
        return "convective_storm"
    if f["front_pts"] > 0 and cf > 0.6 and f["flashes"] < MIN_FLASHES:
        return "frontal_band"
    if f["flashes"] > 0:
        return "other"
    if cf < 0.03:
        return "clear"
    liq, ht = f["liquid_frac"], f["top_height_med"]
    if cf > 0.85 and liq > 0.9 and ht < 1500:
        return "low_stratus_fog"
    if cf > 0.5 and f["ice_frac"] > 0.8 and ht > 7000 and f["cold_frac"] < 0.1:
        return "cirrus"
    if cf > 0.85 and f["bt13_std"] < 5.0 and f["cold_frac"] < 0.02:
        return "stratiform"
    if 0.15 < cf < 0.7 and liq > 0.8 and ht < 3500 and f["bt13_min"] > 260:
        return "cumulus"
    return "other"
