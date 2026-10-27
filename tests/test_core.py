import datetime as dt

import numpy as np
import pytest

from satwx import ancillary, goes, labels
from satwx.fetch import bt_to_png, pick_satellite, png_to_bt
from satwx.model import expected_calibration_error, fit_temperature, softmax


def g19_grid():
    # GOES-19 full-disk 2 km grid (scan angles from the CMIPF file attributes)
    x = -0.151844 + 5.6e-05 * np.arange(goes.FULL_DISK_2KM)
    y = 0.151844 - 5.6e-05 * np.arange(goes.FULL_DISK_2KM)
    return goes.FixedGrid(-75.0, 35786023.0, 6378137.0, 6356752.31414, x, y)


def test_subsatellite_point_is_grid_centre():
    g = g19_grid()
    lat, lon = g.latlon(np.array([0.0]), np.array([0.0]))
    assert abs(lat[0]) < 1e-9 and abs(lon[0] + 75.0) < 1e-9
    r, c, vis = g.pixel(0.0, -75.0)
    assert vis and abs(r - 2711.5) < 1 and abs(c - 2711.5) < 1


@pytest.mark.parametrize("lat,lon", [(25.5, -80.2), (-30, -40), (45, -120), (10, -20)])
def test_projection_round_trip(lat, lon):
    g = g19_grid()
    x, y, vis = g.scan_angles(lat, lon)
    assert vis
    la, lo = g.latlon(x, y)
    assert abs(la - lat) < 1e-6 and abs(lo - lon) < 1e-6


def test_far_side_not_visible():
    assert not g19_grid().scan_angles(0.0, 105.0)[2]


def test_space_pixels_are_nan():
    la, lo = g19_grid().latlon(np.array([0.151]), np.array([0.151]))
    assert np.isnan(la[0]) and np.isnan(lo[0])


def test_scan_start_parsing():
    k = "ABI-L2-CMIPF/2025/250/18/OR_ABI-L2-CMIPF-M6C13_G19_s20252501800207_e2025250_c1.nc"
    assert goes.scan_start(k) == dt.datetime(2025, 9, 7, 18, 0, 20)


CODSUS = """ASUS02 KWBC 080300
CODSUS
1223 AM EDT MON SEP 08 2025

VALID 090803Z
HIGHS 1017 3031077 1017 2521014
COLD WK 3380 3381 3281
3182 3183
TROF 2978 3078 3177
WARM 4211077 4221060
"""


def test_parse_codsus():
    a = ancillary.parse_codsus(CODSUS)
    (valid, fronts), = a.items()
    assert valid == dt.datetime(2025, 9, 8, 3)
    kinds = [k for k, _ in fronts]
    assert kinds == ["COLD", "WARM"]           # troughs are skipped
    cold = fronts[0][1]
    assert cold.shape == (5, 2) and tuple(cold[0]) == (33.0, -80.0)   # continuation line joined
    assert tuple(fronts[1][1][0]) == (42.1, -107.7)                   # high-res 7-digit point


def test_densify_spacing():
    d = ancillary.densify(np.array([[30.0, -90.0], [30.0, -88.0]]), 0.25)
    assert len(d) == 9 and np.allclose(np.diff(d[:, 1]), 0.25)


def feat(**kw):
    base = dict(cloud_frac=0.5, liquid_frac=0.5, ice_frac=0.5, top_height_med=5000.0, bt13_min=250.0,
                bt13_std=8.0, cold_frac=0.0, flashes=0, tc_dist_km=np.inf, front_pts=0)
    base.update(kw)
    return base


@pytest.mark.parametrize("kw,expected", [
    (dict(cloud_frac=0.01), "clear"),
    (dict(cloud_frac=0.4, liquid_frac=0.95, top_height_med=1200, bt13_min=280), "cumulus"),
    (dict(cloud_frac=0.95, liquid_frac=0.95, top_height_med=800), "low_stratus_fog"),
    (dict(cloud_frac=0.95, liquid_frac=0.2, top_height_med=4000, bt13_std=3), "stratiform"),
    (dict(cloud_frac=0.8, ice_frac=0.9, top_height_med=10000), "cirrus"),
    (dict(cloud_frac=0.8, flashes=50, cold_frac=0.3), "convective_storm"),
    (dict(cloud_frac=0.9, flashes=50, cold_frac=0.3, tc_dist_km=100), "tropical_cyclone"),
    (dict(cloud_frac=0.9, front_pts=4), "frontal_band"),
    (dict(cloud_frac=0.6, flashes=3), "other"),
])
def test_label_rules(kw, expected):
    assert labels.label(feat(**kw)) == expected


def test_png_round_trip(tmp_path):
    bt = np.linspace(190, 310, 64 * 64).reshape(64, 64)
    bt_to_png(bt, tmp_path / "t.png")
    back = png_to_bt(tmp_path / "t.png")
    assert back.shape == (1, 64, 64) and np.abs(back[0] - bt).max() < 0.4


def test_pick_satellite():
    assert pick_satellite(-80) == "goes19" and pick_satellite(-150) == "goes18"


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 5000)
    logits = rng.normal(size=(5000, 4))
    logits[np.arange(5000), y] += 1.0
    logits *= 4  # overconfident
    t = fit_temperature(logits, y)
    assert t > 2
    assert expected_calibration_error(softmax(logits / t), y) < expected_calibration_error(softmax(logits), y) / 2
