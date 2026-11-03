import datetime as dt
import os

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


def test_cli_on_npy_tile(tmp_path, capsys):
    import json
    from satwx import cli

    rng = np.random.default_rng(0)
    np.save(tmp_path / "t.npy", (rng.normal(285, 3, (4, 64, 64))).astype(np.float32))
    cli.main([str(tmp_path / "t.npy"), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert set(out["probabilities"]) == set(labels.CLASSES)
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert out["model"] == "satwx_ir4.pt" and out["prediction"] == max(out["probabilities"], key=out["probabilities"].get)


def test_cli_rejects_wrong_shape(tmp_path):
    from satwx import cli

    np.save(tmp_path / "bad.npy", np.zeros((4, 32, 32), np.float32))
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "bad.npy")])


@pytest.mark.skipif(not os.environ.get("SATWX_NETWORK_TESTS"), reason="set SATWX_NETWORK_TESTS=1 (downloads ~25 MB)")
def test_remote_window_matches_full_file(tmp_path):
    from satwx.fetch import fetch_tile
    from satwx.scene import TILE, model_input

    t = dt.datetime(2025, 10, 28, 12)
    x, meta = fetch_tile(17.5, -78.1, t, "goes19")
    key = goes.find_key("goes19", "ABI-L2-CMIPF", t, "C13")
    full, _ = goes.read_var(goes.download("goes19", key, tmp_path), "CMI")
    r0, c0 = meta["row"], meta["col"]
    ref = model_input(full[None, r0:r0 + TILE, c0:c0 + TILE])[0]
    assert np.allclose(x[2], ref, atol=1e-3)


def test_codsus_front_ends_at_next_bulletin_header():
    text = CODSUS + "$$\n000\nASUS02 KWBC 080600\nCODSUS\n1223 AM EDT MON SEP 08 2025\n"
    (_, fronts), = ancillary.parse_codsus(text).items()
    warm = fronts[-1][1]
    assert warm.shape == (2, 2) and (warm[:, 1] > -180).all()


def test_codsus_year_rolls_over_on_new_year():
    text = "ASUS01 KWBC 010300\nCODSUS\n1000 PM EST WED DEC 31 2025\n\nVALID 010100Z\nCOLD 3380 3381\n"
    assert list(ancillary.parse_codsus(text)) == [dt.datetime(2026, 1, 1, 0)]


def test_cyclone_interpolation_across_dateline():
    import pandas as pd

    t0 = dt.datetime(2025, 9, 1, 0)
    tracks = pd.DataFrame(dict(SID=["X", "X"], NAME=["T", "T"], NATURE=["TS", "TS"], LAT=[20.0, 20.0],
                               LON=[179.0, -179.0], WIND=[50.0, 50.0],
                               ISO_TIME=pd.to_datetime([t0, t0 + dt.timedelta(hours=6)])))
    (tc,) = ancillary.cyclones_at(tracks, t0 + dt.timedelta(hours=3))
    assert abs(tc["lon"] - 180.0) < 1e-6


def test_split_has_gaps_around_held_out_blocks():
    import sys
    sys.path.insert(0, "scripts")
    from train import split_of

    day = lambda d, suffix="": split_of(f"goes19_{(dt.datetime(2025, 6, 1) + dt.timedelta(days=d)):%Y%m%dT%H%M}{suffix}")
    assert day(0) == "train" and day(50) == "val" and day(70) == "test"
    assert day(46) == "gap" and day(81) == "gap"          # 2 days before val, 1 day after test
    assert day(40) == "train" and day(40, "_tc") == "gap"  # 8 days before val: ok for regular, not for TC scenes


def test_parse_time_latest_and_utc():
    from satwx.cli import parse_time

    assert parse_time("latest") is None and parse_time("NOW") is None
    assert parse_time("2025-09-26T14:00-04:00") == dt.datetime(2025, 9, 26, 18, 0)


def test_fetch_rejects_invalid_coordinates():
    from satwx.fetch import fetch_tile

    with pytest.raises(ValueError):
        fetch_tile(95.0, -80.0, dt.datetime(2025, 9, 26, 18))


def test_http_retries_dropped_connections(monkeypatch):
    import io as _io
    import ssl
    import urllib.request

    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        if len(calls) < 3:
            raise ssl.SSLEOFError("EOF occurred in violation of protocol")

        class R(_io.BytesIO):
            headers = {"Content-Length": "2"}
        return R(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(goes.time, "sleep", lambda s: None)
    assert goes.http("https://example.invalid/x")[1] == b"ok" and len(calls) == 3
