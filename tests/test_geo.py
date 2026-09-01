from img2gps.constants import GRID_N
from img2gps.geo import cell_center, get_cell, haversine_m, offset_from_center


def test_cell_center_roundtrip():
    for cell_id in [0, 11, 55, GRID_N * GRID_N - 1]:
        lat, lon = cell_center(cell_id)
        assert get_cell(lat, lon) == cell_id


def test_offset_center_is_zero():
    lat, lon = cell_center(42)
    off_lat, off_lon = offset_from_center(lat, lon, 42)
    assert abs(off_lat) < 1e-12
    assert abs(off_lon) < 1e-12


def test_haversine_zero():
    assert haversine_m(39.95, -75.19, 39.95, -75.19) == 0
