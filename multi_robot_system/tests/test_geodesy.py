"""Testes Unitários de Rigor Matemático para Geodesia WGS-84, ECEF, NED e AER.

Validação cruzada com a biblioteca de referência da indústria pymap3d.
"""

import numpy as np
import pytest
import pymap3d as pm

from multi_robot_system.core.math.geodesy import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_ned,
    ned_to_ecef,
    geodetic_to_ned,
    ned_to_aer,
    prime_vertical_radius
)

LAT0 = 41.45201666666667
LON0 = -8.81322222222222
H0 = 55.44791126176966

TEST_POINTS = [
    (LAT0, LON0, H0),
    (41.452436, -8.812361, 56.73),
    (41.453116, -8.812056, 28.35),
    (41.451011, -8.814173, 25.03),
    (38.722300, -9.139300, 100.0)
]


@pytest.mark.parametrize("lat,lon,h", TEST_POINTS)
def test_geodetic_to_ecef_vs_pymap3d(lat, lon, h):
    X, Y, Z = geodetic_to_ecef(lat, lon, h)
    X_ref, Y_ref, Z_ref = pm.geodetic2ecef(lat, lon, h)
    assert abs(X - X_ref) < 1e-6
    assert abs(Y - Y_ref) < 1e-6
    assert abs(Z - Z_ref) < 1e-6


@pytest.mark.parametrize("lat,lon,h", TEST_POINTS)
def test_ecef_to_geodetic_roundtrip(lat, lon, h):
    X, Y, Z = geodetic_to_ecef(lat, lon, h)
    lat_r, lon_r, h_r = ecef_to_geodetic(X, Y, Z)
    assert abs(lat_r - lat) < 1e-8
    assert abs(lon_r - lon) < 1e-8
    assert abs(h_r - h) < 1e-4


@pytest.mark.parametrize("lat,lon,h", TEST_POINTS)
def test_geodetic_to_ned_vs_pymap3d(lat, lon, h):
    N, E, D = geodetic_to_ned(lat, lon, h, LAT0, LON0, H0)
    N_ref, E_ref, D_ref = pm.geodetic2ned(lat, lon, h, LAT0, LON0, H0)
    assert abs(N - N_ref) < 1e-5
    assert abs(E - E_ref) < 1e-5
    assert abs(D - D_ref) < 1e-5


def test_origin_is_zero():
    N, E, D = geodetic_to_ned(LAT0, LON0, H0, LAT0, LON0, H0)
    assert abs(N) < 1e-10
    assert abs(E) < 1e-10
    assert abs(D) < 1e-10


def test_cardinal_directions():
    # 100m a Norte
    az, el, rng = ned_to_aer(100.0, 0.0, 0.0)
    assert abs(az - 0.0) < 1e-8
    assert abs(el - 0.0) < 1e-8
    assert abs(rng - 100.0) < 1e-8

    # 100m a Este
    az, el, rng = ned_to_aer(0.0, 100.0, 0.0)
    assert abs(az - 90.0) < 1e-8
    assert abs(el - 0.0) < 1e-8
    assert abs(rng - 100.0) < 1e-8
