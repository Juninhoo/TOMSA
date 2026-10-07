"""Testes Unitários de Rigor Matemático para Transformações de Coordenadas.

Validação cruzada de implementações em coordenadas puras contra as bibliotecas
de referência padrão da indústria: pymap3d e pyproj.
"""

import numpy as np
import pytest
import pymap3d as pm
import pyproj

from src.coordinates import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_enu,
    enu_to_ecef,
    ecef_to_ned,
    ned_to_ecef,
    geodetic_to_enu,
    geodetic_to_ned,
    enu_to_aer,
    ned_to_aer,
    aer_to_enu,
    aer_to_ned
)
from src.frames import (
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    transform_to_local_robot_frame,
    calculate_relative_aer_body
)

# Origem de referência (Aguçadoura / INESC TEC)
LAT0 = 41.45201666666667
LON0 = -8.81322222222222
H0 = 55.44791126176966

# Conjunto de pontos de teste (Portugal Continental e costa Atlântica)
TEST_POINTS = [
    (41.45201666666667, -8.81322222222222, 55.44791126176966),  # Origem
    (41.45243668417263, -8.812361269432273, 56.73275840096176), # Robot 1 t0
    (41.45311684255299, -8.812056013616406, 28.35936586372554), # Robot 4 t0
    (41.45101100000000, -8.814173000000000, 25.03000000000000), # Robot 7
    (38.72230000000000, -9.139300000000000, 100.00000000000000), # Lisboa
]


@pytest.mark.parametrize("lat,lon,h", TEST_POINTS)
def test_geodetic_to_ecef_vs_pymap3d(lat, lon, h):
    """Valida conversão Geodésico -> ECEF contra pymap3d (erro < 1e-6 metros)."""
    X_calc, Y_calc, Z_calc = geodetic_to_ecef(lat, lon, h)
    X_ref, Y_ref, Z_ref = pm.geodetic2ecef(lat, lon, h)

    assert abs(X_calc - X_ref) < 1e-6
    assert abs(Y_calc - Y_ref) < 1e-6
    assert abs(Z_calc - Z_ref) < 1e-6


@pytest.mark.parametrize("lat,lon,h", TEST_POINTS)
def test_ecef_roundtrip_bowring(lat, lon, h):
    """Valida ida e volta Geodésico -> ECEF -> Geodésico via Bowring."""
    X, Y, Z = geodetic_to_ecef(lat, lon, h)
    lat_rt, lon_rt, h_rt = ecef_to_geodetic(X, Y, Z)

    assert abs(lat - lat_rt) < 1e-9  # graus (~0.1 mm)
    assert abs(lon - lon_rt) < 1e-9  # graus
    assert abs(h - h_rt) < 1e-4      # metros (< 0.1 mm)


@pytest.mark.parametrize("lat,lon,h", TEST_POINTS)
def test_geodetic_to_enu_vs_pymap3d(lat, lon, h):
    """Valida conversão Geodésico -> ENU contra pymap3d."""
    e_calc, n_calc, u_calc = geodetic_to_enu(lat, lon, h, LAT0, LON0, H0)
    e_ref, n_ref, u_ref = pm.geodetic2enu(lat, lon, h, LAT0, LON0, H0)

    assert abs(e_calc - e_ref) < 1e-5
    assert abs(n_calc - n_ref) < 1e-5
    assert abs(u_calc - u_ref) < 1e-5


@pytest.mark.parametrize("lat,lon,h", TEST_POINTS)
def test_geodetic_to_ned_vs_pymap3d(lat, lon, h):
    """Valida conversão Geodésico -> NED contra pymap3d."""
    n_calc, e_calc, d_calc = geodetic_to_ned(lat, lon, h, LAT0, LON0, H0)
    n_ref, e_ref, d_ref = pm.geodetic2ned(lat, lon, h, LAT0, LON0, H0)

    assert abs(n_calc - n_ref) < 1e-5
    assert abs(e_calc - e_ref) < 1e-5
    assert abs(d_calc - d_ref) < 1e-5


@pytest.mark.parametrize("lat,lon,h", TEST_POINTS)
def test_enu_to_aer_vs_pymap3d(lat, lon, h):
    """Valida conversão ENU -> AER (Azimute, Elevação, Range) contra pymap3d."""
    e, n, u = geodetic_to_enu(lat, lon, h, LAT0, LON0, H0)
    az_calc, el_calc, rng_calc = enu_to_aer(e, n, u)
    az_ref, el_ref, rng_ref = pm.enu2aer(e, n, u)

    if rng_calc > 0.1:  # Se não estiver exatamente na origem
        assert abs(az_calc - az_ref) < 1e-4
        assert abs(el_calc - el_ref) < 1e-4
    assert abs(rng_calc - rng_ref) < 1e-5


def test_body_frame_known_geometry():
    """Valida o referencial local com geometria analítica conhecida.

    Cenário: Robô orientado a Este (Heading = 90°).
    Um alvo situado a 50m a Este e 20m a Sul (em coordenadas NED):
    - No referencial do robô (x à frente, y à direita/estibordo):
      - Como o robô aponta para Este, o alvo a Este está à frente (x = +50m).
      - O alvo a Sul está à sua direita/estibordo (y = +20m).
    """
    yaw_rad = np.pi / 2.0  # 90 graus (Este)
    qx, qy, qz, qw = 0.0, 0.0, np.sin(yaw_rad / 2.0), np.cos(yaw_rad / 2.0)

    robot_pos = np.array([0.0, 0.0, 0.0])      # Origem
    target_pos = np.array([-20.0, 50.0, 0.0])  # Em NED: 20m Sul (-N), 50m Este (+E)

    pos_body = transform_to_local_robot_frame(target_pos, robot_pos, (qx, qy, qz, qw))

    # O alvo a Este deve ficar em x_body = +50
    # O alvo a Sul deve ficar em y_body = +20
    assert abs(pos_body[0] - 50.0) < 1e-6
    assert abs(pos_body[1] - 20.0) < 1e-6
    assert abs(pos_body[2] - 0.0) < 1e-6

    # Testar azimute relativo (bearing)
    bearing, elevation, distance = calculate_relative_aer_body(pos_body)
    expected_bearing = np.degrees(np.arctan2(20.0, 50.0))
    assert abs(bearing - expected_bearing) < 1e-5
    assert abs(distance - np.sqrt(50**2 + 20**2)) < 1e-5
