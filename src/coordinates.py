"""Módulo de Transformações de Coordenadas Geodésicas e Espaciais.

Implementação de raiz (first-principles) com NumPy para a disciplina
TOMSA (Tópicos de Matemática para Sistemas Autónomos).

Sistemas Suportados:
1. Geodésico WGS-84: Latitude (phi), Longitude (lambda), Altitude (h)
2. ECEF (Earth-Centered, Earth-Fixed): (X, Y, Z)
3. ENU (East-North-Up): Tangente local cartesiano (xEast, yNorth, zUp)
4. NED (North-East-Down): Tangente local cartesiano aeronáutico/marítimo (xNorth, yEast, zDown)
5. AER (Azimuth, Elevation, Range): Coordenadas esféricas relativas locais
"""

from typing import Tuple, Union
import numpy as np

# Constantes do Elipsoide de Referência WGS-84
WGS84_A: float = 6378137.0                   # Semieixo maior (metros)
WGS84_F: float = 1.0 / 298.257223563          # Achatamento (flattening)
WGS84_B: float = WGS84_A * (1.0 - WGS84_F)   # Semieixo menor: ~6356752.314245 m
WGS84_E2: float = 2.0 * WGS84_F - WGS84_F**2 # Primeira excentricidade ao quadrado (~0.00669437999014)
WGS84_E_PRIME2: float = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2) # Segunda excentricidade ao quadrado


def prime_vertical_radius(lat_rad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calcula o Raio de Curvatura Normal no Primeiro Vertical N(phi).

    Fórmula:
        N(phi) = a / sqrt(1 - e^2 * sin^2(phi))
    """
    sin_lat = np.sin(lat_rad)
    return WGS84_A / np.sqrt(1.0 - WGS84_E2 * (sin_lat**2))


def geodetic_to_ecef(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    h_m: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte Coordenadas Geodésicas (WGS-84) para Cartesianas ECEF.

    Equações:
        X = (N(phi) + h) * cos(phi) * cos(lambda)
        Y = (N(phi) + h) * cos(phi) * sin(lambda)
        Z = (N(phi) * (1 - e^2) + h) * sin(phi)
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    h = np.asarray(h_m, dtype=float)

    N = prime_vertical_radius(phi)
    cos_phi = np.cos(phi)

    X = (N + h) * cos_phi * np.cos(lam)
    Y = (N + h) * cos_phi * np.sin(lam)
    Z = (N * (1.0 - WGS84_E2) + h) * np.sin(phi)

    return X, Y, Z


def ecef_to_geodetic(
    x_m: Union[float, np.ndarray],
    y_m: Union[float, np.ndarray],
    z_m: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte Coordenadas ECEF para Geodésicas (WGS-84).

    Utiliza o algoritmo analítico fechado de Bowring (1985), garantindo
    precisão submilimétrica em qualquer ponto da Terra sem iterações lentas.
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    z = np.asarray(z_m, dtype=float)

    p = np.sqrt(x**2 + y**2)
    # Ângulo paramétrico inicial theta
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)

    # Fórmula fechada de Bowring para a latitude
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    num = z + WGS84_E_PRIME2 * WGS84_B * (sin_theta**3)
    den = p - WGS84_E2 * WGS84_A * (cos_theta**3)
    phi = np.arctan2(num, den)

    # Longitude
    lam = np.arctan2(y, x)

    # Altitude
    N = prime_vertical_radius(phi)
    h = (p / np.cos(phi)) - N

    return np.degrees(phi), np.degrees(lam), h


def rotation_matrix_ecef_to_enu(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """Matriz de rotação ortogonal 3x3 do referencial ECEF para o plano local ENU.

    R_ecef2enu = [
        [-sin(lambda0),               cos(lambda0),              0       ],
        [-sin(phi0)*cos(lambda0),    -sin(phi0)*sin(lambda0),    cos(phi0)],
        [ cos(phi0)*cos(lambda0),     cos(phi0)*sin(lambda0),    sin(phi0)]
    ]
    """
    phi0 = np.radians(lat0_deg)
    lam0 = np.radians(lon0_deg)

    sin_phi = np.sin(phi0)
    cos_phi = np.cos(phi0)
    sin_lam = np.sin(lam0)
    cos_lam = np.cos(lam0)

    return np.array([
        [-sin_lam, cos_lam, 0.0],
        [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
        [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi]
    ])


def ecef_to_enu(
    x_m: Union[float, np.ndarray],
    y_m: Union[float, np.ndarray],
    z_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    h0_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte ECEF para o Plano Tangente Local ENU em torno de uma origem."""
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    dx = np.asarray(x_m, dtype=float) - x0
    dy = np.asarray(y_m, dtype=float) - y0
    dz = np.asarray(z_m, dtype=float) - z0

    R = rotation_matrix_ecef_to_enu(lat0_deg, lon0_deg)

    delta = np.stack([dx, dy, dz], axis=0)
    enu = np.tensordot(R, delta, axes=(1, 0))

    return enu[0], enu[1], enu[2]


def enu_to_ecef(
    e_m: Union[float, np.ndarray],
    n_m: Union[float, np.ndarray],
    u_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    h0_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte ENU para ECEF (usando a transposta da matriz ortogonal R^T)."""
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0_m)
    R = rotation_matrix_ecef_to_enu(lat0_deg, lon0_deg)
    R_inv = R.T  # Para matrizes de rotação ortogonais, a inversa é a transposta

    enu = np.stack([np.asarray(e_m), np.asarray(n_m), np.asarray(u_m)], axis=0)
    delta = np.tensordot(R_inv, enu, axes=(1, 0))

    return delta[0] + x0, delta[1] + y0, delta[2] + z0


def ecef_to_ned(
    x_m: Union[float, np.ndarray],
    y_m: Union[float, np.ndarray],
    z_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    h0_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte ECEF para o Plano Tangente Local NED (North-East-Down).

    Relação direta com ENU:
        x_N = y_N (North)
        y_E = x_E (East)
        z_D = -z_U (-Up)
    """
    e, n, u = ecef_to_enu(x_m, y_m, z_m, lat0_deg, lon0_deg, h0_m)
    return n, e, -u


def ned_to_ecef(
    n_m: Union[float, np.ndarray],
    e_m: Union[float, np.ndarray],
    d_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    h0_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte NED para ECEF."""
    return enu_to_ecef(e_m, n_m, -d_m, lat0_deg, lon0_deg, h0_m)


def geodetic_to_enu(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    h_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    h0_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte diretamente Geodésico para ENU (via ECEF)."""
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    return ecef_to_enu(x, y, z, lat0_deg, lon0_deg, h0_m)


def geodetic_to_ned(
    lat_deg: Union[float, np.ndarray],
    lon_deg: Union[float, np.ndarray],
    h_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    h0_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte diretamente Geodésico para NED (via ECEF)."""
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, h_m)
    return ecef_to_ned(x, y, z, lat0_deg, lon0_deg, h0_m)


def enu_to_geodetic(
    e_m: Union[float, np.ndarray],
    n_m: Union[float, np.ndarray],
    u_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    h0_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte ENU para Geodésico."""
    x, y, z = enu_to_ecef(e_m, n_m, u_m, lat0_deg, lon0_deg, h0_m)
    return ecef_to_geodetic(x, y, z)


def ned_to_geodetic(
    n_m: Union[float, np.ndarray],
    e_m: Union[float, np.ndarray],
    d_m: Union[float, np.ndarray],
    lat0_deg: float,
    lon0_deg: float,
    h0_m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte NED para Geodésico."""
    x, y, z = ned_to_ecef(n_m, e_m, d_m, lat0_deg, lon0_deg, h0_m)
    return ecef_to_geodetic(x, y, z)


def enu_to_aer(
    e_m: Union[float, np.ndarray],
    n_m: Union[float, np.ndarray],
    u_m: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte Coordenadas ENU para AER (Azimuth, Elevation, Slant Range).

    Convenções:
        - Azimuth (graus): Ângulo no plano horizontal medido no sentido horário
          a partir do Norte (0° = Norte, 90° = Este, 180° = Sul, 270° = Oeste).
        - Elevation (graus): Ângulo vertical acima do plano horizontal [-90°, 90°].
        - Range (metros): Distância euclidiana tridimensional.
    """
    e = np.asarray(e_m, dtype=float)
    n = np.asarray(n_m, dtype=float)
    u = np.asarray(u_m, dtype=float)

    r_horiz = np.sqrt(e**2 + n**2)
    slant_range = np.sqrt(r_horiz**2 + u**2)

    # Azimute no sentido horário a partir do Norte: atan2(East, North)
    az_rad = np.arctan2(e, n)
    az_deg = np.degrees(az_rad) % 360.0

    # Elevação vertical
    el_rad = np.arctan2(u, r_horiz)
    el_deg = np.degrees(el_rad)

    return az_deg, el_deg, slant_range


def ned_to_aer(
    n_m: Union[float, np.ndarray],
    e_m: Union[float, np.ndarray],
    d_m: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte Coordenadas NED para AER.

    Relação:
        e = e_m, n = n_m, u = -d_m
    """
    return enu_to_aer(e_m, n_m, -np.asarray(d_m, dtype=float))


def aer_to_enu(
    az_deg: Union[float, np.ndarray],
    el_deg: Union[float, np.ndarray],
    range_m: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte Coordenadas Esféricas AER para Cartesianas Locais ENU."""
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    rng = np.asarray(range_m, dtype=float)

    r_horiz = rng * np.cos(el_rad)
    e = r_horiz * np.sin(az_rad)
    n = r_horiz * np.cos(az_rad)
    u = rng * np.sin(el_rad)

    return e, n, u


def aer_to_ned(
    az_deg: Union[float, np.ndarray],
    el_deg: Union[float, np.ndarray],
    range_m: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converte Coordenadas Esféricas AER para Cartesianas Locais NED."""
    e, n, u = aer_to_enu(az_deg, el_deg, range_m)
    return n, e, -u
