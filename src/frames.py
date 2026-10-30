"""Orientação dos robôs e posições relativas.

Conversões entre quaterniões, ângulos de Euler (roll, pitch, yaw) e matrizes de rotação,
e cálculo da posição de um robô vista por outro (distância, azimute, elevação).
"""

from typing import Tuple, Union
import numpy as np


def normalize_quaternion(
    qx: Union[float, np.ndarray],
    qy: Union[float, np.ndarray],
    qz: Union[float, np.ndarray],
    qw: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normaliza um quaternião para que tenha comprimento 1."""
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return qx / norm, qy / norm, qz / norm, qw / norm


def quaternion_to_euler(
    qx: Union[float, np.ndarray],
    qy: Union[float, np.ndarray],
    qz: Union[float, np.ndarray],
    qw: Union[float, np.ndarray],
    degrees: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula os ângulos roll, pitch e yaw a partir de um quaternião (convenção ZYX)."""
    qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)

    # Roll, pitch (limitado a ±90°) e yaw
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    return roll, pitch, yaw


def quaternion_to_rotation_matrix(
    qx: float,
    qy: float,
    qz: float,
    qw: float
) -> np.ndarray:
    """Calcula a matriz de rotação do robô a partir do quaternião.

    Esta matriz passa vetores das coordenadas do robô para NED; a sua transposta faz o inverso.
    """
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm > 1e-12:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    return np.array([
        [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx**2 + qy**2)]
    ])


def transform_to_local_robot_frame(
    target_pos_ned: np.ndarray,
    robot_pos_ned: np.ndarray,
    robot_quat: Tuple[float, float, float, float]
) -> np.ndarray:
    """Esta função calcula a posição de outro robô vista pelo robô selecionado.

    Devolve (x frente, y direita, z baixo) em metros, para um ou vários pontos.
    """
    R_b2ned = quaternion_to_rotation_matrix(*robot_quat)
    R_ned2b = R_b2ned.T

    target_pos = np.asarray(target_pos_ned, dtype=float)
    robot_pos = np.asarray(robot_pos_ned, dtype=float)

    if target_pos.ndim == 1:
        delta_ned = target_pos - robot_pos
        return R_ned2b @ delta_ned
    else:
        # Versão para vários pontos de uma só vez
        delta_ned = target_pos - robot_pos[:, np.newaxis]
        return R_ned2b @ delta_ned


def calculate_relative_aer_body(
    pos_body: np.ndarray
) -> Tuple[float, float, float]:
    """Calcula a direção (0° à frente, +90° à direita, 180° atrás), a elevação e a distância
    de um ponto dado nas coordenadas do robô.
    """
    pos = np.asarray(pos_body, dtype=float)
    x = pos[0]
    y = pos[1]
    z = pos[2] if len(pos) > 2 else 0.0

    r_horiz = np.sqrt(x**2 + y**2)
    slant_range = np.sqrt(r_horiz**2 + z**2)

    rel_bearing_deg = np.degrees(np.arctan2(y, x))

    rel_elevation_deg = np.degrees(np.arctan2(-z, r_horiz))

    return rel_bearing_deg, rel_elevation_deg, slant_range
