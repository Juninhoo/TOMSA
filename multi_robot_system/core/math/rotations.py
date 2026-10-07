"""Módulo Cinemático de Atitude, Quaterniões e Rotações 3D em SO(3).

Implementa operações sobre quaterniões unitários, matrizes de rotação,
convenção intrínseca ZYX de ângulos de Euler (Roll, Pitch, Yaw) e interpolação esférica SLERP.
"""

from typing import Tuple, Union
import numpy as np


def normalize_quaternion(
    qx: Union[float, np.ndarray],
    qy: Union[float, np.ndarray],
    qz: Union[float, np.ndarray],
    qw: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Garante que o quaternião possui norma unitária estrita (||q|| = 1)."""
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    norm = np.where(norm < 1e-15, 1.0, norm)
    return qx / norm, qy / norm, qz / norm, qw / norm


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Converte um quaternião [qx, qy, qz, qw] ou array (N, 4) numa matriz de rotação R in SO(3)."""
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        qx, qy, qz, qw = q
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if norm > 1e-15:
            qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        return np.array([
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx**2 + qy**2)]
        ], dtype=float)

    # Versão em lote para array (N, 4)
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    norms = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    norms = np.where(norms < 1e-15, 1.0, norms)
    qx, qy, qz, qw = qx / norms, qy / norms, qz / norms, qw / norms

    N = q.shape[0]
    R = np.empty((N, 3, 3), dtype=float)
    R[:, 0, 0] = 1.0 - 2.0 * (qy**2 + qz**2)
    R[:, 0, 1] = 2.0 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2.0 * (qx * qz + qy * qw)

    R[:, 1, 0] = 2.0 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1.0 - 2.0 * (qx**2 + qz**2)
    R[:, 1, 2] = 2.0 * (qy * qz - qx * qw)

    R[:, 2, 0] = 2.0 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2.0 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1.0 - 2.0 * (qx**2 + qy**2)
    return R


def quaternion_to_euler_zyx(
    qx: Union[float, np.ndarray],
    qy: Union[float, np.ndarray],
    qz: Union[float, np.ndarray],
    qw: Union[float, np.ndarray],
    degrees: bool = True
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Extrai os ângulos de Euler (Roll, Pitch, Yaw) segundo a convenção aeroespacial ZYX."""
    # Roll (rotação em torno de x)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotação em torno de y) com saturação para evitar NaN em gimbal lock
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (rotação em torno de z)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    if degrees:
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    return roll, pitch, yaw


def euler_zyx_to_rotation_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Constrói a matriz de rotação R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ], dtype=float)


def slerp(q0: np.ndarray, q1: np.ndarray, tau: float) -> np.ndarray:
    """Interpolação Esférica Linear (SLERP) entre quaterniões q0 e q1 com parâmetro tau in [0, 1]."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        res = q0 + tau * (q1 - q0)
        return res / np.linalg.norm(res)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta_t = theta_0 * tau
    sin_theta_t = np.sin(theta_t)

    s0 = np.cos(theta_t) - dot * sin_theta_t / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    return (s0 * q0) + (s1 * q1)
