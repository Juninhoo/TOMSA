"""Módulo de Cinemática, Quaternions e Transformações para o Referencial Local do Robô.

Responsável por:
1. Conversão entre Quaternions unitários, Ângulos de Euler (Roll, Pitch, Yaw)
   e Matrizes de Cossenos Diretores (DCM).
2. Transformação de coordenadas globais (NED/ENU) para o referencial local (Body Frame)
   do robô selecionado (origem no robô, eixos orientados com a sua atitude).
3. Cálculo de grandezas relativas entre robôs: distância relativa, azimute/marcação relativa
   (bearing) e elevação relativa.
"""

from typing import Tuple, Union
import numpy as np


def normalize_quaternion(
    qx: Union[float, np.ndarray],
    qy: Union[float, np.ndarray],
    qz: Union[float, np.ndarray],
    qw: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Garante que o quaternion é unitário (norma = 1)."""
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
    """Converte Quaternion (qx, qy, qz, qw) para Ângulos de Euler (Roll, Pitch, Yaw).

    Convenção: Tait-Bryan Z-Y-X (Yaw-Pitch-Roll) padrão na navegação NED:
        - Roll (phi): rotação em torno do eixo x (longitudinal / caturro/balanço)
        - Pitch (theta): rotação em torno do eixo y (transversal / arfagem)
        - Yaw / Heading (psi): rotação em torno do eixo z (vertical / guinada)
    """
    qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)

    # Roll (rotação em x)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotação em y, limitado a [-pi/2, pi/2])
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (rotação em z / Heading em relação ao Norte no referencial NED)
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
    """Calcula a Matriz de Rotação 3x3 R_body2ned a partir do quaternion unitário.

    R_body2ned mapeia vetores do referencial do corpo (Body) para o referencial de navegação (NED):
        v_NED = R_body2ned * v_body

    Logo, para transformar coordenadas de NED para o referencial do corpo:
        v_body = (R_body2ned)^T * v_NED
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
    """Transforma a posição de um alvo/vizinho expressa em NED para o referencial local (Body) do robô.

    Parâmetros:
        target_pos_ned: array [N, E, D] ou [3, M] da posição do robô vizinho em NED (metros).
        robot_pos_ned: array [N, E, D] da posição do robô selecionado em NED (metros).
        robot_quat: tuplo (qx, qy, qz, qw) da orientação do robô selecionado.

    Retorna:
        pos_body: array [x_fwd, y_stbd, z_down] ou [3, M] no referencial do corpo:
            - x_fwd: distância para a frente (positivo à proa)
            - y_stbd: distância lateral (positivo a estibordo / direita)
            - z_down: distância vertical (positivo para baixo)
    """
    R_b2ned = quaternion_to_rotation_matrix(*robot_quat)
    R_ned2b = R_b2ned.T  # Inversa da rotação

    target_pos = np.asarray(target_pos_ned, dtype=float)
    robot_pos = np.asarray(robot_pos_ned, dtype=float)

    if target_pos.ndim == 1:
        delta_ned = target_pos - robot_pos
        return R_ned2b @ delta_ned
    else:
        # Vetorizado para múltiplos pontos (3, M)
        delta_ned = target_pos - robot_pos[:, np.newaxis]
        return R_ned2b @ delta_ned


def calculate_relative_aer_body(
    pos_body: np.ndarray
) -> Tuple[float, float, float]:
    """Calcula Azimute Relativo (Bearing), Elevação Relativa e Distância no referencial do corpo.

    Convenções do Corpo (Body Frame):
        - Relative Bearing (azimute relativo): ângulo no plano horizontal do robô
          relativo à proa (frente / eixo x):
          0° = Alvo diretamente à frente (Dead Ahead)
          +90° = Alvo a estibordo (Direita)
          -90° / 270° = Alvo a bombordo (Esquerda)
          180° = Alvo à ré (Atrás)
        - Elevation: inclinação vertical em relação ao plano do veículo
        - Range: distância euclidiana direta (metros)
    """
    pos = np.asarray(pos_body, dtype=float)
    x = pos[0]
    y = pos[1]
    z = pos[2] if len(pos) > 2 else 0.0

    r_horiz = np.sqrt(x**2 + y**2)
    slant_range = np.sqrt(r_horiz**2 + z**2)

    # Bearing horizontal relativo ao eixo de avanço (x)
    rel_bearing_deg = np.degrees(np.arctan2(y, x))

    # Elevação relativa (z positivo é para baixo em NED, logo elevação positiva é -z)
    rel_elevation_deg = np.degrees(np.arctan2(-z, r_horiz))

    return rel_bearing_deg, rel_elevation_deg, slant_range
