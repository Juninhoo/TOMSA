"""Carregador Robusto de Telemetria de Odometria ROS (nav_msgs/Odometry em CSV).

Lê os ficheiros de odometria do INESC TEC sem alterar os ficheiros originais no disco.
Trata colunas, ordena cronologicamente preservando registos simultâneos,
extrai as matrizes de covariância e converte para coordenadas NED/ENU locais.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from multi_robot_system.core.math.geodesy import geodetic_to_ned, geodetic_to_enu
from multi_robot_system.core.math.rotations import (
    normalize_quaternion,
    quaternion_to_euler_zyx,
    quaternion_to_rotation_matrix
)
from multi_robot_system.core.math.kalman_rts import (
    TrajectoryKalmanRTS,
    compute_longitudinal_speed,
    compute_turn_rate
)


# Colunas necessárias da mensagem ROS Odometry
ODOM_REQUIRED_COLS = [
    "Time",
    "pose.pose.position.x",
    "pose.pose.position.y",
    "pose.pose.position.z",
    "pose.pose.orientation.x",
    "pose.pose.orientation.y",
    "pose.pose.orientation.z",
    "pose.pose.orientation.w"
]


def parse_covariance_diagonal(cov_series: pd.Series) -> np.ndarray:
    """Extrai os desvios-padrão [sigma_x, sigma_y, sigma_z] a partir da diagonal da covariância (36 elementos)."""
    stds = []
    for entry in cov_series:
        if not isinstance(entry, str):
            stds.append([0.8, 0.8, 0.8])
            continue
        clean = entry.strip("() \n\r\t")
        parts = clean.split(",")
        try:
            var_x = float(parts[0])
            var_y = float(parts[7])
            var_z = float(parts[14])
            stds.append([
                float(np.sqrt(max(0.001, var_x))),
                float(np.sqrt(max(0.001, var_y))),
                float(np.sqrt(max(0.001, var_z)))
            ])
        except (IndexError, ValueError):
            stds.append([0.8, 0.8, 0.8])
    return np.array(stds, dtype=float)


def compute_reference_origin(file_paths: List[str]) -> Tuple[float, float, float]:
    """Origem local calculada a partir dos dados, igual à do visualizador de referência (trajectories/).

    - lat0, lon0: média de todas as amostras de todos os robôs.
    - h_ref (superfície da água): média, entre robôs, da mediana das primeiras 50 amostras de altura.
    """
    lats, lons, h_refs = [], [], []
    for fp in file_paths:
        df = pd.read_csv(fp, usecols=["pose.pose.position.x", "pose.pose.position.y", "pose.pose.position.z"])
        lats.append(df["pose.pose.position.x"].values)
        lons.append(df["pose.pose.position.y"].values)
        h_refs.append(float(df["pose.pose.position.z"].iloc[:50].median()))
    return float(np.mean(np.concatenate(lats))), float(np.mean(np.concatenate(lons))), float(np.mean(h_refs))


def load_single_robot_csv(
    file_path: str,
    robot_id: str,
    label: str,
    color: str,
    origin_lat: float,
    origin_lon: float,
    origin_h: float,
    apply_rts: bool = True,
    process_noise_q: float = 0.05
) -> Dict:
    """Carrega um ficheiro CSV de odometria, ordena por tempo e aplica suavização RTS em memória.

    O ficheiro original em disco permanece 100% inalterado.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro de telemetria não encontrado: {file_path}")

    # Leitura otimizada
    df = pd.read_csv(path)

    # Validação de cabeçalho
    for col in ODOM_REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Ficheiro {path.name} não contém a coluna obrigatória '{col}'")

    # Ordenação estrita por tempo (preservando registos repetidos conforme Q23)
    df = df.sort_values(by="Time", kind="stable").reset_index(drop=True)

    timestamps = df["Time"].values.astype(float)
    lat_deg = df["pose.pose.position.x"].values.astype(float)
    lon_deg = df["pose.pose.position.y"].values.astype(float)
    alt_m = df["pose.pose.position.z"].values.astype(float)

    # Conversão Geodésica para NED e ENU
    N, E, D = geodetic_to_ned(lat_deg, lon_deg, alt_m, origin_lat, origin_lon, origin_h)
    pos_ned_raw = np.stack([N, E, D], axis=0)  # (3, N)

    # Quaterniões e Atitude
    qx = df["pose.pose.orientation.x"].values.astype(float)
    qy = df["pose.pose.orientation.y"].values.astype(float)
    qz = df["pose.pose.orientation.z"].values.astype(float)
    qw = df["pose.pose.orientation.w"].values.astype(float)
    qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)
    quats = np.stack([qx, qy, qz, qw], axis=1)  # (N, 4)

    roll_deg, pitch_deg, yaw_deg = quaternion_to_euler_zyx(qx, qy, qz, qw, degrees=True)
    euler_deg = np.stack([roll_deg, pitch_deg, yaw_deg], axis=1)

    # Matrizes de rotação 3D
    rot_matrices = quaternion_to_rotation_matrix(quats)  # (N, 3, 3)

    # Incerteza da telemetria a partir de pose.covariance
    if "pose.covariance" in df.columns:
        cov_stds = parse_covariance_diagonal(df["pose.covariance"])
    else:
        cov_stds = np.full((len(timestamps), 3), 0.8)

    # Suavização com Filtro de Kalman + Suavizador RTS
    if apply_rts and len(timestamps) > 5:
        smoother = TrajectoryKalmanRTS(process_noise_q=process_noise_q)
        pos_ned_smooth, vel_ned_smooth, cov_diag_smooth = smoother.smooth(
            timestamps, pos_ned_raw, cov_stds
        )
    else:
        pos_ned_smooth = pos_ned_raw
        dt_v = np.gradient(timestamps)
        dt_v = np.where(dt_v <= 0, 1e-4, dt_v)
        vel_ned_smooth = np.gradient(pos_ned_raw, dt_v, axis=1)
        cov_diag_smooth = np.zeros((len(timestamps), 6))

    # Velocidade linear longitudinal e velocidade angular de viragem (Q28)
    v_long = compute_longitudinal_speed(vel_ned_smooth, yaw_deg)
    yaw_rate = compute_turn_rate(yaw_deg, timestamps)

    return {
        "robot_id": robot_id,
        "label": label,
        "color": color,
        "timestamps": timestamps,
        "lat": lat_deg,
        "lon": lon_deg,
        "alt": alt_m,
        "pos_ned": pos_ned_smooth,
        "pos_ned_raw": pos_ned_raw,
        "quaternions": quats,
        "rot_matrices": rot_matrices,
        "euler_deg": euler_deg,
        "vel_ned": vel_ned_smooth,
        "speed_longitudinal": v_long,
        "turn_rate_yaw": yaw_rate,
        "cov_diag": cov_diag_smooth
    }
