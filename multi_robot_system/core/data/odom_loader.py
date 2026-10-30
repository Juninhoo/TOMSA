"""Leitura dos ficheiros de telemetria dos robôs (CSV de odometria ROS).

Lê cada ficheiro sem o modificar, ordena os dados por tempo, converte as posições
para coordenadas locais (NED), calcula a orientação e suaviza a trajetória.
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


# Colunas obrigatórias no ficheiro CSV
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
    """Extrai a incerteza da posição (desvio-padrão em x, y e z) da matriz de covariância de cada linha."""
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


def parse_declared_horizontal_sigma(cov_series: pd.Series) -> np.ndarray:
    """Calcula a incerteza horizontal indicada no ficheiro, em metros: √(var_x + var_y)."""
    out = np.zeros(len(cov_series))
    for k, entry in enumerate(cov_series):
        if not isinstance(entry, str):
            continue
        parts = entry.strip("() \n\r\t").split(",")
        try:
            out[k] = float(np.sqrt(max(0.0, float(parts[0])) + max(0.0, float(parts[7]))))
        except (IndexError, ValueError):
            pass
    return out


def detect_position_jumps(timestamps: np.ndarray, pos_ned_raw: np.ndarray,
                          max_speed_m_s: float = 2.5, group_s: float = 5.0) -> List[Dict]:
    """Esta função deteta saltos de posição nos dados brutos.

    Um salto é um deslocamento entre duas medidas que exigiria uma velocidade acima de
    max_speed_m_s (estes robôs andam a menos de 1 m/s). Saltos próximos no tempo contam como um só.
    """
    t = np.asarray(timestamps, dtype=float)
    if len(t) < 3:
        return []
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(pos_ned_raw, axis=1), axis=0)
    ok = dt > 1e-6
    idx = np.where(ok & (step / np.where(ok, dt, 1.0) > max_speed_m_s))[0]
    groups, cur = [], []
    for j in idx:
        if cur and t[j] - t[cur[-1]] > group_s:
            groups.append(cur)
            cur = []
        cur.append(j)
    if cur:
        groups.append(cur)
    events = []
    for g in groups:
        j0, j1 = g[0], g[-1] + 1
        events.append({
            "t": float(t[j0]),
            "m": float(np.linalg.norm(pos_ned_raw[:, j1] - pos_ned_raw[:, j0])),
            "depth_m": float(pos_ned_raw[2, j0]),
        })
    return events


def compute_reference_origin(file_paths: List[str]) -> Tuple[float, float, float]:
    """Calcula a origem das coordenadas locais a partir dos dados.

    Latitude e longitude: média de todos os pontos de todos os robôs.
    Altitude: nível da superfície da água, estimado com as primeiras medidas de cada robô.
    """
    lats, lons, h_refs = [], [], []
    for fp in file_paths:
        df = pd.read_csv(fp, usecols=["pose.pose.position.x", "pose.pose.position.y", "pose.pose.position.z"])
        lats.append(df["pose.pose.position.x"].values)
        lons.append(df["pose.pose.position.y"].values)
        h_refs.append(float(df["pose.pose.position.z"].iloc[:50].median()))
    return float(np.nanmean(np.concatenate(lats))), float(np.nanmean(np.concatenate(lons))), float(np.nanmean(h_refs))


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
    """Esta função carrega os dados de um robô a partir do seu ficheiro CSV.

    Converte as posições para NED, calcula a orientação, suaviza a trajetória (Kalman + RTS)
    e calcula as velocidades. Devolve um dicionário com todas as séries de dados.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro de telemetria não encontrado: {file_path}")

    # Leitura do ficheiro, verificação das colunas e ordenação por tempo
    df = pd.read_csv(path)

    for col in ODOM_REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Ficheiro {path.name} não contém a coluna obrigatória '{col}'")

    # Ignora as linhas com valores em falta ou ilegíveis (com um aviso)
    df[ODOM_REQUIRED_COLS] = df[ODOM_REQUIRED_COLS].apply(pd.to_numeric, errors="coerce")
    validas = np.isfinite(df[ODOM_REQUIRED_COLS].values).all(axis=1)
    if not validas.all():
        print(f"     [AVISO] {path.name}: {int((~validas).sum())} linhas com valores em falta foram ignoradas")
        df = df[validas]
    if len(df) < 2:
        raise ValueError(f"Ficheiro {path.name}: menos de 2 linhas válidas")

    df = df.sort_values(by="Time", kind="stable").reset_index(drop=True)

    # Registos com o mesmo instante: afastados 1 ms (a interpolação exige tempos sempre crescentes)
    tempos = df["Time"].values.astype(float)
    if np.any(np.diff(tempos) <= 0):
        n_rep = int(np.sum(np.diff(tempos) <= 0))
        for i in range(1, len(tempos)):
            if tempos[i] <= tempos[i - 1]:
                tempos[i] = tempos[i - 1] + 0.001
        df["Time"] = tempos
        print(f"     [AVISO] {path.name}: {n_rep} registos com instante repetido foram afastados 1 ms")

    # Posições: latitude/longitude/altitude convertidas para NED
    timestamps = df["Time"].values.astype(float)
    lat_deg = df["pose.pose.position.x"].values.astype(float)
    lon_deg = df["pose.pose.position.y"].values.astype(float)
    alt_m = df["pose.pose.position.z"].values.astype(float)

    N, E, D = geodetic_to_ned(lat_deg, lon_deg, alt_m, origin_lat, origin_lon, origin_h)
    pos_ned_raw = np.stack([N, E, D], axis=0)

    # Orientação: quaterniões, ângulos de Euler e matrizes de rotação
    qx = df["pose.pose.orientation.x"].values.astype(float)
    qy = df["pose.pose.orientation.y"].values.astype(float)
    qz = df["pose.pose.orientation.z"].values.astype(float)
    qw = df["pose.pose.orientation.w"].values.astype(float)
    qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)
    quats = np.stack([qx, qy, qz, qw], axis=1)

    roll_deg, pitch_deg, yaw_deg = quaternion_to_euler_zyx(qx, qy, qz, qw, degrees=True)
    euler_deg = np.stack([roll_deg, pitch_deg, yaw_deg], axis=1)

    rot_matrices = quaternion_to_rotation_matrix(quats)

    # Incerteza das medidas, seguida da suavização da trajetória (ou derivada simples se desativada)
    if "pose.covariance" in df.columns:
        cov_stds = parse_covariance_diagonal(df["pose.covariance"])
    else:
        cov_stds = np.full((len(timestamps), 3), 0.8)

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

    # Velocidade medida pelo próprio robô, incerteza declarada e saltos de posição (usados no Cockpit)
    twist_cols = ["twist.twist.linear.x", "twist.twist.linear.y", "twist.twist.linear.z"]
    if all(c in df.columns for c in twist_cols):
        vel_body_meas = df[twist_cols].values.astype(float).T
    else:
        vel_body_meas = np.zeros((3, len(timestamps)))

    if "pose.covariance" in df.columns:
        sigma_h_decl = parse_declared_horizontal_sigma(df["pose.covariance"])
    else:
        sigma_h_decl = np.zeros(len(timestamps))
    jump_events = detect_position_jumps(timestamps, pos_ned_raw)

    # Velocidade de avanço e velocidade de rotação
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
        "cov_diag": cov_diag_smooth,
        "vel_body_meas": vel_body_meas,
        "sigma_h_decl": sigma_h_decl,
        "jump_events": jump_events
    }
