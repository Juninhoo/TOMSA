"""Módulo de Carregamento, Pré-Processamento e Sincronização Temporal dos Dados.

Lê ficheiros CSV de mensagens de Odometria ROS anonimizadas da missão INESC TEC (Aguçadoura).
Executa:
1. Extração de Latitude, Longitude, Altitude e Atitude (Quaternions).
2. Conversão para referenciais cartesianos locais NED e ENU em relação à origem da missão.
3. Extração e cálculo de matrizes de covariância e velocidades.
4. Sincronização temporal e interpolação (linear para posições, SLERP para rotações).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp

from src.coordinates import geodetic_to_ned, geodetic_to_enu
from src.frames import quaternion_to_euler, normalize_quaternion


@dataclass
class RobotTrajectory:
    robot_id: str
    label: str
    color: str
    timestamps: np.ndarray       # (N,) segundos
    lat: np.ndarray              # (N,) graus
    lon: np.ndarray              # (N,) graus
    alt: np.ndarray              # (N,) metros
    pos_ned: np.ndarray          # (3, N) [N, E, D] metros
    pos_enu: np.ndarray          # (3, N) [E, N, U] metros
    quaternions: np.ndarray      # (N, 4) [qx, qy, qz, qw]
    euler_deg: np.ndarray        # (N, 3) [roll, pitch, yaw] graus
    linear_vel: np.ndarray       # (N, 3) [vx, vy, vz] m/s
    angular_vel: np.ndarray      # (N, 3) [wx, wy, wz] rad/s
    pos_std_ned: np.ndarray      # (N, 3) [sigma_N, sigma_E, sigma_D] metros

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def total_distance(self) -> float:
        diffs = np.diff(self.pos_ned[:2, :], axis=1)
        return float(np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2)))

    @property
    def mean_speed(self) -> float:
        return float(np.mean(np.linalg.norm(self.linear_vel, axis=1)))


class MissionDataset:
    """Gere as trajetórias de múltiplos robôs com sincronização temporal."""

    def __init__(
        self,
        trajectories: Dict[str, RobotTrajectory],
        origin_lat: float,
        origin_lon: float,
        origin_alt: float
    ):
        self.trajectories = trajectories
        self.origin = (origin_lat, origin_lon, origin_alt)

        # Determinar intervalo de tempo sobreposto comum
        self.common_start = max(traj.timestamps[0] for traj in trajectories.values())
        self.common_end = min(traj.timestamps[-1] for traj in trajectories.values())
        self.has_overlap = self.common_end > self.common_start

        # Interpoladores temporais pré-calculados para cada robô
        self._slerps: Dict[str, Slerp] = {}
        for r_id, traj in self.trajectories.items():
            rotations = R.from_quat(traj.quaternions)
            self._slerps[r_id] = Slerp(traj.timestamps, rotations)

    def get_state_at(self, t: float) -> Dict[str, Dict]:
        """Devolve o estado interpolado (posição, orientação, velocidade) de todos os robôs no instante t."""
        states = {}
        for r_id, traj in self.trajectories.items():
            if t < traj.timestamps[0] or t > traj.timestamps[-1]:
                continue

            # Interpolação linear da posição NED
            n = float(np.interp(t, traj.timestamps, traj.pos_ned[0]))
            e = float(np.interp(t, traj.timestamps, traj.pos_ned[1]))
            d = float(np.interp(t, traj.timestamps, traj.pos_ned[2]))

            # Interpolação SLERP da atitude
            rot = self._slerps[r_id]([t])
            quat = rot.as_quat()[0]  # [qx, qy, qz, qw]
            euler = rot.as_euler("xyz", degrees=True)[0]  # [roll, pitch, yaw]

            # Velocidades lineares interpoladas
            vx = float(np.interp(t, traj.timestamps, traj.linear_vel[:, 0]))
            vy = float(np.interp(t, traj.timestamps, traj.linear_vel[:, 1]))
            vz = float(np.interp(t, traj.timestamps, traj.linear_vel[:, 2]))

            # Incerteza interpolada
            sig_n = float(np.interp(t, traj.timestamps, traj.pos_std_ned[:, 0]))
            sig_e = float(np.interp(t, traj.timestamps, traj.pos_std_ned[:, 1]))

            states[r_id] = {
                "pos_ned": np.array([n, e, d]),
                "pos_enu": np.array([e, n, -d]),
                "quat": quat,
                "euler_deg": euler,
                "heading_deg": euler[2] % 360.0,
                "linear_vel": np.array([vx, vy, vz]),
                "pos_std_ned": np.array([sig_n, sig_e]),
                "color": traj.color,
                "label": traj.label
            }
        return states

    def generate_synchronized_timeseries(self, dt: float = 1.0) -> Tuple[np.ndarray, pd.DataFrame]:
        """Gera uma tabela regular sincronizada com passo dt para exportação e análise."""
        if not self.has_overlap:
            # Se não houver sobreposição, usa o intervalo global
            t_min = min(traj.timestamps[0] for traj in self.trajectories.values())
            t_max = max(traj.timestamps[-1] for traj in self.trajectories.values())
        else:
            t_min = self.common_start
            t_max = self.common_end

        t_grid = np.arange(t_min, t_max + 1e-5, dt)
        rows = []

        for t in t_grid:
            states = self.get_state_at(t)
            row = {"time": t, "rel_time": t - t_min}
            for r_id, state in states.items():
                p = state["pos_ned"]
                h = state["heading_deg"]
                row[f"{r_id}_N"] = p[0]
                row[f"{r_id}_E"] = p[1]
                row[f"{r_id}_D"] = p[2]
                row[f"{r_id}_heading"] = h
                row[f"{r_id}_speed"] = float(np.linalg.norm(state["linear_vel"]))
            rows.append(row)

        df_sync = pd.DataFrame(rows)
        return t_grid, df_sync


def load_robot_csv(
    file_path: str,
    robot_id: str,
    label: str,
    color: str,
    lat0: float,
    lon0: float,
    h0: float
) -> RobotTrajectory:
    """Lê um ficheiro CSV de odometria do robô e calcula todas as coordenadas."""
    path = Path(file_path)
    if not path.is_file():
        for candidate in [
            Path("data") / file_path,
            Path("data") / Path(file_path).name,
            Path("multi_robot_system/data") / file_path,
            Path("multi_robot_system/data") / Path(file_path).name,
        ]:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"Ficheiro de dados não encontrado: {file_path}")

    # Leitura otimizada do CSV com pandas
    cols_to_use = [
        "Time",
        "pose.pose.position.x",
        "pose.pose.position.y",
        "pose.pose.position.z",
        "pose.pose.orientation.x",
        "pose.pose.orientation.y",
        "pose.pose.orientation.z",
        "pose.pose.orientation.w",
        "twist.twist.linear.x",
        "twist.twist.linear.y",
        "twist.twist.linear.z",
        "twist.twist.angular.x",
        "twist.twist.angular.y",
        "twist.twist.angular.z",
        "pose.covariance"
    ]

    df = pd.read_csv(path, usecols=cols_to_use)

    # Ordenar rigorosamente por timestamp
    df = df.sort_values(by="Time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Time"]).reset_index(drop=True)

    timestamps = df["Time"].values.astype(float)
    lat = df["pose.pose.position.x"].values.astype(float)
    lon = df["pose.pose.position.y"].values.astype(float)
    alt = df["pose.pose.position.z"].values.astype(float)

    # Converter geodésico para NED e ENU
    N, E, D = geodetic_to_ned(lat, lon, alt, lat0, lon0, h0)
    pos_ned = np.stack([N, E, D], axis=0)
    pos_enu = np.stack([E, N, -D], axis=0)

    # Quaternions
    qx = df["pose.pose.orientation.x"].values.astype(float)
    qy = df["pose.pose.orientation.y"].values.astype(float)
    qz = df["pose.pose.orientation.z"].values.astype(float)
    qw = df["pose.pose.orientation.w"].values.astype(float)
    qx, qy, qz, qw = normalize_quaternion(qx, qy, qz, qw)
    quats = np.stack([qx, qy, qz, qw], axis=1)

    # Ângulos de Euler
    roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw, degrees=True)
    euler_deg = np.stack([roll, pitch, yaw], axis=1)

    # Velocidades
    vx = df["twist.twist.linear.x"].values.astype(float)
    vy = df["twist.twist.linear.y"].values.astype(float)
    vz = df["twist.twist.linear.z"].values.astype(float)
    linear_vel = np.stack([vx, vy, vz], axis=1)

    wx = df["twist.twist.angular.x"].values.astype(float)
    wy = df["twist.twist.angular.y"].values.astype(float)
    wz = df["twist.twist.angular.z"].values.astype(float)
    angular_vel = np.stack([wx, wy, wz], axis=1)

    # Extração rápida da diagonal da covariância (var_x, var_y, var_z)
    def parse_diag_cov(cov_str):
        clean = cov_str.strip("() \n\r")
        parts = clean.split(",")
        var_x = float(parts[0])
        var_y = float(parts[7])
        var_z = float(parts[14])
        return [np.sqrt(max(0.0, var_x)), np.sqrt(max(0.0, var_y)), np.sqrt(max(0.0, var_z))]

    cov_stds = np.array([parse_diag_cov(c) for c in df["pose.covariance"].values])

    return RobotTrajectory(
        robot_id=robot_id,
        label=label,
        color=color,
        timestamps=timestamps,
        lat=lat,
        lon=lon,
        alt=alt,
        pos_ned=pos_ned,
        pos_enu=pos_enu,
        quaternions=quats,
        euler_deg=euler_deg,
        linear_vel=linear_vel,
        angular_vel=angular_vel,
        pos_std_ned=cov_stds
    )


def load_mission_dataset(config_path: str = "config/config.yaml") -> Tuple[MissionDataset, "MissionConfig"]:
    """Função utilitária de conveniência para carregar o conjunto completo da missão."""
    from src.config_loader import load_config
    config = load_config(config_path)

    trajectories: Dict[str, RobotTrajectory] = {}
    for r in config.robots:
        traj = load_robot_csv(
            file_path=r.file,
            robot_id=r.id,
            label=r.label,
            color=r.color,
            lat0=config.origin.latitude,
            lon0=config.origin.longitude,
            h0=config.origin.altitude
        )
        trajectories[r.id] = traj

    dataset = MissionDataset(
        trajectories=trajectories,
        origin_lat=config.origin.latitude,
        origin_lon=config.origin.longitude,
        origin_alt=config.origin.altitude
    )

    return dataset, config
