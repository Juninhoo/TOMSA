"""Estruturas de Dados de Trajetória e Conjunto de Dados da Missão Multi-Robô.

Gere as trajetórias individuais e o alinhamento temporal global para replay sincronizado.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp

from multi_robot_system.core.math.proximity import evaluate_proximity_and_cpa, CPAResult


@dataclass
class RobotTrajectory:
    robot_id: str
    label: str
    color: str
    timestamps: np.ndarray          # (N,) segundos
    lat: np.ndarray                 # (N,) graus
    lon: np.ndarray                 # (N,) graus
    alt: np.ndarray                 # (N,) metros
    pos_ned: np.ndarray             # (3, N) [N, E, D] metros suavizados
    quaternions: np.ndarray         # (N, 4) [qx, qy, qz, qw]
    rot_matrices: np.ndarray        # (N, 3, 3)
    euler_deg: np.ndarray           # (N, 3) [roll, pitch, yaw] graus
    vel_ned: np.ndarray             # (3, N) [vN, vE, vD] m/s
    speed_longitudinal: np.ndarray  # (N,) m/s
    turn_rate_yaw: np.ndarray       # (N,) deg/s
    cov_diag: np.ndarray            # (N, 6)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def total_distance_m(self) -> float:
        diffs = np.diff(self.pos_ned[:2, :], axis=1)
        return float(np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2)))

    @property
    def max_depth_m(self) -> float:
        return float(np.max(self.pos_ned[2]))

    @property
    def mean_speed_m_s(self) -> float:
        return float(np.mean(np.abs(self.speed_longitudinal)))


class MissionDataset:
    """Gere as trajetórias conjuntas de múltiplos veículos com sincronização temporal."""

    def __init__(
        self,
        trajectories: Dict[str, RobotTrajectory],
        origin_lat: float,
        origin_lon: float,
        origin_h: float
    ):
        self.trajectories = trajectories
        self.origin = (origin_lat, origin_lon, origin_h)

        # Intervalo temporal global sobreposto
        self.common_start = max(traj.timestamps[0] for traj in trajectories.values())
        self.common_end = min(traj.timestamps[-1] for traj in trajectories.values())
        self.has_overlap = self.common_end > self.common_start

        # Tempo de referência de início absoluto (t0)
        self.t0_mission = min(traj.timestamps[0] for traj in trajectories.values())
        self.t_end_mission = max(traj.timestamps[-1] for traj in trajectories.values())

        # SLERP pré-calculado para cada robô para rotações suaves
        self._slerp_interpolators: Dict[str, Slerp] = {}
        for r_id, traj in self.trajectories.items():
            rotations = R.from_quat(traj.quaternions)
            self._slerp_interpolators[r_id] = Slerp(traj.timestamps, rotations)

    def get_state_at_time(self, t: float) -> Dict[str, Dict]:
        """Devolve o estado cinemático e atitude interpolados de todos os robôs no instante t."""
        states = {}
        for r_id, traj in self.trajectories.items():
            if t < traj.timestamps[0] or t > traj.timestamps[-1]:
                continue

            # Posições NED interpoladas
            N = float(np.interp(t, traj.timestamps, traj.pos_ned[0]))
            E = float(np.interp(t, traj.timestamps, traj.pos_ned[1]))
            D = float(np.interp(t, traj.timestamps, traj.pos_ned[2]))

            # Velocidades interpoladas
            vN = float(np.interp(t, traj.timestamps, traj.vel_ned[0]))
            vE = float(np.interp(t, traj.timestamps, traj.vel_ned[1]))
            vD = float(np.interp(t, traj.timestamps, traj.vel_ned[2]))

            # Atitude por SLERP
            rot = self._slerp_interpolators[r_id]([t])
            quat = rot.as_quat()[0]  # [qx, qy, qz, qw]
            euler = rot.as_euler("xyz", degrees=True)[0]  # [roll, pitch, yaw]
            R_mat = rot.as_matrix()[0]

            # Velocidade longitudinal e taxa angular
            v_long = float(np.interp(t, traj.timestamps, traj.speed_longitudinal))
            yaw_rate = float(np.interp(t, traj.timestamps, traj.turn_rate_yaw))

            states[r_id] = {
                "robot_id": r_id,
                "label": traj.label,
                "color": traj.color,
                "pos_ned": np.array([N, E, D]),
                "vel_ned": np.array([vN, vE, vD]),
                "quaternion": quat,
                "euler_deg": euler,
                "R_wb": R_mat,
                "speed_longitudinal": v_long,
                "turn_rate_yaw": yaw_rate,
                "depth_m": D
            }
        return states

    def evaluate_cpa_at_time(self, t: float, collision_threshold_m: float = 10.0) -> List[CPAResult]:
        """Calcula o estado de proximidade e CPA entre todos os robôs ativos no instante t."""
        states = self.get_state_at_time(t)
        return evaluate_proximity_and_cpa(states, collision_threshold_m=collision_threshold_m)

    def export_synchronized_csv(self, output_path: str, dt_step_s: float = 0.5) -> str:
        """Exporta um CSV mestre com todas as trajetórias sincronizadas na grelha temporal comum."""
        start_t = self.common_start if self.has_overlap else self.t0_mission
        end_t = self.common_end if self.has_overlap else self.t_end_mission
        times = np.arange(start_t, end_t + 1e-6, dt_step_s)

        rows = []
        for t in times:
            states = self.get_state_at_time(t)
            row = {"time_utc": t, "mission_elapsed_s": t - self.t0_mission}
            for r_id, st in states.items():
                prefix = r_id.replace(" ", "_").lower()
                row[f"{prefix}_north_m"] = st["pos_ned"][0]
                row[f"{prefix}_east_m"] = st["pos_ned"][1]
                row[f"{prefix}_depth_m"] = st["pos_ned"][2]
                row[f"{prefix}_speed_mps"] = st["speed_longitudinal"]
                row[f"{prefix}_heading_deg"] = st["euler_deg"][2]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return output_path
