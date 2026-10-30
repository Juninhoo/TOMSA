"""Estruturas de dados da missão.

RobotTrajectory guarda os dados de um robô; MissionDataset junta todos os robôs
e permite obter o estado de cada um em qualquer instante.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp

from multi_robot_system.core.math.proximity import evaluate_proximity_and_cpa, CPAResult


@dataclass
class RobotTrajectory:
    """Todos os dados de um robô, já suavizados.

    Cada campo é uma série com um valor por instante (timestamps, em segundos):
    posição geográfica e local NED (Norte, Este, profundidade), orientação, velocidades,
    incertezas e saltos de posição. Ângulos em graus, distâncias em metros.
    """
    robot_id: str
    label: str
    color: str
    timestamps: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    pos_ned: np.ndarray
    quaternions: np.ndarray
    rot_matrices: np.ndarray
    euler_deg: np.ndarray
    vel_ned: np.ndarray
    speed_longitudinal: np.ndarray
    turn_rate_yaw: np.ndarray
    cov_diag: np.ndarray
    vel_body_meas: Optional[np.ndarray] = None
    sigma_h_decl: Optional[np.ndarray] = None
    jump_events: Optional[list] = None
    pos_ned_raw: Optional[np.ndarray] = None

    @property
    def duration_s(self) -> float:
        """Duração do registo do robô, em segundos."""
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def total_distance_m(self) -> float:
        """Distância horizontal total percorrida pelo robô, em metros."""
        diffs = np.diff(self.pos_ned[:2, :], axis=1)
        return float(np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2)))

    @property
    def max_depth_m(self) -> float:
        """Profundidade máxima atingida pelo robô, em metros."""
        return float(np.max(self.pos_ned[2]))

    @property
    def mean_speed_m_s(self) -> float:
        """Velocidade média de avanço do robô, em m/s."""
        return float(np.mean(np.abs(self.speed_longitudinal)))


class MissionDataset:
    """Conjunto de todos os robôs da missão, com o tempo sincronizado."""

    def __init__(
        self,
        trajectories: Dict[str, RobotTrajectory],
        origin_lat: float,
        origin_lon: float,
        origin_h: float
    ):
        self.trajectories = trajectories
        self.origin = (origin_lat, origin_lon, origin_h)

        # Janela de tempo comum a todos os robôs e duração total da missão
        self.common_start = max(traj.timestamps[0] for traj in trajectories.values())
        self.common_end = min(traj.timestamps[-1] for traj in trajectories.values())
        self.has_overlap = self.common_end > self.common_start

        self.t0_mission = min(traj.timestamps[0] for traj in trajectories.values())
        self.t_end_mission = max(traj.timestamps[-1] for traj in trajectories.values())

        # Prepara a interpolação da orientação de cada robô
        self._slerp_interpolators: Dict[str, Slerp] = {}
        for r_id, traj in self.trajectories.items():
            rotations = R.from_quat(traj.quaternions)
            self._slerp_interpolators[r_id] = Slerp(traj.timestamps, rotations)

    def get_state_at_time(self, t: float) -> Dict[str, Dict]:
        """Esta função devolve o estado de cada robô ativo no instante t.

        Interpola a posição, a velocidade, a orientação (SLERP) e as velocidades de avanço e rotação.
        """
        states = {}
        for r_id, traj in self.trajectories.items():
            if t < traj.timestamps[0] or t > traj.timestamps[-1]:
                continue

            # Interpolação da posição, velocidade, orientação e velocidades no instante t
            N = float(np.interp(t, traj.timestamps, traj.pos_ned[0]))
            E = float(np.interp(t, traj.timestamps, traj.pos_ned[1]))
            D = float(np.interp(t, traj.timestamps, traj.pos_ned[2]))

            vN = float(np.interp(t, traj.timestamps, traj.vel_ned[0]))
            vE = float(np.interp(t, traj.timestamps, traj.vel_ned[1]))
            vD = float(np.interp(t, traj.timestamps, traj.vel_ned[2]))

            rot = self._slerp_interpolators[r_id]([t])
            quat = rot.as_quat()[0]
            euler = rot.as_euler("xyz", degrees=True)[0]
            R_mat = rot.as_matrix()[0]

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
        """Calcula a proximidade e o CPA entre todos os robôs ativos no instante t."""
        states = self.get_state_at_time(t)
        return evaluate_proximity_and_cpa(states, collision_threshold_m=collision_threshold_m)

    def export_synchronized_csv(self, output_path: str, dt_step_s: float = 0.5) -> str:
        """Exporta um CSV com as posições de todos os robôs numa grelha de tempo regular."""
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
