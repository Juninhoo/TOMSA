"""Filtro de Kalman Linear Cinemático e Suavizador Ótimo de Rauch-Tung-Striebel (RTS).

Implementa a reconstrução ótima em espaço de estados para as posições e velocidades
3D dos submarinos, utilizando o modelo CWNA (Continuous White Noise Acceleration)
e alimentando a matriz de covariância R com os valores medidos da telemetria ROS.
"""

from typing import Tuple, Optional
import numpy as np


class TrajectoryKalmanRTS:
    """Estimador e suavizador estocástico bidirecional de trajetórias 3D."""

    def __init__(self, process_noise_q: float = 0.05, default_meas_std: float = 0.8):
        """process_noise_q: quanto o filtro admite que a velocidade varia entre amostras (maior = segue mais as medidas).
        default_meas_std: desvio padrão da posição [m] quando o CSV não traz incerteza (maior = alisa mais)."""
        self.q = process_noise_q
        self.default_meas_std = default_meas_std

    def smooth(
        self,
        timestamps: np.ndarray,
        positions_ned: np.ndarray,
        meas_cov_stds: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Executa a passagem progressiva de Kalman e a passagem regressiva RTS.

        Args:
            timestamps: Array (N,) com tempos estritamente crescentes em segundos.
            positions_ned: Array (3, N) com [N, E, D] em metros.
            meas_cov_stds: Opcional, Array (N, 3) com [sigma_N, sigma_E, sigma_D] em metros.

        Returns:
            smoothed_pos: Array (3, N) de posições suavizadas.
            smoothed_vel: Array (3, N) de velocidades estimadas (vN, vE, vD) em m/s.
            smoothed_cov_diag: Array (N, 6) com as variâncias das posições e velocidades.
        """
        N_pts = len(timestamps)
        if N_pts < 2:
            zeros_v = np.zeros_like(positions_ned)
            zeros_cov = np.zeros((N_pts, 6))
            return positions_ned, zeros_v, zeros_cov

        # Alocação de memória para estruturas do Filtro de Kalman
        x_pred = np.zeros((N_pts, 6))
        P_pred = np.zeros((N_pts, 6, 6))
        x_filt = np.zeros((N_pts, 6))
        P_filt = np.zeros((N_pts, 6, 6))
        F_list = np.zeros((N_pts, 6, 6))

        # Matriz de observação H (observa posições N, E, D)
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        HT = H.T

        # Estado inicial (k=0)
        dt_init = max(1e-3, timestamps[1] - timestamps[0])
        v0 = (positions_ned[:, 1] - positions_ned[:, 0]) / dt_init
        x_filt[0] = np.array([
            positions_ned[0, 0], positions_ned[1, 0], positions_ned[2, 0],
            v0[0], v0[1], v0[2]
        ])

        P_filt[0] = np.diag([
            (meas_cov_stds[0, 0] if meas_cov_stds is not None else self.default_meas_std)**2,
            (meas_cov_stds[0, 1] if meas_cov_stds is not None else self.default_meas_std)**2,
            (meas_cov_stds[0, 2] if meas_cov_stds is not None else self.default_meas_std)**2,
            4.0, 4.0, 4.0  # Incerteza inicial da velocidade
        ])

        x_pred[0] = x_filt[0]
        P_pred[0] = P_filt[0]

        # -------------------------------------------------------------
        # 1. Passagem Progressiva (Forward Kalman Filter)
        # -------------------------------------------------------------
        I6 = np.eye(6)
        for k in range(1, N_pts):
            dt = max(1e-4, timestamps[k] - timestamps[k - 1])

            # Matriz de transição F
            F = np.eye(6)
            F[0, 3] = F[1, 4] = F[2, 5] = dt
            F_list[k] = F

            # Matriz de ruído de processo Q (CWNA)
            q = self.q
            dt2 = dt**2
            dt3 = dt**3
            Q = np.zeros((6, 6))
            for i in range(3):
                Q[i, i] = (dt3 / 3.0) * q
                Q[i, i + 3] = (dt2 / 2.0) * q
                Q[i + 3, i] = (dt2 / 2.0) * q
                Q[i + 3, i + 3] = dt * q

            # Predição
            x_p = F @ x_filt[k - 1]
            P_p = F @ P_filt[k - 1] @ F.T + Q

            x_pred[k] = x_p
            P_pred[k] = P_p

            # Medição e ruído R
            z_k = positions_ned[:, k]
            if meas_cov_stds is not None and not np.isnan(meas_cov_stds[k]).any():
                sig = np.clip(meas_cov_stds[k], 0.05, 20.0)
                R_k = np.diag(sig**2)
            else:
                R_k = np.eye(3) * (self.default_meas_std**2)

            # Atualização (Update)
            y_k = z_k - H @ x_p  # Inovação
            S_k = H @ P_p @ HT + R_k
            K_k = P_p @ HT @ np.linalg.inv(S_k)

            x_filt[k] = x_p + K_k @ y_k
            P_filt[k] = (I6 - K_k @ H) @ P_p

        # -------------------------------------------------------------
        # 2. Passagem Regressiva (Backward Rauch-Tung-Striebel Smoother)
        # -------------------------------------------------------------
        x_smooth = np.zeros((N_pts, 6))
        P_smooth = np.zeros((N_pts, 6, 6))

        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for k in range(N_pts - 2, -1, -1):
            P_p_next = P_pred[k + 1]
            F_next = F_list[k + 1]

            # Ganho do suavizador: C_k = P_filt[k] * F_{k+1}^T * (P_pred[k+1])^-1
            C_k = P_filt[k] @ F_next.T @ np.linalg.inv(P_p_next)

            x_smooth[k] = x_filt[k] + C_k @ (x_smooth[k + 1] - x_pred[k + 1])
            P_smooth[k] = P_filt[k] + C_k @ (P_smooth[k + 1] - P_p_next) @ C_k.T

        smoothed_pos = x_smooth[:, :3].T  # (3, N)
        smoothed_vel = x_smooth[:, 3:].T  # (3, N)
        smoothed_cov_diag = np.diagonal(P_smooth, axis1=1, axis2=2)  # (N, 6)

        return smoothed_pos, smoothed_vel, smoothed_cov_diag


def compute_longitudinal_speed(vel_ned: np.ndarray, yaw_deg: np.ndarray) -> np.ndarray:
    """Calcula a velocidade linear orientada na direção do movimento longitudinal (m/s).

    Args:
        vel_ned: Array (3, N) com [vN, vE, vD].
        yaw_deg: Array (N,) com o ângulo de proa em graus (Norte=0°, Este=90°).
    """
    yaw_rad = np.radians(yaw_deg)
    head_x = np.cos(yaw_rad)  # vetor unitário no plano horizontal N
    head_y = np.sin(yaw_rad)  # vetor unitário no plano horizontal E

    # Produto escalar com o vetor de velocidade horizontal
    v_long = vel_ned[0] * head_x + vel_ned[1] * head_y
    return v_long


def compute_turn_rate(yaw_deg: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Calcula a velocidade angular de viragem (yaw rate) em graus por segundo."""
    if len(timestamps) < 2:
        return np.zeros_like(yaw_deg)
    yaw_unwrapped = np.unwrap(np.radians(yaw_deg))
    dt = np.diff(timestamps)
    dt = np.where(dt <= 1e-6, 1e-3, dt)
    rates = np.diff(yaw_unwrapped) / dt
    yaw_rate = np.concatenate([[rates[0]], rates])
    return np.degrees(yaw_rate)
