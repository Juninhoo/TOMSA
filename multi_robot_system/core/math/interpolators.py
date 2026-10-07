"""Comparador Rigoroso de Interpoladores de Trajetória: Linear vs Spline Cúbica vs Kalman/RTS.

Quantifica o desempenho dos três métodos sobre os dados reais de telemetria do INESC TEC,
calculando métricas de erro numérico (RMSE, Desvio Máximo e Tempo Computacional).
"""

import time
from typing import Dict, Tuple
import numpy as np
from scipy.interpolate import CubicSpline

from multi_robot_system.core.math.kalman_rts import TrajectoryKalmanRTS


def compare_interpolators_on_trajectory(
    timestamps: np.ndarray,
    positions_ned: np.ndarray,
    subsample_factor: int = 5
) -> Dict[str, Dict[str, float]]:
    """Compara os três métodos retirando pontos para validação cruzada (subamostragem).

    Args:
        timestamps: Array (N,) com tempos.
        positions_ned: Array (3, N) com [N, E, D].
        subsample_factor: Fator de subamostragem (treina com 1 a cada K pontos e testa nos restantes).

    Returns:
        Dicionário com métricas de cada método: RMSE (m), Max_Dev (m) e Time_ms (ms).
    """
    N_total = len(timestamps)
    if N_total < 20:
        return {}

    # Subconjunto de treino e teste
    train_idx = np.arange(0, N_total, subsample_factor)
    test_mask = np.ones(N_total, dtype=bool)
    test_mask[train_idx] = False
    test_idx = np.where(test_mask)[0]

    t_train = timestamps[train_idx]
    pos_train = positions_ned[:, train_idx]

    t_test = timestamps[test_idx]
    pos_ground_truth = positions_ned[:, test_idx]

    results = {}

    # 1. Interpolação Linear por Partes
    t0 = time.perf_counter()
    lin_pred = np.empty_like(pos_ground_truth)
    for dim in range(3):
        lin_pred[dim] = np.interp(t_test, t_train, pos_train[dim])
    dt_linear = (time.perf_counter() - t0) * 1000.0

    err_lin = np.linalg.norm(lin_pred - pos_ground_truth, axis=0)
    results["Linear"] = {
        "rmse_m": float(np.sqrt(np.mean(err_lin**2))),
        "max_err_m": float(np.max(err_lin)),
        "mean_err_m": float(np.mean(err_lin)),
        "compute_time_ms": dt_linear,
        "continuity": "C0"
    }

    # 2. Spline Cúbica Natural (C2)
    t0 = time.perf_counter()
    cs_pred = np.empty_like(pos_ground_truth)
    for dim in range(3):
        spline = CubicSpline(t_train, pos_train[dim], bc_type="natural")
        cs_pred[dim] = spline(t_test)
    dt_spline = (time.perf_counter() - t0) * 1000.0

    err_spline = np.linalg.norm(cs_pred - pos_ground_truth, axis=0)
    results["Spline Cúbica"] = {
        "rmse_m": float(np.sqrt(np.mean(err_spline**2))),
        "max_err_m": float(np.max(err_spline)),
        "mean_err_m": float(np.mean(err_spline)),
        "compute_time_ms": dt_spline,
        "continuity": "C2"
    }

    # 3. Filtro de Kalman + Suavizador RTS
    t0 = time.perf_counter()
    smoother = TrajectoryKalmanRTS(process_noise_q=0.08, default_meas_std=0.5)
    smooth_pos, _, _ = smoother.smooth(t_train, pos_train)
    # Interpolação no teste a partir do estado suave
    rts_pred = np.empty_like(pos_ground_truth)
    for dim in range(3):
        spline_rts = CubicSpline(t_train, smooth_pos[dim], bc_type="natural")
        rts_pred[dim] = spline_rts(t_test)
    dt_rts = (time.perf_counter() - t0) * 1000.0

    err_rts = np.linalg.norm(rts_pred - pos_ground_truth, axis=0)
    results["Kalman + RTS"] = {
        "rmse_m": float(np.sqrt(np.mean(err_rts**2))),
        "max_err_m": float(np.max(err_rts)),
        "mean_err_m": float(np.mean(err_rts)),
        "compute_time_ms": dt_rts,
        "continuity": "C1 Estocástica (Ótima)"
    }

    return results


def format_benchmark_table_markdown(results: Dict[str, Dict[str, float]]) -> str:
    """Gera uma tabela formatada em Markdown com o resultado do benchmark."""
    header = "| Método | Classe de Continuidade | RMSE (m) | Desvio Máx. (m) | Erro Médio (m) | Tempo (ms) |\n"
    separator = "|---|---|---|---|---|---|\n"
    rows = []
    for method, m in results.items():
        rows.append(
            f"| **{method}** | {m.get('continuity', '')} | {m['rmse_m']:.4f} m | "
            f"{m['max_err_m']:.4f} m | {m['mean_err_m']:.4f} m | {m['compute_time_ms']:.2f} ms |"
        )
    return header + separator + "\n".join(rows)
