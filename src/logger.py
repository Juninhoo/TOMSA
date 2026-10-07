"""Módulo de Registo (Logging) e Exportação de Métricas da Missão.

Regista a execução do sistema, gera ficheiros de log e exporta resumos de telemetria
(distâncias percorridas, velocidades, distâncias mínimas inter-robôs / CPA).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from src.data_loader import MissionDataset


def setup_logger(log_file_path: str = "output/mission_execution.log") -> logging.Logger:
    """Configura o sistema de logging com saída para ficheiro e consola."""
    path = Path(log_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("TOMSA")
    logger.setLevel(logging.INFO)

    # Limpar handlers anteriores para evitar duplicações
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def calculate_mission_metrics(dataset: MissionDataset, dt: float = 1.0) -> Dict:
    """Calcula estatísticas de desempenho, distâncias totais e aproximações mínimas (CPA)."""
    metrics = {
        "mission": {
            "origin": {
                "latitude": dataset.origin[0],
                "longitude": dataset.origin[1],
                "altitude": dataset.origin[2]
            },
            "num_robots": len(dataset.trajectories),
            "common_overlap_duration_s": float(dataset.common_end - dataset.common_start),
            "common_start_timestamp": float(dataset.common_start),
            "common_end_timestamp": float(dataset.common_end)
        },
        "robots": {},
        "inter_robot_proximity": []
    }

    # Estatísticas individuais por robô
    for r_id, traj in dataset.trajectories.items():
        speeds = np.linalg.norm(traj.linear_vel, axis=1)
        metrics["robots"][r_id] = {
            "label": traj.label,
            "total_points": int(len(traj.timestamps)),
            "duration_s": traj.duration,
            "total_distance_m": round(traj.total_distance, 2),
            "mean_speed_ms": round(float(np.mean(speeds)), 3),
            "max_speed_ms": round(float(np.max(speeds)), 3),
            "start_coords_deg": [float(traj.lat[0]), float(traj.lon[0]), float(traj.alt[0])],
            "end_coords_deg": [float(traj.lat[-1]), float(traj.lon[-1]), float(traj.alt[-1])]
        }

    # Análise de proximidade entre todos os pares durante o intervalo comum
    robot_ids = list(dataset.trajectories.keys())
    t_grid = np.arange(dataset.common_start, dataset.common_end, dt)

    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            r_a = robot_ids[i]
            r_b = robot_ids[j]

            distances = []
            for t in t_grid:
                st = dataset.get_state_at(t)
                pa = st[r_a]["pos_ned"]
                pb = st[r_b]["pos_ned"]
                dist = np.linalg.norm(pa - pb)
                distances.append(dist)

            distances = np.array(distances)
            min_dist_idx = np.argmin(distances)
            min_dist = float(distances[min_dist_idx])
            cpa_time = float(t_grid[min_dist_idx])

            metrics["inter_robot_proximity"].append({
                "robot_a": r_a,
                "robot_b": r_b,
                "closest_point_of_approach_m": round(min_dist, 2),
                "cpa_relative_time_s": round(cpa_time - dataset.common_start, 1),
                "cpa_timestamp": round(cpa_time, 2),
                "mean_separation_m": round(float(np.mean(distances)), 2)
            })

    return metrics


def export_mission_reports(
    dataset: MissionDataset,
    metrics_path: str = "output/mission_metrics.json",
    csv_path: str = "output/synchronized_trajectories.csv",
    logger: Optional[logging.Logger] = None
) -> Tuple[str, str]:
    """Gera e grava o JSON de métricas e o CSV sincronizado."""
    if logger is None:
        logger = setup_logger()

    logger.info("A calcular métricas globais e aproximações mínimas (CPA)...")
    metrics = calculate_mission_metrics(dataset)

    # Gravar JSON
    m_path = Path(metrics_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas gravadas em: {metrics_path}")

    # Gravar CSV sincronizado
    logger.info("A gerar e exportar série temporal sincronizada...")
    _, df_sync = dataset.generate_synchronized_timeseries(dt=1.0)
    c_path = Path(csv_path)
    c_path.parent.mkdir(parents=True, exist_ok=True)
    df_sync.to_csv(c_path, index=False)
    logger.info(f"Trajetórias sincronizadas exportadas em: {csv_path} ({len(df_sync)} amostras)")

    return str(m_path), str(c_path)
