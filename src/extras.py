"""Módulo de Funcionalidades Avançadas e Extensões ("Algo Extra").

Contém implementações de elevado valor técnico para sistemas autónomos:
1. Elipses de Incerteza (Covariance Ellipses) a 95% de confiança através da
   decomposição espetral (autovalores e autovetores) da matriz de covariância 6x6 do ROS.
2. Monitorização Dinâmica de Proximidade e Cálculo do Ponto de Maior Aproximação (CPA - Closest Point of Approach).
3. Exportação para GeoJSON (compatível com Google Earth, QGIS e Leaflet/Mapbox).
4. Visualização 3D da trajetória e perfis batimétricos/altimétricos.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.data_loader import MissionDataset, RobotTrajectory


def calculate_covariance_ellipse(
    cov_matrix_2x2: np.ndarray,
    scale_sigma: float = 2.4477  # 95% para distribuição qui-quadrado bidimensional
) -> Tuple[float, float, float]:
    """Calcula os semi-eixos maior e menor e o ângulo de inclinação da elipse de incerteza.

    Retorna:
        semi_major_m: comprimento do semi-eixo maior
        semi_minor_m: comprimento do semi-eixo menor
        angle_deg: ângulo de rotação em graus
    """
    eigvals, eigvecs = np.linalg.eigh(cov_matrix_2x2)

    # Garantir que os autovalores são positivos
    eigvals = np.maximum(eigvals, 1e-12)

    # Ordenar por autovalor decrescente
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    semi_major = scale_sigma * np.sqrt(eigvals[0])
    semi_minor = scale_sigma * np.sqrt(eigvals[1])

    # Ângulo do autovetor principal em relação ao eixo horizontal
    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    angle_deg = np.degrees(angle_rad)

    return float(semi_major), float(semi_minor), float(angle_deg)


def plot_3d_mission_trajectories(
    dataset: MissionDataset,
    output_file: Optional[str] = "docs/figures/mission_trajectories_3d.pdf"
) -> Tuple[plt.Figure, plt.Axes]:
    """Gera visualização 3D das trajetórias com profundidade/altitude real."""
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    for r_id, traj in dataset.trajectories.items():
        # Em NED: x = North, y = East, z = Down (-z é Up/Altitude)
        north = traj.pos_ned[0]
        east = traj.pos_ned[1]
        altitude = -traj.pos_ned[2]  # Convertido para cota positiva

        ax.plot(east, north, altitude, color=traj.color, linewidth=1.8, label=traj.label)
        ax.scatter([east[0]], [north[0]], [altitude[0]], color=traj.color, marker="o", s=50)
        ax.scatter([east[-1]], [north[-1]], [altitude[-1]], color=traj.color, marker="s", s=60)

    # Origem
    ax.scatter([0], [0], [0], color="black", marker="P", s=100, label="Origem de Superfície (0,0,0)")

    ax.set_title("Visualização Tridimensional das Trajetórias (Aguçadoura)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Este (East) [m]", fontsize=9)
    ax.set_ylabel("Norte (North) [m]", fontsize=9)
    ax.set_zlabel("Altitude / Cota Relativa [m]", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    if output_file:
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")
        fig.savefig(out_p.with_suffix(".png"), dpi=300, bbox_inches="tight")

    return fig, ax


def plot_inter_robot_distance_matrix(
    dataset: MissionDataset,
    output_file: Optional[str] = "docs/figures/proximity_analysis.pdf",
    alert_threshold_m: float = 30.0
) -> Tuple[plt.Figure, plt.Axes]:
    """Gera análise gráfica contínua da distância entre todos os pares de robôs com deteção de CPA."""
    robot_ids = list(dataset.trajectories.keys())
    t_min = dataset.common_start
    t_max = dataset.common_end
    dt = 1.0
    t_grid = np.arange(t_min, t_max, dt)
    t_min_rel = (t_grid - t_min) / 60.0  # Em minutos

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Linha do limiar de perigo
    ax.axhline(alert_threshold_m, color="red", linestyle="--", linewidth=1.5,
               label=f"Limiar de Alerta de Proximidade ({alert_threshold_m:.0f}m)")

    pair_colors = ["#E63946", "#2A9D8F", "#457B9D"]
    color_idx = 0

    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            r_a = robot_ids[i]
            r_b = robot_ids[j]

            dists = []
            for t in t_grid:
                st = dataset.get_state_at(t)
                pa = st[r_a]["pos_ned"]
                pb = st[r_b]["pos_ned"]
                dists.append(np.linalg.norm(pa - pb))

            dists = np.array(dists)
            min_idx = np.argmin(dists)
            cpa_dist = dists[min_idx]
            cpa_t = t_min_rel[min_idx]

            color = pair_colors[color_idx % len(pair_colors)]
            color_idx += 1

            label = f"{r_a} ↔ {r_b} (CPA: {cpa_dist:.1f}m aos {cpa_t:.1f}min)"
            ax.plot(t_min_rel, dists, color=color, linewidth=1.6, label=label)
            ax.scatter([cpa_t], [cpa_dist], color=color, marker="*", s=140, edgecolors="black", zorder=5)

    ax.set_title("Evolução Temporal da Distância Inter-Robôs e Ponto de Maior Aproximação (CPA)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Tempo Decorrido na Missão [minutos]", fontsize=10)
    ax.set_ylabel("Distância Euclidiana [metros]", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    if output_file:
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")
        fig.savefig(out_p.with_suffix(".png"), dpi=300, bbox_inches="tight")

    return fig, ax


def export_trajectories_to_geojson(
    dataset: MissionDataset,
    output_path: str = "output/mission_trajectories.geojson"
) -> str:
    """Exporta as trajetórias para o formato padrão GeoJSON (WGS-84 RFC 7946).

    Permite importar diretamente as rotas no Google Earth, QGIS, ArcGIS ou mapas web.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    features = []

    # 1. Ponto da Origem
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [dataset.origin[1], dataset.origin[0], dataset.origin[2]]  # [Lon, Lat, Alt]
        },
        "properties": {
            "name": "Origem da Missão (Aguçadoura)",
            "description": "Ponto de referência em superfície"
        }
    })

    # 2. Linhas de trajetória por robô
    for r_id, traj in dataset.trajectories.items():
        coords = [[float(lon), float(lat), float(alt)] for lon, lat, alt in zip(traj.lon, traj.lat, traj.alt)]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "robot_id": r_id,
                "label": traj.label,
                "color": traj.color,
                "total_points": len(traj.timestamps),
                "duration_seconds": traj.duration,
                "total_distance_meters": traj.total_distance
            }
        })

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2, ensure_ascii=False)

    return str(out_p)
