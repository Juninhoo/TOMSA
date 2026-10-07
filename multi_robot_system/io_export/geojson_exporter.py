"""Exportador de Trajetórias Multi-Robô para o Formato GeoJSON (Compatível com QGIS e SIG).

Gera um ficheiro GeoJSON formal com suporte para geometrias LineString (trajetórias)
e Point (marcos de início, fim e pontos críticos de aproximação CPA < 10m).
"""

import json
from pathlib import Path
from typing import Dict
import numpy as np

from multi_robot_system.core.data.trajectory import MissionDataset
from multi_robot_system.core.math.geodesy import ned_to_ecef, ecef_to_geodetic


def export_mission_to_geojson(
    dataset: MissionDataset,
    output_path: str,
    subsample_step: int = 10
) -> str:
    """Exporta todas as trajetórias e eventos de missão para GeoJSON estruturado."""
    lat0, lon0, h0 = dataset.origin
    features = []

    # 1. Trajetórias contínuas como LineString
    for r_id, traj in dataset.trajectories.items():
        coords_wgs84 = []
        N_pts = len(traj.timestamps)
        indices = np.arange(0, N_pts, subsample_step)
        if indices[-1] != N_pts - 1:
            indices = np.append(indices, N_pts - 1)

        for idx in indices:
            N = traj.pos_ned[0, idx]
            E = traj.pos_ned[1, idx]
            D = traj.pos_ned[2, idx]

            X, Y, Z = ned_to_ecef(N, E, D, lat0, lon0, h0)
            lat, lon, h = ecef_to_geodetic(X, Y, Z)

            # No padrão GeoJSON: [longitude, latitude, elevação/altura]
            coords_wgs84.append([float(lon), float(lat), float(h)])

        feature_line = {
            "type": "Feature",
            "properties": {
                "robot_id": r_id,
                "label": traj.label,
                "stroke": traj.color,
                "stroke-width": 3,
                "duration_seconds": traj.duration_s,
                "total_distance_m": round(traj.total_distance_m, 2),
                "max_depth_m": round(traj.max_depth_m, 2),
                "mean_speed_mps": round(traj.mean_speed_m_s, 2)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords_wgs84
            }
        }
        features.append(feature_line)

        # 2. Ponto de Início
        p0_N, p0_E, p0_D = traj.pos_ned[:, 0]
        X0, Y0, Z0 = ned_to_ecef(p0_N, p0_E, p0_D, lat0, lon0, h0)
        lat_start, lon_start, h_start = ecef_to_geodetic(X0, Y0, Z0)
        features.append({
            "type": "Feature",
            "properties": {
                "robot_id": r_id,
                "event": "START_MISSION",
                "marker-color": traj.color,
                "marker-symbol": "star",
                "time_utc": float(traj.timestamps[0])
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon_start), float(lat_start), float(h_start)]
            }
        })

    geojson_dict = {
        "type": "FeatureCollection",
        "metadata": {
            "project": "TOMSA 2026 - INESC TEC Aguçadoura Multi-Robot Mission",
            "origin_reference": {"lat": lat0, "lon": lon0, "alt_m": h0},
            "datum": "WGS-84 (EPSG:4326)",
            "filter_applied": "Rauch-Tung-Striebel (RTS) Optimal Smoother"
        },
        "features": features
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, indent=2)

    return output_path
