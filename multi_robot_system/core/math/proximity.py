"""Análise de Proximidade, Geometria Relativa AER e Ponto de Maior Aproximação (CPA).

Implementa:
1. Matriz de distâncias euclidianas relativas entre múltiplos robôs.
2. Grandezas esféricas de visada (Azimuth, Elevation, Range - AER) no referencial próprio (Body Frame).
3. Solução analítica fechada para o Closest Point of Approach (CPA) com alerta crítico para distâncias < 10m.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CPAResult:
    robot_i: str
    robot_j: str
    current_distance_m: float
    time_to_cpa_s: float
    cpa_distance_m: float
    is_collision_risk: bool  # True se cpa_distance_m < 10m ou current_distance_m < 10m
    relative_aer: Tuple[float, float, float]  # (az_deg, el_deg, range_m)


def compute_relative_body_vector(
    pos_i: np.ndarray,
    R_wb_i: np.ndarray,
    pos_j: np.ndarray
) -> np.ndarray:
    """Transforma a posição do robô j para o referencial próprio do corpo do robô i.

    p_j^(i) = (R_wb_i)^T * (pos_j - pos_i).
    """
    diff_w = pos_j - pos_i
    return R_wb_i.T @ diff_w


def vector_to_aer(p_body: np.ndarray) -> Tuple[float, float, float]:
    """Converte um vetor cartesiano no referencial do corpo (x frente, y direita, z baixo)

    em coordenadas esféricas relativas de visada (Azimuth, Elevation, Slant Range).
    """
    x, y, z = p_body[0], p_body[1], p_body[2]
    slant_range = float(np.sqrt(x**2 + y**2 + z**2))

    # Azimute no plano horizontal do corpo (-180° a +180°, 0° à frente, +90° estibordo)
    az_deg = float(np.degrees(np.arctan2(y, x)))

    # Elevação (-90° a +90°, positiva acima do horizonte do corpo)
    horiz_dist = float(np.sqrt(x**2 + y**2))
    el_deg = float(np.degrees(np.arctan2(-z, horiz_dist)))

    return az_deg, el_deg, slant_range


def compute_analytic_cpa(
    p1: np.ndarray,
    v1: np.ndarray,
    p2: np.ndarray,
    v2: np.ndarray,
    lookahead_limit_s: float = 60.0
) -> Tuple[float, float]:
    """Calcula analiticamente o instante (t_cpa em s) e a distância mínima (d_cpa em m).

    Baseado na cinemática linear: r(t) = Delta_p + Delta_v * t.
    """
    dp = p2 - p1
    dv = v2 - v1

    dv_sq = float(np.dot(dv, dv))
    dp_sq = float(np.dot(dp, dp))

    # Se a velocidade relativa for praticamente nula, distância mantém-se constante
    if dv_sq < 1e-8:
        return 0.0, float(np.sqrt(max(0.0, dp_sq)))

    # t_cpa = - (dp . dv) / ||dv||^2
    t_cpa_raw = -float(np.dot(dp, dv)) / dv_sq

    # Se t_cpa for negativo, os robôs estão a afastar-se; a menor distância futura é agora (t=0)
    if t_cpa_raw <= 0.0:
        t_cpa = 0.0
        d_cpa = float(np.sqrt(max(0.0, dp_sq)))
    else:
        t_cpa = min(t_cpa_raw, lookahead_limit_s)
        p_cpa = dp + dv * t_cpa
        d_cpa = float(np.linalg.norm(p_cpa))

    return t_cpa, d_cpa


def evaluate_proximity_and_cpa(
    robot_states: Dict[str, Dict],
    collision_threshold_m: float = 10.0,
    lookahead_limit_s: float = 60.0
) -> List[CPAResult]:
    """Avalia o risco de aproximação entre todos os pares de robôs ativos num instante t."""
    results = []
    robot_ids = sorted(robot_states.keys())
    n = len(robot_ids)

    for i in range(n):
        id_i = robot_ids[i]
        st_i = robot_states[id_i]
        pos_i = np.array(st_i["pos_ned"], dtype=float)
        vel_i = np.array(st_i["vel_ned"], dtype=float)
        R_i = np.array(st_i["R_wb"], dtype=float)

        for j in range(i + 1, n):
            id_j = robot_ids[j]
            st_j = robot_states[id_j]
            pos_j = np.array(st_j["pos_ned"], dtype=float)
            vel_j = np.array(st_j["vel_ned"], dtype=float)

            # Vetor e AER no referencial do robô i
            p_rel_i = compute_relative_body_vector(pos_i, R_i, pos_j)
            aer = vector_to_aer(p_rel_i)

            curr_dist = aer[2]
            t_cpa, d_cpa = compute_analytic_cpa(pos_i, vel_i, pos_j, vel_j, lookahead_limit_s)

            # Alerta de colisão se a distância atual ou a distância no CPA for inferior a 10m
            is_risk = (curr_dist < collision_threshold_m) or (d_cpa < collision_threshold_m and t_cpa < 20.0)

            results.append(CPAResult(
                robot_i=id_i,
                robot_j=id_j,
                current_distance_m=curr_dist,
                time_to_cpa_s=t_cpa,
                cpa_distance_m=d_cpa,
                is_collision_risk=is_risk,
                relative_aer=aer
            ))

    return results
