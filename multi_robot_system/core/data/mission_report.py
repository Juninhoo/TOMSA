"""Relatório da missão: encontros entre robôs e saltos de posição.

Compara as posições dos robôs segundo a segundo (distância 3D) para encontrar
os momentos em que se aproximaram. Os tempos contam a partir do início da missão.
"""

from itertools import combinations
from typing import Dict, List

import numpy as np


def _level(d_min: float, danger_m: float, warning_m: float) -> str:
    """Classifica um encontro pela distância mínima: perigo, atenção ou simples aproximação."""
    if d_min < danger_m:
        return "perigo"
    if d_min < warning_m:
        return "atencao"
    return "aproximacao"


def compute_mission_report(dataset, danger_m: float = 10.0, warning_m: float = 30.0,
                           encounter_m: float = 60.0, dt_s: float = 1.0) -> Dict:
    """Esta função calcula o relatório de encontros entre todos os pares de robôs.

    Para cada par: distância mínima, tempo passado abaixo de cada limiar e lista de encontros.
    Devolve um dicionário pronto a gravar em JSON.
    """
    t0 = float(dataset.t0_mission)
    ids = sorted(dataset.trajectories.keys())

    # Resumo de cada robô e lista de saltos de posição
    robots: List[Dict] = []
    jumps: List[Dict] = []
    for rid in ids:
        tr = dataset.trajectories[rid]
        ts = np.asarray(tr.timestamps, dtype=float)
        robot_jumps = list(tr.jump_events or [])
        robots.append({
            "id": rid, "label": tr.label,
            "start": float(ts[0] - t0), "end": float(ts[-1] - t0),
            "n": int(len(ts)), "max_depth": float(np.max(tr.pos_ned[2])),
            "n_jumps": len(robot_jumps),
        })
        for e in robot_jumps:
            jumps.append({"robot": rid, "t": float(e["t"] - t0), "m": float(e["m"]), "depth": float(e["depth_m"])})
    jumps.sort(key=lambda j: j["t"])

    # Para cada par de robôs: distâncias na janela de tempo comum
    pairs: List[Dict] = []
    for a, b in combinations(ids, 2):
        ta, tb = dataset.trajectories[a], dataset.trajectories[b]
        lo = max(float(ta.timestamps[0]), float(tb.timestamps[0]))
        hi = min(float(ta.timestamps[-1]), float(tb.timestamps[-1]))
        entry = {"a": a, "b": b, "overlap": None, "d_min": None, "t_min": None,
                 "below": {"w60": 0.0, "w30": 0.0, "w10": 0.0}, "episodes": []}
        if hi - lo >= dt_s:
            times = np.arange(lo, hi + 1e-9, dt_s)
            pa = np.vstack([np.interp(times, ta.timestamps, ta.pos_ned[k]) for k in range(3)])
            pb = np.vstack([np.interp(times, tb.timestamps, tb.pos_ned[k]) for k in range(3)])
            d = np.linalg.norm(pa - pb, axis=0)
            i_min = int(np.argmin(d))
            entry["overlap"] = [float(lo - t0), float(hi - t0)]
            entry["d_min"] = float(d[i_min])
            entry["t_min"] = float(times[i_min] - t0)
            entry["below"] = {"w60": float(np.sum(d < encounter_m) * dt_s),
                              "w30": float(np.sum(d < warning_m) * dt_s),
                              "w10": float(np.sum(d < danger_m) * dt_s)}
            # Encontros: períodos seguidos com distância abaixo de encounter_m
            inside = np.concatenate(([False], d < encounter_m, [False]))
            edges = np.diff(inside.astype(int))
            starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
            for s, e in zip(starts, ends):
                seg = d[s:e + 1]
                k = int(np.argmin(seg))
                t_start, t_end = float(times[s] - t0), float(times[e] - t0)
                near_jump = any(j["robot"] in (a, b) and t_start - 5.0 <= j["t"] <= t_end + 5.0 for j in jumps)
                entry["episodes"].append({
                    "t0": t_start, "t1": t_end, "t_min": float(times[s + k] - t0), "d_min": float(seg[k]),
                    "level": _level(float(seg[k]), danger_m, warning_m), "jump_near": bool(near_jump),
                })
        pairs.append(entry)

    return {
        "robots": robots, "pairs": pairs, "jumps": jumps,
        "thresholds": {"danger": float(danger_m), "warning": float(warning_m), "encounter": float(encounter_m)},
        "dt_s": float(dt_s),
    }
