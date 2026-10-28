"""Testes do relatório da missão (encontros, resumo por par e saltos)."""

from types import SimpleNamespace

import numpy as np

from multi_robot_system.core.data.mission_report import compute_mission_report


def _traj(ts, pos, jumps=None):
    return SimpleNamespace(label="x", timestamps=np.asarray(ts, float), pos_ned=np.asarray(pos, float), jump_events=jumps)


def test_single_pass_encounter():
    t = np.arange(0.0, 201.0, 1.0)
    # A parado na origem; B passa a 5 m de distância (Este) com 1 m/s, a cruzar o Norte 0 aos 100 s
    a = _traj(t, np.vstack([np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)]))
    b = _traj(t, np.vstack([100.0 - t, np.full_like(t, 5.0), np.zeros_like(t)]),
              jumps=[{"t": 1000.0 + 100.0, "m": 3.0, "depth_m": 1.0}])
    ds = SimpleNamespace(t0_mission=1000.0, trajectories={"A": a, "B": b})
    ds.trajectories["A"].timestamps = ds.trajectories["A"].timestamps + 1000.0
    ds.trajectories["B"].timestamps = ds.trajectories["B"].timestamps + 1000.0
    rep = compute_mission_report(ds)
    p = rep["pairs"][0]
    assert abs(p["d_min"] - 5.0) < 1e-9 and abs(p["t_min"] - 100.0) < 1e-9
    # abaixo de 10 m: |N| < sqrt(75) = 8.66 -> cerca de 17 s; abaixo de 30 m: cerca de 59 s
    assert abs(p["below"]["w10"] - 17.0) <= 1.0
    assert abs(p["below"]["w30"] - 59.0) <= 1.0
    assert len(p["episodes"]) == 1
    ep = p["episodes"][0]
    assert ep["level"] == "perigo" and ep["jump_near"] is True
    assert ep["t0"] < 100.0 < ep["t1"]
    assert rep["jumps"][0]["robot"] == "B" and abs(rep["jumps"][0]["t"] - 100.0) < 1e-9


def test_no_overlap_gives_empty_pair():
    ta, tb = np.arange(0.0, 50.0), np.arange(100.0, 150.0)
    z = lambda t: np.vstack([np.zeros_like(t)] * 3)
    ds = SimpleNamespace(t0_mission=0.0, trajectories={"A": _traj(ta, z(ta)), "B": _traj(tb, z(tb))})
    p = compute_mission_report(ds)["pairs"][0]
    assert p["overlap"] is None and p["d_min"] is None and p["episodes"] == []
