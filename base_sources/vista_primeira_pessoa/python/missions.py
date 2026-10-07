"""Missoes de demonstracao e leitor de CSV. Todas em NED (z positivo = profundidade, em metros)."""
import csv
import re
from pathlib import Path
import numpy as np


def _polyline(points, speed, dt):
    """Percorre uma linha poligonal a velocidade constante; devolve (t, pos)."""
    pts = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s_cum[-1] / speed + dt, dt)
    s = np.minimum(t * speed, s_cum[-1])
    pos = np.column_stack([np.interp(s, s_cum, pts[:, k]) for k in range(3)])
    return t, pos


def demo_missions():
    """Tres drones com taxas de amostragem diferentes. Devolve [dict(name, kind="ned", t, pos), ...]."""
    # Alfa: varrimento "lawnmower" a 20 m de profundidade
    wp = [(0, 0, 20), (80, 0, 20), (80, 15, 20), (0, 15, 20), (0, 30, 20), (80, 30, 20), (80, 45, 20), (0, 45, 20)]
    t1, p1 = _polyline(wp, speed=2.4, dt=2.0)

    # Bravo: helice descendente (8 m -> 40 m) em torno de (40, 25)
    t2 = np.arange(0, 150 + 1e-9, 1.5)
    w = 2 * np.pi / 50
    p2 = np.column_stack([40 + 25 * np.cos(w * t2), 25 + 25 * np.sin(w * t2), 8 + (32 / 150) * t2])

    # Charlie: oito (lemniscata) com oscilacao de profundidade
    t3 = np.arange(0, 150 + 1e-9, 1.0)
    a = 2 * np.pi * t3 / 150
    p3 = np.column_stack([40 + 35 * np.sin(a), 25 + 20 * np.sin(2 * a), 30 + 8 * np.sin(2 * np.pi * t3 / 75)])

    return [dict(name=n, kind="ned", t=t, pos=p) for n, t, p in (("Alfa", t1, p1), ("Bravo", t2, p2), ("Charlie", t3, p3))]


ODOM_COLS = ["Time", "pose.pose.position.x", "pose.pose.position.y", "pose.pose.position.z",
             "pose.pose.orientation.x", "pose.pose.orientation.y", "pose.pose.orientation.z",
             "pose.pose.orientation.w"]


def load_csv(path):
    """Le um CSV de trajetoria. Devolve um dict:

    - kind "ned":  cabecalho t,x,y,z (NED, metros)  -> pos (N,3)
    - kind "odom": exportacao ROS Odometry (colunas pose.pose.position.x/y/z e orientation.x/y/z/w).
                   Nos teus ficheiros x = latitude (graus), y = longitude (graus), z = altura elipsoidal (m, para cima);
                   a orientacao e um quaternion corpo->NED. -> lla (N,3) e quats (N,4) = x,y,z,w
    """
    path = Path(path)
    name = re.sub(r"_s\d+$", "", path.stem)
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        header = rd.fieldnames or []
        if "pose.pose.position.x" in header:
            miss = [c for c in ODOM_COLS if c not in header]
            if miss:
                raise ValueError(f"{path.name}: faltam colunas {miss}")
            rows = np.array([[float(r[c]) for c in ODOM_COLS] for r in rd])
            order = np.argsort(rows[:, 0], kind="stable")
            rows = rows[order]
            keep = np.concatenate([[True], np.diff(rows[:, 0]) > 0])      # tempos estritamente crescentes
            rows = rows[keep]
            return dict(name=name, kind="odom", t=rows[:, 0], lla=rows[:, 1:4], quats=rows[:, 4:8])
    d = np.genfromtxt(path, delimiter=",", names=True)
    return dict(name=name, kind="ned", t=d["t"], pos=np.column_stack([d["x"], d["y"], d["z"]]))
