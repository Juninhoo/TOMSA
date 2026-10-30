"""Testes para CSV novos: nomes dos robots, linhas em falta e instantes repetidos."""
from pathlib import Path

import numpy as np
import pandas as pd

from multi_robot_system.core.data.odom_loader import load_single_robot_csv
from multi_robot_system.main import robots_from_files

DADOS = Path(__file__).resolve().parents[1] / "data" / "robot4_s1.csv"


def _carregar(csv):
    return load_single_robot_csv(str(csv), "Robot 4", "Robot 4", "#2A9D8F", 41.4523, -8.8125, 56.5)


def test_instantes_repetidos_e_linhas_em_falta(tmp_path):
    df = pd.read_csv(DADOS).iloc[:400].copy()
    df.loc[5, "pose.pose.position.x"] = np.nan                 # linha com valor em falta
    df = pd.concat([df, df.iloc[10:13]])                       # três registos repetidos
    csv = tmp_path / "sujo.csv"
    df.to_csv(csv, index=False)
    r = _carregar(csv)
    t = r["timestamps"]
    assert np.all(np.diff(t) > 0)
    assert len(t) == 400 - 1 + 3
    assert np.all(np.isfinite(r["pos_ned"]))


def test_csv_sem_covariancia_nem_twist(tmp_path):
    df = pd.read_csv(DADOS).iloc[:300]
    df = df[[c for c in df.columns if not c.startswith("twist") and c != "pose.covariance"]]
    csv = tmp_path / "simples.csv"
    df.to_csv(csv, index=False)
    r = _carregar(csv)
    assert len(r["timestamps"]) == 300


def test_nomes_dos_robots_a_partir_dos_ficheiros(tmp_path):
    for n in ["robot7_s2.csv", "Robot12.csv", "auv_alpha.csv"]:
        (tmp_path / n).write_text("x")
    robots = robots_from_files([str(tmp_path)])
    ids = sorted(r["id"] for r in robots)
    assert ids == ["Robot 12", "Robot 7", "auv_alpha"]
