"""Testes Unitários para o Cálculo Analítico do Ponto de Maior Aproximação (CPA)."""

import numpy as np
import pytest

from multi_robot_system.core.math.proximity import compute_analytic_cpa


def test_head_on_collision_cpa():
    # Robô 1 em (0, 0, 0) a avançar para Norte a 2 m/s
    p1 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([2.0, 0.0, 0.0])

    # Robô 2 em (100, 0, 0) a avançar para Sul a 3 m/s
    p2 = np.array([100.0, 0.0, 0.0])
    v2 = np.array([-3.0, 0.0, 0.0])

    # Devem colidir exatamente em t = 100 / (2 + 3) = 20 s com distância d = 0m
    t_cpa, d_cpa = compute_analytic_cpa(p1, v1, p2, v2)
    assert abs(t_cpa - 20.0) < 1e-6
    assert abs(d_cpa - 0.0) < 1e-6


def test_perpendicular_miss_cpa():
    # Robô 1 no eixo x em (-50, 0, 0) a 5 m/s para Este (+y)
    p1 = np.array([0.0, -50.0, 0.0])
    v1 = np.array([0.0, 5.0, 0.0])

    # Robô 2 no eixo y em (0, -10, 0) parado
    p2 = np.array([8.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 0.0])

    # Robô 1 passa a 8 metros em t = 10s
    t_cpa, d_cpa = compute_analytic_cpa(p1, v1, p2, v2)
    assert abs(t_cpa - 10.0) < 1e-6
    assert abs(d_cpa - 8.0) < 1e-6
    # Como 8m < 10m, ativa risco crítico
    assert d_cpa < 10.0


def test_diverging_robots_cpa():
    # Robôs a afastarem-se
    p1 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([-1.0, 0.0, 0.0])

    p2 = np.array([20.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])

    # Como estão a afastar-se, a menor distância futura é agora (t=0s, d=20m)
    t_cpa, d_cpa = compute_analytic_cpa(p1, v1, p2, v2)
    assert t_cpa == 0.0
    assert abs(d_cpa - 20.0) < 1e-6
