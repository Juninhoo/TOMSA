"""Testes do cálculo do ponto de maior aproximação (CPA)."""

import numpy as np
import pytest

from multi_robot_system.core.math.proximity import compute_analytic_cpa


def test_head_on_collision_cpa():
    """Dois robôs em rota de colisão frontal (2 m/s e 3 m/s, a 100 m): colidem aos 20 s, a 0 m."""
    p1 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([2.0, 0.0, 0.0])

    p2 = np.array([100.0, 0.0, 0.0])
    v2 = np.array([-3.0, 0.0, 0.0])

    t_cpa, d_cpa = compute_analytic_cpa(p1, v1, p2, v2)
    assert abs(t_cpa - 20.0) < 1e-6
    assert abs(d_cpa - 0.0) < 1e-6


def test_perpendicular_miss_cpa():
    """Um robô passa ao lado de outro parado: distância mínima de 8 m aos 10 s, com alerta de risco."""
    p1 = np.array([0.0, -50.0, 0.0])
    v1 = np.array([0.0, 5.0, 0.0])

    p2 = np.array([8.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 0.0])

    t_cpa, d_cpa = compute_analytic_cpa(p1, v1, p2, v2)
    assert abs(t_cpa - 10.0) < 1e-6
    assert abs(d_cpa - 8.0) < 1e-6
    assert d_cpa < 10.0


def test_diverging_robots_cpa():
    """Dois robôs a afastarem-se: a distância mínima é a atual (t = 0 s, 20 m)."""
    p1 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([-1.0, 0.0, 0.0])

    p2 = np.array([20.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])

    t_cpa, d_cpa = compute_analytic_cpa(p1, v1, p2, v2)
    assert t_cpa == 0.0
    assert abs(d_cpa - 20.0) < 1e-6
