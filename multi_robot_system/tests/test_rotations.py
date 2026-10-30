"""Testes das rotações: quaterniões, matrizes de rotação, ângulos de Euler e SLERP."""

import numpy as np
import pytest

from multi_robot_system.core.math.rotations import (
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_to_euler_zyx,
    euler_zyx_to_rotation_matrix,
    slerp
)


def test_quaternion_orthonormality():
    """Verifica que as matrizes obtidas de quaterniões são rotações válidas (RᵀR = I e det(R) = 1)."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        q = rng.normal(size=4)
        qx, qy, qz, qw = normalize_quaternion(q[0], q[1], q[2], q[3])
        assert abs(qx**2 + qy**2 + qz**2 + qw**2 - 1.0) < 1e-12

        R = quaternion_to_rotation_matrix(np.array([qx, qy, qz, qw]))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_euler_roundtrip():
    """Verifica que ângulos -> matriz -> quaternião -> ângulos devolve os ângulos iniciais."""
    rolls = [-45.0, 0.0, 30.0, 60.0]
    pitchs = [-40.0, -15.0, 0.0, 20.0, 45.0]
    yaws = [0.0, 45.0, 90.0, 180.0, -120.0]

    for r in rolls:
        for p in pitchs:
            for y in yaws:
                R = euler_zyx_to_rotation_matrix(np.radians(r), np.radians(p), np.radians(y))
                from scipy.spatial.transform import Rotation
                quat = Rotation.from_matrix(R).as_quat()
                r_calc, p_calc, y_calc = quaternion_to_euler_zyx(quat[0], quat[1], quat[2], quat[3], degrees=True)
                assert abs(r - r_calc) < 1e-6
                assert abs(p - p_calc) < 1e-6
                dyaw = (y - y_calc + 180.0) % 360.0 - 180.0
                assert abs(dyaw) < 1e-6


def test_slerp_endpoints():
    """Verifica a SLERP nas extremidades (tau = 0 e 1) e a meio caminho (45°)."""
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])

    assert np.allclose(slerp(q0, q1, 0.0), q0, atol=1e-12)
    assert np.allclose(slerp(q0, q1, 1.0), q1, atol=1e-12)

    q_mid = slerp(q0, q1, 0.5)
    expected_mid = np.array([0.0, 0.0, np.sin(np.pi/8), np.cos(np.pi/8)])
    assert np.allclose(q_mid, expected_mid, atol=1e-6)
