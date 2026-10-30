"""Testes do filtro de Kalman e do suavizador RTS."""

import numpy as np
import pytest

from multi_robot_system.core.math.kalman_rts import TrajectoryKalmanRTS


def test_rts_reduces_noise():
    """Junta ruído a uma trajetória em linha reta e verifica que o suavizador
    reduz o erro em mais de 40 % e recupera a velocidade verdadeira.
    """
    t = np.linspace(0, 50, 101)
    true_n = 2.0 * t
    true_e = -1.0 * t
    true_d = 0.2 * t

    true_pos = np.stack([true_n, true_e, true_d], axis=0)

    rng = np.random.default_rng(123)
    noisy_pos = true_pos + rng.normal(0, 1.0, size=true_pos.shape)

    smoother = TrajectoryKalmanRTS(process_noise_q=0.01, default_meas_std=1.0)
    smooth_pos, smooth_vel, cov_diag = smoother.smooth(t, noisy_pos)

    rmse_noisy = np.sqrt(np.mean((noisy_pos - true_pos)**2))
    rmse_smooth = np.sqrt(np.mean((smooth_pos - true_pos)**2))

    assert rmse_smooth < rmse_noisy
    assert rmse_smooth < 0.6 * rmse_noisy

    mean_vx = np.mean(smooth_vel[0, 10:-10])
    mean_vy = np.mean(smooth_vel[1, 10:-10])
    mean_vz = np.mean(smooth_vel[2, 10:-10])

    assert abs(mean_vx - 2.0) < 0.2
    assert abs(mean_vy - (-1.0)) < 0.2
    assert abs(mean_vz - 0.2) < 0.1
