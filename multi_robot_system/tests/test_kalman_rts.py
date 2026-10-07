"""Testes Unitários para o Filtro de Kalman e Suavizador RTS."""

import numpy as np
import pytest

from multi_robot_system.core.math.kalman_rts import TrajectoryKalmanRTS


def test_rts_reduces_noise():
    # Cria trajetória linear pura com ruído gaussiano adicionado
    t = np.linspace(0, 50, 101)  # dt = 0.5s
    true_n = 2.0 * t  # v = 2 m/s
    true_e = -1.0 * t
    true_d = 0.2 * t

    true_pos = np.stack([true_n, true_e, true_d], axis=0)

    # Ruído com sigma = 1.0 m
    rng = np.random.default_rng(123)
    noisy_pos = true_pos + rng.normal(0, 1.0, size=true_pos.shape)

    smoother = TrajectoryKalmanRTS(process_noise_q=0.01, default_meas_std=1.0)
    smooth_pos, smooth_vel, cov_diag = smoother.smooth(t, noisy_pos)

    # O erro RMSE da posição suavizada deve ser estritamente menor do que o sinal ruidoso
    rmse_noisy = np.sqrt(np.mean((noisy_pos - true_pos)**2))
    rmse_smooth = np.sqrt(np.mean((smooth_pos - true_pos)**2))

    assert rmse_smooth < rmse_noisy
    assert rmse_smooth < 0.6 * rmse_noisy  # Atenuação de ruído superior a 40%

    # Verifica velocidade estimada (deve aproximar-se de 2, -1, 0.2 m/s)
    mean_vx = np.mean(smooth_vel[0, 10:-10])
    mean_vy = np.mean(smooth_vel[1, 10:-10])
    mean_vz = np.mean(smooth_vel[2, 10:-10])

    assert abs(mean_vx - 2.0) < 0.2
    assert abs(mean_vy - (-1.0)) < 0.2
    assert abs(mean_vz - 0.2) < 0.1
