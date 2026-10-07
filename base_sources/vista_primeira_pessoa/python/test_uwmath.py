"""Testes da matematica (C++). Correr:  python test_uwmath.py"""
import math
import numpy as np
import uwmath as uw


def arr(R):
    return R.to_numpy()


def test_rotation_properties():
    rng = np.random.default_rng(1)
    for _ in range(100):
        r, p, y = rng.uniform(-3, 3), rng.uniform(-1.4, 1.4), rng.uniform(-3, 3)
        R = arr(uw.from_euler_zyx(r, p, y))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1) < 1e-12
        r2, p2, y2 = uw.to_euler_zyx(uw.from_euler_zyx(r, p, y))
        assert np.allclose([r, p, y], [r2, p2, y2], atol=1e-9)


def test_pose_roundtrip():
    pose = uw.Pose(uw.from_euler_zyx(0.2, -0.3, 1.1), uw.Vec3(10, 5, 20))
    p = uw.Vec3(3, -4, 7)
    q = pose.to_body(pose.to_world(p))
    assert np.allclose(q.to_tuple(), p.to_tuple(), atol=1e-12)


def test_rodrigues_90deg():
    R = arr(uw.rodrigues(uw.Vec3(0, 0, 1), math.pi / 2))
    assert np.allclose(R, arr(uw.rot_z(math.pi / 2)), atol=1e-12)


def test_pdf_example():
    """Exemplo numerico da seccao 8 do PDF."""
    cam = uw.Camera.from_fov(640, 480, math.radians(90))
    drone = uw.Pose(uw.from_euler_zyx(0, 0, math.pi / 2), uw.Vec3(10, 5, 20))
    a = cam.project(drone, uw.Vec3(10, 15, 20))
    b = cam.project(drone, uw.Vec3(10, 15, 25))
    assert np.allclose([a.u, a.v, a.zc], [320, 240, 10], atol=1e-9)
    assert np.allclose([b.u, b.v, b.zc], [320, 400, 10], atol=1e-9)
    assert a.visible and b.visible
    behind = cam.project(drone, uw.Vec3(10, -5, 20))
    assert not behind.visible


def test_camera_vectors():
    cam = uw.Camera.from_fov(640, 480, math.radians(90))
    drone = uw.Pose(uw.from_euler_zyx(0, 0, math.pi / 2), uw.Vec3(0, 0, 0))
    assert np.allclose(cam.forward_world(drone).to_tuple(), (0, 1, 0), atol=1e-12)   # Este
    assert np.allclose(cam.up_world(drone).to_tuple(), (0, 0, -1), atol=1e-12)       # cima = -z NED


def test_trajectory_heading():
    t = [0, 1, 2, 3]
    p = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]], float)  # a andar para Este
    tr = uw.Trajectory(t, p)
    _, pitch, yaw = uw.to_euler_zyx(tr.pose_at(1.5).R)
    assert abs(yaw - math.pi / 2) < 1e-9 and abs(pitch) < 1e-9
    assert abs(tr.speed_at(1.5) - 1) < 1e-12
    q = tr.pose_at(1.5).t
    assert np.allclose(q.to_tuple(), (0, 1.5, 0))
    p2 = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2]], float)             # Norte e a descer
    _, pitch, yaw = uw.to_euler_zyx(uw.Trajectory([0, 1, 2], p2).pose_at(1).R)
    assert abs(yaw) < 1e-9 and pitch < 0                                  # nariz para baixo


def test_batch():
    cam = uw.Camera.from_fov(640, 480, math.radians(90))
    drone = uw.Pose(uw.from_euler_zyx(0.1, 0.2, 0.3), uw.Vec3(1, 2, 3))
    pts = np.random.default_rng(0).uniform(-10, 10, (50, 3))
    out = cam.world_to_camera_array(drone, pts)
    for i in range(50):
        one = cam.world_to_camera(drone, uw.Vec3(*pts[i])).to_tuple()
        assert np.allclose(out[i], one)


def test_quaternion():
    a = 0.7
    R = arr(uw.from_quaternion(0, 0, math.sin(a / 2), math.cos(a / 2)))
    assert np.allclose(R, arr(uw.rot_z(a)), atol=1e-12)
    # mesmo resultado que ZYX com yaw/pitch/roll
    r, p, y = 0.3, -0.2, 1.2
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    q = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
    assert np.allclose(arr(uw.from_quaternion(*q)), arr(uw.from_euler_zyx(r, p, y)), atol=1e-12)
    # q e -q sao a mesma rotacao; escala e ignorada
    assert np.allclose(arr(uw.from_quaternion(*[-2 * c for c in q])), arr(uw.from_quaternion(*q)), atol=1e-12)


def test_geodesy_ned():
    f = uw.LocalFrameNED(41.45, -8.81, 0.0)
    n = f.to_ned(41.45 + 0.001, -8.81, 0.0)           # 0.001 grau a Norte ~ 111 m
    assert 110 < n.x < 112.5 and abs(n.y) < 1e-3 and abs(n.z) < 0.01
    e = f.to_ned(41.45, -8.81 + 0.001, 0.0)           # 0.001 grau a Este ~ 83 m a esta latitude
    assert 82 < e.y < 84 and abs(e.x) < 0.01
    up = f.to_ned(41.45, -8.81, 10.0)                 # subir 10 m => z NED = -10
    assert abs(up.z + 10) < 1e-6 and abs(up.x) < 1e-6
    R = f.R().to_numpy()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(R) - 1) < 1e-12
    p = f.to_ned(41.452, -8.812, 3.0)
    back = f.ned_to_ecef(p)
    assert np.allclose(back.to_tuple(), uw.lla_to_ecef(41.452, -8.812, 3.0).to_tuple(), atol=1e-6)
    arr3 = f.to_ned_array(np.array([[41.452, -8.812, 3.0]]))
    assert np.allclose(arr3[0], p.to_tuple())


def test_trajectory_quaternions():
    t = [0, 1, 2]
    p = np.zeros((3, 3))
    qz = lambda a: (0, 0, math.sin(a / 2), math.cos(a / 2))
    q = np.array([qz(0), qz(0.5), qz(1.0)])
    tr = uw.Trajectory(t, p, q)
    _, _, yaw = uw.to_euler_zyx(tr.pose_at(0.5).R)
    assert abs(yaw - 0.25) < 1e-3                       # nlerp ~ slerp para angulos pequenos
    q2 = np.array([qz(0), (0, 0, -math.sin(0.25), -math.cos(0.25)), qz(1.0)])   # sinal trocado: mesma rotacao
    _, _, yaw2 = uw.to_euler_zyx(uw.Trajectory(t, p, q2).pose_at(0.5).R)
    assert abs(yaw2 - 0.25) < 1e-3                      # entre 0 e 0.5 rad, a meio = 0.25 rad


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
