// bindings.cpp - expoe uwmath.hpp ao Python (pybind11).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "uwmath.hpp"

namespace py = pybind11;
using uw::Camera;
using uw::Mat3;
using uw::Pose;
using uw::Trajectory;
using uw::Vec3;

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Aplica f a cada linha (x,y,z) de um array (N,3) -> novo array (N,3).
template <typename F>
static Array map_rows(const Array& a, F f) {
    if (a.ndim() != 2 || a.shape(1) != 3) throw std::invalid_argument("expected array of shape (N,3)");
    py::ssize_t n = a.shape(0);
    Array out({n, (py::ssize_t)3});
    auto in = a.unchecked<2>();
    auto o = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        Vec3 r = f(Vec3(in(i, 0), in(i, 1), in(i, 2)));
        o(i, 0) = r.x; o(i, 1) = r.y; o(i, 2) = r.z;
    }
    return out;
}

static Array to_numpy(const Mat3& M) {
    Array out({(py::ssize_t)3, (py::ssize_t)3});
    auto o = out.mutable_unchecked<2>();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) o(i, j) = M.m[i][j];
    return out;
}

PYBIND11_MODULE(uwmath, m) {
    m.doc() = "Matematica do visualizador de trajetorias (NED, corpo, camara).";

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<double, double, double>(), py::arg("x") = 0, py::arg("y") = 0, py::arg("z") = 0)
        .def_readwrite("x", &Vec3::x).def_readwrite("y", &Vec3::y).def_readwrite("z", &Vec3::z)
        .def("__add__", &Vec3::operator+).def("__sub__", [](const Vec3& a, const Vec3& b) { return a - b; })
        .def("__mul__", [](const Vec3& a, double s) { return a * s; })
        .def("__neg__", [](const Vec3& a) { return -a; })
        .def("dot", &Vec3::dot).def("cross", &Vec3::cross).def("norm", &Vec3::norm)
        .def("normalized", &Vec3::normalized)
        .def("to_tuple", [](const Vec3& a) { return py::make_tuple(a.x, a.y, a.z); })
        .def("__repr__", [](const Vec3& a) {
            return "Vec3(" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + ")";
        });

    py::class_<Mat3>(m, "Mat3")
        .def(py::init<>())
        .def_static("identity", &Mat3::identity)
        .def_static("from_columns", &Mat3::from_columns)
        .def("column", &Mat3::column).def("transpose", &Mat3::transpose).def("det", &Mat3::det)
        .def("__matmul__", [](const Mat3& a, const Mat3& b) { return a * b; })
        .def("__matmul__", [](const Mat3& a, const Vec3& v) { return a * v; })
        .def("to_numpy", &to_numpy);

    m.def("rot_x", &uw::rot_x).def("rot_y", &uw::rot_y).def("rot_z", &uw::rot_z);
    m.def("from_euler_zyx", &uw::from_euler_zyx, py::arg("roll"), py::arg("pitch"), py::arg("yaw"));
    m.def("to_euler_zyx", [](const Mat3& R) { auto e = uw::to_euler_zyx(R); return py::make_tuple(e[0], e[1], e[2]); },
          "Devolve (roll, pitch, yaw) em radianos.");
    m.def("from_quaternion", &uw::from_quaternion, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("w"),
          "Quaternion (x,y,z,w) -> matriz de rotacao 3x3.");
    m.def("lla_to_ecef", &uw::lla_to_ecef, py::arg("lat_deg"), py::arg("lon_deg"), py::arg("h"));
    m.def("skew", &uw::skew).def("rodrigues", &uw::rodrigues).def("orthonormalize", &uw::orthonormalize);

    py::class_<uw::LocalFrameNED>(m, "LocalFrameNED")
        .def(py::init<double, double, double>(), py::arg("lat0_deg"), py::arg("lon0_deg"), py::arg("h0"))
        .def("to_ned", &uw::LocalFrameNED::to_ned, py::arg("lat_deg"), py::arg("lon_deg"), py::arg("h"))
        .def("ecef_to_ned", &uw::LocalFrameNED::ecef_to_ned).def("ned_to_ecef", &uw::LocalFrameNED::ned_to_ecef)
        .def("R", [](const uw::LocalFrameNED& f) { return f.R(); })
        .def("to_ned_array", [](const uw::LocalFrameNED& f, const Array& a) {   // (N,3): lat, lon, h
            return map_rows(a, [&](Vec3 v) { return f.to_ned(v.x, v.y, v.z); });
        });

    py::class_<Pose>(m, "Pose")
        .def(py::init<>()).def(py::init<const Mat3&, const Vec3&>(), py::arg("R"), py::arg("t"))
        .def_readwrite("R", &Pose::R).def_readwrite("t", &Pose::t)
        .def("to_world", &Pose::to_world).def("to_body", &Pose::to_body)
        .def("inverse", &Pose::inverse).def("compose", &Pose::compose)
        .def("to_world_array", [](const Pose& p, const Array& a) { return map_rows(a, [&](Vec3 v) { return p.to_world(v); }); })
        .def("to_body_array", [](const Pose& p, const Array& a) { return map_rows(a, [&](Vec3 v) { return p.to_body(v); }); });

    py::class_<uw::Projection>(m, "Projection")
        .def_readonly("visible", &uw::Projection::visible)
        .def_readonly("u", &uw::Projection::u).def_readonly("v", &uw::Projection::v)
        .def_readonly("zc", &uw::Projection::zc);

    py::class_<Camera>(m, "Camera")
        .def(py::init<double, double, double, double, int, int>(),
             py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("width"), py::arg("height"))
        .def_static("from_fov", &Camera::from_fov, py::arg("width"), py::arg("height"), py::arg("fov_h_rad"))
        .def_readwrite("body_to_cam", &Camera::body_to_cam)
        .def("pose_in_world", &Camera::pose_in_world)
        .def("world_to_camera", &Camera::world_to_camera)
        .def("world_to_camera_array", [](const Camera& c, const Pose& d, const Array& a) {
            Pose wc = c.pose_in_world(d);
            return map_rows(a, [&](Vec3 v) { return wc.to_body(v); });
        })
        .def("project", &Camera::project, py::arg("drone"), py::arg("p_w"), py::arg("near") = 0.1)
        .def("forward_world", &Camera::forward_world).def("up_world", &Camera::up_world);

    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init([](const std::vector<double>& t, const Array& p) {
            if (p.ndim() != 2 || p.shape(1) != 3) throw std::invalid_argument("positions must be (N,3)");
            std::vector<Vec3> pos;
            auto in = p.unchecked<2>();
            for (py::ssize_t i = 0; i < in.shape(0); ++i) pos.emplace_back(in(i, 0), in(i, 1), in(i, 2));
            return Trajectory(t, pos);
        }), py::arg("times"), py::arg("positions"))
        .def(py::init([](const std::vector<double>& t, const Array& p, const Array& q) {
            if (p.ndim() != 2 || p.shape(1) != 3) throw std::invalid_argument("positions must be (N,3)");
            if (q.ndim() != 2 || q.shape(1) != 4) throw std::invalid_argument("quaternions must be (N,4) = x,y,z,w");
            std::vector<Vec3> pos;
            std::vector<std::array<double, 4>> quats;
            auto ip = p.unchecked<2>();
            auto iq = q.unchecked<2>();
            for (py::ssize_t i = 0; i < ip.shape(0); ++i) pos.emplace_back(ip(i, 0), ip(i, 1), ip(i, 2));
            for (py::ssize_t i = 0; i < iq.shape(0); ++i) quats.push_back({iq(i, 0), iq(i, 1), iq(i, 2), iq(i, 3)});
            return Trajectory(t, pos, quats);
        }), py::arg("times"), py::arg("positions"), py::arg("quaternions_xyzw"))
        .def("positions_at", [](const Trajectory& tr, const std::vector<double>& ts) {
            Array out({(py::ssize_t)ts.size(), (py::ssize_t)3});
            auto o = out.mutable_unchecked<2>();
            for (size_t i = 0; i < ts.size(); ++i) {
                Vec3 p = tr.pose_at(ts[i]).t;
                o(i, 0) = p.x; o(i, 1) = p.y; o(i, 2) = p.z;
            }
            return out;
        })
        .def_property_readonly("t_start", &Trajectory::t_start)
        .def_property_readonly("t_end", &Trajectory::t_end)
        .def("pose_at", &Trajectory::pose_at).def("speed_at", &Trajectory::speed_at);
}
