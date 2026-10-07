// uwmath.hpp - matematica do visualizador (Vec3, Mat3, Pose, Camera, Trajectory).
// Header-only, C++17. Convencoes: NED (x Norte, y Este, z Baixo), corpo (x frente,
// y direita, z baixo), camara (x direita, y baixo, z frente). Matrizes 3x3 (sem 4x4).
// Cada formula esta deduzida em docs/fundamentos_matematicos.pdf (seccao indicada).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace uw {

constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------- Vec3
class Vec3 {
public:
    double x, y, z;
    Vec3(double x_ = 0, double y_ = 0, double z_ = 0) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double norm() const { return std::sqrt(dot(*this)); }
    Vec3 normalized() const {
        double n = norm();
        if (n < 1e-15) throw std::runtime_error("Vec3: cannot normalize zero vector");
        return *this * (1.0 / n);
    }
};

// ---------------------------------------------------------------- Mat3
class Mat3 {
public:
    double m[3][3];  // m[linha][coluna]
    Mat3() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) m[i][j] = (i == j) ? 1.0 : 0.0;
    }
    static Mat3 identity() { return Mat3(); }
    // Constroi com as colunas dadas (eixos de um referencial escritos no outro).
    static Mat3 from_columns(const Vec3& a, const Vec3& b, const Vec3& c) {
        Mat3 r;
        r.m[0][0] = a.x; r.m[1][0] = a.y; r.m[2][0] = a.z;
        r.m[0][1] = b.x; r.m[1][1] = b.y; r.m[2][1] = b.z;
        r.m[0][2] = c.x; r.m[1][2] = c.y; r.m[2][2] = c.z;
        return r;
    }
    Vec3 column(int j) const { return {m[0][j], m[1][j], m[2][j]}; }
    Mat3 transpose() const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
        return r;
    }
    Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0;
                for (int k = 0; k < 3; ++k) s += m[i][k] * o.m[k][j];
                r.m[i][j] = s;
            }
        return r;
    }
    Vec3 operator*(const Vec3& v) const {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }
    Mat3 operator+(const Mat3& o) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = m[i][j] + o.m[i][j];
        return r;
    }
    Mat3 operator*(double s) const {
        Mat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = m[i][j] * s;
        return r;
    }
    double det() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
};

// ------------------------------------------------- rotacoes (seccao 4 do PDF)
inline Mat3 rot_x(double a) {
    Mat3 r; double c = std::cos(a), s = std::sin(a);
    r.m[1][1] = c; r.m[1][2] = -s; r.m[2][1] = s; r.m[2][2] = c;
    return r;
}
inline Mat3 rot_y(double a) {
    Mat3 r; double c = std::cos(a), s = std::sin(a);
    r.m[0][0] = c; r.m[0][2] = s; r.m[2][0] = -s; r.m[2][2] = c;
    return r;
}
inline Mat3 rot_z(double a) {
    Mat3 r; double c = std::cos(a), s = std::sin(a);
    r.m[0][0] = c; r.m[0][1] = -s; r.m[1][0] = s; r.m[1][1] = c;
    return r;
}
// R_wb = Rz(yaw) * Ry(pitch) * Rx(roll)
inline Mat3 from_euler_zyx(double roll, double pitch, double yaw) {
    return rot_z(yaw) * rot_y(pitch) * rot_x(roll);
}
// Devolve {roll, pitch, yaw}. Em gimbal lock (cos(pitch)~0) fixa roll=0.
inline std::array<double, 3> to_euler_zyx(const Mat3& R) {
    double pitch = std::atan2(-R.m[2][0], std::hypot(R.m[0][0], R.m[1][0]));
    double roll, yaw;
    if (std::hypot(R.m[0][0], R.m[1][0]) < 1e-9) {
        roll = 0.0;
        yaw = std::atan2(-R.m[0][1], R.m[1][1]);
    } else {
        yaw = std::atan2(R.m[1][0], R.m[0][0]);
        roll = std::atan2(R.m[2][1], R.m[2][2]);
    }
    return {roll, pitch, yaw};
}
// Matriz antissimetrica [w]x tal que [w]x v = w x v.
inline Mat3 skew(const Vec3& w) {
    Mat3 r;
    r.m[0][0] = 0;    r.m[0][1] = -w.z; r.m[0][2] = w.y;
    r.m[1][0] = w.z;  r.m[1][1] = 0;    r.m[1][2] = -w.x;
    r.m[2][0] = -w.y; r.m[2][1] = w.x;  r.m[2][2] = 0;
    return r;
}
// exp([w]x dt) = I + sin(a) K + (1-cos(a)) K^2, a=|w|dt, K=[w/|w|]x (Rodrigues).
inline Mat3 rodrigues(const Vec3& w, double dt) {
    double n = w.norm();
    double a = n * dt;
    if (a < 1e-12) return Mat3::identity() + skew(w) * dt;  // 1a ordem
    Mat3 K = skew(w * (1.0 / n));
    return Mat3::identity() + K * std::sin(a) + (K * K) * (1.0 - std::cos(a));
}
// Re-ortonormaliza as colunas (Gram-Schmidt) para travar erro numerico acumulado.
inline Mat3 orthonormalize(const Mat3& R) {
    Vec3 a = R.column(0).normalized();
    Vec3 b = R.column(1);
    b = (b - a * a.dot(b)).normalized();
    Vec3 c = a.cross(b);
    return Mat3::from_columns(a, b, c);
}

// ------------------------------------------- quaternion (seccao 9 do PDF)
// q = (x, y, z, w) = (sin(a/2) n, cos(a/2)): rotacao de angulo a em torno do eixo n.
// R = I + 2 w [v]x + 2 [v]x^2, com v = (x, y, z)  (Rodrigues com meio-angulo).
inline Mat3 from_quaternion(double x, double y, double z, double w) {
    double n = std::sqrt(x * x + y * y + z * z + w * w);
    if (n < 1e-12) throw std::invalid_argument("from_quaternion: zero quaternion");
    x /= n; y /= n; z /= n; w /= n;
    Mat3 K = skew(Vec3(x, y, z));
    return Mat3::identity() + K * (2.0 * w) + (K * K) * 2.0;
}

// -------------------------------------------- geodesia WGS84 (seccao 3 do PDF)
constexpr double WGS84_A = 6378137.0;           // semieixo maior (m)
constexpr double WGS84_E2 = 6.69437999014e-3;   // excentricidade^2

// latitude, longitude (graus), altura elipsoidal h (m, positiva para cima) -> ECEF
inline Vec3 lla_to_ecef(double lat_deg, double lon_deg, double h) {
    double la = lat_deg * PI / 180.0, lo = lon_deg * PI / 180.0;
    double s = std::sin(la), c = std::cos(la);
    double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * s * s);
    return {(N + h) * c * std::cos(lo), (N + h) * c * std::sin(lo), (N * (1.0 - WGS84_E2) + h) * s};
}

// Referencial local NED com origem em (lat0, lon0, h0). x Norte, y Este, z Baixo.
class LocalFrameNED {
public:
    LocalFrameNED(double lat0_deg, double lon0_deg, double h0)
        : origin_(lla_to_ecef(lat0_deg, lon0_deg, h0)) {
        double la = lat0_deg * PI / 180.0, lo = lon0_deg * PI / 180.0;
        double sa = std::sin(la), ca = std::cos(la), so = std::sin(lo), co = std::cos(lo);
        // linhas = Norte, Este, Baixo escritos em ECEF
        R_.m[0][0] = -sa * co; R_.m[0][1] = -sa * so; R_.m[0][2] = ca;
        R_.m[1][0] = -so;      R_.m[1][1] = co;       R_.m[1][2] = 0;
        R_.m[2][0] = -ca * co; R_.m[2][1] = -ca * so; R_.m[2][2] = -sa;
    }
    Vec3 ecef_to_ned(const Vec3& p_ecef) const { return R_ * (p_ecef - origin_); }
    Vec3 ned_to_ecef(const Vec3& p_ned) const { return R_.transpose() * p_ned + origin_; }
    Vec3 to_ned(double lat_deg, double lon_deg, double h) const {
        return ecef_to_ned(lla_to_ecef(lat_deg, lon_deg, h));
    }
    const Mat3& R() const { return R_; }
private:
    Vec3 origin_;
    Mat3 R_;
};

// ---------------------------------------------------------------- Pose
// Referencial b dentro de w: p_w = R p_b + t   (seccao 5)
class Pose {
public:
    Mat3 R;
    Vec3 t;
    Pose() {}
    Pose(const Mat3& R_, const Vec3& t_) : R(R_), t(t_) {}
    Vec3 to_world(const Vec3& p_b) const { return R * p_b + t; }
    Vec3 to_body(const Vec3& p_w) const { return R.transpose() * (p_w - t); }
    Pose inverse() const {
        Mat3 Rt = R.transpose();
        return Pose(Rt, -(Rt * t));
    }
    // (this) o (o): se this=T_wb e o=T_bc entao resultado = T_wc
    Pose compose(const Pose& o) const { return Pose(R * o.R, R * o.t + t); }
};

// -------------------------------------------------------------- Camera
struct Projection {
    bool visible;   // z_c > near e dentro da imagem
    double u, v;    // pixel (so valido se z_c > 0)
    double zc;      // profundidade
};

class Camera {
public:
    double fx, fy, cx, cy;
    int width, height;
    Pose body_to_cam;  // T_bc

    Camera(double fx_, double fy_, double cx_, double cy_, int w, int h)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_), width(w), height(h),
          body_to_cam(default_R_bc(), Vec3(0, 0, 0)) {}

    // camara x=y_b, y=z_b, z=x_b  =>  colunas de R_bc
    static Mat3 default_R_bc() {
        return Mat3::from_columns(Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(1, 0, 0));
    }
    // f_x = (W/2) / tan(FOV/2)
    static Camera from_fov(int w, int h, double fov_h_rad) {
        double f = (w / 2.0) / std::tan(fov_h_rad / 2.0);
        return Camera(f, f, w / 2.0, h / 2.0, w, h);
    }
    Pose pose_in_world(const Pose& drone) const { return drone.compose(body_to_cam); }
    Vec3 world_to_camera(const Pose& drone, const Vec3& p_w) const {
        return pose_in_world(drone).to_body(p_w);   // R_wc^T (p_w - t_wc)
    }
    Projection project(const Pose& drone, const Vec3& p_w, double near = 0.1) const {
        Vec3 c = world_to_camera(drone, p_w);
        Projection pr{false, 0, 0, c.z};
        if (c.z > 1e-12) {
            pr.u = fx * c.x / c.z + cx;
            pr.v = fy * c.y / c.z + cy;
            pr.visible = (c.z > near) && pr.u >= 0 && pr.u < width && pr.v >= 0 && pr.v < height;
        }
        return pr;
    }
    Vec3 forward_world(const Pose& drone) const { return pose_in_world(drone).R.column(2); }
    Vec3 up_world(const Pose& drone) const { return -(pose_in_world(drone).R.column(1)); }
};

// ---------------------------------------------------------- Trajectory
// Amostras (t, p) em NED. Sem quaternions, a orientacao e deduzida da velocidade:
//   yaw = atan2(vy,vx), pitch = atan2(-vz, hypot(vx,vy)), roll = 0.
// Com quaternions medidos (x,y,z,w; corpo->NED), usa-se esses, interpolados por nlerp.
class Trajectory {
public:
    Trajectory(std::vector<double> times, std::vector<Vec3> pos)
        : t_(std::move(times)), p_(std::move(pos)) {
        if (t_.size() != p_.size() || t_.size() < 2)
            throw std::invalid_argument("Trajectory: need >=2 samples, same length");
        for (size_t i = 1; i < t_.size(); ++i)
            if (!(t_[i] > t_[i - 1]))
                throw std::invalid_argument("Trajectory: times must be strictly increasing");
        compute_attitude();
    }
    Trajectory(std::vector<double> times, std::vector<Vec3> pos, std::vector<std::array<double, 4>> quats)
        : Trajectory(std::move(times), std::move(pos)) {
        if (quats.size() != p_.size()) throw std::invalid_argument("Trajectory: quats size mismatch");
        for (auto& q : quats) {
            double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            if (n < 1e-12) throw std::invalid_argument("Trajectory: zero quaternion");
            for (double& c : q) c /= n;
        }
        q_ = std::move(quats);
    }
    double t_start() const { return t_.front(); }
    double t_end() const { return t_.back(); }
    size_t size() const { return t_.size(); }

    Pose pose_at(double t) const {
        size_t i; double s;
        locate(t, i, s);
        Vec3 pos = p_[i] + (p_[i + 1] - p_[i]) * s;
        if (!q_.empty()) {
            // nlerp: q = normalizar((1-s) a + s b), com b -> -b se a.b < 0 (caminho curto)
            const auto &a = q_[i], &b = q_[i + 1];
            double d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
            double sg = (d < 0) ? -1.0 : 1.0;
            double q[4];
            for (int k = 0; k < 4; ++k) q[k] = (1.0 - s) * a[k] + s * sg * b[k];
            return Pose(from_quaternion(q[0], q[1], q[2], q[3]), pos);
        }
        double dyaw = wrap(yaw_[i + 1] - yaw_[i]);
        double yaw = yaw_[i] + s * dyaw;
        double pitch = pitch_[i] + s * (pitch_[i + 1] - pitch_[i]);
        return Pose(from_euler_zyx(0.0, pitch, yaw), pos);
    }
    double speed_at(double t) const {
        size_t i; double s;
        locate(t, i, s);
        return (p_[i + 1] - p_[i]).norm() / (t_[i + 1] - t_[i]);
    }

private:
    std::vector<double> t_;
    std::vector<Vec3> p_;
    std::vector<double> yaw_, pitch_;
    std::vector<std::array<double, 4>> q_;   // vazio => orientacao pela velocidade

    static double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

    void locate(double t, size_t& i, double& s) const {
        t = std::min(std::max(t, t_.front()), t_.back());
        auto it = std::upper_bound(t_.begin(), t_.end(), t);
        i = (it == t_.begin()) ? 0 : size_t(it - t_.begin()) - 1;
        if (i + 1 >= t_.size()) i = t_.size() - 2;
        s = (t - t_[i]) / (t_[i + 1] - t_[i]);
    }
    void compute_attitude() {
        size_t n = t_.size();
        yaw_.assign(n, 0.0);
        pitch_.assign(n, 0.0);
        std::vector<bool> ok(n, false);
        for (size_t j = 0; j < n; ++j) {
            size_t a = (j == 0) ? 0 : j - 1, b = (j + 1 == n) ? n - 1 : j + 1;
            Vec3 v = (p_[b] - p_[a]) * (1.0 / (t_[b] - t_[a]));
            if (v.norm() > 1e-6) {
                yaw_[j] = std::atan2(v.y, v.x);
                pitch_[j] = std::atan2(-v.z, std::hypot(v.x, v.y));
                ok[j] = true;
            }
        }
        // Parado: nao ha direcao. Mantem a do ponto anterior (ou do primeiro em movimento).
        size_t first = 0;
        while (first < n && !ok[first]) ++first;
        if (first == n) return;
        for (size_t j = 0; j < first; ++j) { yaw_[j] = yaw_[first]; pitch_[j] = pitch_[first]; }
        for (size_t j = first + 1; j < n; ++j)
            if (!ok[j]) { yaw_[j] = yaw_[j - 1]; pitch_[j] = pitch_[j - 1]; }
    }
};

}  // namespace uw
