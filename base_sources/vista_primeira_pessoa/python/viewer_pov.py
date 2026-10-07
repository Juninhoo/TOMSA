"""Visualizador 3D em 1a pessoa (Plotly). Carregar 1, 2, 3... muda o drone.

A matematica (geodesia, quaternions, referenciais, camara) vem do modulo C++ `uwmath`.
Corre um pequeno servidor local que calcula cada vista a pedido e abre o browser.

Uso:  python viewer_pov.py --csv robot1_s1.csv robot4_s1.csv robot7_s1.csv
      python viewer_pov.py            (missoes de demonstracao)
"""
import argparse
import json
import math
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import plotly.graph_objects as go

try:
    import uwmath as uw
except ImportError:
    sys.exit("Falta compilar o modulo C++. Na raiz do projeto:  python setup.py build_ext -b python")

from missions import demo_missions, load_csv

COLORS = ["#ffb703", "#fb5607", "#8ecae6", "#b5e48c", "#e0aaff", "#ff99c8", "#caf0f8", "#f1fa8c", "#adb5bd"]
TRAIL_PTS = 60


# --------------------------------------------------------------- geometria
def grid_lines(cx, cy, z, radius, step):
    """Grelha horizontal em NED a profundidade z, alinhada a multiplos de `step` em volta de (cx, cy)."""
    c = np.arange(-radius, radius + 1e-9, step)
    ox, oy = round(cx / step) * step, round(cy / step) * step
    rows = []
    for g in c:
        rows += [[ox + a, oy + g, z] for a in c] + [[np.nan] * 3]
    for g in c:
        rows += [[ox + g, oy + a, z] for a in c] + [[np.nan] * 3]
    return np.array(rows)


def to_display(cam_xyz):
    """Camara (x direita, y baixo, z frente) -> eixos do Plotly (X direita, Y frente, Z cima)."""
    return np.column_stack([cam_xyz[:, 0], cam_xyz[:, 2], -cam_xyz[:, 1]])


def clean(a):
    a = np.round(a, 2)
    return [None if np.isnan(v) else float(v) for v in a]


def deg(rad):
    return f"{int(round(math.degrees(rad))):+d}°"


def clock(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


# ------------------------------------------------------------------ cena
class Scene:
    def __init__(self, entries, args):
        self.names = [e["name"] for e in entries]
        self.N = len(entries)
        self.t0 = min(float(e["t"][0]) for e in entries)            # origem do tempo (epoch ou 0)
        self.args = args
        self.radius = args.radius
        self.step = args.grid
        self.cam = uw.Camera.from_fov(1280, 720, math.radians(args.fov))

        odom = [e for e in entries if e["kind"] == "odom"]
        self.frame = None
        if odom:
            lat0 = float(np.mean(np.concatenate([e["lla"][:, 0] for e in odom])))
            lon0 = float(np.mean(np.concatenate([e["lla"][:, 1] for e in odom])))
            h0 = args.surface_h if args.surface_h is not None else float(
                np.median(np.concatenate([e["lla"][:20, 2] for e in odom])))
            self.frame = uw.LocalFrameNED(lat0, lon0, h0)
            print(f"origem NED: lat {lat0:.6f}  lon {lon0:.6f}  superficie h = {h0:.2f} m (z NED = -(h - h0))")

        self.trajs = []
        for e in entries:
            t = np.ascontiguousarray(e["t"] - self.t0, float)
            if e["kind"] == "odom":
                ned = self.frame.to_ned_array(np.ascontiguousarray(e["lla"]))
                self.trajs.append(uw.Trajectory(list(t), ned, np.ascontiguousarray(e["quats"])))
            else:
                self.trajs.append(uw.Trajectory(list(t), np.ascontiguousarray(e["pos"], float)))
        self.t_end = max(tr.t_end for tr in self.trajs)
        self.T = int(self.t_end / args.dt) + 1
        common0 = max(tr.t_start for tr in self.trajs)
        common1 = min(tr.t_end for tr in self.trajs)
        self.i_init = int(math.ceil(common0 / args.dt)) if common0 < common1 else 0
        print(f"{self.N} drones, {self.T} instantes de {args.dt:g} s ({self.t_end:.0f} s no total)")

    def active(self, j, t):
        return self.trajs[j].t_start <= t <= self.trajs[j].t_end

    def speed(self, j, t, w=5.0):
        tr = self.trajs[j]
        a, b = max(t - w, tr.t_start), min(t + w, tr.t_end)
        if b - a < 1e-9:
            return 0.0
        return (tr.pose_at(b).t - tr.pose_at(a).t).norm() / (b - a)

    def compute(self, k, i):
        """Vista do drone k no instante i: coordenadas de cada traco (eixos Plotly) + HUD."""
        t = i * self.args.dt
        tr_k = self.trajs[k]
        dpose = tr_k.pose_at(t)                                    # fora do intervalo: usa 1a/ultima pose
        wc = lambda arr: to_display(self.cam.world_to_camera_array(dpose, np.ascontiguousarray(arr, float)))
        xs, ys, zs = [], [], []

        def add(arr):
            if arr is None or len(arr) == 0:
                xs.append([]); ys.append([]); zs.append([])
            else:
                d = wc(arr)
                xs.append(clean(d[:, 0])); ys.append(clean(d[:, 1])); zs.append(clean(d[:, 2]))

        zk = dpose.t.z
        for z in (self.args.seabed, 0.0):                          # 0: fundo (opcional), 1: superficie
            ok = z is not None and abs(z - zk) < self.radius
            add(grid_lines(dpose.t.x, dpose.t.y, z, self.radius, self.step) if ok else None)

        lines = []
        for j in range(self.N):
            trj = self.trajs[j]
            if t >= trj.t_start:
                ts = np.linspace(trj.t_start, min(t, trj.t_end), TRAIL_PTS)
                add(np.array(trj.positions_at(list(ts))))
            else:
                add(None)
            col = COLORS[j % len(COLORS)]
            if j == k:
                add(None)
                continue
            if not self.active(j, t):
                add(None)
                when = (f"começa às {clock(self.t0 + trj.t_start)}" if t < trj.t_start
                        else f"terminou às {clock(self.t0 + trj.t_end)}")
                lines.append(f"<span style='color:{col}'>■</span> {self.names[j]}: sem dados ({when})")
                continue
            pj = trj.pose_at(t)
            add(np.array([pj.to_world(uw.Vec3(-1.2, 0, 0)).to_tuple(), pj.t.to_tuple(),
                          pj.to_world(uw.Vec3(1.2, 0, 0)).to_tuple()]))
            c = self.cam.world_to_camera(dpose, pj.t)
            pr = self.cam.project(dpose, pj.t)
            az = math.degrees(math.atan2(c.x, c.z))
            el = math.degrees(math.atan2(-c.y, math.hypot(c.x, c.z)))
            where = f"pixel ({pr.u:.0f}, {pr.v:.0f})" if pr.visible else "fora do campo"
            lines.append(f"<span style='color:{col}'>■</span> {self.names[j]}: {c.norm():.0f} m · "
                         f"az {az:+.0f}° · el {el:+.0f}° · prof. {pj.t.z:.1f} m · {where}")

        roll, pitch, yaw = uw.to_euler_zyx(dpose.R)
        head = (f"<b style='color:{COLORS[k % len(COLORS)]}'>{self.names[k]}</b> &nbsp; "
                f"{clock(self.t0 + t)} UTC &nbsp; (t = {t:.0f} s)<br>")
        if not self.active(k, t):
            head += "<span style='color:#ff6b6b'>sem dados neste instante: a mostrar a pose do registo mais próximo</span><br>"
        head += (f"NED ({dpose.t.x:.1f}, {dpose.t.y:.1f}) m · <b>profundidade {zk:.1f} m</b><br>"
                 f"yaw {deg(yaw)} · pitch {deg(pitch)} · roll {deg(roll)} · {self.speed(k, t):.2f} m/s<br>")
        return dict(f=[xs, ys, zs], hud=head + "<br>".join(lines))


# ---------------------------------------------------------------- pagina
def build_html(scene):
    N = scene.N
    first = scene.compute(0, scene.i_init)
    traces = [go.Scatter3d(name="fundo", mode="lines", line=dict(color="#7a5c3a", width=2)),
              go.Scatter3d(name="superficie", mode="lines", line=dict(color="#7fd1ff", width=1))]
    for j in range(N):
        col = COLORS[j % len(COLORS)]
        traces.append(go.Scatter3d(name=f"rasto {scene.names[j]}", mode="lines", line=dict(color=col, width=4)))
        traces.append(go.Scatter3d(name=scene.names[j], mode="lines+markers+text", line=dict(color=col, width=9),
                                   marker=dict(color=col, size=[0, 8, 0]), text=["", scene.names[j], ""],
                                   textfont=dict(color=col, size=13), textposition="top center"))
    for tr, x, y, z in zip(traces, *first["f"]):
        tr.x, tr.y, tr.z, tr.hoverinfo, tr.showlegend = x, y, z, "skip", False
    R = scene.radius
    ax = dict(range=[-R, R], visible=False, showbackground=False)
    camera = dict(eye=dict(x=0, y=-0.01, z=0), center=dict(x=0, y=0.3, z=0), up=dict(x=0, y=0, z=1),
                  projection=dict(type="perspective"))
    fig = go.Figure(traces)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#04202e", showlegend=False,
                      scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube", bgcolor="#04202e", camera=camera))
    meta = dict(N=N, T=scene.T, names=scene.names, ntraces=len(traces), camera=camera, i_init=scene.i_init,
                first=first, dt=scene.args.dt)
    js = JS.replace("__META__", json.dumps(meta, separators=(",", ":")))
    return fig.to_html(full_html=True, include_plotlyjs=True, post_script=js, default_height="100vh",
                       default_width="100%", config=dict(displaylogo=False, scrollZoom=True))


JS = r"""
(function () {
  var gd = document.getElementById('{plot_id}');
  var M = __META__;
  var cur = 0, ti = M.i_init, playing = false, interval = 100, reqId = 0, cache = {}, order = [];
  var idx = []; for (var q = 0; q < M.ntraces; q++) idx.push(q);

  var css = document.createElement('style');
  css.textContent = "body{margin:0;background:#04202e;font-family:system-ui,sans-serif}" +
    ".hud{position:fixed;left:14px;top:12px;color:#e8f4f8;font-size:13px;line-height:1.5;background:rgba(2,16,24,.65);padding:10px 14px;border-radius:8px;pointer-events:none;max-width:560px}" +
    ".bar{position:fixed;left:0;right:0;bottom:0;display:flex;gap:8px;align-items:center;padding:10px 14px;background:rgba(2,16,24,.75);color:#cfe8f1;font-size:13px}" +
    ".bar button{background:#0b3a52;color:#e8f4f8;border:1px solid #1d6a8f;border-radius:6px;padding:5px 10px;cursor:pointer}" +
    ".bar button.on{background:#ffb703;color:#04202e;border-color:#ffb703;font-weight:700}" +
    ".bar input[type=range]{flex:1}";
  document.head.appendChild(css);
  var hud = document.createElement('div'); hud.className = 'hud'; document.body.appendChild(hud);
  var bar = document.createElement('div'); bar.className = 'bar'; document.body.appendChild(bar);
  var btns = [];
  M.names.forEach(function (n, k) {
    var b = document.createElement('button'); b.textContent = (k + 1) + ' · ' + n;
    b.onclick = function () { select(k); }; bar.appendChild(b); btns.push(b);
  });
  var play = document.createElement('button'); play.textContent = '▶ / ❚❚ (espaço)'; play.onclick = toggle; bar.appendChild(play);
  var slider = document.createElement('input'); slider.type = 'range'; slider.min = 0; slider.max = M.T - 1; slider.value = ti;
  slider.oninput = function () { ti = parseInt(slider.value); render(); }; bar.appendChild(slider);
  var info = document.createElement('span'); info.textContent = '←/→ passo · R vista · +/− velocidade'; bar.appendChild(info);

  function load(k, i) {
    var key = k + '_' + i;
    if (cache[key]) return Promise.resolve(cache[key]);
    return fetch('/frame?k=' + k + '&i=' + i).then(function (r) { return r.json(); }).then(function (d) {
      cache[key] = d; order.push(key); if (order.length > 400) delete cache[order.shift()]; return d;
    });
  }
  function apply(d) {
    Plotly.restyle(gd, {x: d.f[0], y: d.f[1], z: d.f[2]}, idx);
    hud.innerHTML = d.hud; slider.value = ti;
    btns.forEach(function (b, k) { b.className = (k === cur) ? 'on' : ''; });
  }
  function render() {
    var id = ++reqId;
    return load(cur, ti).then(function (d) { if (id === reqId) apply(d); });
  }
  function resetCam() { Plotly.relayout(gd, {'scene.camera': M.camera}); }
  function select(k) { if (k >= 0 && k < M.N) { cur = k; render(); resetCam(); } }
  function step(d) { ti = Math.min(Math.max(ti + d, 0), M.T - 1); return render(); }
  function tick() {
    if (!playing) return;
    if (ti >= M.T - 1) ti = -1;
    step(1).then(function () { if (playing) setTimeout(tick, interval); });
  }
  function toggle() { playing = !playing; if (playing) tick(); }
  window.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var k = e.key;
    if (k >= '1' && k <= '9') select(parseInt(k) - 1);
    else if (k === ' ') { e.preventDefault(); toggle(); }
    else if (k === 'ArrowRight') step(1);
    else if (k === 'ArrowLeft') step(-1);
    else if (k === 'r' || k === 'R') resetCam();
    else if (k === '+' || k === '=') interval = Math.max(interval / 2, 20);
    else if (k === '-') interval = Math.min(interval * 2, 1000);
  });
  apply(M.first);
})();
"""


# ---------------------------------------------------------------- servidor
def make_handler(scene, html):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                self._send(200, "text/html; charset=utf-8", html)
            elif u.path == "/frame":
                try:
                    q = parse_qs(u.query)
                    k = min(max(int(q["k"][0]), 0), scene.N - 1)
                    i = min(max(int(q["i"][0]), 0), scene.T - 1)
                    self._send(200, "application/json", json.dumps(scene.compute(k, i), separators=(",", ":")).encode())
                except Exception as ex:
                    self._send(400, "text/plain", str(ex).encode())
            else:
                self._send(404, "text/plain", b"not found")

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", help="um CSV por drone (ROS Odometry, ou cabecalho t,x,y,z em NED)")
    ap.add_argument("--radius", type=float, help="alcance de visao (m); omissao 150 (CSV) ou 60 (demo)")
    ap.add_argument("--grid", type=float, help="espacamento da grelha (m); omissao 25 (CSV) ou 10 (demo)")
    ap.add_argument("--fov", type=float, default=70.0, help="FOV horizontal da camara virtual (graus), usado no HUD")
    ap.add_argument("--dt", type=float, default=2.0, help="passo do slider (s)")
    ap.add_argument("--surface-h", type=float, help="altura elipsoidal da superficie do mar (m); omissao: estimada")
    ap.add_argument("--seabed", type=float, help="profundidade do fundo (m) para desenhar uma grelha; omissao: nenhuma")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    entries = [load_csv(p) for p in a.csv] if a.csv else demo_missions()
    if len(entries) > 9:
        sys.exit("Maximo de 9 drones (teclas 1-9).")
    real = bool(a.csv)
    a.radius = a.radius or (150.0 if real else 60.0)
    a.grid = a.grid or (25.0 if real else 10.0)
    if not real and a.seabed is None:
        a.seabed = 55.0

    scene = Scene(entries, a)
    html = build_html(scene).encode()
    srv = None
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(scene, html))
            break
        except OSError:
            continue
    if srv is None:
        sys.exit("Nao consegui abrir uma porta local.")
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"a servir em {url}   (Ctrl+C para sair)")
    if not a.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
