"""Motor de Renderização Tridimensional em Plotly para Vistas Global 3D, POV de Câmara e Linhas Temporais.

Implementa:
1. Vista Global 3D com batimetria (seabed), plano da água (superfície) e rasto progressivo por robô.
2. Vista em 1ª Pessoa (POV) com câmara orientada pelo submarino ativo e mira tática sobre os outros veículos no FOV.
3. Gráfico 2D da evolução temporal das distâncias relativas com limiar crítico de 10m.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multi_robot_system.core.data.trajectory import MissionDataset
from multi_robot_system.core.math.proximity import CPAResult
from multi_robot_system.core.math.rotations import euler_zyx_to_rotation_matrix


# -----------------------------------------------------------------------------
# 1. Grelhas de Fundo do Mar e Superfície da Água
# -----------------------------------------------------------------------------
def build_seabed_and_surface_traces(
    center_e: float,
    center_n: float,
    radius_m: float,
    seabed_depth_m: float
) -> Tuple[go.Scatter3d, go.Scatter3d]:
    """Gera as grelhas wireframe da superfície da água (D=0) e do fundo do mar."""
    grid_res = 12
    e_vals = np.linspace(center_e - radius_m, center_e + radius_m, grid_res)
    n_vals = np.linspace(center_n - radius_m, center_n + radius_m, grid_res)

    lines_surface_e, lines_surface_n, lines_surface_d = [], [], []
    lines_seabed_e, lines_seabed_n, lines_seabed_d = [], [], []

    for e in e_vals:
        lines_surface_e.extend([e, e, None])
        lines_surface_n.extend([n_vals[0], n_vals[-1], None])
        lines_surface_d.extend([0.0, 0.0, None])

        lines_seabed_e.extend([e, e, None])
        lines_seabed_n.extend([n_vals[0], n_vals[-1], None])
        lines_seabed_d.extend([seabed_depth_m, seabed_depth_m, None])

    for n in n_vals:
        lines_surface_e.extend([e_vals[0], e_vals[-1], None])
        lines_surface_n.extend([n, n, None])
        lines_surface_d.extend([0.0, 0.0, None])

        lines_seabed_e.extend([e_vals[0], e_vals[-1], None])
        lines_seabed_n.extend([n, n, None])
        lines_seabed_d.extend([seabed_depth_m, seabed_depth_m, None])

    trace_surface = go.Scatter3d(
        x=lines_surface_e, y=lines_surface_n, z=lines_surface_d,
        mode="lines",
        line=dict(color="rgba(0, 180, 216, 0.35)", width=1.5),
        name="Superfície do Mar (D=0m)",
        hoverinfo="skip"
    )

    trace_seabed = go.Scatter3d(
        x=lines_seabed_e, y=lines_seabed_n, z=lines_seabed_d,
        mode="lines",
        line=dict(color="rgba(108, 117, 125, 0.4)", width=1.5),
        name=f"Fundo do Mar ({seabed_depth_m:.1f}m)",
        hoverinfo="skip"
    )

    return trace_surface, trace_seabed


def compute_submarine_wireframe(
    e: float, n: float, d: float,
    yaw_deg: float, pitch_deg: float,
    length: float = 8.0, wingspan: float = 3.5, height: float = 2.5
) -> Tuple[List[float], List[float], List[float]]:
    """Calcula a geometria tridimensional do casco do submarino (nariz, cauda, aletas e leme dorsal)."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    fe = np.sin(yaw) * np.cos(pitch)
    fn = np.cos(yaw) * np.cos(pitch)
    fd = -np.sin(pitch)

    re = np.cos(yaw)
    rn = -np.sin(yaw)
    rd = 0.0

    ue = np.sin(yaw) * np.sin(pitch)
    un = np.cos(yaw) * np.sin(pitch)
    ud = -np.cos(pitch)

    nose = np.array([e + length * fe, n + length * fn, d + length * fd])
    stern = np.array([e - 0.5 * length * fe, n - 0.5 * length * fn, d - 0.5 * length * fd])
    port = stern - 0.5 * wingspan * np.array([re, rn, rd])
    stbd = stern + 0.5 * wingspan * np.array([re, rn, rd])
    rudder = stern + height * np.array([ue, un, ud])

    pts = np.vstack([nose, stern, port, stbd, stern, rudder, nose])
    return pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist()


# Eixos do referencial do corpo (Body Frame): x = proa (frente), y = estibordo (direita), z = quilha (baixo).
# Convenção de cores habitual em robótica: x vermelho, y verde, z azul.
BODY_AXIS_NAMES = ("x", "y", "z")
BODY_AXIS_COLORS = ("#ff3b3b", "#2ee66b", "#3b8bff")
BODY_AXIS_LENGTH_M = 10.0


def compute_body_axes(
    e: float, n: float, d: float,
    roll_deg: float, pitch_deg: float, yaw_deg: float,
    length: float = BODY_AXIS_LENGTH_M
) -> List[Tuple[List[float], List[float], List[float]]]:
    """Calcula as 3 setas (x, y, z) do referencial do corpo, coladas ao robô e rodadas com a sua atitude.

    Cada coluna da matriz de rotação R = Rz(yaw) Ry(pitch) Rx(roll) é um eixo do corpo escrito no
    referencial NED. Devolve, para cada eixo, as listas (E, N, D) do ponto de origem (centro do robô)
    e do ponto da ponta da seta, prontas para o Scatter3d (x = Este, y = Norte, z = Profundidade).
    """
    r, p, y = np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)
    R = euler_zyx_to_rotation_matrix(r, p, y)
    origin = np.array([n, e, d])  # ordem NED
    axes = []
    for k in range(3):
        tip = origin + length * R[:, k]
        axes.append(([e, float(tip[1])], [n, float(tip[0])], [d, float(tip[2])]))
    return axes


def get_scene_geometry(dataset: MissionDataset, seabed_margin_m: float = 5.0, top_margin_m: float = 3.0) -> dict:
    """Caixa 3D partilhada pela Vista Global e pela Vista POV (eixos fixos, escala métrica 1:1).

    O JS usa estes valores para converter metros -> coordenadas normalizadas da câmara do Plotly.
    """
    all_n = np.concatenate([t.pos_ned[0] for t in dataset.trajectories.values()])
    all_e = np.concatenate([t.pos_ned[1] for t in dataset.trajectories.values()])
    all_d = np.concatenate([t.pos_ned[2] for t in dataset.trajectories.values()])
    radius = float(max(np.ptp(all_n), np.ptp(all_e))) / 2.0 + 30.0
    d_bottom = float(np.max(all_d)) + seabed_margin_m
    d_top = -top_margin_m
    return {
        "cE": float(np.mean(all_e)), "cN": float(np.mean(all_n)),
        "cD": (d_bottom + d_top) / 2.0, "radius": radius, "span": 2.0 * radius,
        "d_bottom": d_bottom, "d_top": d_top,
        "ratio_z": (d_bottom - d_top) / (2.0 * radius),
    }


# -----------------------------------------------------------------------------
# 2. Vista Global 3D (Com rasto progressivo)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Modelo 3D do submarino (casco de torpedo com 4 aletas na cauda), partilhado entre Python e JavaScript
# -----------------------------------------------------------------------------
SUB_HALF_LENGTH_M = 5.0     # o casco vai de -5 m (cauda) a +5 m (proa) em torno da posição do robot
SUB_RADIUS_M = 1.15
_SUB_MESH_CACHE = {}


def _sub_radius(x: float) -> float:
    """Perfil do casco: nariz elíptico, corpo cilíndrico, cauda a afinar."""
    R = SUB_RADIUS_M
    if x > 3.0:
        return R * float(np.sqrt(max(0.0, 1.0 - ((x - 3.0) / 2.0) ** 2)))
    if x < -2.0:
        return R * max(0.0, 1.0 - ((-2.0 - x) / 3.0) ** 2)
    return R


def build_submarine_mesh(n_phi: int = 14) -> dict:
    """Malha do submarino no referencial do corpo (x proa, y estibordo, z quilha), em metros.

    Devolve listas x, y, z (vértices) e i, j, k (triângulos). As aletas têm vértices duplicados com a
    ordem invertida, para se verem iluminadas dos dois lados.
    """
    if n_phi in _SUB_MESH_CACHE:
        return _SUB_MESH_CACHE[n_phi]
    L = SUB_HALF_LENGTH_M
    stations = np.unique(np.concatenate([np.linspace(-L, -2.0, 8), np.linspace(-2.0, 3.0, 6), np.linspace(3.0, L, 9)]))[1:-1]
    verts = [(-L, 0.0, 0.0)]                                   # ponta da cauda
    for x in stations:
        r = _sub_radius(float(x))
        for q in range(n_phi):
            a = 2.0 * np.pi * q / n_phi
            verts.append((float(x), r * float(np.cos(a)), r * float(np.sin(a))))
    verts.append((L, 0.0, 0.0))                                # ponta da proa
    tris = []
    ns = len(stations)
    ring = lambda s, q: 1 + s * n_phi + (q % n_phi)
    for q in range(n_phi):                                     # leque da cauda
        tris.append((0, ring(0, q + 1), ring(0, q)))
    for s in range(ns - 1):                                    # corpo
        for q in range(n_phi):
            a, b, c, d = ring(s, q), ring(s, q + 1), ring(s + 1, q + 1), ring(s + 1, q)
            tris.append((a, b, c))
            tris.append((a, c, d))
    last = len(verts) - 1
    for q in range(n_phi):                                     # leque da proa
        tris.append((last, ring(ns - 1, q), ring(ns - 1, q + 1)))
    # Aletas em cruz na cauda: dorsal, ventral, bombordo, estibordo
    for (uy, uz) in ((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)):
        quad = [(-3.2, _sub_radius(-3.2)), (-4.9, _sub_radius(-4.9)), (-4.8, _sub_radius(-4.8) + 1.7), (-4.0, _sub_radius(-4.0) + 1.7)]
        base = len(verts)
        for x, rad in quad:
            verts.append((x, uy * rad, uz * rad))
        for x, rad in quad:
            verts.append((x, uy * rad, uz * rad))
        tris.append((base, base + 1, base + 2)); tris.append((base, base + 2, base + 3))
        tris.append((base + 4, base + 6, base + 5)); tris.append((base + 4, base + 7, base + 6))
    mesh = {
        "x": [round(v[0], 3) for v in verts], "y": [round(v[1], 3) for v in verts], "z": [round(v[2], 3) for v in verts],
        "i": [t[0] for t in tris], "j": [t[1] for t in tris], "k": [t[2] for t in tris],
    }
    _SUB_MESH_CACHE[n_phi] = mesh
    return mesh


def orient_submarine_mesh(mesh: dict, e: float, n: float, d: float, roll_deg: float, pitch_deg: float, yaw_deg: float, scale: float = 1.0):
    """Roda a malha com a atitude (roll, pitch, yaw) e põe-na na posição (Este, Norte, Profundidade). Devolve x, y, z do Plotly."""
    ph, th, ps = np.radians([roll_deg, pitch_deg, yaw_deg])
    cph, sph, cth, sth, cps, sps = np.cos(ph), np.sin(ph), np.cos(th), np.sin(th), np.cos(ps), np.sin(ps)
    R = np.array([
        [cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
        [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
        [-sth,      sph * cth,                   cph * cth],
    ])   # corpo para NED (Rz yaw, Ry pitch, Rx roll): as colunas são os eixos x, y, z do corpo
    body = np.vstack([mesh["x"], mesh["y"], mesh["z"]]) * scale
    ned = R @ body + np.array([[n], [e], [d]])
    return ned[1].tolist(), ned[0].tolist(), ned[2].tolist()


def _mix_white(hex_color: str, f: float) -> str:
    """Clareia uma cor #rrggbb misturando com branco (f de 0 a 1)."""
    c = hex_color.lstrip("#")
    r, g, b = (int(c[k:k + 2], 16) for k in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(round(v + (255 - v) * f)) for v in (r, g, b))


def create_global_3d_figure(
    dataset: MissionDataset,
    current_time: float,
    selected_robot_id: str,
    seabed_margin_m: float = 5.0
) -> go.Figure:
    """Constrói a cena 3D panorâmica da missão no instante current_time.

    O rasto da trajetória desenha-se progressivamente (sem rasto no início, revelando-se com o avanço temporal).
    Traces:
      - 0: Superfície
      - 1: Fundo do mar
      - Para cada robô i (ordenado):
          2 + 3*i: Linha de Rasto (Progressivo)
          3 + 3*i: Marcador de Posição
          4 + 3*i: Casco 3D do Submarino (Modelo Tridimensional Completo)
      - 2 + 3*N + i: Rota completa de cada robô (fantasma ténue)
      - 2 + 4*N + 3*i + a: Seta do eixo a (0 = x proa, 1 = y estibordo, 2 = z quilha) do robô i,
        colada ao robô e rodada com a sua atitude (roll, pitch, yaw)
      - 2 + 7*N + i: Previsão a 60 s do robô i (reta com a velocidade estimada; preenchida em JS)
    (N = número de robôs. A ordem é usada pelo JavaScript do dashboard em web_server.py.)
    """
    fig = go.Figure()

    geo = get_scene_geometry(dataset, seabed_margin_m)
    center_n, center_e, max_range, seabed_d = geo["cN"], geo["cE"], geo["radius"], geo["d_bottom"]

    # Superfície e Fundo (Traces 0 e 1)
    tr_surf, tr_sea = build_seabed_and_surface_traces(center_e, center_n, max_range, seabed_d)
    fig.add_trace(tr_surf)
    fig.add_trace(tr_sea)

    # Estados dos robôs no instante atual
    current_states = dataset.get_state_at_time(current_time)

    # Trajetórias progressivas de cada robô em ordem estritamente determinística
    sorted_robots = sorted(dataset.trajectories.keys())
    body_pose = {}  # posição e atitude de cada robô, para desenhar os eixos x, y, z no fim
    for r_id in sorted_robots:
        traj = dataset.trajectories[r_id]
        # Filtra pontos até ao tempo atual (rasto progressivo conforme Q9)
        mask = traj.timestamps <= current_time
        pts_n = traj.pos_ned[0, mask].tolist()
        pts_e = traj.pos_ned[1, mask].tolist()
        pts_d = traj.pos_ned[2, mask].tolist()

        if len(pts_e) == 0:
            pts_n = [float(traj.pos_ned[0, 0])]
            pts_e = [float(traj.pos_ned[1, 0])]
            pts_d = [float(traj.pos_ned[2, 0])]

        # Trace 2 + 3*i: Rasto progressivo
        fig.add_trace(go.Scatter3d(
            x=pts_e, y=pts_n, z=pts_d,
            mode="lines",
            name=f"{r_id} (Rasto)",
            line=dict(color=traj.color, width=4),
            hovertemplate=f"<b>{r_id}</b><br>Este: %{{x:.1f}} m<br>Norte: %{{y:.1f}} m<br>Profundidade: %{{z:.1f}} m<extra></extra>"
        ))

        # Obter estado no tempo atual
        if r_id in current_states:
            st = current_states[r_id]
            curr_n, curr_e, curr_d = float(st["pos_ned"][0]), float(st["pos_ned"][1]), float(st["pos_ned"][2])
            roll_deg = float(st["euler_deg"][0])
            yaw_deg = float(st["euler_deg"][2])
            pitch_deg = float(st["euler_deg"][1])
            spd = float(st["speed_longitudinal"])
            lbl = st["label"]
        else:
            curr_n, curr_e, curr_d = float(traj.pos_ned[0, 0]), float(traj.pos_ned[1, 0]), float(traj.pos_ned[2, 0])
            roll_deg = float(traj.euler_deg[0, 0])
            yaw_deg = float(traj.euler_deg[0, 2])
            pitch_deg = float(traj.euler_deg[0, 1])
            spd = 0.0
            lbl = traj.label

        body_pose[r_id] = (curr_e, curr_n, curr_d, roll_deg, pitch_deg, yaw_deg)

        is_sel = (r_id == selected_robot_id)
        marker_size = 8 if is_sel else 6

        # Trace 3 + 3*i: Marcador de posição do veículo
        fig.add_trace(go.Scatter3d(
            x=[curr_e], y=[curr_n], z=[curr_d - 2.2],
            mode="markers+text",
            name=f"{r_id} (Atual)",
            marker=dict(size=marker_size, color=traj.color, symbol="diamond",
                        line=dict(color="#ffffff" if is_sel else traj.color, width=2)),
            text=[f"<b>{r_id}</b><br>{spd:.1f} m/s"],
            textposition="top center",
            hovertemplate=f"<b>{lbl}</b><br>N: {curr_n:.1f} m | E: {curr_e:.1f} m<br>Prof: {curr_d:.1f} m | Proa: {yaw_deg:.1f}°<extra></extra>"
        ))

        # Trace 4 + 3*i: Modelo 3D do Submarino (Casco, asas e leme orientados em 3D)
        mesh = build_submarine_mesh()
        sub_x, sub_y, sub_z = orient_submarine_mesh(mesh, curr_e, curr_n, curr_d, roll_deg, pitch_deg, yaw_deg)
        fig.add_trace(go.Mesh3d(
            x=sub_x, y=sub_y, z=sub_z, i=mesh["i"], j=mesh["j"], k=mesh["k"],
            name=f"{r_id} (Casco 3D)",
            showlegend=False,
            hoverinfo="skip",
            color=_mix_white(traj.color, 0.4) if is_sel else traj.color,
            flatshading=False,
            lighting=dict(ambient=0.55, diffuse=0.85, specular=0.35, roughness=0.45, fresnel=0.2),
            lightposition=dict(x=0, y=-2000, z=3000)
        ))

    # Rota completa de cada robô (fantasma ténue) - traces 2 + 3*N + i. O rasto brilhante revela-se por cima.
    for r_id in sorted_robots:
        traj = dataset.trajectories[r_id]
        step = max(1, traj.pos_ned.shape[1] // 1500)
        fig.add_trace(go.Scatter3d(
            x=traj.pos_ned[1, ::step].tolist(), y=traj.pos_ned[0, ::step].tolist(), z=traj.pos_ned[2, ::step].tolist(),
            mode="lines", name=f"{r_id} (Rota completa)", showlegend=False, hoverinfo="skip",
            opacity=0.22, line=dict(color=traj.color, width=2)
        ))

    # Eixos x, y, z do corpo de cada robô (traces 2 + 4*N + 3*i + a): a origem fica no centro do robô e as
    # setas rodam com ele. A letra do eixo aparece na ponta da seta.
    for r_id in sorted_robots:
        e0, n0, d0, roll0, pitch0, yaw0 = body_pose[r_id]
        for a, (ax_x, ax_y, ax_z) in enumerate(compute_body_axes(e0, n0, d0, roll0, pitch0, yaw0)):
            fig.add_trace(go.Scatter3d(
                x=ax_x, y=ax_y, z=ax_z,
                mode="lines+text",
                name=f"{r_id} (eixo {BODY_AXIS_NAMES[a]})",
                showlegend=False,
                hoverinfo="skip",
                text=["", f"<b>{BODY_AXIS_NAMES[a]}</b>"],
                textposition="top center",
                textfont=dict(color=BODY_AXIS_COLORS[a], size=13),
                line=dict(color=BODY_AXIS_COLORS[a], width=7)
            ))

    # Previsão a 60 s (traces 2 + 7*N + i): começa vazia, o JavaScript preenche-a a cada instante
    for r_id in sorted_robots:
        traj = dataset.trajectories[r_id]
        e0, n0, d0 = body_pose[r_id][0], body_pose[r_id][1], body_pose[r_id][2]
        fig.add_trace(go.Scatter3d(
            x=[e0], y=[n0], z=[d0], mode="lines", name=f"{r_id} (Previsão 60 s)", showlegend=False,
            hoverinfo="skip", line=dict(color=traj.color, width=5, dash="dash")
        ))

    # Layout Dark Mode com Profundidade Invertida (autoranging reversed em Z para maior profundidade = mais em baixo)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        scene=dict(
            xaxis=dict(title="Este (m)", range=[center_e - max_range, center_e + max_range], backgroundcolor="#111827", gridcolor="#374151"),
            yaxis=dict(title="Norte (m)", range=[center_n - max_range, center_n + max_range], backgroundcolor="#111827", gridcolor="#374151"),
            # intervalo [fundo, superfície]: maior profundidade em baixo
            zaxis=dict(title="Profundidade (m)", range=[geo["d_bottom"], geo["d_top"]], backgroundcolor="#111827", gridcolor="#374151"),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=geo["ratio_z"])
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(17, 24, 39, 0.7)")
    )
    return fig


# -------------------------------------------------------------
# 3. Vista em 3ª Pessoa - a mesma cena 3D, com a câmara atrás e acima do robô ativo
# -------------------------------------------------------------
def create_pov_camera_figure(
    dataset: MissionDataset,
    current_time: float,
    active_robot_id: str,
    fov_deg: float = 70.0
) -> go.Figure:
    """Cena 3D idêntica à Vista Global (mesmos traces e ordem), usada na Vista 3ª Pessoa.

    A câmara (eye/center) é posicionada em JS a cada frame atrás e acima do robô ativo, a olhar para a proa
    (rumo alisado). Por cima da cena, o dashboard desenha um radar (canto superior esquerdo) e um painel de
    estatísticas do robô ativo (canto inferior esquerdo). O nome da função mantém-se por compatibilidade.
    """
    fig = create_global_3d_figure(dataset, current_time, active_robot_id)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig


# -------------------------------------------------------------
# 4. Gráfico Temporal 2D de Proximidade (com limiar 10m)
# -------------------------------------------------------------
PROX_PAIR_COLORS = ["#9085e9", "#d55181", "#e0a030"]   # uma cor por par de robots (distinguíveis em fundo escuro)


def _fmt_mission_time(sec: float) -> str:
    """Segundos como minutos e segundos (62m 9.0s), igual ao relógio do navegador."""
    m = int(sec // 60)
    return f"{m}m {sec - 60 * m:.1f}s"


def _add_hline(fig: go.Figure, y: float, color: str, text: str, below: bool) -> None:
    """Linha horizontal tracejada com rótulo no canto direito (escrita à mão: add_hline não funciona antes de haver traços)."""
    fig.add_shape(type="line", xref="x domain", yref="y", x0=0, x1=1, y0=y, y1=y,
                  line=dict(color=color, dash="dash", width=1.5))
    fig.add_annotation(xref="x domain", yref="y", x=1, y=y, text=text, showarrow=False, xanchor="right",
                       yanchor="top" if below else "bottom", font=dict(color=color, size=11))


def _add_band(fig: go.Figure, y0: float, y1: float, color: str, opacity: float) -> None:
    """Faixa horizontal colorida (zona de perigo ou de aviso) atrás das curvas, escrita à mão como _add_hline."""
    fig.add_shape(type="rect", xref="x domain", yref="y", x0=0, x1=1, y0=y0, y1=y1,
                  fillcolor=color, opacity=opacity, line_width=0, layer="below")


def create_proximity_timeline_figure(
    dataset: MissionDataset,
    current_time: float,
    collision_threshold_m: float = 10.0,
    dt_step_s: float = 2.0,
    warning_threshold_m: float = 30.0
) -> go.Figure:
    """Gráfico 2D da evolução temporal das distâncias relativas entre robôs.

    - Uma cor por par, com o nome escrito no fim de cada linha e o mínimo de cada par assinalado.
    - Faixas de aviso (abaixo de warning_threshold_m) e de perigo (abaixo de collision_threshold_m).
    - Por baixo, uma barra por robô a mostrar quando tem dados (fora dela o par não existe, não é um erro).
    - Botões para mudar a escala vertical: 0 a 400 m, ampliação 0 a 60 m e logarítmica.
    IMPORTANTE: a ordem das formas (shapes) é usada pelo JavaScript: shapes[1] é o cursor do tempo atual.
    """
    start_t = dataset.t0_mission
    end_t = dataset.t_end_mission
    times = np.arange(start_t, end_t, dt_step_s)

    r_ids = sorted(dataset.trajectories.keys())
    pairs = [(r_ids[i], r_ids[j]) for i in range(len(r_ids)) for j in range(i + 1, len(r_ids))]
    t_rel_list = times - dataset.t0_mission

    # Posições interpoladas (NaN fora da janela de operação de cada robô)
    pos = {}
    pos_raw = {}   # posições brutas (sem RTS), se existirem
    for r_id, traj in dataset.trajectories.items():
        p = np.vstack([np.interp(times, traj.timestamps, traj.pos_ned[k]) for k in range(3)])
        p[:, (times < traj.timestamps[0]) | (times > traj.timestamps[-1])] = np.nan
        pos[r_id] = p
        if getattr(traj, "pos_ned_raw", None) is not None:
            q = np.vstack([np.interp(times, traj.timestamps, traj.pos_ned_raw[k]) for k in range(3)])
            q[:, (times < traj.timestamps[0]) | (times > traj.timestamps[-1])] = np.nan
            pos_raw[r_id] = q

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18], vertical_spacing=0.06)

    # Limiar crítico (forma 0) e cursor do tempo atual (forma 1): esta ordem não pode mudar
    _add_hline(fig, collision_threshold_m, "#e63946", f"CRÍTICO (< {collision_threshold_m:g} m)", below=True)
    curr_t_rel = current_time - dataset.t0_mission
    fig.add_shape(type="line", xref="x", yref="paper", x0=curr_t_rel, x1=curr_t_rel, y0=0, y1=1,
                  line=dict(color="#00b4d8", width=2))

    # Faixas de perigo e de aviso e linha dos 30 m
    _add_band(fig, 0.01, collision_threshold_m, "#e63946", 0.16)
    _add_band(fig, collision_threshold_m, warning_threshold_m, "#f4a261", 0.12)
    _add_hline(fig, warning_threshold_m, "#f4a261", f"AVISO (< {warning_threshold_m:g} m)", below=False)

    # Uma curva por par, com rótulo direto no fim e o mínimo assinalado
    smooth_idx, marker_idx, raw_idx = [], [], []   # índices dos traços, para o botão suavizado / bruto
    for idx, (p1, p2) in enumerate(pairs):
        dist = np.linalg.norm(pos[p1] - pos[p2], axis=0)
        color = PROX_PAIR_COLORS[idx % len(PROX_PAIR_COLORS)]
        short = f"{p1.replace('Robot ', 'R')} – {p2.replace('Robot ', 'R')}"
        fig.add_trace(go.Scatter(
            x=t_rel_list, y=dist, mode="lines", name=f"{p1} – {p2}",
            line=dict(color=color, width=2.5),
            hovertemplate="<b>" + short + "</b><br>t = %{x:.0f} s<br>distância = %{y:.1f} m<extra></extra>"
        ), row=1, col=1)
        smooth_idx.append(len(fig.data) - 1)
        if p1 in pos_raw and p2 in pos_raw:
            dist_raw = np.linalg.norm(pos_raw[p1] - pos_raw[p2], axis=0)
            fig.add_trace(go.Scatter(
                x=t_rel_list, y=dist_raw, mode="lines", name=f"{p1} – {p2} (bruto)", visible=False, showlegend=False,
                line=dict(color=color, width=1.4, dash="dot"),
                hovertemplate="<b>" + short + " (bruto)</b><br>t = %{x:.0f} s<br>distância = %{y:.1f} m<extra></extra>"
            ), row=1, col=1)
            raw_idx.append(len(fig.data) - 1)
        valid = np.where(~np.isnan(dist))[0]
        if len(valid) == 0:
            continue
        i_min, i_end = valid[np.argmin(dist[valid])], valid[-1]
        fig.add_trace(go.Scatter(
            x=[t_rel_list[i_min]], y=[dist[i_min]], mode="markers+text",
            text=[f"mín {dist[i_min]:.1f} m"], textposition="top center", textfont=dict(color="#e0e1dd", size=11),
            marker=dict(size=9, color="#111827", line=dict(color=color, width=2)),
            showlegend=False, hoverinfo="skip"
        ), row=1, col=1)
        marker_idx.append(len(fig.data) - 1)
        fig.add_annotation(x=t_rel_list[i_end], y=dist[i_end], text="<b>" + short + "</b>", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color="#e0e1dd", size=12), row=1, col=1)

    # Barras de presença: quando cada robô tem dados
    n = len(r_ids)
    for k, r_id in enumerate(r_ids):
        traj = dataset.trajectories[r_id]
        a, b = float(traj.timestamps[0] - dataset.t0_mission), float(traj.timestamps[-1] - dataset.t0_mission)
        fig.add_trace(go.Scatter(
            x=[a, b], y=[n - 1 - k, n - 1 - k], mode="lines", line=dict(color=traj.color, width=9),
            showlegend=False,
            hovertemplate=f"<b>{r_id}</b><br>com dados de {_fmt_mission_time(a)} a {_fmt_mission_time(b)}<extra></extra>"
        ), row=2, col=1)
    fig.update_yaxes(tickmode="array", tickvals=list(range(n)), ticktext=[r.replace("Robot ", "R") for r in reversed(r_ids)],
                     range=[-0.7, n - 0.3], showgrid=False, zeroline=False, title_text="Dados", row=2, col=1)

    fig.update_yaxes(title_text="Distância de separação (m)", range=[0, 410], row=1, col=1)
    fig.update_xaxes(title_text="Tempo decorrido na missão (s)", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        title="<b>Distância entre robots ao longo da missão</b>",
        autosize=True,
        margin=dict(l=60, r=50, t=80, b=50),
        updatemenus=[dict(
            type="buttons", direction="right", x=1.0, xanchor="right", y=1.0, yanchor="bottom",
            bgcolor="#cbd5e1", bordercolor="#94a3b8", font=dict(color="#0b0f19", size=12),
            buttons=[
                dict(label="0 a 400 m", method="relayout", args=[{"yaxis.type": "linear", "yaxis.range": [0, 410]}]),
                dict(label="Ampliar 0 a 60 m", method="relayout", args=[{"yaxis.type": "linear", "yaxis.range": [0, 60]}]),
                dict(label="Escala logarítmica", method="relayout", args=[{"yaxis.type": "log", "yaxis.range": [float(np.log10(5.0)), float(np.log10(500.0))]}]),
            ],
        )],
    )
    if raw_idx:
        managed = smooth_idx + marker_idx + raw_idx

        def _vis(smooth, marker, raw):
            # lista de visibilidade, por ordem: traços suavizados, marcadores, traços brutos (usada pelos 3 botões)
            return [smooth] * len(smooth_idx) + [marker] * len(marker_idx) + [raw] * len(raw_idx)

        menus = list(fig.layout.updatemenus)
        menus.append(dict(
            type="buttons", direction="right", x=0.0, xanchor="left", y=1.0, yanchor="bottom",
            bgcolor="#cbd5e1", bordercolor="#94a3b8", font=dict(color="#0b0f19", size=12),
            buttons=[
                dict(label="Suavizado (RTS)", method="restyle", args=[{"visible": _vis(True, True, False)}, managed]),
                dict(label="Bruto (CSV)", method="restyle", args=[{"visible": _vis(False, False, True)}, managed]),
                dict(label="Ambos", method="restyle", args=[{"visible": _vis(True, True, True)}, managed]),
            ],
        ))
        fig.update_layout(updatemenus=menus)
    return fig
