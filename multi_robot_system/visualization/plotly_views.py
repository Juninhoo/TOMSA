"""Motor de Renderização Tridimensional em Plotly para Vistas Global 3D, POV de Câmara e Linhas Temporais.

Implementa:
1. Vista Global 3D com batimetria (seabed), plano da água (superfície) e rasto progressivo por robô.
2. Vista em 1ª Pessoa (POV) com câmara orientada pelo submarino ativo e mira tática sobre os outros veículos no FOV.
3. Gráfico 2D da evolução temporal das distâncias relativas com limiar crítico de 10m.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go

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
            x=[curr_e], y=[curr_n], z=[curr_d],
            mode="markers+text",
            name=f"{r_id} (Atual)",
            marker=dict(size=marker_size, color=traj.color, symbol="diamond",
                        line=dict(color="#ffffff" if is_sel else traj.color, width=2)),
            text=[f"<b>{r_id}</b><br>{spd:.1f} m/s"],
            textposition="top center",
            hovertemplate=f"<b>{lbl}</b><br>N: {curr_n:.1f} m | E: {curr_e:.1f} m<br>Prof: {curr_d:.1f} m | Proa: {yaw_deg:.1f}°<extra></extra>"
        ))

        # Trace 4 + 3*i: Modelo 3D do Submarino (Casco, asas e leme orientados em 3D)
        sub_x, sub_y, sub_z = compute_submarine_wireframe(curr_e, curr_n, curr_d, yaw_deg, pitch_deg)
        fig.add_trace(go.Scatter3d(
            x=sub_x, y=sub_y, z=sub_z,
            mode="lines",
            name=f"{r_id} (Casco 3D)",
            showlegend=False,
            line=dict(color="#ffffff" if is_sel else traj.color, width=4)
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
# 3. Vista em 1ª Pessoa (POV) - a mesma cena 3D, com a câmara dentro do robô ativo
# -------------------------------------------------------------
def create_pov_camera_figure(
    dataset: MissionDataset,
    current_time: float,
    active_robot_id: str,
    fov_deg: float = 70.0
) -> go.Figure:
    """Cena 3D idêntica à Vista Global (mesmos traces e ordem), usada em 1ª pessoa.

    A câmara (eye/center/up) é posicionada em JS a cada frame no nariz do robô ativo, a olhar na direção da
    proa e com o 'up' do corpo (roll). O casco do próprio robô fica invisível; os outros são desenhados maiores.
    """
    fig = create_global_3d_figure(dataset, current_time, active_robot_id)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    return fig


# -------------------------------------------------------------
# 4. Gráfico Temporal 2D de Proximidade (com limiar 10m)
# -------------------------------------------------------------
def create_proximity_timeline_figure(
    dataset: MissionDataset,
    current_time: float,
    collision_threshold_m: float = 10.0,
    dt_step_s: float = 2.0
) -> go.Figure:
    """Gera o gráfico 2D da evolução temporal das distâncias relativas entre robôs."""
    fig = go.Figure()
    start_t = dataset.t0_mission
    end_t = dataset.t_end_mission
    times = np.arange(start_t, end_t, dt_step_s)

    r_ids = sorted(dataset.trajectories.keys())
    pairs = [(r_ids[i], r_ids[j]) for i in range(len(r_ids)) for j in range(i + 1, len(r_ids))]
    t_rel_list = times - dataset.t0_mission

    # Posições interpoladas (NaN fora da janela de operação de cada robô)
    pos = {}
    for r_id, traj in dataset.trajectories.items():
        p = np.vstack([np.interp(times, traj.timestamps, traj.pos_ned[k]) for k in range(3)])
        p[:, (times < traj.timestamps[0]) | (times > traj.timestamps[-1])] = np.nan
        pos[r_id] = p
    distances = {f"{p1} - {p2}": np.linalg.norm(pos[p1] - pos[p2], axis=0) for p1, p2 in pairs}

    colors = ["#f4a261", "#e76f51", "#2a9d8f", "#e63946"]
    for idx, (label, dist_vals) in enumerate(distances.items()):
        fig.add_trace(go.Scatter(
            x=t_rel_list, y=dist_vals,
            mode="lines",
            name=label,
            line=dict(color=colors[idx % len(colors)], width=2.5)
        ))

    # Limiar crítico de colisão (< 10m conforme Q39)
    fig.add_hline(
        y=collision_threshold_m,
        line_dash="dash",
        line_color="#e63946",
        annotation_text="LIMIAR CRÍTICO (< 10m)",
        annotation_position="bottom right",
        annotation_font_color="#e63946"
    )

    # Marcador de tempo atual
    curr_t_rel = current_time - dataset.t0_mission
    fig.add_vline(
        x=curr_t_rel,
        line_color="#00b4d8",
        line_width=2,
        annotation_text="Tempo Atual",
        annotation_position="top"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        title="<b>Evolução Temporal da Distância Relativa Inter-Robôs</b>",
        xaxis_title="Tempo Decorrido na Missão (s)",
        yaxis_title="Distância de Separação (m)",
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=40)
    )
    return fig
